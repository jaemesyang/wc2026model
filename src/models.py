"""
models.py — Elo baseline and Poisson goal model.

Elo:
  elo_win_prob(rating_home, rating_away, neutral) -> (p_home, p_draw, p_away)
  update_elo(...)  -> (new_home_rating, new_away_rating)

Poisson:
  PoissonModel.fit(df)
  PoissonModel.predict_scoreline_probs(home, away, neutral) -> (max_goals+1)² matrix
  PoissonModel.predict_outcome_probs(home, away, neutral) -> (p_home, p_draw, p_away)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_CACHE = DATA_DIR / "poisson_params.json"

# --- Elo constants ---

ELO_K = 40
HOME_ADVANTAGE_ELO = 100  # Elo points added when not neutral

# Draw probability peaks at ~0.30 for even matchups and shrinks with rating gap.
# Calibrated so that a 200-point gap gives ~20 % draw probability.
_DRAW_PEAK = 0.30
_DRAW_WIDTH = 200.0


def elo_win_prob(
    rating_home: float,
    rating_away: float,
    neutral: bool = True,
) -> tuple:
    """
    Convert Elo ratings to match outcome probabilities.

    Returns (p_home_win, p_draw, p_away_win) summing to 1.0.
    """
    advantage = 0.0 if neutral else HOME_ADVANTAGE_ELO
    diff = (rating_home + advantage) - rating_away
    expected = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    # Gaussian draw term: wide when diff ≈ 0, decays with rating gap.
    p_draw = _DRAW_PEAK * np.exp(-0.5 * (diff / _DRAW_WIDTH) ** 2)

    p_home = expected - p_draw / 2.0
    p_away = (1.0 - expected) - p_draw / 2.0

    # Floor to keep probabilities legal before normalising.
    p_home = max(0.02, p_home)
    p_away = max(0.02, p_away)
    p_draw = max(0.05, p_draw)

    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def update_elo(
    rating_home: float,
    rating_away: float,
    score_home: int,
    score_away: int,
    neutral: bool = True,
    k: float = ELO_K,
) -> tuple:
    """
    Update Elo ratings after a match.

    Returns (new_rating_home, new_rating_away).
    Uses a goal-difference multiplier matching World Football Elo conventions.
    """
    advantage = 0.0 if neutral else HOME_ADVANTAGE_ELO
    expected_home = 1.0 / (1.0 + 10.0 ** (-((rating_home + advantage) - rating_away) / 400.0))

    if score_home > score_away:
        actual = 1.0
    elif score_home == score_away:
        actual = 0.5
    else:
        actual = 0.0

    gd = abs(score_home - score_away)
    if gd <= 1:
        mult = 1.0
    elif gd == 2:
        mult = 1.5
    else:
        mult = (11.0 + gd) / 8.0

    delta = k * mult * (actual - expected_home)
    return rating_home + delta, rating_away - delta


# --- Poisson model ---


class PoissonModel:
    """
    Dixon-Coles-style independent Poisson goals model.

    Each team has log-scale attack and defense parameters:
      μ_home = exp(intercept + att_home − def_away + home_adv × (1 − neutral))
      μ_away = exp(intercept + att_away − def_home)

    Fit by maximum likelihood with exponential time-weighting so recent
    matches carry more influence (half-life set to 3 years by default).
    """

    def __init__(self) -> None:
        self.attack: dict = {}
        self.defense: dict = {}
        self.home_advantage: float = 0.0
        self.intercept: float = 0.0
        self.teams: list = []
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        min_date: Optional[str] = None,
        halflife_days: int = 365 * 3,
    ) -> "PoissonModel":
        """
        Fit model parameters on a match DataFrame.

        df must have columns: date, home_team, away_team,
                              home_score, away_score, neutral.
        min_date: drop matches before this ISO date string.
        halflife_days: exponential time-weight half-life.
        """
        if min_date:
            df = df[df["date"] >= pd.Timestamp(min_date)].copy()

        df = df.dropna(subset=["home_score", "away_score"]).copy()
        df = df[df["home_score"].apply(lambda x: x == int(x))]
        df = df[df["away_score"].apply(lambda x: x == int(x))]

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        max_date = df["date"].max()
        days_ago = (max_date - df["date"]).dt.days.values.astype(float)
        weights = np.exp(-np.log(2) * days_ago / halflife_days)

        home_idx = df["home_team"].map(idx).values
        away_idx = df["away_team"].map(idx).values
        h_goals = df["home_score"].values.astype(int)
        a_goals = df["away_score"].values.astype(int)
        is_neutral = df["neutral"].astype(float).values

        # Parameter layout: [att_0..n-1, def_0..n-1, home_adv, intercept]
        x0 = np.zeros(2 * n + 2)
        mean_goals = (df["home_score"].mean() + df["away_score"].mean()) / 2
        x0[-1] = np.log(max(mean_goals, 0.5))

        def neg_ll(params):
            att = params[:n]
            defe = params[n : 2 * n]
            home_adv = params[-2]
            intercept = params[-1]

            # Identifiability: zero-sum constraints (applied inline, not in params)
            att = att - att.mean()
            defe = defe - defe.mean()

            mu_h = np.exp(intercept + att[home_idx] - defe[away_idx] + home_adv * (1 - is_neutral))
            mu_a = np.exp(intercept + att[away_idx] - defe[home_idx])

            mu_h = np.clip(mu_h, 1e-6, 20.0)
            mu_a = np.clip(mu_a, 1e-6, 20.0)

            ll = weights * (
                poisson.logpmf(h_goals, mu_h) + poisson.logpmf(a_goals, mu_a)
            )
            return -ll.sum()

        result = minimize(
            neg_ll,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-6},
        )

        att = result.x[:n] - result.x[:n].mean()
        defe = result.x[n : 2 * n] - result.x[n : 2 * n].mean()

        self.attack = {t: float(att[i]) for i, t in enumerate(teams)}
        self.defense = {t: float(defe[i]) for i, t in enumerate(teams)}
        self.home_advantage = float(result.x[-2])
        self.intercept = float(result.x[-1])
        self.teams = teams
        self._fitted = True

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _expected_goals(self, home: str, away: str, neutral: bool = True) -> tuple:
        """Return (mu_home, mu_away) expected goals. Falls back to global mean."""
        att_h = self.attack.get(home, 0.0)
        def_h = self.defense.get(home, 0.0)
        att_a = self.attack.get(away, 0.0)
        def_a = self.defense.get(away, 0.0)

        home_bonus = 0.0 if neutral else self.home_advantage
        mu_home = np.exp(self.intercept + att_h - def_a + home_bonus)
        mu_away = np.exp(self.intercept + att_a - def_h)
        return float(mu_home), float(mu_away)

    def predict_scoreline_probs(
        self,
        home: str,
        away: str,
        neutral: bool = True,
        max_goals: int = 10,
    ) -> np.ndarray:
        """
        Return an (max_goals+1) × (max_goals+1) matrix where
        P[i, j] = P(home scores i goals, away scores j goals).
        Rows = home goals, columns = away goals.
        """
        mu_h, mu_a = self._expected_goals(home, away, neutral)
        goals = np.arange(max_goals + 1)
        p_h = poisson.pmf(goals, mu_h)
        p_a = poisson.pmf(goals, mu_a)
        mat = np.outer(p_h, p_a)
        return mat / mat.sum()  # renormalise after truncation

    def predict_outcome_probs(
        self,
        home: str,
        away: str,
        neutral: bool = True,
        max_goals: int = 10,
    ) -> tuple:
        """Return (p_home_win, p_draw, p_away_win)."""
        mat = self.predict_scoreline_probs(home, away, neutral, max_goals)
        p_home = float(np.tril(mat, -1).sum())   # home goals > away goals
        p_draw = float(np.trace(mat))
        p_away = float(np.triu(mat, 1).sum())    # away goals > home goals
        return p_home, p_draw, p_away

    def has_team(self, team: str) -> bool:
        return team in self.attack

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = MODEL_CACHE
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "attack": self.attack,
            "defense": self.defense,
            "home_advantage": self.home_advantage,
            "intercept": self.intercept,
            "teams": self.teams,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PoissonModel":
        if path is None:
            path = MODEL_CACHE
        with open(path) as fh:
            data = json.load(fh)
        m = cls()
        m.attack = data["attack"]
        m.defense = data["defense"]
        m.home_advantage = data["home_advantage"]
        m.intercept = data["intercept"]
        m.teams = data["teams"]
        m._fitted = True
        return m


# --- Quick sanity check ---

if __name__ == "__main__":
    print("Elo sanity check:")
    ph, pd_, pa = elo_win_prob(2000, 1500)
    print(f"  2000 vs 1500 (neutral): home={ph:.2%}  draw={pd_:.2%}  away={pa:.2%}")
    ph, pd_, pa = elo_win_prob(1500, 1500)
    print(f"  1500 vs 1500 (neutral): home={ph:.2%}  draw={pd_:.2%}  away={pa:.2%}")
    ph, pd_, pa = elo_win_prob(1500, 1500, neutral=False)
    print(f"  1500 vs 1500 (home):    home={ph:.2%}  draw={pd_:.2%}  away={pa:.2%}")

    print("\nPoisson model requires fit on data — run simulate.py to test end-to-end.")
