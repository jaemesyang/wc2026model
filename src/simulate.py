"""
simulate.py — Monte Carlo group-stage simulation for WC 2026.

Paste the real draw into GROUPS below once it is announced.
Run:  python simulate.py
Output: predictions/<timestamp>_group_stage.csv

Tiebreaker order (FIFA rules):
  1. Points
  2. Goal difference (overall)
  3. Goals scored (overall)
  4. Head-to-head points (among tied subset)
  5. Head-to-head goal difference (among tied subset)
  6. (Lots — resolved by random shuffle in simulation)
"""

import sys
import random
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Allow running directly from src/ or from project root
sys.path.insert(0, str(Path(__file__).parent))

from ingest import load_historical_results, scrape_elo_ratings
from models import PoissonModel, elo_win_prob

PREDICTIONS_DIR = Path(__file__).parent.parent / "predictions"
N_SIMS = 20_000

# ---------------------------------------------------------------------------
# 2026 FIFA World Cup group draw
# Replace with the official draw once confirmed.
# 12 groups × 4 teams = 48 teams.
# ---------------------------------------------------------------------------
GROUPS: dict = {
    "A": ["USA", "Panama", "Honduras", "Colombia"],
    "B": ["Mexico", "Ecuador", "Jamaica", "Martinique"],
    "C": ["Argentina", "Chile", "Peru", "Venezuela"],
    "D": ["Uruguay", "Bolivia", "Paraguay", "Costa Rica"],
    "E": ["Brazil", "Colombia", "Bolivia", "El Salvador"],
    "F": ["France", "Belgium", "Austria", "Slovakia"],
    "G": ["England", "Netherlands", "Germany", "Switzerland"],
    "H": ["Spain", "Portugal", "Croatia", "Serbia"],
    "I": ["Morocco", "Senegal", "Egypt", "Tunisia"],
    "J": ["Cameroon", "Nigeria", "Ghana", "Ivory Coast"],
    "K": ["Japan", "South Korea", "Australia", "Iran"],
    "L": ["Saudi Arabia", "Qatar", "Iraq", "United Arab Emirates"],
}

# Teams with no Poisson history fall back to this default rating.
_DEFAULT_ELO = 1500


def _get_elo(elo_ratings: dict, team: str) -> float:
    return float(elo_ratings.get(team, _DEFAULT_ELO))


def simulate_match(
    home: str,
    away: str,
    model: PoissonModel,
    elo_ratings: dict,
    neutral: bool = True,
) -> tuple:
    """
    Sample one match outcome. Returns (home_goals, away_goals).

    Uses Poisson joint distribution when both teams are in the fitted model;
    falls back to Elo-based outcome sampling otherwise.
    """
    if model.has_team(home) and model.has_team(away):
        mat = model.predict_scoreline_probs(home, away, neutral=neutral)
        flat = mat.flatten()
        idx = np.random.choice(len(flat), p=flat)
        n = mat.shape[0]
        return int(idx // n), int(idx % n)

    # Elo fallback: sample outcome, draw a plausible scoreline.
    r_h = _get_elo(elo_ratings, home)
    r_a = _get_elo(elo_ratings, away)
    p_h, p_d, p_a = elo_win_prob(r_h, r_a, neutral=neutral)
    outcome = np.random.choice(3, p=[p_h, p_d, p_a])
    if outcome == 0:   # home win
        hg = np.random.randint(1, 4)
        ag = np.random.randint(0, hg)
    elif outcome == 1: # draw
        g = np.random.randint(0, 3)
        hg, ag = g, g
    else:              # away win
        ag = np.random.randint(1, 4)
        hg = np.random.randint(0, ag)
    return int(hg), int(ag)


# ---------------------------------------------------------------------------
# Group simulation and tiebreakers
# ---------------------------------------------------------------------------

def _h2h_stats(tied: list, h2h: dict) -> dict:
    """Head-to-head points and goal difference among a subset of teams."""
    stats = {}
    for t in tied:
        pts = sum(h2h[t][opp]["points"] for opp in tied if opp != t)
        gd = sum(h2h[t][opp]["gd"] for opp in tied if opp != t)
        stats[t] = (pts, gd)
    return stats


def _sort_tied_group(tied: list, records: dict, h2h: dict) -> list:
    """
    Sort a list of tied-on-points teams by FIFA tiebreaker cascade.
    Falls back to random shuffle when all criteria are equal.
    """
    if len(tied) == 1:
        return tied

    # 1. Goal difference (overall)
    by_gd = sorted(tied, key=lambda t: -records[t]["gd"])
    if by_gd[0] != by_gd[-1] or records[by_gd[0]]["gd"] != records[by_gd[-1]]["gd"]:
        # At least one team separable by GD — but may still have internal ties.
        return _sort_with_key(tied, records, h2h, lambda t: -records[t]["gd"])

    # 2. Goals scored (overall)
    if len(set(records[t]["gf"] for t in tied)) > 1:
        return _sort_with_key(tied, records, h2h, lambda t: -records[t]["gf"])

    # 3. Head-to-head points among tied teams
    h2h_st = _h2h_stats(tied, h2h)
    if len(set(v[0] for v in h2h_st.values())) > 1:
        return sorted(tied, key=lambda t: (-h2h_st[t][0], -h2h_st[t][1]))

    # 4. Head-to-head GD among tied teams
    if len(set(v[1] for v in h2h_st.values())) > 1:
        return sorted(tied, key=lambda t: -h2h_st[t][1])

    # 5. Drawing of lots (random)
    shuffled = tied[:]
    random.shuffle(shuffled)
    return shuffled


def _sort_with_key(tied: list, records: dict, h2h: dict, key) -> list:
    """Sort with key; recursively break remaining ties within each tied cluster."""
    ranked = sorted(tied, key=key)
    result = []
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and key(ranked[j]) == key(ranked[i]):
            j += 1
        cluster = ranked[i:j]
        if len(cluster) > 1:
            cluster = _sort_tied_group(cluster, records, h2h)
        result.extend(cluster)
        i = j
    return result


def simulate_group(
    group_teams: list,
    model: PoissonModel,
    elo_ratings: dict,
) -> pd.DataFrame:
    """
    Simulate a round-robin group. Returns a standings DataFrame with columns:
      team, rank, points, gd, gf, ga
    """
    records = {
        t: {"points": 0, "gd": 0, "gf": 0, "ga": 0}
        for t in group_teams
    }
    h2h: dict = {
        t: {opp: {"points": 0, "gd": 0} for opp in group_teams if opp != t}
        for t in group_teams
    }

    for home, away in combinations(group_teams, 2):
        hg, ag = simulate_match(home, away, model, elo_ratings, neutral=True)

        records[home]["gf"] += hg
        records[home]["ga"] += ag
        records[home]["gd"] += hg - ag
        records[away]["gf"] += ag
        records[away]["ga"] += hg
        records[away]["gd"] += ag - hg

        if hg > ag:
            records[home]["points"] += 3
            h2h[home][away]["points"] += 3
            h2h[home][away]["gd"] += hg - ag
            h2h[away][home]["gd"] += ag - hg
        elif hg == ag:
            records[home]["points"] += 1
            records[away]["points"] += 1
            h2h[home][away]["points"] += 1
            h2h[away][home]["points"] += 1
            # GD remains 0 for both in a draw
        else:
            records[away]["points"] += 3
            h2h[away][home]["points"] += 3
            h2h[away][home]["gd"] += ag - hg
            h2h[home][away]["gd"] += hg - ag

    # Initial sort by points; then apply tiebreakers within clusters.
    by_pts = sorted(group_teams, key=lambda t: -records[t]["points"])

    sorted_teams = []
    i = 0
    while i < len(by_pts):
        j = i + 1
        while j < len(by_pts) and records[by_pts[j]]["points"] == records[by_pts[i]]["points"]:
            j += 1
        cluster = by_pts[i:j]
        sorted_teams.extend(_sort_tied_group(cluster, records, h2h))
        i = j

    rows = []
    for rank, team in enumerate(sorted_teams, 1):
        r = records[team]
        rows.append({
            "team": team,
            "rank": rank,
            "points": r["points"],
            "gd": r["gd"],
            "gf": r["gf"],
            "ga": r["ga"],
        })
    return pd.DataFrame(rows)


def select_best_thirds(third_entries: list) -> list:
    """
    Select the 8 best third-place teams from 12 groups.

    Ranking criteria (FIFA): points → GD → GF → (omit fair play / lots in sim).
    Returns a list of 8 team names.
    """
    df = pd.DataFrame(third_entries)
    df = df.sort_values(
        ["points", "gd", "gf"], ascending=False
    ).reset_index(drop=True)
    return df.head(8)["team"].tolist()


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_group_stage(
    model: PoissonModel,
    elo_ratings: dict,
    n_sims: int = N_SIMS,
) -> pd.DataFrame:
    """
    Monte Carlo the group stage n_sims times.

    Returns a DataFrame with one row per team, sorted by advance_prob:
      team, group, advance_prob, top2_prob, via_third_prob
    """
    all_teams = [(team, group) for group, teams in GROUPS.items() for team in teams]
    advance_counts: dict = {t: 0 for t, _ in all_teams}
    top2_counts: dict = {t: 0 for t, _ in all_teams}

    for _ in range(n_sims):
        third_entries = []

        for group_name, teams in GROUPS.items():
            standings = simulate_group(teams, model, elo_ratings)

            top2 = standings[standings["rank"] <= 2]["team"].tolist()
            for t in top2:
                advance_counts[t] += 1
                top2_counts[t] += 1

            row3 = standings[standings["rank"] == 3].iloc[0]
            third_entries.append({
                "team": row3["team"],
                "group": group_name,
                "points": int(row3["points"]),
                "gd": int(row3["gd"]),
                "gf": int(row3["gf"]),
            })

        best_thirds = select_best_thirds(third_entries)
        for t in best_thirds:
            advance_counts[t] += 1

    rows = []
    for team, group in all_teams:
        via_third = advance_counts[team] - top2_counts[team]
        rows.append({
            "team": team,
            "group": group,
            "advance_prob": round(advance_counts[team] / n_sims, 4),
            "top2_prob": round(top2_counts[team] / n_sims, 4),
            "via_third_prob": round(via_third / n_sims, 4),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("advance_prob", ascending=False)
        .reset_index(drop=True)
    )


def write_predictions(df: pd.DataFrame, label: str = "group_stage") -> Path:
    """Write a timestamped CSV to predictions/."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PREDICTIONS_DIR / f"{ts}_{label}.csv"
    df.to_csv(out_path, index=False)
    print(f"Predictions written → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(n_sims: int = N_SIMS, min_date: Optional[str] = "2010-01-01") -> tuple:
    print("Loading historical results...")
    df = load_historical_results()
    print(f"  {len(df):,} matches loaded")

    print("Loading Elo ratings (cached or scraped)...")
    elo_series = scrape_elo_ratings()
    elo_ratings = elo_series.to_dict()
    print(f"  {len(elo_ratings)} teams with Elo ratings")

    print(f"Fitting Poisson model on data from {min_date}...")
    model = PoissonModel()
    model.fit(df, min_date=min_date)
    model.save()
    print(f"  Fitted on {len(model.teams)} teams")

    print(f"\nRunning {n_sims:,} group-stage simulations...")
    results = run_group_stage(model, elo_ratings, n_sims=n_sims)

    print("\nTop 20 by advancement probability:")
    print(results.head(20).to_string(index=False))

    out_path = write_predictions(results, label="group_stage")
    return results, out_path


if __name__ == "__main__":
    main()
