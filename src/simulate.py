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

PREDICTIONS_DIR = Path(__file__).parent.parent / "output"
N_SIMS = 20_000

# ---------------------------------------------------------------------------
# 2026 FIFA World Cup group draw
# Replace with the official draw once confirmed.
# 12 groups × 4 teams = 48 teams.
# ---------------------------------------------------------------------------
GROUPS: dict = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
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
    if records[by_gd[0]]["gd"] != records[by_gd[-1]]["gd"]:
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
) -> tuple:
    """
    Monte Carlo the group stage n_sims times.

    Returns (advance_df, rank_df):
      advance_df — one row per team, sorted by advance_prob
                   columns: team, group, advance_prob, top2_prob, via_third_prob
      rank_df    — one row per team with per-position finish probabilities
                   columns: team, group, p1, p2, p3, p4, mode_rank, expected_rank
    """
    all_teams = [(team, group) for group, teams in GROUPS.items() for team in teams]
    advance_counts: dict = {t: 0 for t, _ in all_teams}
    top2_counts:   dict = {t: 0 for t, _ in all_teams}
    rank_counts:   dict = {t: {1: 0, 2: 0, 3: 0, 4: 0} for t, _ in all_teams}

    for _ in range(n_sims):
        third_entries = []

        for group_name, teams in GROUPS.items():
            standings = simulate_group(teams, model, elo_ratings)

            for _, row in standings.iterrows():
                rank_counts[row["team"]][int(row["rank"])] += 1

            top2 = standings[standings["rank"] <= 2]["team"].tolist()
            for t in top2:
                advance_counts[t] += 1
                top2_counts[t] += 1

            row3 = standings[standings["rank"] == 3].iloc[0]
            third_entries.append({
                "team":   row3["team"],
                "group":  group_name,
                "points": int(row3["points"]),
                "gd":     int(row3["gd"]),
                "gf":     int(row3["gf"]),
            })

        best_thirds = select_best_thirds(third_entries)
        for t in best_thirds:
            advance_counts[t] += 1

    advance_rows = []
    rank_rows = []
    for team, group in all_teams:
        via_third = advance_counts[team] - top2_counts[team]
        advance_rows.append({
            "team":           team,
            "group":          group,
            "advance_prob":   round(advance_counts[team] / n_sims, 4),
            "top2_prob":      round(top2_counts[team] / n_sims, 4),
            "via_third_prob": round(via_third / n_sims, 4),
        })

        rc = rank_counts[team]
        p1, p2, p3, p4 = rc[1]/n_sims, rc[2]/n_sims, rc[3]/n_sims, rc[4]/n_sims
        mode_rank = max(rc, key=rc.__getitem__)
        rank_rows.append({
            "team":          team,
            "group":         group,
            "p1":            round(p1, 4),
            "p2":            round(p2, 4),
            "p3":            round(p3, 4),
            "p4":            round(p4, 4),
            "mode_rank":     mode_rank,
            "expected_rank": round(1*p1 + 2*p2 + 3*p3 + 4*p4, 3),
        })

    advance_df = (
        pd.DataFrame(advance_rows)
        .sort_values("advance_prob", ascending=False)
        .reset_index(drop=True)
    )
    rank_df = pd.DataFrame(rank_rows)
    return advance_df, rank_df


def write_predictions(
    df: pd.DataFrame,
    label: str = "group_stage",
    out_path: Optional[Path] = None,
) -> Path:
    """Write predictions CSV to predictions/. Uses a fixed out_path if given, else timestamped."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = PREDICTIONS_DIR / f"{ts}_{label}.csv"
    df.to_csv(out_path, index=False)
    print(f"Predictions written → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Human-readable group-stage output
# ---------------------------------------------------------------------------

_NAME_W = 25          # column width for team names (fits "Bosnia and Herzegovina")
_COL_W  = 7           # width of each percentage column
_RANK_LABEL = {1: "1st ✓", 2: "2nd ✓", 3: "3rd", 4: "4th"}


def _pct(p: float) -> str:
    return f"{p*100:.1f}%".rjust(_COL_W)


def _build_group_summary(rank_df: pd.DataFrame, advance_df: pd.DataFrame) -> pd.DataFrame:
    """Merge rank and via_third info; add projected-position label; sort by expected_rank."""
    df = rank_df.merge(advance_df[["team", "via_third_prob"]], on="team")
    df["projected"] = df["mode_rank"].map(_RANK_LABEL)
    return df.sort_values(["group", "expected_rank"]).reset_index(drop=True)


def _make_group_block(group_name: str, summary: pd.DataFrame) -> str:
    gdf = summary[summary["group"] == group_name]
    header = (
        f"{'GROUP ' + group_name:<{_NAME_W}}  "
        f"{'P(1st)':>{_COL_W}}  {'P(2nd)':>{_COL_W}}  "
        f"{'P(3rd)':>{_COL_W}}  {'P(4th)':>{_COL_W}}  Projected"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for _, row in gdf.iterrows():
        lines.append(
            f"{row['team']:<{_NAME_W}}  "
            f"{_pct(row['p1'])}  {_pct(row['p2'])}  "
            f"{_pct(row['p3'])}  {_pct(row['p4'])}  "
            f"{row['projected']}"
        )
    return "\n".join(lines)


def _make_thirds_note(advance_df: pd.DataFrame) -> str:
    """
    Identify the 8 most likely third-place qualifiers across all 12 groups.
    Each group contributes exactly one projected third (highest via_third_prob).
    Those 12 thirds are then ranked and the top 8 are the projected qualifiers.
    """
    thirds = (
        advance_df[advance_df["via_third_prob"] > 0]
        .sort_values(["group", "via_third_prob"], ascending=[True, False])
        .groupby("group")
        .first()
        .reset_index()
        [["group", "team", "via_third_prob"]]
        .sort_values("via_third_prob", ascending=False)
        .reset_index(drop=True)
    )
    in_top8 = thirds.head(8)
    out = thirds.tail(len(thirds) - 8)

    width = 50
    lines = [
        "=" * width,
        "PROJECTED THIRD-PLACE QUALIFIERS  (best 8 of 12)",
        "=" * width,
        "IN:",
    ]
    for _, r in in_top8.iterrows():
        lines.append(f"  Group {r['group']}: {r['team']:<24}  ({r['via_third_prob']*100:.1f}%)")
    if not out.empty:
        lines.append("OUT:")
        for _, r in out.iterrows():
            lines.append(f"  Group {r['group']}: {r['team']:<24}  ({r['via_third_prob']*100:.1f}%)")
    lines.append("")
    lines.append("✓ = auto-qualifies (top 2 in group)")
    return "\n".join(lines)


def print_group_tables(rank_df: pd.DataFrame, advance_df: pd.DataFrame) -> None:
    summary = _build_group_summary(rank_df, advance_df)
    print("\n" + "=" * 65)
    print("GROUP STAGE PROJECTIONS  (20,000 simulations)")
    print("=" * 65)
    for group_name in GROUPS:
        print()
        print(_make_group_block(group_name, summary))
    print()
    print(_make_thirds_note(advance_df))


def write_group_readable_csv(
    rank_df: pd.DataFrame,
    advance_df: pd.DataFrame,
    path: Path,
) -> Path:
    summary = _build_group_summary(rank_df, advance_df)
    out = summary[["group", "team", "p1", "p2", "p3", "p4",
                   "mode_rank", "expected_rank", "via_third_prob", "projected"]]
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Group predictions CSV → {path}")
    return path


def write_group_readable_txt(
    rank_df: pd.DataFrame,
    advance_df: pd.DataFrame,
    path: Path,
) -> Path:
    summary = _build_group_summary(rank_df, advance_df)
    blocks = []
    blocks.append("WC 2026 GROUP STAGE PROJECTIONS  (20,000 Monte Carlo simulations)")
    blocks.append("=" * 65)
    for group_name in GROUPS:
        blocks.append(_make_group_block(group_name, summary))
    blocks.append(_make_thirds_note(advance_df))

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Group predictions TXT  → {path}")
    return path


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
    advance_df, rank_df = run_group_stage(model, elo_ratings, n_sims=n_sims)

    print("\nTop 10 by advancement probability:")
    print(advance_df.head(10).to_string(index=False))

    print_group_tables(rank_df, advance_df)

    today = datetime.now().strftime("%Y-%m-%d")
    # pre_tournament stays in predictions/ as the locked commitment artifact
    _pred_dir = Path(__file__).parent.parent / "predictions"
    adv_path  = _pred_dir / f"group_stage_pre_tournament_{today}.csv"
    csv_path  = PREDICTIONS_DIR / f"group_predictions_readable_{today}.csv"
    txt_path  = PREDICTIONS_DIR / f"group_predictions_{today}.txt"

    write_predictions(advance_df, out_path=adv_path)
    write_group_readable_csv(rank_df, advance_df, csv_path)
    write_group_readable_txt(rank_df, advance_df, txt_path)

    return advance_df, rank_df


if __name__ == "__main__":
    main()
