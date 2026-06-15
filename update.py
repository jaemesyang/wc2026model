#!/usr/bin/env python3
"""
update.py — Record a live WC 2026 result and refresh group-stage predictions.

Commands:
    python update.py record <home_team> <home_score> <away_score> <away_team> [date]
    python update.py simulate
    python update.py delete <home_team> <away_team> [date]

After every 'record', the simulation automatically re-runs: real scores are
locked in for played matches and Monte Carlo is used for the rest.
"""

import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest import load_wc_results, record_result, scrape_elo_ratings, WC_RESULTS_CSV
from models import PoissonModel
from simulate import (
    GROUPS,
    N_SIMS,
    PREDICTIONS_DIR,
    _sort_tied_group,
    precompute_match_matrices,
    presample_match_results,
    select_best_thirds,
    simulate_match,
    write_predictions,
    write_group_readable_csv,
    write_group_readable_txt,
    print_group_tables,
)


# ---------------------------------------------------------------------------
# Known-result injection
# ---------------------------------------------------------------------------

def _load_known_results() -> dict:
    """Return {(home_team, away_team): (home_goals, away_goals)} for recorded results."""
    df = load_wc_results()
    return {
        (row["home_team"], row["away_team"]): (int(row["home_score"]), int(row["away_score"]))
        for _, row in df.iterrows()
    }


# ---------------------------------------------------------------------------
# Partial simulation (locks in known results, simulates the rest)
# ---------------------------------------------------------------------------

def _simulate_group_partial_fast(
    group_teams: list,
    presampled: dict,
    sim_idx: int,
    known_results: dict,
    model: PoissonModel,
    elo_ratings: dict,
) -> list:
    """
    Hot-path partial group sim: known results fixed, rest from presampled.
    Returns list of (team, rank, points, gd, gf, ga) tuples — no pandas.
    """
    records = {t: {"points": 0, "gd": 0, "gf": 0, "ga": 0} for t in group_teams}
    h2h: dict = {
        t: {opp: {"points": 0, "gd": 0} for opp in group_teams if opp != t}
        for t in group_teams
    }

    for home, away in combinations(group_teams, 2):
        if (home, away) in known_results:
            hg, ag = known_results[(home, away)]
        elif (away, home) in known_results:
            ag, hg = known_results[(away, home)]
        elif (home, away) in presampled:
            hg = presampled[(home, away)][0][sim_idx]
            ag = presampled[(home, away)][1][sim_idx]
        else:
            hg, ag = simulate_match(home, away, model, elo_ratings, neutral=True)

        records[home]["gf"] += hg; records[home]["ga"] += ag; records[home]["gd"] += hg - ag
        records[away]["gf"] += ag; records[away]["ga"] += hg; records[away]["gd"] += ag - hg

        if hg > ag:
            records[home]["points"] += 3
            h2h[home][away]["points"] += 3
            h2h[home][away]["gd"] += hg - ag
            h2h[away][home]["gd"] += ag - hg
        elif hg == ag:
            records[home]["points"] += 1; records[away]["points"] += 1
            h2h[home][away]["points"] += 1; h2h[away][home]["points"] += 1
        else:
            records[away]["points"] += 3
            h2h[away][home]["points"] += 3
            h2h[away][home]["gd"] += ag - hg
            h2h[home][away]["gd"] += hg - ag

    by_pts = sorted(group_teams, key=lambda t: -records[t]["points"])
    sorted_teams: list = []
    i = 0
    while i < len(by_pts):
        j = i + 1
        while j < len(by_pts) and records[by_pts[j]]["points"] == records[by_pts[i]]["points"]:
            j += 1
        sorted_teams.extend(_sort_tied_group(by_pts[i:j], records, h2h))
        i = j

    return [
        (team, rank, records[team]["points"], records[team]["gd"],
         records[team]["gf"], records[team]["ga"])
        for rank, team in enumerate(sorted_teams, 1)
    ]


def _run_group_stage_partial(
    model: PoissonModel,
    elo_ratings: dict,
    known_results: dict,
    n_sims: int = N_SIMS,
) -> tuple:
    """
    Monte Carlo group stage with real results locked in.
    Returns (advance_df, rank_df) — same schema as simulate.run_group_stage.
    """
    cache      = precompute_match_matrices(model)
    presampled = presample_match_results(cache, n_sims)

    all_teams = [(team, group) for group, teams in GROUPS.items() for team in teams]
    advance_counts: dict = {t: 0 for t, _ in all_teams}
    top2_counts:   dict = {t: 0 for t, _ in all_teams}
    rank_counts:   dict = {t: {1: 0, 2: 0, 3: 0, 4: 0} for t, _ in all_teams}

    for sim_idx in range(n_sims):
        third_entries = []

        for group_name, teams in GROUPS.items():
            standings = _simulate_group_partial_fast(
                teams, presampled, sim_idx, known_results, model, elo_ratings
            )
            for team, rank, pts, gd, gf, ga in standings:
                rank_counts[team][rank] += 1
                if rank <= 2:
                    advance_counts[team] += 1
                    top2_counts[team] += 1
                elif rank == 3:
                    third_entries.append({
                        "team": team, "group": group_name,
                        "points": pts, "gd": gd, "gf": gf,
                    })

        for t in select_best_thirds(third_entries):
            advance_counts[t] += 1

    advance_rows, rank_rows = [], []
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
        rank_rows.append({
            "team":          team,
            "group":         group,
            "p1":            round(p1, 4),
            "p2":            round(p2, 4),
            "p3":            round(p3, 4),
            "p4":            round(p4, 4),
            "mode_rank":     max(rc, key=rc.__getitem__),
            "expected_rank": round(1*p1 + 2*p2 + 3*p3 + 4*p4, 3),
        })

    advance_df = (
        pd.DataFrame(advance_rows)
        .sort_values("advance_prob", ascending=False)
        .reset_index(drop=True)
    )
    return advance_df, pd.DataFrame(rank_rows)


# ---------------------------------------------------------------------------
# Refresh — orchestrates the full record → simulate → write cycle
# ---------------------------------------------------------------------------

def refresh_predictions(
    n_sims: int = N_SIMS,
    baseline_csv: Path = None,
) -> Path:
    """
    Load the saved Poisson model, inject any recorded WC results, Monte Carlo
    the remaining matches, and write a new timestamped prediction file.

    If baseline_csv is given, prints a before/after comparison for every team
    in groups that have at least one known result.

    Returns the path of the written advance-probability CSV.
    """
    print("Loading Poisson model from cache...")
    model = PoissonModel.load()

    print("Loading Elo ratings (cached or scraped)...")
    elo_ratings = scrape_elo_ratings().to_dict()

    known = _load_known_results()
    n_played = len(known)
    print(f"\nKnown WC 2026 results: {n_played}")
    for (h, a), (hg, ag) in known.items():
        print(f"  {h} {hg}–{ag} {a}")

    print(f"\nRunning {n_sims:,} simulations ({n_played} result(s) locked in)...")
    advance_df, rank_df = _run_group_stage_partial(model, elo_ratings, known, n_sims=n_sims)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    adv_path = PREDICTIONS_DIR / f"group_stage_update_{ts}.csv"
    csv_path = PREDICTIONS_DIR / f"group_predictions_readable_{ts}.csv"
    txt_path = PREDICTIONS_DIR / f"group_predictions_{ts}.txt"

    write_predictions(advance_df, out_path=adv_path)
    write_group_readable_csv(rank_df, advance_df, csv_path)
    write_group_readable_txt(rank_df, advance_df, txt_path)

    # Before/after comparison for affected groups
    if baseline_csv and Path(baseline_csv).exists() and known:
        _print_comparison(advance_df, baseline_csv, known)

    print("\nTop 10 by advancement probability:")
    print(advance_df.head(10).to_string(index=False))
    print_group_tables(rank_df, advance_df)

    return adv_path


def _print_comparison(new_df: pd.DataFrame, baseline_path: Path, known: dict) -> None:
    """Print before/after advance_prob for teams in groups with known results."""
    baseline = pd.read_csv(baseline_path)

    affected_groups = set()
    for group, teams in GROUPS.items():
        for team in teams:
            if any(h == team or a == team for h, a in known):
                affected_groups.add(group)

    print("\n--- Advancement probability changes (affected groups) ---")
    for group in sorted(affected_groups):
        teams = GROUPS[group]
        print(f"\n  Group {group}:")
        for team in teams:
            before_row = baseline[baseline["team"] == team]
            after_row  = new_df[new_df["team"] == team]
            if before_row.empty or after_row.empty:
                continue
            before = before_row.iloc[0]["advance_prob"]
            after  = after_row.iloc[0]["advance_prob"]
            delta  = after - before
            sign   = "+" if delta >= 0 else ""
            print(f"    {team:<28}  {before:.1%} → {after:.1%}  ({sign}{delta:+.1%})")
    print()


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent
_PREDICTIONS_DIR = _ROOT / "predictions"


def _latest_pretournament_csv() -> Path:
    """
    Return the locked pre-tournament CSV.
    Checks project root first (canonical location), then predictions/ as fallback.
    """
    root_candidates = sorted(_ROOT.glob("group_stage_pre_tournament_*.csv"))
    if root_candidates:
        return root_candidates[-1]
    pred_candidates = sorted(_PREDICTIONS_DIR.glob("group_stage_pre_tournament_*.csv"))
    return pred_candidates[-1] if pred_candidates else None


def cmd_record(args: list) -> None:
    """record "HOME hg AWAY ag" [...] [--date YYYY-MM-DD]"""
    if not args:
        sys.exit('Usage: update.py record "HOME hg AWAY ag" [...] [--date YYYY-MM-DD]')

    from score import _parse_result_str, _resolve_team

    date_val = datetime.now().strftime("%Y-%m-%d")
    result_strs = []
    i = 0
    while i < len(args):
        if args[i] == "--date" and i + 1 < len(args):
            date_val = args[i + 1]
            i += 2
        else:
            result_strs.append(args[i])
            i += 1

    for s in result_strs:
        try:
            home_raw, hg, away_raw, ag = _parse_result_str(s)
            home = _resolve_team(home_raw)
            away = _resolve_team(away_raw)
        except ValueError as e:
            print(f"ERROR '{s}': {e}", file=sys.stderr)
            continue
        record_result({
            "date":       date_val,
            "home_team":  home,
            "away_team":  away,
            "home_score": hg,
            "away_score": ag,
        })

    refresh_predictions(baseline_csv=_latest_pretournament_csv())


def cmd_simulate(_args: list) -> None:
    """Re-run simulation from currently recorded results, no new recording."""
    refresh_predictions(baseline_csv=_latest_pretournament_csv())


def cmd_delete(args: list) -> None:
    """delete <home_team> <away_team> [date] — remove a recorded result and optionally re-simulate."""
    if len(args) < 2:
        sys.exit("Usage: update.py delete <home_team> <away_team> [date]")

    home_team   = args[0]
    away_team   = args[1]
    date_filter = args[2] if len(args) >= 3 else None

    if not WC_RESULTS_CSV.exists():
        print("No recorded results file found — nothing to delete.")
        return

    df = pd.read_csv(WC_RESULTS_CSV)
    mask = (df["home_team"] == home_team) & (df["away_team"] == away_team)
    if date_filter:
        mask &= df["date"] == date_filter

    n_removed = mask.sum()
    if n_removed == 0:
        print(f"No result found for {home_team} vs {away_team}" +
              (f" on {date_filter}" if date_filter else "") + ".")
        return

    df = df[~mask].reset_index(drop=True)
    if df.empty:
        WC_RESULTS_CSV.unlink()
        print(f"Removed {n_removed} result(s). Results file deleted (now empty).")
    else:
        df.to_csv(WC_RESULTS_CSV, index=False)
        print(f"Removed {n_removed} result(s): {home_team} vs {away_team}.")


_COMMANDS = {
    "record":   cmd_record,
    "simulate": cmd_simulate,
    "delete":   cmd_delete,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        print(__doc__.strip())
        sys.exit(0)
    _COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
