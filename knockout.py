#!/usr/bin/env python3
"""
knockout.py — Monte Carlo single-elimination bracket for WC 2026 knockout stage.

Usage:
    python knockout.py [--sims N]

Fixed bracket M73–Final. Draws resolved by Elo-weighted penalty shootout
clamped to [0.40, 0.60]. Reuses precomputed scoreline-matrix cache.
"""

import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest import scrape_elo_ratings
from models import PoissonModel
from simulate import PREDICTIONS_DIR

OUTPUT_DIR = PREDICTIONS_DIR

N_SIMS = 20_000
_PENALTY_FACTOR = 0.0003   # p = 0.5 + factor × elo_diff, clamped below
_PENALTY_MIN    = 0.40
_PENALTY_MAX    = 0.60

# ---------------------------------------------------------------------------
# Fixed bracket (Round of 32 → Final)
# ---------------------------------------------------------------------------

# R32 matchups in bracket order — index = 0-based match number (M73=0, M74=1, …)
R32 = [
    ("South Africa",           "Canada"),                  # M73  idx 0
    ("Germany",                "Paraguay"),                # M74  idx 1
    ("Netherlands",            "Morocco"),                 # M75  idx 2
    ("Brazil",                 "Japan"),                   # M76  idx 3
    ("France",                 "Sweden"),                  # M77  idx 4
    ("Ivory Coast",            "Norway"),                  # M78  idx 5
    ("Mexico",                 "Ecuador"),                 # M79  idx 6
    ("England",                "DR Congo"),                # M80  idx 7
    ("United States",          "Bosnia and Herzegovina"), # M81  idx 8
    ("Belgium",                "Senegal"),                 # M82  idx 9
    ("Colombia",               "Ghana"),                   # M83  idx 10
    ("Spain",                  "Austria"),                 # M84  idx 11
    ("Switzerland",            "Algeria"),                 # M85  idx 12
    ("Argentina",              "Cape Verde"),              # M86  idx 13
    ("Portugal",               "Croatia"),                 # M87  idx 14
    ("Australia",              "Egypt"),                   # M88  idx 15
]

# R16 pairs: (i, j) → r32_winners[i] vs r32_winners[j]
R16_PAIRS = [
    (1,  4),   # M89: W74 vs W77
    (0,  2),   # M90: W73 vs W75
    (3,  5),   # M91: W76 vs W78
    (6,  7),   # M92: W79 vs W80
    (14, 10),  # M93: W87 vs W83
    (8,  9),   # M94: W81 vs W82
    (13, 15),  # M95: W86 vs W88
    (12, 11),  # M96: W85 vs W84
]

# QF pairs: (i, j) → r16_winners[i] vs r16_winners[j]
QF_PAIRS = [
    (0, 1),  # M97:  W89 vs W90
    (4, 5),  # M98:  W93 vs W94
    (2, 3),  # M99:  W91 vs W92
    (6, 7),  # M100: W95 vs W96
]

# SF pairs: (i, j) → qf_winners[i] vs qf_winners[j]
SF_PAIRS = [
    (0, 1),  # M101: W97  vs W98
    (2, 3),  # M102: W99  vs W100
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _ko_winner(home: str, away: str, hg: int, ag: int, elo: dict) -> str:
    """Return winner; resolve draws via Elo-weighted penalty shootout."""
    if hg > ag:
        return home
    if ag > hg:
        return away
    r_h = elo.get(home, 1500)
    r_a = elo.get(away, 1500)
    p = max(_PENALTY_MIN, min(_PENALTY_MAX, 0.5 + _PENALTY_FACTOR * (r_h - r_a)))
    return home if random.random() < p else away


def _sample_match(home: str, away: str, ko_cache: dict, elo: dict) -> str:
    """Sample one knockout match result from precomputed matrix."""
    flat, n = ko_cache[(home, away)]
    idx = int(np.random.choice(len(flat), p=flat))
    return _ko_winner(home, away, idx // n, idx % n, elo)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _build_ko_cache(model: PoissonModel, teams: list) -> dict:
    """Precompute scoreline matrices for all directed pairs among knockout teams."""
    cache = {}
    for t1 in teams:
        for t2 in teams:
            if t1 != t2 and model.has_team(t1) and model.has_team(t2):
                mat = model.predict_scoreline_probs(t1, t2, neutral=True)
                cache[(t1, t2)] = (mat.flatten(), mat.shape[0])
    return cache


def _presample_r32(ko_cache: dict, n_sims: int) -> dict:
    """Vectorised presample for the 16 fixed R32 matchups."""
    presampled = {}
    for home, away in R32:
        if (home, away) in ko_cache:
            flat, n = ko_cache[(home, away)]
            idxs = np.random.choice(len(flat), size=n_sims, p=flat)
            presampled[(home, away)] = ((idxs // n).tolist(), (idxs % n).tolist())
    return presampled


def run_knockout(model: PoissonModel, elo: dict, n_sims: int = N_SIMS) -> pd.DataFrame:
    """
    Monte Carlo the full knockout bracket.

    Returns DataFrame with columns: team, R16, QF, SF, Final, Win
    (each column = fraction of simulations the team reached that round).
    Sorted descending by Win.
    """
    ko_teams = sorted(set(t for pair in R32 for t in pair))

    print(f"Precomputing scoreline matrices for {len(ko_teams)} teams...")
    ko_cache = _build_ko_cache(model, ko_teams)

    print("Presampling R32 matchups...")
    r32_pre = _presample_r32(ko_cache, n_sims)

    reach = {t: {"r16": 0, "qf": 0, "sf": 0, "final": 0, "win": 0} for t in ko_teams}

    print(f"Running {n_sims:,} bracket simulations...")
    for sim in range(n_sims):
        # R32 — use presampled scorelines
        r32_w = []
        for home, away in R32:
            if (home, away) in r32_pre:
                hg = r32_pre[(home, away)][0][sim]
                ag = r32_pre[(home, away)][1][sim]
                winner = _ko_winner(home, away, hg, ag, elo)
            else:
                winner = _sample_match(home, away, ko_cache, elo)
            r32_w.append(winner)

        # R16 — inline sampling; R32 winners now "reached R16"
        r16_w = []
        for i, j in R16_PAIRS:
            h, a = r32_w[i], r32_w[j]
            reach[h]["r16"] += 1
            reach[a]["r16"] += 1
            r16_w.append(_sample_match(h, a, ko_cache, elo))

        # QF
        qf_w = []
        for i, j in QF_PAIRS:
            h, a = r16_w[i], r16_w[j]
            reach[h]["qf"] += 1
            reach[a]["qf"] += 1
            qf_w.append(_sample_match(h, a, ko_cache, elo))

        # SF
        sf_w = []
        for i, j in SF_PAIRS:
            h, a = qf_w[i], qf_w[j]
            reach[h]["sf"] += 1
            reach[a]["sf"] += 1
            sf_w.append(_sample_match(h, a, ko_cache, elo))

        # Final
        h, a = sf_w[0], sf_w[1]
        reach[h]["final"] += 1
        reach[a]["final"] += 1
        champ = _sample_match(h, a, ko_cache, elo)
        reach[champ]["win"] += 1

    rows = []
    for team in ko_teams:
        r = reach[team]
        rows.append({
            "team":  team,
            "R16":   round(r["r16"]   / n_sims, 4),
            "QF":    round(r["qf"]    / n_sims, 4),
            "SF":    round(r["sf"]    / n_sims, 4),
            "Final": round(r["final"] / n_sims, 4),
            "Win":   round(r["win"]   / n_sims, 4),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("Win", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_sims = N_SIMS
    args = sys.argv[1:]
    if "--sims" in args:
        idx = args.index("--sims")
        if idx + 1 < len(args):
            n_sims = int(args[idx + 1])

    print("Loading Poisson model...")
    model = PoissonModel.load()

    print("Loading Elo ratings...")
    elo = scrape_elo_ratings().to_dict()

    df = run_knockout(model, elo, n_sims=n_sims)

    print("\nTop 10 by championship probability:")
    top10 = df.head(10).copy()
    for col in ("R16", "QF", "SF", "Final", "Win"):
        top10[col] = top10[col].map(lambda x: f"{x:.1%}")
    print(top10.to_string(index=False))

    # Save to output/
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"knockout_probs_{ts}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
