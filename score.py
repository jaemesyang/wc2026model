#!/usr/bin/env python3
"""
score.py — Batch-score WC 2026 results against pre-tournament Poisson predictions.

Usage:
    python score.py "ARG 1 KSA 2" "MEX 0 POL 0" "FRA 4 AUS 1"
    python score.py --summary

Accepts multi-word team names or 3-letter codes.
Scores locked against poisson_params_pretournament.json (snapshotted on first run).
"""

import math
import re
import shutil
import sys
import difflib
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest import record_result
from models import PoissonModel
from simulate import GROUPS

DATA_DIR = Path(__file__).parent / "data"
SCORES_CSV = DATA_DIR / "wc2026_scores.csv"

_MODEL_CACHE         = DATA_DIR / "poisson_params.json"
_PRETOURNAMENT_MODEL = DATA_DIR / "poisson_params_pretournament.json"

_SCORES_COLS = [
    "match_id", "home", "away", "hg", "ag",
    "outcome", "p_home", "p_draw", "p_away",
    "brier", "logloss", "date_scored",
]

# ---------------------------------------------------------------------------
# Team name resolution
# ---------------------------------------------------------------------------

_WC_TEAMS = [t for teams in GROUPS.values() for t in teams]

_CODE_TO_NAME: dict = {
    # A
    "MEX": "Mexico",
    "RSA": "South Africa", "ZAF": "South Africa", "SAF": "South Africa",
    "KOR": "South Korea",
    "CZE": "Czech Republic", "CZR": "Czech Republic",
    # B
    "CAN": "Canada",
    "BIH": "Bosnia and Herzegovina", "BOS": "Bosnia and Herzegovina",
    "QAT": "Qatar",
    "SUI": "Switzerland", "SWI": "Switzerland",
    # C
    "BRA": "Brazil",
    "MAR": "Morocco", "MOR": "Morocco",
    "HAI": "Haiti", "HTI": "Haiti",
    "SCO": "Scotland",
    # D
    "USA": "United States",
    "PAR": "Paraguay",
    "AUS": "Australia",
    "TUR": "Turkey",
    # E
    "GER": "Germany",
    "CUW": "Curaçao", "CUR": "Curaçao",
    "CIV": "Ivory Coast",
    "ECU": "Ecuador",
    # F
    "NED": "Netherlands", "HOL": "Netherlands",
    "JPN": "Japan",
    "SWE": "Sweden",
    "TUN": "Tunisia",
    # G
    "BEL": "Belgium",
    "EGY": "Egypt",
    "IRN": "Iran",
    "NZL": "New Zealand",
    # H
    "ESP": "Spain",
    "CPV": "Cape Verde",
    "KSA": "Saudi Arabia", "SAU": "Saudi Arabia",
    "URU": "Uruguay",
    # I
    "FRA": "France",
    "SEN": "Senegal",
    "IRQ": "Iraq",
    "NOR": "Norway",
    # J
    "ARG": "Argentina",
    "ALG": "Algeria", "DZA": "Algeria",
    "AUT": "Austria",
    "JOR": "Jordan",
    # K
    "POR": "Portugal",
    "COD": "DR Congo", "CGO": "DR Congo", "DRC": "DR Congo",
    "UZB": "Uzbekistan",
    "COL": "Colombia",
    # L
    "ENG": "England",
    "CRO": "Croatia", "HRV": "Croatia",
    "GHA": "Ghana",
    "PAN": "Panama",
}


def _resolve_team(raw: str) -> str:
    """
    Map raw input (code or full/partial name) to canonical WC 2026 team name.
    Raises ValueError listing candidates when ambiguous or not found.
    """
    raw = raw.strip()

    # Exact match
    if raw in _WC_TEAMS:
        return raw

    # Case-insensitive exact match
    raw_up = raw.upper()
    for t in _WC_TEAMS:
        if t.upper() == raw_up:
            return t

    # 3-letter code / alias
    looked_up = _CODE_TO_NAME.get(raw_up)
    if looked_up and looked_up in _WC_TEAMS:
        return looked_up

    # Fuzzy match
    matches = difflib.get_close_matches(raw, _WC_TEAMS, n=3, cutoff=0.5)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        opts = ", ".join(f'"{m}"' for m in matches)
        raise ValueError(f"Ambiguous: '{raw}' matches {opts} — use the exact name.")

    raise ValueError(
        f"No WC 2026 team found for '{raw}'.\n"
        f"  Known codes: {sorted(_CODE_TO_NAME)}\n"
        f"  Known names: {_WC_TEAMS}"
    )


# ---------------------------------------------------------------------------
# Result string parsing
# ---------------------------------------------------------------------------

def _parse_result_str(s: str) -> tuple:
    """
    Parse 'HOME hg AWAY ag' (tokens and numbers) into
    (home_raw, home_goals, away_raw, away_goals).
    HOME/AWAY may be multi-word; the two numeric tokens are the goals.
    """
    s = s.strip()
    nums = [(m.start(), m.end(), int(m.group())) for m in re.finditer(r"\b\d+\b", s)]
    if len(nums) < 2:
        raise ValueError(f"Need two numbers in '{s}'")
    i1_s, i1_e, hg = nums[0]
    i2_s, i2_e, ag = nums[-1]
    home_raw = s[:i1_s].strip()
    away_raw = s[i1_e:i2_s].strip()
    if not home_raw:
        raise ValueError(f"No home team before first number in '{s}'")
    if not away_raw:
        raise ValueError(f"No away team between numbers in '{s}'")
    return home_raw, hg, away_raw, ag


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _outcome(hg: int, ag: int) -> str:
    if hg > ag:
        return "H"
    if hg == ag:
        return "D"
    return "A"


def _brier(probs: tuple, outcome: str) -> float:
    """Multiclass Brier: sum_o (p_o - y_o)^2.  Range [0, 2]."""
    p_h, p_d, p_a = probs
    y = {"H": (1, 0, 0), "D": (0, 1, 0), "A": (0, 0, 1)}[outcome]
    return float(sum((p - yi) ** 2 for p, yi in zip((p_h, p_d, p_a), y)))


def _logloss(probs: tuple, outcome: str) -> float:
    """-ln(p_actual), clipped to avoid -inf."""
    p_h, p_d, p_a = probs
    p_actual = {"H": p_h, "D": p_d, "A": p_a}[outcome]
    return float(-math.log(max(p_actual, 1e-15)))


def _match_id(home: str, away: str) -> str:
    return f"{home}_vs_{away}"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_pretournament_model() -> PoissonModel:
    """
    Load the locked pre-tournament model.
    On first call: snapshots current data/poisson_params.json to
    data/poisson_params_pretournament.json so scores stay comparable
    even if simulate.py is re-run later.
    """
    if not _PRETOURNAMENT_MODEL.exists():
        shutil.copy(_MODEL_CACHE, _PRETOURNAMENT_MODEL)
        print(f"Locked pre-tournament model → {_PRETOURNAMENT_MODEL}")
    return PoissonModel.load(_PRETOURNAMENT_MODEL)


# ---------------------------------------------------------------------------
# Scores persistence
# ---------------------------------------------------------------------------

def _load_scores() -> pd.DataFrame:
    if SCORES_CSV.exists():
        df = pd.read_csv(SCORES_CSV)
        return df
    return pd.DataFrame(columns=_SCORES_COLS)


def _save_scores(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORES_CSV, index=False)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_HDR = (
    f"{'Match':<36}  {'Out'}  "
    f"{'P(H)':>6}  {'P(D)':>6}  {'P(A)':>6}  "
    f"{'Brier':>6}  {'LogLoss':>7}"
)
_SEP = "─" * len(_HDR)


def _format_row(r: dict) -> str:
    match_str = f"{r['home']} {r['hg']}–{r['ag']} {r['away']}"
    return (
        f"{match_str:<36}  {r['outcome']:>3}  "
        f"{r['p_home']:>6.1%}  {r['p_draw']:>6.1%}  {r['p_away']:>6.1%}  "
        f"{r['brier']:>6.4f}  {r['logloss']:>7.4f}"
    )


def _print_table(rows: list) -> None:
    print()
    print(_HDR)
    print(_SEP)
    for r in rows:
        print(_format_row(r))


def _print_aggregates(df: pd.DataFrame) -> None:
    n = len(df)
    mean_b  = df["brier"].mean()
    mean_ll = df["logloss"].mean()
    print(_SEP)
    print(f"N={n}  Mean Brier: {mean_b:.4f}  Mean LogLoss: {mean_ll:.4f}")


# ---------------------------------------------------------------------------
# Core scoring logic
# ---------------------------------------------------------------------------

def score_results(result_strs: list) -> None:
    model = _load_pretournament_model()
    scores_df = _load_scores()
    existing_ids = set(scores_df["match_id"]) if not scores_df.empty else set()

    new_rows = []
    today = date.today().isoformat()

    for s in result_strs:
        # Parse
        try:
            home_raw, hg, away_raw, ag = _parse_result_str(s)
        except ValueError as e:
            print(f"PARSE ERROR '{s}': {e}", file=sys.stderr)
            continue

        # Resolve team names
        try:
            home = _resolve_team(home_raw)
        except ValueError as e:
            print(f"TEAM ERROR (home) in '{s}': {e}", file=sys.stderr)
            continue
        try:
            away = _resolve_team(away_raw)
        except ValueError as e:
            print(f"TEAM ERROR (away) in '{s}': {e}", file=sys.stderr)
            continue

        mid = _match_id(home, away)

        # Idempotency check
        if mid in existing_ids:
            print(f"SKIP (already scored): {home} vs {away}")
            continue

        outcome = _outcome(hg, ag)
        probs   = model.predict_outcome_probs(home, away, neutral=True)
        brier   = _brier(probs, outcome)
        ll      = _logloss(probs, outcome)

        row = {
            "match_id":    mid,
            "home":        home,
            "away":        away,
            "hg":          hg,
            "ag":          ag,
            "outcome":     outcome,
            "p_home":      round(probs[0], 4),
            "p_draw":      round(probs[1], 4),
            "p_away":      round(probs[2], 4),
            "brier":       round(brier, 4),
            "logloss":     round(ll, 4),
            "date_scored": today,
        }
        new_rows.append(row)
        existing_ids.add(mid)

        # Keep wc2026_results.csv in sync for update.py re-simulation
        try:
            record_result({
                "date":       today,
                "home_team":  home,
                "away_team":  away,
                "home_score": hg,
                "away_score": ag,
            })
        except Exception as exc:
            print(f"  Warning: could not sync to wc2026_results.csv: {exc}", file=sys.stderr)

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=_SCORES_COLS)
        scores_df = pd.concat(
            [scores_df, new_df], ignore_index=True
        ) if not scores_df.empty else new_df
        _save_scores(scores_df)
        _print_table(new_rows)

    _print_aggregates(scores_df) if not scores_df.empty else print("No scored matches yet.")


# ---------------------------------------------------------------------------
# --summary command
# ---------------------------------------------------------------------------

def cmd_summary() -> None:
    df = _load_scores()
    if df.empty:
        print("No scored matches yet.")
        return
    rows = df.to_dict("records")
    _print_table(rows)
    _print_aggregates(df)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__.strip())
        sys.exit(0)
    if args[0] == "--summary":
        cmd_summary()
    else:
        score_results(args)


if __name__ == "__main__":
    main()
