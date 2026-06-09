"""
evaluate.py — prediction logging and calibration scoring.

Workflow:
  1. After each simulate.py run, call log_predictions_from_csv() to record
     every team's advancement probability.
  2. After each stage completes, call record_outcome() for each team.
  3. compute_scores() returns Brier and RPS for all resolved predictions.
  4. plot_scores() saves a calibration chart to predictions/.

Scores:
  Brier score (binary): (p - actual)²  — lower is better; naive=0.25
  Ranked Probability Score (binary = Brier for two outcomes)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
PREDICTIONS_DIR = Path(__file__).parent.parent / "predictions"
EVAL_LOG = DATA_DIR / "evaluation_log.json"


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

def brier_score(p: float, outcome: bool) -> float:
    """Binary Brier score: (p − actual)². Range [0, 1]; lower is better."""
    return (p - float(outcome)) ** 2


def ranked_probability_score(probs: list, outcome_idx: int) -> float:
    """
    RPS for an ordinal outcome with K categories.

    probs: list of K probabilities summing to 1 (e.g. [p_home, p_draw, p_away])
    outcome_idx: index of the realised category (0-based)

    For binary predictions probs=[p, 1-p] this equals the Brier score.
    """
    k = len(probs)
    cum_pred = np.cumsum(probs)
    cum_actual = np.zeros(k)
    cum_actual[outcome_idx:] = 1.0
    return float(np.mean((cum_pred - cum_actual) ** 2))


# ---------------------------------------------------------------------------
# Prediction log CRUD
# ---------------------------------------------------------------------------

def _load_log() -> list:
    if EVAL_LOG.exists():
        with open(EVAL_LOG) as fh:
            return json.load(fh)
    return []


def _save_log(log: list) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_LOG, "w") as fh:
        json.dump(log, fh, indent=2)


def log_prediction(
    team: str,
    event: str,
    predicted_prob: float,
    source_file: Optional[str] = None,
) -> dict:
    """
    Append one prediction to the evaluation log.

    team:           team name
    event:          human-readable event label, e.g. "advance_group_stage"
    predicted_prob: probability assigned to the event occurring
    source_file:    path to the predictions CSV this came from (optional)

    Returns the logged entry dict.
    """
    if not 0.0 <= predicted_prob <= 1.0:
        raise ValueError(f"predicted_prob must be in [0, 1], got {predicted_prob}")

    entry = {
        "logged_at": datetime.now().isoformat(),
        "team": team,
        "event": event,
        "predicted_prob": float(predicted_prob),
        "source_file": str(source_file) if source_file else None,
        "actual_outcome": None,
        "scored_at": None,
    }
    log = _load_log()
    log.append(entry)
    _save_log(log)
    return entry


def log_predictions_from_csv(
    csv_path: Path,
    event: str = "advance_group_stage",
    prob_col: str = "advance_prob",
) -> int:
    """
    Bulk-log advancement probabilities from a predictions CSV.

    Skips teams already logged for the same (team, event, source_file).
    Returns the number of new entries logged.
    """
    df = pd.read_csv(csv_path)
    if "team" not in df.columns or prob_col not in df.columns:
        raise ValueError(f"CSV must have 'team' and '{prob_col}' columns")

    log = _load_log()
    existing = {
        (e["team"], e["event"], e["source_file"])
        for e in log
    }

    new_entries = []
    for _, row in df.iterrows():
        key = (row["team"], event, str(csv_path))
        if key not in existing:
            new_entries.append({
                "logged_at": datetime.now().isoformat(),
                "team": row["team"],
                "event": event,
                "predicted_prob": float(row[prob_col]),
                "source_file": str(csv_path),
                "actual_outcome": None,
                "scored_at": None,
            })

    if new_entries:
        log.extend(new_entries)
        _save_log(log)
        print(f"Logged {len(new_entries)} predictions from {csv_path.name}")
    else:
        print("No new predictions to log (already recorded).")

    return len(new_entries)


def record_outcome(team: str, event: str, advanced: bool) -> None:
    """
    Mark whether a team actually achieved the predicted event.

    Updates the most recent unscored entry matching (team, event).
    """
    log = _load_log()
    matched = False
    for entry in reversed(log):
        if (
            entry["team"] == team
            and entry["event"] == event
            and entry["actual_outcome"] is None
        ):
            entry["actual_outcome"] = bool(advanced)
            entry["scored_at"] = datetime.now().isoformat()
            matched = True
            break

    if not matched:
        print(f"Warning: no unscored prediction found for {team} / {event}")
        return

    _save_log(log)
    symbol = "✓" if advanced else "✗"
    print(f"Recorded outcome {symbol}  {team}  [{event}]")


def record_outcomes_from_dict(outcomes: dict, event: str = "advance_group_stage") -> None:
    """
    Convenience: outcomes = {team_name: True/False}.

    Example:
        record_outcomes_from_dict({
            "France": True, "Belgium": False, ...
        })
    """
    for team, advanced in outcomes.items():
        record_outcome(team, event, advanced)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_scores() -> pd.DataFrame:
    """
    Compute Brier and RPS for all resolved predictions.

    Returns a DataFrame with columns:
      logged_at, team, event, predicted_prob, actual_outcome,
      brier_score, rps, source_file
    Sorted by logged_at ascending.
    """
    log = _load_log()
    resolved = [e for e in log if e["actual_outcome"] is not None]

    if not resolved:
        print("No resolved predictions yet. Record outcomes with record_outcome().")
        return pd.DataFrame()

    rows = []
    for e in resolved:
        p = e["predicted_prob"]
        actual = bool(e["actual_outcome"])
        bs = brier_score(p, actual)
        rps = ranked_probability_score([p, 1.0 - p], 0 if actual else 1)
        rows.append({
            "logged_at": e["logged_at"],
            "team": e["team"],
            "event": e["event"],
            "predicted_prob": p,
            "actual_outcome": int(actual),
            "brier_score": round(bs, 6),
            "rps": round(rps, 6),
            "source_file": e.get("source_file"),
        })

    df = pd.DataFrame(rows)
    df["logged_at"] = pd.to_datetime(df["logged_at"])
    return df.sort_values("logged_at").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scores(
    df: Optional[pd.DataFrame] = None,
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Plot cumulative mean Brier and RPS over prediction log time.
    Saves a PNG to predictions/ and returns the path.
    """
    if df is None:
        df = compute_scores()
    if df is None or df.empty:
        print("Nothing to plot.")
        return None

    df = df.sort_values("logged_at").reset_index(drop=True)
    df["cum_brier"] = df["brier_score"].expanding().mean()
    df["cum_rps"] = df["rps"].expanding().mean()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("WC 2026 Model Calibration", fontsize=13, fontweight="bold")

    axes[0].plot(df.index + 1, df["cum_brier"], marker="o", ms=4, color="steelblue", label="Cumulative Brier")
    axes[0].axhline(0.25, color="gray", ls="--", lw=1, label="Naive baseline (0.25)")
    axes[0].set_ylabel("Brier Score")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(bottom=0)

    axes[1].plot(df.index + 1, df["cum_rps"], marker="o", ms=4, color="tomato", label="Cumulative RPS")
    axes[1].axhline(0.25, color="gray", ls="--", lw=1)
    axes[1].set_ylabel("RPS")
    axes[1].set_xlabel("Prediction index (chronological)")
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()

    if out_path is None:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = PREDICTIONS_DIR / f"{ts}_calibration.png"

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Calibration plot saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary() -> None:
    df = compute_scores()
    if df.empty:
        return

    print(f"\n{'─'*50}")
    print(f"  Resolved predictions : {len(df)}")
    print(f"  Mean Brier score     : {df['brier_score'].mean():.4f}  (naive=0.2500)")
    print(f"  Mean RPS             : {df['rps'].mean():.4f}")
    print(f"  Correct direction    : {(df['predicted_prob'] > 0.5) == df['actual_outcome'].astype(bool)}")

    by_event = df.groupby("event")[["brier_score", "rps"]].mean()
    print("\n  By event:")
    print(by_event.to_string())
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    print_summary()
    df = compute_scores()
    if not df.empty:
        plot_scores(df)
