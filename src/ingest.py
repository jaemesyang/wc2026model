"""
ingest.py — data loading, Elo scraping, and live result recording.

Data sources:
  data/results.csv   Kaggle "international football results 1872-present"
                     https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
  data/elo_ratings.csv   scraped from eloratings.net (auto-cached, 7-day TTL)
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
ELO_CACHE = DATA_DIR / "elo_ratings.csv"
RESULTS_CSV = DATA_DIR / "results.csv"
WC_RESULTS_CSV = DATA_DIR / "wc2026_results.csv"

_ELO_TTL_DAYS = 7
_REQUEST_THROTTLE = 2.0  # seconds between HTTP requests


def load_historical_results(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Kaggle international football results CSV and return a clean
    match table with columns:
      date, home_team, away_team, home_score, away_score,
      tournament, city, country, neutral
    """
    if csv_path is None:
        candidates = sorted(DATA_DIR.glob("results*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No results CSV found in {DATA_DIR}.\n"
                "Download from:\n"
                "  https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017\n"
                "and save as data/results.csv"
            )
        csv_path = candidates[0]

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()

    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])

    if "neutral" not in df.columns:
        df["neutral"] = False
    df["neutral"] = df["neutral"].astype(bool)

    for col in ("tournament", "city", "country"):
        if col not in df.columns:
            df[col] = ""

    df = df[["date", "home_team", "away_team", "home_score", "away_score",
             "tournament", "city", "country", "neutral"]]

    return df.sort_values("date").reset_index(drop=True)


def scrape_elo_ratings(force_refresh: bool = False) -> "pd.Series":
    """
    Return a Series of {team_name: elo_rating} from eloratings.net.
    Caches locally for _ELO_TTL_DAYS days.
    """
    if ELO_CACHE.exists() and not force_refresh:
        age = (datetime.now() - datetime.fromtimestamp(ELO_CACHE.stat().st_mtime)).days
        if age < _ELO_TTL_DAYS:
            return pd.read_csv(ELO_CACHE, index_col="team")["elo"]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": "wc2026-model/1.0 (research; non-commercial)"
    }

    # Primary: HTML table on the world rankings page
    ratings = _scrape_html(headers)

    # Fallback: try the JSON endpoint some versions of the site expose
    if not ratings:
        ratings = _scrape_json(headers)

    if not ratings:
        if ELO_CACHE.exists():
            print("Warning: scrape failed, using stale cache.")
            return pd.read_csv(ELO_CACHE, index_col="team")["elo"]
        raise RuntimeError(
            "Could not scrape eloratings.net and no local cache exists.\n"
            "Check your internet connection or add ratings manually to data/elo_ratings.csv."
        )

    df = (
        pd.DataFrame(ratings, columns=["team", "elo"])
        .drop_duplicates("team")
        .set_index("team")
    )
    df.to_csv(ELO_CACHE)
    print(f"Scraped {len(df)} Elo ratings → {ELO_CACHE}")
    return df["elo"]


def _scrape_html(headers: dict) -> list:
    """
    Scrape Elo ratings from eloratings.net TSV data files.

    The site is a JS SPA — no HTML table is served. Data comes from two TSV
    endpoints:
      en.teams.tsv  — code<TAB>name[<TAB>shorter variants...]
      World.tsv     — local_rank<TAB>global_rank<TAB>code<TAB>elo<TAB>...
    """
    base = "https://www.eloratings.net"
    try:
        time.sleep(_REQUEST_THROTTLE)
        teams_resp = requests.get(f"{base}/en.teams.tsv", headers=headers, timeout=30)
        teams_resp.raise_for_status()
        teams_resp.encoding = "utf-8"
    except Exception as e:
        print(f"en.teams.tsv fetch failed: {e}")
        return []

    code_to_name: dict = {}
    for line in teams_resp.text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and not parts[0].endswith("_loc"):
            code_to_name[parts[0]] = parts[1]

    try:
        time.sleep(_REQUEST_THROTTLE)
        world_resp = requests.get(f"{base}/World.tsv", headers=headers, timeout=30)
        world_resp.raise_for_status()
        world_resp.encoding = "utf-8"
    except Exception as e:
        print(f"World.tsv fetch failed: {e}")
        return []

    rows = []
    for line in world_resp.text.splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            code = parts[2]
            elo = int(parts[3])
            team = code_to_name.get(code, code)
            if 900 <= elo <= 2400 and team:
                rows.append((team, elo))
        except (ValueError, IndexError):
            continue

    return rows


def _scrape_json(headers: dict) -> list:
    """Fallback: try JSON endpoint that some builds of eloratings.net expose."""
    url = "https://www.eloratings.net/World.json"
    try:
        time.sleep(_REQUEST_THROTTLE)
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"JSON fetch failed: {e}")
        return []

    rows = []
    entries = data if isinstance(data, list) else data.get("teams", [])
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        team = entry.get("name") or entry.get("team") or entry.get("n")
        elo = entry.get("rating") or entry.get("elo") or entry.get("r")
        if team and elo:
            try:
                rows.append((str(team), int(elo)))
            except (ValueError, TypeError):
                continue
    return rows


def record_result(match: dict, csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Append a real tournament match result to the local WC 2026 results store.
    Deduplicates by (date, home_team, away_team).

    Required keys: date, home_team, away_team, home_score, away_score
    Optional keys: tournament (default "FIFA World Cup"), city, country,
                   neutral (default True)

    Example:
        record_result({
            "date": "2026-06-12",
            "home_team": "Mexico",
            "away_team": "USA",
            "home_score": 1,
            "away_score": 2,
        })
    """
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(match.keys())
    if missing:
        raise ValueError(f"match dict missing required keys: {missing}")

    if csv_path is None:
        csv_path = WC_RESULTS_CSV

    row = {
        "date": pd.Timestamp(match["date"]).strftime("%Y-%m-%d"),
        "home_team": str(match["home_team"]),
        "away_team": str(match["away_team"]),
        "home_score": int(match["home_score"]),
        "away_score": int(match["away_score"]),
        "tournament": str(match.get("tournament", "FIFA World Cup")),
        "city": str(match.get("city", "")),
        "country": str(match.get("country", "")),
        "neutral": bool(match.get("neutral", True)),
    }

    df_new = pd.DataFrame([row])

    if Path(csv_path).exists():
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=["date", "home_team", "away_team"], keep="last"
        )
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_combined = df_new

    df_combined = df_combined.sort_values("date").reset_index(drop=True)
    df_combined.to_csv(csv_path, index=False)

    print(
        f"Recorded: {row['home_team']} {row['home_score']}–"
        f"{row['away_score']} {row['away_team']} ({row['date']})"
    )
    return df_combined


def load_wc_results(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load recorded WC 2026 results. Returns empty DataFrame if none yet."""
    if csv_path is None:
        csv_path = WC_RESULTS_CSV
    if not Path(csv_path).exists():
        return pd.DataFrame(columns=[
            "date", "home_team", "away_team", "home_score", "away_score",
            "tournament", "city", "country", "neutral"
        ])
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def debug_elo_html() -> None:
    """Fetch eloratings.net/World and print the first 3000 chars of raw HTML."""
    url = "https://www.eloratings.net/World"
    headers = {"User-Agent": "wc2026-model/1.0 (research; non-commercial)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    print(f"Status: {resp.status_code}  Content-Type: {resp.headers.get('Content-Type', '?')}")
    print(f"Response length: {len(resp.text):,} chars\n")
    print("--- first 3000 chars ---")
    print(resp.text[:3000])
    print("--- end ---")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        debug_elo_html()
        sys.exit(0)

    print("Loading historical results...")
    df = load_historical_results()
    print(f"  {len(df):,} matches, {df['date'].min().date()} – {df['date'].max().date()}")
    print(f"  {df['home_team'].nunique()} unique teams")

    print("\nFetching Elo ratings (may take a moment)...")
    elo = scrape_elo_ratings()
    print(f"  Top 5:\n{elo.head()}")
