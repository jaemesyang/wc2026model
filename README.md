# wc2026-model

Probabilistic forecasting model for the 2026 FIFA World Cup.

## Tournament format

- 48 teams across 12 groups of 4 (full round-robin within each group)
- Top 2 from each group advance automatically → 24 teams
- Best 8 third-place finishers advance → 8 more teams
- Round of 32 → Round of 16 → Quarter-finals → Semi-finals → Final (July 19, 2026)

## Model overview

### Stage 1 — Elo baseline

Team strength is seeded from current Elo ratings scraped from [eloratings.net](https://www.eloratings.net).
The Elo model converts rating differences to win/draw/loss probabilities via the standard logistic
function, with a draw probability term that peaks for closely matched teams.

### Stage 2 — Poisson goals model

A Dixon-Coles-style Poisson model is fit on Kaggle's
[international football results 1872–present](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
dataset. Each team has a log attack strength and log defense strength; the expected goals for each
side are:

```
μ_home = exp(intercept + attack_home − defense_away + home_advantage × (1 − neutral))
μ_away = exp(intercept + attack_away − defense_home)
```

Parameters are fit by maximum likelihood with exponential time-weighting (half-life ≈ 3 years) so
that recent form matters more than old results. Where a team has no historical data, the model falls
back to the Elo-based outcome sampler.

### Stage 3 — Monte Carlo group simulation

20,000 independent simulations of all 12 groups. Each match outcome is sampled from the Poisson
joint goal distribution. Group standings use the correct FIFA tiebreaker order:

1. Points
2. Goal difference
3. Goals scored
4. Head-to-head points (among tied teams only)
5. Head-to-head goal difference (among tied teams)

The 8 best third-place teams are selected by points → GD → GF across all 12 groups.

## Data sources

| Source | What | Where |
|--------|------|--------|
| Kaggle – martj42 | Historical match results (1872–present) | `data/results.csv` (download manually) |
| eloratings.net | Current team Elo ratings | Scraped and cached in `data/elo_ratings.csv` |

## Predictions

Every time `simulate.py` is run, it writes a timestamped CSV to `predictions/`. These files are
committed to git **before each tournament stage** so the full prediction history is preserved.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the Kaggle CSV and place it at `data/results.csv`:
[international-football-results-from-1872-to-2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

Then run the full pipeline:

```bash
cd src
python simulate.py        # fits model, runs 20k sims, writes predictions/
python evaluate.py        # score predictions once results exist
```

To record a real match result as it happens:

```python
from ingest import record_result
record_result({
    "date": "2026-06-12",
    "home_team": "Mexico",
    "away_team": "USA",
    "home_score": 1,
    "away_score": 2,
    "tournament": "FIFA World Cup",
    "neutral": True,
})
```

## Roadmap

- [x] Elo baseline + Poisson goal model
- [x] Monte Carlo group stage (20k sims) with correct tiebreakers
- [x] Live result ingestion via `record_result()`
- [x] Calibration scoring (Brier, RPS) and plots
- [ ] Knockout stage simulation
- [ ] Dixon–Coles low-score correction
- [ ] XGBoost meta-model on top of Poisson features

## What's not built yet

Dixon–Coles correction and XGBoost ensemble are explicitly deferred until the Elo + Poisson +
group-simulation pipeline is validated on the first matchday results.
