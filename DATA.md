# Data & how to expand it

GoalEdge's goal model is fitted on **real international results across multiple
competitions**, not just the World Cup. Everything is merged, de-duplicated, and
weighted automatically by the loader (`backend/data/loader.py`).

## What's bundled now
- `seed_matches.csv` — World Cup history.
- `extra_results.csv` — a curated, high-confidence set of **Euros, Copa América,
  Nations League finals, and key friendlies** (2016–2024).

Current coverage: ~263 matches, 62 teams, 5 competitions.

## How matches are weighted
Each match gets `weight = competition_weight × recency_weight`:
- **Competition** — friendlies count least (0.45), qualifiers/Nations League in
  the middle, World Cup / Euros / Copa América most (1.05–1.15). Tune in
  `COMPETITION_WEIGHT`.
- **Recency** — exponential decay, 3-year half-life. Older games matter less.

So a recent Euro final shapes the model far more than a 2014 friendly — exactly
how a good analyst weighs evidence.

## Add the FULL results history (recommended next step)
For complete coverage (~45,000 internationals, 1872–present, every tournament
and friendly), drop in the well-known public dataset:

1. Get `results.csv` from the **martj42/international_results** dataset
   (GitHub / Kaggle: "International football results 1872–present").
2. Create the folder `backend/data/extra/` and place the CSV inside it.
3. That's it — the loader auto-detects any `*.csv` in `backend/data/extra/`,
   merges and weights it. Restart the API.

Required columns (the martj42 file already matches): `date, home_team,
away_team, home_score, away_score, tournament, neutral`. Team names should match
across files (e.g. "United States", "Czechia", "South Korea").

> Tip: Claude Code can fetch this CSV and drop it in `backend/data/extra/`
> for you, or wire a live API into `loader.py`'s fetch hook.

## Real corner / card / shot data (the pluggable layer)
Public results files only contain **scores**. Corners, cards, shots, etc. come
from paid stats APIs (API-Football paid, Opta) or scraping. GoalEdge keeps these
**pluggable**:

- `backend/data/team_profiles.csv` holds per-team real stat tendencies
  (`avg_shots, avg_sot, avg_corners, avg_fouls, avg_yellows, avg_offsides,
  possession_bias`). A starter sample for top teams is included.
- When a team is in that file, the simulator uses its real-ish rates (scaled by
  current attacking intensity); otherwise it models them from goal output.
- To ground all teams: have Claude Code pull season stat averages from a stats
  API and append rows to `team_profiles.csv`. No code change needed.

## What's modelled vs. data-driven (honesty note)
- **Data-driven:** goal expectations (weighted Poisson), recent form, W/D/L,
  scorelines, over/unders, BTTS, clean sheets.
- **Profile/data-driven when available, else modelled:** shots, shots on target,
  corners, fouls, cards, offsides, possession.
- **Input (default neutral):** recovery / rest days — feed real fixture
  congestion here when you have it.
