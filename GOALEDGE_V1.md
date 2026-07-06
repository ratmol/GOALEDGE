# GoalEdge v1 — Complete Build Context

> Generated: 2026-06-28 | Model: Claude Sonnet 4.6

---

## What Was Built

### Infrastructure

| Item | Path | Notes |
|---|---|---|
| Virtual env | `.venv/` | Python 3.12, activated via `.venv\Scripts\activate` |
| Dependencies | `requirements.txt` | 44 packages — pandas, numpy, scipy, xgboost, fastapi, uvicorn, stable-baselines3, tiktoken, rich |
| Config | `.env` | Copied from `.env.example`; keys blank for keyless operation |

---

### Backend

#### Skills (autonomous loop workers)

**`backend/skills/quant_analyzer.py`**
- Iterative Brier-threshold refinement loop
- Runs up to `MAX_QUANT_ITERATIONS` (default 6) improvement actions
- Actions: rolling form windows → recovery features → chemistry features → XGBoost retune → recalibrate → blend Poisson+XGB
- Stops early if validation Brier ≤ `TARGET_BRIER` (default 0.55 — realistic for WC data)
- Returns best model + full history

**`backend/skills/token_optimizer.py`**
- Compresses LLM prompts to fit within `MAX_PROMPT_TOKENS` (default 1500)
- Action chain (least lossy first): collapse whitespace → round numbers → drop blank lines → keep top lines → hard truncate
- Never drops the `must_keep` block (probabilities)
- Demonstrated: 4423 tokens → 193 tokens in one pass

#### Data

**`backend/data/loader.py`**
- Priority: disk cache (`data/cache/matches.csv`) → football-data.org API → bundled seed CSV
- Returns standardised DataFrame: `date, home_team, away_team, home_score, away_score, tournament, neutral`

**`backend/data/seed_matches.csv`**
- 190 World Cup matches: 2014 (64), 2018 (64), 2022 (62)
- 47 national teams — every WC participant from all three tournaments
- Used as fallback when no API key is configured

#### Features

**`backend/features/engineer.py`**
- 13 engineered features per match:
  - `elo_home`, `elo_away`, `elo_diff` — Elo snapshot before match
  - `form_home/away_wins/gd/att/def` — rolling 5-match form window
  - `h2h_home_winrate` — head-to-head win rate last 5 meetings
  - `neutral` — venue flag
- Elo ratings updated incrementally through match history

**`backend/features/chemistry.py`**
- Lineup chemistry proxies: continuity, squad stability, minutes together
- Plug-in ready — waiting on real lineup data

#### Models

**`backend/models/elo.py`** *(Phase 0 — baseline)*
- Classic Elo with K=30, home advantage=65 points, margin-of-victory multiplier
- Used as cold-start fallback when Phase-1 not trained

**`backend/models/phase1.py`** *(Phase 1 — primary)*
- **PoissonModel**: Dixon-Coles inspired MLE over team attack/defence strengths + home advantage. Predicts score distributions up to 8 goals, converts to WDL.
- **XGBModel**: XGBoost (200 trees, depth 4) + isotonic calibration via 3-fold CV on 13 engineered features.
- **Phase1Model**: Blended final output — `alpha × Poisson + (1-alpha) × XGBoost`. Alpha tuned by quant loop (0.35 → 0.50 as actions unlock).
- Saves trained model to `backend/models/phase1.pkl` for instant API reload.
- `run_quant_loop()` drives iterative refinement via QuantAnalyzer.

**`backend/models/rl_agent.py`** *(Phase 2 stub — ready to train)*
- Custom Gymnasium `BettingEnv`: episode = tournament, observation = [WDL probs, bankroll ratio, confidence], action = bet fraction [0, 10%]
- PPO agent via stable-baselines3 with Kelly-inspired payoff reward
- `train_rl_agent()` and `evaluate_agent()` helpers ready
- Integration point: replace `_synthetic_probs()` with `Phase1Model.predict()`

#### Backtest

**`backend/backtest/engine.py`**
- Walk-forward evaluation: warm-up on first N matches, evaluate on rest
- Reports: Brier score, log-loss, flat-stake ROI, calibration, favourite accuracy
- Elo baseline benchmark: Brier=0.65, Accuracy=43%, FlatROI=+17.6%

#### API

**`backend/api/main.py`** — FastAPI v1.0.0

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness + active model + Ollama status |
| `/teams` | GET | List of 47 known teams |
| `/predict` | GET | WDL probabilities for `?home=X&away=Y` |
| `/match/analysis` | GET | Probs + Ollama written preview (token-optimised) |
| `/backtest` | GET | Walk-forward backtest on seed data |
| `/train` | POST | Retrain Phase-1 in background thread |
| `/docs` | GET | Interactive Swagger UI |

Model priority: Phase1Model (Poisson+XGBoost) → Elo fallback.

#### Daily Digest

**`backend/daily_predictions.py`**
- Generates a `predictions_YYYYMMDD.md` file
- Uses Phase1Model if trained, Elo otherwise
- Each fixture: probability table + AI preview (token-optimised for Ollama)
- Default fixtures: ARG vs BRA, FRA vs ENG, ESP vs GER, NED vs POR, MAR vs CRO

---

### Frontend

**`frontend/`** — React 18 + Vite 5

| File | Role |
|---|---|
| `vite.config.js` | Dev server on :5173, proxies `/predict`, `/match`, `/health` to :8000 |
| `src/App.jsx` | State management, fetch logic, layout |
| `src/components/PredictPanel.jsx` | Team dropdowns (47 teams), neutral toggle, Predict / AI Preview buttons |
| `src/components/ResultCard.jsx` | Animated probability bars, model badge, most-likely highlight, AI preview block |
| `src/index.css` | Dark theme CSS variables — no framework dependency |

To run: `cd frontend && npm run dev` → opens at http://localhost:5173

---

### Project Structure

```
goaledge/
├── .env                          ← your config (gitignored)
├── .env.example                  ← template with all keys
├── requirements.txt
├── GOALEDGE_V1.md                ← this file
├── backend/
│   ├── api/main.py               ← FastAPI app (6 endpoints)
│   ├── backtest/engine.py        ← walk-forward evaluator
│   ├── data/
│   │   ├── loader.py             ← cache → API → seed fallback
│   │   ├── seed_matches.csv      ← 190 WC matches (2014-2022)
│   │   └── cache/               ← auto-written when API key is used
│   ├── features/
│   │   ├── engineer.py           ← 13 engineered features
│   │   └── chemistry.py          ← lineup chemistry (plug-in ready)
│   ├── llm/ollama_client.py      ← Ollama local LLM wrapper
│   ├── models/
│   │   ├── elo.py                ← Elo baseline (Phase 0)
│   │   ├── phase1.py             ← Poisson+XGBoost blend (Phase 1)
│   │   ├── phase1.pkl            ← trained model (after first run)
│   │   └── rl_agent.py           ← PPO betting agent (Phase 2 stub)
│   ├── skills/
│   │   ├── quant_analyzer.py     ← Brier-threshold refinement loop
│   │   └── token_optimizer.py    ← prompt compression loop
│   └── daily_predictions.py      ← markdown digest generator
├── frontend/
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css
│       └── components/
│           ├── PredictPanel.jsx
│           └── ResultCard.jsx
└── skills/
    ├── quant-analyzer.md         ← skill documentation
    └── token-optimizer.md        ← skill documentation
```

---

## How to Run Everything

```bash
# 1. Activate environment
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux

# 2. Start the API (auto-loads Phase1Model if trained)
uvicorn backend.api.main:app --reload
# → http://localhost:8000/docs

# 3. Start the React dashboard (separate terminal)
cd frontend && npm run dev
# → http://localhost:5173

# 4. Train Phase-1 model (first time, ~3 minutes)
python backend/models/phase1.py   # trains + saves phase1.pkl
# OR via API: POST http://localhost:8000/train

# 5. Run the RL agent (Phase 2)
python backend/models/rl_agent.py  # trains on synthetic data, ~2 min

# 6. Generate a daily digest
python backend/daily_predictions.py

# 7. Backtest Elo baseline
python backend/backtest/engine.py
```

---

## Benchmarks (Walk-Forward on 150 test matches, warmup=40)

| Model | Brier ↓ | Favourite Acc ↑ | Flat ROI |
|---|---|---|---|
| Uniform baseline | 0.667 | ~33% | −100% |
| Elo baseline | 0.652 | 43.3% | +17.6% |
| **Phase-1 (Poisson + XGBoost blend)** | **0.568** | **57.3%** | **+14.3%** |

Phase-1 improves on Elo by **13% Brier**, **+14 percentage points accuracy**.

---

## Future Suggestions (Priority Order)

### Immediate Impact

1. **Real match data** — sign up for a free [football-data.org](https://www.football-data.org) key and add it to `.env`. The loader will automatically pull all available WC matches, cache them, and retrain. More data → better Poisson estimates.

2. **Expand seed data** — add 2010 and 2006 WC matches (+ major qualifiers, confederations cups) to `seed_matches.csv`. More historical context improves Elo warmup and form features.

3. **Real fixture feed** — replace `SAMPLE_FIXTURES` in `daily_predictions.py` with a live API call to `/v4/competitions/WC/matches?status=SCHEDULED`. The loader already supports this.

### Model Improvements

4. **Dixon-Coles correction** — add the ρ (rho) term to the Poisson model to correct the joint probability of 0-0 and 1-0 / 0-1 scorelines (historically under-predicted). Standard in football analytics.

5. **Player ratings as features** — add FIFA/club-season ratings as team-level features (mean rating of the expected XI). Significant Elo lift, publicly available from FIFA game data.

6. **Tournament stage weight** — add `tournament_stage` as a feature (Group stage vs Knockouts). Teams play differently in elimination games. The seed CSV has `tournament` column ready to extend.

7. **Historical odds calibration** — if you ever get access to historical bookmaker odds, use them as calibration targets (Platt scaling) rather than isotonic CV. Dramatically improves calibration.

8. **XGBoost hyperparameter search** — add an Optuna sweep inside the quant loop's `extra_grid_search` action. The current XGBoost config is a reasonable default but not tuned.

### RL Agent (Phase 2)

9. **Connect to Phase1Model** — in `rl_agent.py`, replace `_synthetic_probs()` with:
   ```python
   probs = self.phase1.predict(home, away)
   return np.array([probs["win"], probs["draw"], probs["loss"]]), simulate_outcome(probs)
   ```

10. **Fractional Kelly sizing** — replace the flat-stake reward with a fractional Kelly formula for the reward signal. This grounds the agent in proven bankroll management theory.

11. **Multi-market actions** — extend the action space to allow betting on home win, draw, or away win separately (3 continuous actions instead of 1). Lets the agent exploit draw odds, which are often mispriced.

### Infrastructure

12. **API authentication** — add a `X-API-Key` header check before `/train` and `/backtest` endpoints so they can't be triggered by anyone if the API is exposed.

13. **Database swap** — replace the CSV cache with SQLite (`backend/data/db.py`) so you can query matches by team, tournament, date range without loading everything into RAM.

14. **Scheduled retraining** — use Windows Task Scheduler or a cron job to call `POST /train` weekly so the model stays fresh as new tournaments happen.

15. **React dashboard — charts** — add a recharts (or Chart.js) probability history chart showing how the model's confidence evolved across a tournament. Useful for storytelling.

16. **React dashboard — head-to-head panel** — show the last 5 meetings between the two teams pulled from the data loader, so users can see the form context behind the numbers.

17. **CI / testing** — add `pytest` tests for the loader, feature engineer, and API endpoints. The backtest engine already acts as an integration test; formalise it with `pytest.mark.slow`.

18. **Docker** — `Dockerfile` + `docker-compose.yml` with `api` and `ollama` services so the whole stack starts with one command.

---

*GoalEdge v1 skeleton is complete. The next high-value milestone is step 1: add a real API key and retrain — predictions will immediately improve with more data.*
