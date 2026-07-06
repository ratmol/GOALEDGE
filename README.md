# ⚽💰 GoalEdge — The Gaffer's Match Lab

GoalEdge does **not** just pick a winner. It simulates the **entire match** —
goals, expected goals, shots, shots on target, possession, corners, fouls,
yellow/red cards, offsides — and prices every market (BTTS, over/unders, clean
sheets, most-likely scorelines) via a 20,000-run Monte Carlo engine. A free local
LLM (Ollama) turns the numbers into a written scout report.


> **NEW in v3 — the gambler's rebuild (2026-07-05).** Dixon-Coles draw
> correction, weighted MLE, chronological (out-of-time) validation, fixed
> inference features, and a full **value-bet engine**: de-vigged edge, EV,
> fractional Kelly staking, live odds via The Odds API, and WC-2026 fixtures
> with fair odds on the dashboard. Read **RESEARCH.md** for what works and
> what doesn't. To unlock the full ~48k-match training set run once:
> `python scripts/download_data.py` then `python train.py`.

> **NEW in v2:** full-stats `MatchSimulator` (`backend/models/match_simulator.py`),
> rich `/match/full` + `/match/scout` API endpoints, and a redesigned single-file
> dashboard (`frontend/app.html`) fronted by the persona below.

---

## 1. The Persona — "The Gaffer"

> **The Gaffer** — a hybrid football tactician and quantitative trader.
> *"I don't pick winners — I price the whole match. Goals, corners, cards,
> territory. Tell me the two sides and I'll show you how the 90 minutes breathe."*
>
> Reads a match like a manager, prices it like a trader: every output is a
> probability or a distribution, never a flat verdict.

**Football brain**
- Reads tactics, fixture congestion, travel/recovery load, momentum and streaks.
- Knows that xG > goals, that home/neutral venue matters, and that a "form table"
  beats a league table for short-term prediction.
- Treats injuries, squad depth, and rest days as first-class signals.

**Finance brain**
- Frames every prediction as a probability, not a verdict ("62% win" not "they win").
- Thinks in expected value, calibration, and edge vs. the implied odds of the market.
- Applies risk discipline: confidence intervals, Kelly-style sizing, and never
  confusing a confident model with a correct one.

**Operating principles**
1. **Probabilities, not prophecies.** Every output is a calibrated distribution.
2. **Backtest before you trust.** No model ships without out-of-sample validation.
3. **Explainability matters.** A prediction without a "why" is a coin flip with a logo.
4. **Free-first.** Default to zero-cost data + local inference (Ollama) before paid APIs.

---

## 2. Skills & Capabilities

| Domain | What it does |
|---|---|
| **Data engineering** | Pulls + normalizes data from free football APIs, StatsBomb open data, and CSV datasets into a unified schema. |
| **Feature engineering** | Builds recovery/rest features, rolling form (xG, points, goal diff), Elo, streaks, head-to-head, squad availability. |
| **Predictive modeling** | Poisson/Dixon-Coles for scorelines + gradient-boosting (XGBoost/LightGBM) for win/draw/loss probabilities. |
| **Reinforcement learning** | (Phase 2) An agent that "stakes" predictions and learns a policy from match outcomes, optimizing calibrated EV. |
| **LLM analysis (Ollama)** | Generates natural-language match previews and reasoning from the model's feature vector — free + local. |
| **Quant Analyzer (skill)** | Mathematically rigorous module that *loops* — refines features/model until validation Brier ≤ target. See `skills/quant-analyzer.md`. |
| **Token Optimizer (skill)** | *Loops* to compress every LLM prompt under a token budget without losing signal. See `skills/token-optimizer.md`. |
| **Calibration & backtesting** | Brier score, log-loss, reliability diagrams, walk-forward validation. |
| **API / app delivery** | FastAPI backend serving predictions; React frontend dashboard. |

> **Team chemistry** is a first-class feature (`backend/features/chemistry.py`):
> lineup continuity, squad stability, and shared on-pitch minutes — public-data
> proxies for how well a side actually gels.

---

## 3. Data Sources (free-first, per your selection)

We use a layered strategy so the app works fully offline but can pull live data.

**A. Free football APIs (live / recent)**
- [football-data.org](https://www.football-data.org/) — free tier, fixtures,
  results, standings. Requires a free API key (rate-limited).
- [API-Football](https://www.api-football.com/) — free tier (100 req/day) for
  fixtures, lineups, injuries, statistics.

**B. StatsBomb Open Data (rich, event-level)**
- [statsbomb/open-data](https://github.com/statsbomb/open-data) — free, includes
  past World Cups. Great for xG, player events, and recovery/load proxies.

**C. CSV / manual datasets (offline, reproducible)**
- Kaggle historical World Cup results, FIFA rankings, and squad CSVs.
- Used to seed Elo, head-to-head, and to run fully offline backtests.

> **Recovery data note:** true biometric recovery isn't public. We proxy it with
> days-since-last-match, minutes load, travel distance between host cities, and
> fixture congestion — all derivable from the sources above.

---

## 4. Model Approach (phased, per your selection)

### Phase 1 — Classic ML + Ollama explanations
- **Win/Draw/Loss:** gradient-boosting classifier on engineered features.
- **Scorelines:** Dixon-Coles / Poisson for exact-score and over/under.
- **Elo baseline:** sanity-check + cold-start for teams with little data.
- **Ollama:** feed the model's features + probabilities to a local LLM
  (e.g. `llama3.1` or `mistral`) to write the match preview and reasoning.
  Zero cost, runs on your machine.

### Phase 2 — Reinforcement learning
- **Setup:** episodic — each match is a step; the agent outputs a predicted
  distribution / "stake," receives reward based on calibration + EV vs. outcome.
- **Algorithms:** start with contextual bandits → move to PPO (Stable-Baselines3)
  if we model a tournament as a sequence.
- **Reward design:** negative log-loss / Brier-based reward so the agent is
  rewarded for *calibration*, not just picking winners.
- **Why phased:** Phase 1 gives a strong, explainable baseline. RL only earns its
  place if it beats that baseline out-of-sample.

---

## 5. LLM / Inference: Ollama (free, local) vs. API

**Recommended: Ollama (no cost).**
- Install: https://ollama.com → `ollama pull llama3.1` (or `mistral`, `phi3`).
- Runs locally, no API key, no per-token cost — perfect for match previews and
  reasoning over the feature vector.
- The backend calls `http://localhost:11434/api/generate`.

**When a paid API helps:** larger context, higher quality long-form analysis, or
if you don't want to run a local model. The code will keep the LLM layer behind a
single interface so you can swap Ollama ↔ a hosted API with one config change.

> ⚠️ **Important:** the LLM does **not** make the prediction. The math model
> produces probabilities; the LLM only explains them. This keeps predictions
> reproducible and auditable.

---

## 6. Connectors & API Requests

**Outbound API calls the app will make**
- `GET` football-data.org / API-Football → fixtures, lineups, injuries, stats
  (needs free API key in `.env`).
- `GET` StatsBomb open-data (raw GitHub JSON, no key).
- `POST` `localhost:11434` → Ollama for text generation (no key).

**Internal API (what the app serves)** — FastAPI:
- `GET /predict?home=ARG&away=FRA` → probabilities + expected scoreline.
- `GET /match/{id}/analysis` → Ollama-generated preview.
- `GET /standings`, `GET /form/{team}` → supporting data.

**Connectors (your Cowork tools):** if you later want predictions delivered to
Slack, Google Sheets, or a calendar (e.g. "post tomorrow's predictions each
morning"), those can be wired via Cowork connectors + a scheduled task. Just say
the word and I'll search the connector registry.

**Secrets:** all API keys live in a `.env` file (git-ignored). Never committed.

---

## 7. Proposed Architecture

```
worldcup-predictor/
├── backend/
│   ├── data/            # loaders: football_data.py, statsbomb.py, csv_loader.py
│   ├── features/        # rest, form, elo, congestion, h2h
│   ├── models/          # poisson.py, xgb_model.py, elo.py, rl/ (phase 2)
│   ├── llm/             # ollama_client.py (swappable interface)
│   ├── api/             # FastAPI routes
│   └── backtest/        # walk-forward validation + calibration metrics
├── frontend/            # React dashboard (matchups, probabilities, previews)
├── data_cache/          # downloaded + CSV datasets (git-ignored)
├── .env.example
└── README.md
```

**Stack:** Python (FastAPI, pandas, scikit-learn, XGBoost, Stable-Baselines3) +
React (Vite). Ollama for local LLM.

---

## 8. Roadmap

- [ ] **M0 – Scaffold:** repo structure, `.env.example`, dependency setup.
- [ ] **M1 – Data:** loaders for CSV + StatsBomb + one free API; unified schema.
- [ ] **M2 – Features:** rest/recovery, rolling form, Elo, streaks, H2H.
- [ ] **M3 – Phase 1 model:** Poisson + XGBoost, with backtest + calibration.
- [ ] **M4 – Ollama layer:** match previews from feature vectors.
- [ ] **M5 – API + React UI:** serve predictions, build the dashboard.
- [ ] **M6 – Phase 2 RL:** bandit → PPO agent, benchmarked vs. M3 baseline.
- [ ] **M7 – Delivery:** optional connectors + scheduled daily predictions.

---

## 9. Decisions (locked in)

1. **Scope:** World Cup **+ club/league matches** as extra training data — more
   samples = a better-calibrated model.
2. **Team chemistry:** included as a feature (lineup continuity, squad stability,
   minutes-played-together).
3. **API keys:** designed around **free tiers**; add your keys to `.env` later
   (`.env.example` + signup links provided).
4. **Ollama:** not yet installed — please install from https://ledger… → run
   `ollama pull llama3.1`. (Install: https://ollama.com) The app **runs without it** (probabilities only) and
   lights up previews once it's running.
5. **Delivery:** daily predictions pushed on a schedule. Because text is
   generated **locally by Ollama**, this costs ~no API tokens. See
   `backend/daily_predictions.py`.

## 10. Skills that *loop until a threshold* (your request)

Unlike static instructions, both custom skills run iterative loops:

- **Quant Analyzer** keeps applying improvement actions (form windows → recovery →
  chemistry → tuning → recalibration → blending) until validation
  **Brier ≤ `TARGET_BRIER`** (default 0.20) or it hits `MAX_QUANT_ITERATIONS`.
- **Token Optimizer** keeps applying compression actions until the prompt is
  **≤ `MAX_PROMPT_TOKENS`** (default 1500), never dropping the must-keep
  probabilities.

Both are configurable in `.env` and have runnable demos:
`python backend/skills/quant_analyzer.py` and
`python backend/skills/token_optimizer.py`.

---

*Built with a tactician's eye and a trader's discipline. Predictions are
probabilities — bet responsibly, and trust the backtest.*
