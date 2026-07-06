# GoalEdge — Setup & "What you need to do"

This is everything **you** need to run the project after unzipping `goaledge.zip`.

---

## 0. Prerequisites (install once)
- **Python 3.10+** — check: `python --version`
- **pip** — comes with Python
- (Optional, for the dashboard later) **Node.js 18+**
- (Optional, for free LLM previews) **Ollama** — https://ollama.com

---

## 1. Unzip
Unzip `goaledge.zip` somewhere, then open a terminal in that folder.
You should see `backend/`, `skills/`, `frontend/`, `README.md`, `requirements.txt`.

---

## 2. Create a virtual environment + install deps
```bash
# from the project root
python -m venv .venv

# activate it:
#   Windows:  .venv\Scripts\activate
#   macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

---

## 3. Set up your config keys
```bash
# copy the template, then edit .env
cp .env.example .env      # Windows: copy .env.example .env
```
Open `.env` and fill in **free** API keys (optional for now — the app runs on the
Elo baseline without them):
- `FOOTBALL_DATA_API_KEY` → register free: https://www.football-data.org/client/register
- `API_FOOTBALL_KEY` → free tier (100/day): https://www.api-football.com/

The thresholds (`TARGET_BRIER`, `MAX_PROMPT_TOKENS`, etc.) already have defaults.

---

## 4. (Optional) Install Ollama for free local previews
```bash
# after installing from https://ollama.com:
ollama pull llama3.1
ollama serve        # usually starts automatically
```
Without Ollama the app still works — it just returns probabilities and a note
instead of written previews.

---

## 5. Run things

**Test the two looping skills (no setup needed):**
```bash
python backend/skills/quant_analyzer.py     # loops until Brier <= target
python backend/skills/token_optimizer.py    # loops until prompt <= budget
```

**Run the API:**
```bash
uvicorn backend.api.main:app --reload
```
Then open:
- http://localhost:8000/docs  (interactive API)
- http://localhost:8000/predict?home=ARG&away=BRA
- http://localhost:8000/health  (shows if Ollama is detected)

**Generate a daily prediction digest:**
```bash
python backend/daily_predictions.py
# writes predictions_YYYYMMDD.md
```

---

## 6. What's still TO BUILD (stubs waiting for real code)
These are intentionally left as next steps:

1. **Data loaders** (`backend/data/`) — pull from football-data.org / API-Football
   / StatsBomb open data / CSV and normalize into one schema (club + league + intl).
2. **Phase-1 model** (`backend/models/`) — Poisson/Dixon-Coles + XGBoost that the
   Quant Analyzer loop actually trains/refines on real data.
3. **Backtesting** (`backend/backtest/`) — walk-forward validation + calibration.
4. **React dashboard** (`frontend/`) — currently an empty scaffold.
5. **Phase-2 RL agent** (`backend/models/rl/`) — bandit → PPO once Phase 1 is solid.

---

## 7. Recommended order for you (or for me to continue)
1. Get the API running (Step 5) to confirm the skeleton works.
2. Add ONE data loader (start with StatsBomb or a CSV — no API key needed).
3. Wire that data into the Quant Analyzer loop with the Poisson+XGBoost model.
4. Add the React dashboard.
5. Layer in RL.

When you're ready, send me the zip back (or just say "continue") and I'll build
the data loaders + Phase-1 model next.

---

## Troubleshooting
- **`ModuleNotFoundError`** → make sure the venv is activated and run commands
  from the project root.
- **Ollama preview says "not running"** → run `ollama serve` and `ollama pull llama3.1`.
- **API 422 errors** → check query params, e.g. `?home=ARG&away=BRA`.
