# GoalEdge: The Gaffer's Match Lab

A World Cup and international football match simulator with a built-in AI analyst and a value-betting engine.

**Live demo:** [goaledge-ten.vercel.app](https://goaledge-ten.vercel.app)
**Stack:** Python (FastAPI) plus a single self-contained frontend, deployed on Vercel.

> Status: active development. Core simulator, analyst, and value engine are live. The reinforcement-learning staking agent is in progress. Contributions are welcome (see [Contributing](#contributing)).

---

## What it does

GoalEdge doesn't just pick a winner. It simulates the whole match: goals, expected goals, shots, possession, corners, cards, and offsides, then prices every market (1X2, over/unders, BTTS, clean sheets, most-likely scorelines) from a 20,000-run Monte Carlo engine.

Three things sit on top of that engine:

1. **The simulator.** A weighted Poisson model that produces full outcome distributions per fixture, not a single verdict.
2. **The Gaffer.** An AI analyst that reads the simulation and writes a plain-English scouting report for the match.
3. **The value engine.** It compares the model against real bookmaker odds, strips the margin, and reports edge, expected value, and a fractional-Kelly stake so you know when there is value (and when there isn't).

The goal: price the entire 90 minutes the way a trader prices a market, then explain the reasoning in words.

## How it works

**The model.** A weighted Poisson model fitted on roughly 19,800 real international results: World Cups, Euros, Copa América, Nations League, and friendlies. Competition and recency weighting mean a recent knockout counts for more than a decade-old friendly. Each fixture runs 20,000 Monte Carlo simulations to produce win/draw/loss, xG, shots, possession, corners, cards, scorelines, and the derived markets. Recent form and rest-day recovery feed in, and team-specific tendencies come from a pluggable profile table.

**The Gaffer (AI analyst).** A retrieval-augmented analyst. A BM25 retriever pulls relevant team and concept notes, and the analyst injects those plus the live simulation into one prompt to produce a written read. It runs on a hosted LLM with automatic provider fallback, and degrades gracefully to a deterministic summary if no model is reachable, so the analysis layer never hard-fails.

**The value engine.** It accepts bookmaker odds in decimal, American, or fractional format, recovers fair probabilities by removing the margin, and reports edge, expected value, and a fractional-Kelly stake for every outcome.

## Quickstart

```bash
git clone https://github.com/ratmol/GOALEDGE
cd GOALEDGE
pip install -r requirements.txt
cp .env.example .env          # add any API keys you have (all optional)
uvicorn backend.api.main:app --reload
```

Open http://localhost:8000. The app runs with zero API keys: you get the full simulator and value tools. Add keys to enable the live AI analyst and live scores. For model training and the RL work, also install `requirements-dev.txt`.

Environment variables (all optional):

| Variable | Purpose |
|:---|:---|
| `CEREBRAS_API_KEY` or `OPENROUTER_API_KEY` | AI analyst (The Gaffer) |
| `FOOTBALL_DATA_API_KEY` | Live scores and fixtures |
| `ODDS_API_KEY` | Live bookmaker odds for the value engine |
| `TRAIN_TOKEN` | Shared secret required to call `POST /train` (leave unset to disable) |
| `RATE_MAX_REQUESTS`, `RATE_WINDOW_SEC` | Rate limit on quota-bound endpoints |

## Project structure

```
backend/
  api/         FastAPI routes (main.py)
  models/      match_simulator.py, elo.py, phase1.py, rl_agent.py
  features/    form, chemistry, engineered signals
  value/       value engine and odds handling
  llm/         provider chain + analyst
  rag/         BM25 retriever and corpus
  data/        loaders and CSV datasets
  backtest/    walk-forward validation and calibration
frontend/
  app.html     single self-contained dashboard served at /
scripts/       training and utility scripts
```

## API tour

| Endpoint | What you get |
|:---|:---|
| `GET /predict?home=Argentina&away=France` | Win/draw/loss probabilities |
| `GET /match/full` | Full simulated stat profile and markets |
| `GET /match/scout` | Full profile plus The Gaffer's written read |
| `GET /value/manual` | Your odds in, edge/EV/Kelly out |
| `GET /value/live` | Live odds via The Odds API, value assessment out |
| `GET /health/llm?ping=true` | Analyst diagnostics |

## Contributing

Contributions are genuinely welcome, whether that is code, data, bug reports, or ideas. This is a solo project that is more fun with others involved.

**Getting started:**

1. Fork the repo and create a branch: `git checkout -b feature/your-idea`.
2. Follow the Quickstart above to run it locally.
3. Keep changes focused, and add a short note in the PR describing what and why.
4. Open a pull request against `main`.

**Good first contributions:**

* Add or clean up team profiles in the data files.
* Improve the retrieval corpus so The Gaffer's reads are sharper.
* Add tests around the value engine math (de-vig, EV, Kelly).
* Expand the dataset with more international results, with sources cited.
* UI and accessibility improvements to `frontend/app.html`.

**Ground rules:**

* Keep the deployed runtime dependencies slim. Heavy training and RL libraries belong in `requirements-dev.txt`, imported lazily.
* Never commit secrets. All keys live in `.env`, which is gitignored.
* If you touch the model or value math, include a quick before/after so reviewers can sanity-check it.

Not sure where to start? Open an issue describing what you want to work on and we can scope it together.

## Roadmap

Planned and in progress:

* **RL staking agent (in progress):** a reinforcement-learning agent that decides how much to stake, not just what to bet. Each match is a step, the agent sizes stakes from the model's probabilities and the market odds, and it is rewarded on log bankroll growth. Trained locally and benchmarked against flat-stake and Kelly baselines, then served through a dedicated endpoint.
* **Club and league football:** extend beyond internationals to domestic leagues so the model trains on far more matches and can price club fixtures.
* **Semantic retrieval:** upgrade The Gaffer from BM25 to embedding-based search for more relevant scouting notes.
* **Dark mode and a photoreal matchday UI:** a night-stadium theme and richer visuals.
* **Test coverage and CI:** unit tests around the model and value math, wired into GitHub Actions.

## Responsible use

GoalEdge is a modelling and analysis tool, not betting advice. Every output is a probability or a distribution, never a certainty: models can be confident and wrong. If you bet, only bet what you can afford to lose.

## License

See [LICENSE](LICENSE). If none is present yet, treat the code as all-rights-reserved until one is added, and open an issue if you would like to use it.
