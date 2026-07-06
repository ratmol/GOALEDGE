# Deploying GoalEdge to Vercel

## What was fixed
Vercel never found the ASGI app: the previous attempt used a `[tool.vercel]`
entrypoint in `pyproject.toml`, which Vercel does not read (and a bare
`[project]` table can also confuse its dependency install). The working,
documented pattern is:

1. **`api/index.py`** — a 4-line shim exposing `backend.api.main:app`.
   Vercel auto-builds anything in `api/` as a Python serverless function.
2. **`vercel.json`** — rewrites every route to that function, so `/`,
   `/health`, `/value/*` etc. all hit the FastAPI app.
3. **`pyproject.toml` deleted** — it only existed for the fake entrypoint.
4. `requirements.txt` stays slim (fastapi, pandas, numpy, requests,
   python-dotenv, tiktoken). Heavy ML libs remain in `requirements-dev.txt`
   for local training; on Vercel the app falls back to the fitted Elo model.
   Verified locally with scipy/sklearn/xgboost blocked: app imports, /health,
   /predict, /value and /wc2026/fixtures all work; cold start ~2 s.

## Deploy steps
1. Commit these new files and push to `main`.
2. Redeploy on Vercel. The build should now find `backend.api.main:app` and
   install only the slim requirements.
3. The dashboard (`frontend/app.html`) is served by the API itself at `/` — no
   separate frontend build is needed.

## Set these Environment Variables in Vercel (Project → Settings → Env)
Only if you use the paid/live features (all optional — the app runs without them):
- `ODDS_API_KEY` — for `/value/live` (the-odds-api.com free tier).
- `FOOTBALL_DATA_API_KEY` — optional live results fetch.
- `KELLY_FRACTION` — staking fraction (default in code).

Do **not** set `OLLAMA_HOST` on Vercel (see caveat).

## Important caveats
- **The llama analyst won't work in the cloud.** Ollama runs locally on *your*
  machine; Vercel has no `localhost:11434`. On Vercel, "Ask the Gaffer" returns
  the deterministic offline summary, and Scout reports show the numbers-only
  fallback. Everything else — full match stats, markets, value/EV, backtest —
  works. To get the conversational analyst live you'd need a host that can run
  Ollama (your own machine, or a GPU/VM host), not Vercel.
- **Cold starts.** On a cold function the app loads the ~20k-match dataset and
  fits the model (~1–3 s). Fine for Vercel's limits; just the first request
  after idle is slower. (Optional future tweak: load the dataset once and share
  it between the Elo warm-up and the simulator instead of loading twice.)
- **`.env` is git-ignored** (confirmed) — set secrets as Vercel env vars, never
  commit them.

## Quick local check before pushing
```bash
pip install -r requirements.txt
uvicorn backend.api.main:app --reload
# open http://localhost:8000  → run a match, check /health
```
