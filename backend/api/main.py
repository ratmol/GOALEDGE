"""
FastAPI app — GoalEdge v3 · "The Gaffer's Match Lab".

  GET  /health                  — status
  GET  /teams                   — teams present in the data
  GET  /predict                 — WDL probabilities (legacy/simple)
  GET  /match/analysis          — WDL + Ollama preview (legacy)
  GET  /match/full              — FULL stat profile (goals, xG, shots,
                                  possession, corners, cards, fouls, offsides,
                                  form, recovery + markets)
  GET  /match/scout             — full profile + Ollama scout report
  GET  /value/manual            — your bookmaker odds → edge, EV, Kelly stake
  GET  /value/live              — live odds (The Odds API) → edge, EV, Kelly
  GET  /value/market            — side markets (over/under, BTTS) vs your odds
  GET  /wc2026/fixtures         — upcoming WC2026 fixtures + model fair odds
  GET  /backtest                — walk-forward backtest
  POST /train                   — (re)train Phase-1 model in background

Run:  uvicorn backend.api.main:app --reload   (serves frontend/app.html at /)
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from collections import defaultdict, deque

from fastapi import (FastAPI, BackgroundTasks, Body, HTTPException, Request,
                     Header)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.append(str(Path(__file__).resolve().parents[2]))

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DIST = _PROJECT_ROOT / "frontend" / "dist"
_DASHBOARD = _PROJECT_ROOT / "frontend" / "app.html"

from backend.models.elo import EloModel
from backend.models.match_simulator import (build_default_simulator,
                                            MatchSimulator, TeamModel)
from backend.llm.analyst import Analyst
from backend.llm.provider import get_llm, describe as llm_describe, diagnose as llm_diagnose
from backend.rag.retriever import build_retriever
from backend.llm.ollama_client import OllamaClient, write_preview
from backend.skills.token_optimizer import TokenOptimizer
from backend.value.engine import assess_1x2, assess_market
from backend.value import odds_api
from backend.live import scores as live_scores

app = FastAPI(title="GoalEdge — The Gaffer's Match Lab", version="3.0.0")
N_SIMS = int(os.getenv("N_SIMS", "20000"))
_origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
_allowed_origins = ["*"] if _origins_env == "*" else [
    o.strip() for o in _origins_env.split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins,
                   allow_methods=["GET", "POST"], allow_headers=["*"])

# --- Best-effort in-memory rate limiting on quota-bound endpoints. ------------
# Serverless instances are ephemeral, so this is a soft guard against casual
# abuse (and runaway API cost), not a hard security control. Tune via env.
_RL_WINDOW = float(os.getenv("RATE_WINDOW_SEC", "60"))
_RL_MAX = int(os.getenv("RATE_MAX_REQUESTS", "60"))
_RL_HITS: dict = defaultdict(deque)
_RL_PATHS = ("/live", "/value/live", "/match/scout", "/analyst/ask", "/train")


@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    path = request.url.path
    if any(path.startswith(pfx) for pfx in _RL_PATHS):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        hits = _RL_HITS[ip]
        while hits and now - hits[0] > _RL_WINDOW:
            hits.popleft()
        if len(hits) >= _RL_MAX:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=429,
                                content={"detail": "rate limit exceeded, slow down"})
        hits.append(now)
    return await call_next(request)

_elo = EloModel()
# Load the match history ONCE and share it between the Elo warm-up and the
# simulator (avoids reading the ~20k-row dataset twice on cold start).
try:
    from backend.data.loader import load_matches
    _df = load_matches(verbose=False)
    for _r in _df.sort_values("date").itertuples(index=False):
        _elo.update(_r.home_team, _r.away_team, int(_r.home_score),
                    int(_r.away_score), bool(getattr(_r, "neutral", True)))
    _simulator = MatchSimulator(TeamModel().fit(_df), n_sims=N_SIMS)
    # Active national teams only: >=15 matches since 2015, CONIFA excluded.
    _recent = _df[(_df["date"] >= "2015-01-01")
                  & ~_df["tournament"].astype(str).str.contains("CONIFA", case=False, na=False)]
    import pandas as _pd
    _counts = _pd.concat([_recent["home_team"], _recent["away_team"]]).value_counts()
    _active_teams = sorted(_counts[_counts >= 15].index)
except Exception as _e:
    print(f"[api] startup data load failed, falling back: {_e}")
    _simulator = build_default_simulator(n_sims=N_SIMS)
    _active_teams = None
_phase1 = None
_training_active = False


def _try_load_phase1():
    global _phase1
    try:
        from backend.models.phase1 import Phase1Model, MODEL_PATH
        if MODEL_PATH.exists():
            _phase1 = Phase1Model.load()
            print("[api] Phase1Model loaded")
    except Exception as e:
        print(f"[api] Phase1Model not loaded: {e}")


_try_load_phase1()


def _predict(home, away, neutral):
    if _phase1 is not None:
        try:
            return _phase1.predict(home, away, neutral), "phase1_blend"
        except Exception:
            pass
    return _poisson_wdl(home, away, neutral), "poisson_dc"


def _poisson_wdl(home, away, neutral, max_goals: int = 9):
    """Analytic WDL from the simulator's form/venue-adjusted goal rates.
    Pure math — replaces the crude Elo draw-band heuristic on Vercel."""
    import math
    lam_h, lam_a = _simulator.adjusted_lambdas(home, away, neutral)
    ph = [math.exp(-lam_h) * lam_h ** i / math.factorial(i) for i in range(max_goals)]
    pa = [math.exp(-lam_a) * lam_a ** i / math.factorial(i) for i in range(max_goals)]
    win = sum(ph[i] * pa[j] for i in range(max_goals) for j in range(i))
    draw = sum(ph[i] * pa[i] for i in range(max_goals))
    loss = sum(ph[j] * pa[i] for i in range(max_goals) for j in range(i))
    s = win + draw + loss
    return {"win": win / s, "draw": draw / s, "loss": loss / s}


_llm = get_llm()
_retriever = build_retriever(_simulator.tm)
_optimizer = TokenOptimizer()
_analyst = Analyst(_simulator, _llm, retriever=_retriever)


def _require_teams(home: str, away: str):
    """Boundary validation: reject same-team / unknown-team requests."""
    if home == away:
        raise HTTPException(status_code=400,
                            detail="Pick two different teams")
    missing = [t for t in (home, away) if t not in set(_simulator.tm.teams)]
    if missing:
        raise HTTPException(status_code=404,
                            detail=f"Unknown team(s): {', '.join(missing)}. See /teams.")


def _require_odds(*odds):
    if any(o is not None and o <= 1.0 for o in odds):
        raise HTTPException(status_code=422,
                            detail="Decimal odds must be greater than 1.0")


_odds_cache: dict = {}   # (home, away) -> (ts, result); warm-instance TTL cache


def _cached_consensus(home: str, away: str, ttl: float = 60.0):
    key = (home, away); now = time.time()
    hit = _odds_cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    res = odds_api.consensus_odds(home, away)
    _odds_cache[key] = (now, res)
    return res


@app.get("/health")
def health():
    return {"status": "ok",
            "model": "phase1_blend" if _phase1 else "elo_baseline",
            "simulator": "monte_carlo",
            "teams": len(_simulator.tm.teams),
            "matches": _simulator.tm.n_matches,
            "llm_ready": _llm.available(),
            "llm_provider": llm_describe(_llm),
            "odds_api": odds_api.available(),
            "training_active": _training_active}


@app.get("/teams")
def teams():
    ts = _active_teams or _simulator.tm.teams
    return {"teams": ts, "count": len(ts)}


@app.get("/predict")
def predict(home: str, away: str, neutral: bool = True):
    probs, model = _predict(home, away, neutral)
    return {"home": home, "away": away, "neutral": neutral,
            "probabilities": {k: round(v, 3) for k, v in probs.items()},
            "model": model}


@app.get("/match/analysis")
def analysis(home: str, away: str, neutral: bool = True):
    probs, model = _predict(home, away, neutral)
    must = (f"PROBS {home} win={probs['win']:.2f} draw={probs['draw']:.2f} "
            f"{away} win={probs['loss']:.2f}")
    ctx = (f"Match: {home} vs {away} ({'neutral' if neutral else 'home'}).\n"
           f"Model: {model}. Write a short preview.")
    opt = _optimizer.optimize(ctx, must_keep=must)
    return {"home": home, "away": away, "neutral": neutral,
            "probabilities": {k: round(v, 3) for k, v in probs.items()},
            "model": model, "preview": write_preview(_llm, opt.prompt),
            "prompt_tokens": opt.final_tokens}


@app.get("/match/full")
def match_full(home: str, away: str, neutral: bool = True,
               rest_home: float = 4.0, rest_away: float = 4.0):
    """Full match-stat profile + form, recovery and markets."""
    _require_teams(home, away)
    return _simulator.simulate(home, away, neutral, rest_home, rest_away)


@app.get("/match/scout")
def match_scout(home: str, away: str, neutral: bool = True,
                rest_home: float = 4.0, rest_away: float = 4.0):
    """Full stat profile + an Ollama-written scout report."""
    _require_teams(home, away)
    sim = _simulator.simulate(home, away, neutral, rest_home, rest_away)
    o, m, ts = sim["outcome"], sim["markets"], sim["team_stats"]
    must = (f"{home} win {o['home_win']}% / draw {o['draw']}% / "
            f"{away} win {o['away_win']}%. xG {sim['expected_goals']['home']}-"
            f"{sim['expected_goals']['away']}.")
    ctx = (f"Match: {home} vs {away} ({'neutral' if neutral else 'home'}).\n"
           f"Form index {home} {sim['form']['home']['rating']}, "
           f"{away} {sim['form']['away']['rating']}.\n"
           f"Possession {ts['possession']['home']}-{ts['possession']['away']}. "
           f"Shots {ts['shots']['home']}-{ts['shots']['away']}. "
           f"Corners {ts['corners']['home']}-{ts['corners']['away']}.\n"
           f"Markets: total goals {m['total_goals_avg']}, "
           f"over2.5 {m['over_2_5_goals']}%, BTTS {m['btts_yes']}%, "
           f"over10.5 corners {m['over_10_5_corners']}%.\n"
           "Write a punchy scout report on how this plays out.")
    opt = _optimizer.optimize(ctx, must_keep=must)
    sim["scout_report"] = write_preview(_llm, opt.prompt)
    sim["prompt_tokens"] = opt.final_tokens
    return sim


@app.get("/backtest")
def backtest(warmup: int = 40):
    from backend.data.loader import load_matches
    from backend.backtest.engine import run_backtest
    df = load_matches(verbose=False)

    def _wrap(home, away, neutral=True):
        probs, _ = _predict(home, away, neutral)
        return probs

    result = run_backtest(df, _wrap, warmup=warmup)
    return {"model": "phase1_blend" if _phase1 else "elo_baseline",
            "note": ("in-sample where the model was trained on this history — "
                     "trust the out-of-time metrics from /train instead"),
            "n_matches": result.n, "brier": round(result.brier, 4),
            "logloss": round(result.logloss, 4),
            "flat_roi": round(result.flat_roi, 4),
            "favourite_accuracy": round(
                sum(d["correct"] for d in result.details) / result.n, 3),
            "calibration": {k: {"expected": round(v[0], 3),
                                "actual": round(v[1], 3)}
                            for k, v in result.calibration.items()}}


@app.post("/train")
def train(background_tasks: BackgroundTasks, x_train_token: str = Header(default="")):
    # Retraining is expensive, so it requires a shared secret. If TRAIN_TOKEN is
    # unset the endpoint is disabled entirely (safe default for public deploys).
    _expected = os.getenv("TRAIN_TOKEN", "")
    if not _expected or x_train_token != _expected:
        raise HTTPException(status_code=403,
                            detail="training endpoint disabled or invalid token")
    global _training_active
    if _training_active:
        raise HTTPException(status_code=409, detail="Training already in progress")

    def _do_train():
        global _phase1, _training_active
        _training_active = True
        try:
            from backend.models.phase1 import run_quant_loop
            _phase1 = run_quant_loop()
            print("[api] Phase1Model retrained")
        except Exception as e:
            print(f"[api] training failed: {e}")
        finally:
            _training_active = False

    background_tasks.add_task(_do_train)
    return {"status": "training started", "endpoint": "/health to poll"}


@app.get("/value/manual")
def value_manual(home: str, away: str, odds_home: float, odds_draw: float,
                 odds_away: float, neutral: bool = True,
                 bankroll: float = 100.0, kelly_fraction: float = 0.25):
    """Type in your bookmaker's 1X2 odds -> edge, EV and Kelly stake."""
    _require_teams(home, away)
    _require_odds(odds_home, odds_draw, odds_away)
    probs, model = _predict(home, away, neutral)
    res = assess_1x2(probs, odds_home, odds_draw, odds_away,
                     home=home, away=away, bankroll=bankroll,
                     kelly_fraction=kelly_fraction)
    res["model"] = model
    res["model_probs"] = {k: round(v, 4) for k, v in probs.items()}
    return res


@app.get("/value/live")
def value_live(home: str, away: str, neutral: bool = True,
               bankroll: float = 100.0, kelly_fraction: float = 0.25,
               price: str = "best"):
    """Pull live odds from The Odds API (ODDS_API_KEY in .env) and assess."""
    if not odds_api.available():
        raise HTTPException(status_code=503,
                            detail="ODDS_API_KEY not set — use /value/manual "
                                   "or add a free key from the-odds-api.com")
    _require_teams(home, away)
    found = _cached_consensus(home, away)
    if found is None:
        raise HTTPException(status_code=404,
                            detail=f"No live odds found for {home} vs {away}")
    if "error" in found:
        raise HTTPException(status_code=502, detail=found["error"])
    o = found["best"] if price == "best" else found["median"]
    if not all(o.get(k) for k in ("home", "draw", "away")):
        raise HTTPException(status_code=404, detail="Incomplete 1X2 prices")
    probs, model = _predict(home, away, neutral)
    res = assess_1x2(probs, o["home"], o["draw"], o["away"],
                     home=home, away=away, bankroll=bankroll,
                     kelly_fraction=kelly_fraction)
    res.update({"model": model, "odds_source": found, "price_used": price,
                "model_probs": {k: round(v, 4) for k, v in probs.items()}})
    return res


@app.get("/value/market")
def value_market(home: str, away: str, market: str, odds_yes: float,
                 odds_no: float | None = None, neutral: bool = True,
                 bankroll: float = 100.0, kelly_fraction: float = 0.25):
    """Assess a side market (over_2_5_goals, btts_yes, ...) vs your odds.
    Probabilities come from the Monte Carlo simulator."""
    _require_teams(home, away)
    _require_odds(odds_yes, odds_no)
    sim = _simulator.simulate(home, away, neutral)
    mkts = sim["markets"]
    if market not in mkts:
        raise HTTPException(status_code=400,
                            detail=f"Unknown market. One of: "
                                   f"{[k for k in mkts if k.startswith(('over', 'btts'))]}")
    p_yes = float(mkts[market]) / 100.0
    probs = {"yes": p_yes, "no": 1.0 - p_yes}
    odds = {"yes": odds_yes}
    if odds_no:
        odds["no"] = odds_no
    res = assess_market(probs, odds, bankroll=bankroll,
                        kelly_fraction=kelly_fraction)
    res.update({"market": market, "model_prob_yes": round(p_yes, 4),
                "note": "simulator-derived probability — secondary stats are "
                        "modelled, weight these edges accordingly"})
    return res


@app.get("/wc2026/fixtures")
def wc2026_fixtures(predict_odds: bool = False):
    """Upcoming WC2026 fixtures (backend/data/wc2026_fixtures.csv) with
    model probabilities and fair (no-vig) odds for each."""
    import pandas as pd
    fpath = _PROJECT_ROOT / "backend" / "data" / "wc2026_fixtures.csv"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="fixtures file missing")
    df = pd.read_csv(fpath, comment="#")
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    df = df[pd.to_datetime(df["date"]) >= today]
    out = []
    for r in df.itertuples(index=False):
        probs, model = _predict(r.home_team, r.away_team, bool(r.neutral))
        fair = {k: (round(1.0 / v, 2) if v > 0.01 else None)
                for k, v in probs.items()}
        row = {"date": str(r.date), "stage": r.stage,
               "home": r.home_team, "away": r.away_team,
               "neutral": bool(r.neutral), "model": model,
               "probs": {k: round(v, 3) for k, v in probs.items()},
               "fair_odds": {"home": fair["win"], "draw": fair["draw"],
                             "away": fair["loss"]}}
        if predict_odds and odds_api.available():
            row["live_odds"] = odds_api.consensus_odds(r.home_team, r.away_team)
        out.append(row)
    return {"fixtures": out, "count": len(out),
            "odds_api": odds_api.available()}


@app.post("/analyst/ask")
def analyst_ask(payload: dict = Body(...)):
    """Ask The Gaffer a natural-language question; it calls the simulator as a
    tool and explains the numbers. Falls back gracefully if Ollama is offline."""
    q = (payload.get("question") or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")
    return _analyst.ask(q, payload.get("home"), payload.get("away"))


@app.get("/live")
def live():
    """Live / recently-finished / upcoming matches (football-data.org, cached).
    Always 200 with an `available` flag so the UI shows a note, not an error."""
    return live_scores.board()


if _DIST.exists() and (_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="assets")


@app.get("/health/llm")
def health_llm(ping: bool = False):
    """Diagnose why the analyst LLM is/isn't active. ?ping=true actually calls
    the model and returns its reply or the exact error (no secrets leaked)."""
    info = llm_diagnose()
    info["provider"] = llm_describe(_llm)
    info["ready"] = _llm.available()
    if ping and _llm.available():
        try:
            info["ping"] = _llm.generate("Reply with exactly: OK")[:120]
        except Exception as e:
            info["ping_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return info


@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend(full_path: str):
    if _DASHBOARD.exists():
        return FileResponse(str(_DASHBOARD))
    if _DIST.exists():
        base = _DIST.resolve()
        target = (base / full_path).resolve()
        # Prevent path traversal: only serve files that live inside _DIST.
        if (target == base or base in target.parents) and target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(_DIST / "index.html"))
    return {"detail": "frontend not built"}
