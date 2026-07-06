"""
The Odds API client (free tier: 500 requests/month) — live 1X2 odds.

Sign up at https://the-odds-api.com, put ODDS_API_KEY in .env.
Endpoint: GET /v4/sports/soccer_fifa_world_cup/odds
          ?regions=eu&markets=h2h&oddsFormat=decimal&apiKey=...

We take the MEDIAN price across bookmakers as the consensus line (robust to
one book posting a stale/trap price) and also report the best available price
per outcome, since EV should be computed at the price you can actually get.
"""
from __future__ import annotations

import os
import statistics
import time

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY", "")
SPORT_KEY = os.getenv("ODDS_API_SPORT", "soccer_fifa_world_cup")
BASE = "https://api.the-odds-api.com/v4/sports"
_cache: dict = {"t": 0.0, "data": None}
CACHE_SECONDS = 600          # be gentle with the free quota


def available() -> bool:
    return bool(API_KEY)


def fetch_events(force: bool = False) -> list[dict]:
    """Raw events with per-bookmaker h2h odds. Cached for CACHE_SECONDS."""
    if not API_KEY:
        raise RuntimeError("ODDS_API_KEY not set in .env")
    now = time.time()
    if not force and _cache["data"] is not None and now - _cache["t"] < CACHE_SECONDS:
        return _cache["data"]
    r = requests.get(
        f"{BASE}/{SPORT_KEY}/odds",
        params={"regions": "eu,us", "markets": "h2h",
                "oddsFormat": "decimal", "apiKey": API_KEY},
        timeout=20)
    r.raise_for_status()
    _cache["data"], _cache["t"] = r.json(), now
    return _cache["data"]


def consensus_odds(home: str, away: str) -> dict | None:
    """Median + best decimal odds for a fixture; None if not found."""
    try:
        events = fetch_events()
    except Exception as exc:
        return {"error": str(exc)}

    def norm(s: str) -> str:
        return s.lower().replace(".", "").strip()

    for ev in events:
        names = {norm(ev.get("home_team", "")), norm(ev.get("away_team", ""))}
        if not ({norm(home), norm(away)} <= names):
            continue
        prices: dict[str, list[float]] = {"home": [], "draw": [], "away": []}
        for bm in ev.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                for o in mkt.get("outcomes", []):
                    n = norm(o.get("name", ""))
                    if n == norm(ev["home_team"]):
                        prices["home"].append(float(o["price"]))
                    elif n == norm(ev["away_team"]):
                        prices["away"].append(float(o["price"]))
                    elif n == "draw":
                        prices["draw"].append(float(o["price"]))
        if not any(prices.values()):
            continue
        # our caller's "home" may be the API's away team — map by name
        flip = norm(home) == norm(ev.get("away_team", ""))
        med = {k: (round(statistics.median(v), 3) if v else None)
               for k, v in prices.items()}
        best = {k: (round(max(v), 3) if v else None) for k, v in prices.items()}
        if flip:
            med = {"home": med["away"], "draw": med["draw"], "away": med["home"]}
            best = {"home": best["away"], "draw": best["draw"], "away": best["home"]}
        return {"event": f'{ev.get("home_team")} vs {ev.get("away_team")}',
                "commence_time": ev.get("commence_time"),
                "n_bookmakers": len(ev.get("bookmakers", [])),
                "median": med, "best": best}
    return None
