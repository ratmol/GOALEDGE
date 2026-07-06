"""
Live scores — a thin, cached client for football-data.org /v4/matches.

Decoupled from the historical model: this only pulls today's / recent matches
and categorises them into live / finished / upcoming, like a Google scoreboard.

Free tier is ~10 requests/minute, so results are cached (default 30s TTL) — the
frontend can poll every 30s without ever breaching the limit. Falls back
gracefully (available=False) when no FOOTBALL_DATA_API_KEY is set.
"""
from __future__ import annotations

import os
import time
from datetime import date, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
BASE = "https://api.football-data.org/v4/matches"

_LIVE = {"IN_PLAY", "PAUSED", "LIVE"}
_DONE = {"FINISHED", "AWARDED"}
_SOON = {"SCHEDULED", "TIMED"}

_cache: dict = {}   # key -> (ts, payload); warm-instance TTL cache


def available() -> bool:
    return bool(API_KEY)


def _normalize(m: dict) -> dict:
    ft = (m.get("score") or {}).get("fullTime") or {}
    comp = m.get("competition") or {}
    return {
        "id": m.get("id"),
        "competition": comp.get("name", ""),
        "emblem": comp.get("emblem"),
        "home": (m.get("homeTeam") or {}).get("name", "TBD"),
        "away": (m.get("awayTeam") or {}).get("name", "TBD"),
        "home_score": ft.get("home"),
        "away_score": ft.get("away"),
        "status": m.get("status", ""),
        "utc_date": m.get("utcDate", ""),
    }


def _categorize(matches: list[dict]) -> dict:
    live, finished, upcoming = [], [], []
    for m in matches:
        n = _normalize(m)
        s = n["status"]
        if s in _LIVE:
            live.append(n)
        elif s in _DONE:
            finished.append(n)
        elif s in _SOON:
            upcoming.append(n)
    finished.sort(key=lambda x: x["utc_date"], reverse=True)
    upcoming.sort(key=lambda x: x["utc_date"])
    return {"live": live, "finished": finished[:12], "upcoming": upcoming[:12]}


def board(ttl: float = 30.0) -> dict:
    """Cached scoreboard: {available, live, finished, upcoming, fetched_at}."""
    if not available():
        return {"available": False, "live": [], "finished": [], "upcoming": [],
                "note": "Set FOOTBALL_DATA_API_KEY (free at football-data.org) "
                        "to show live scores."}

    now = time.time()
    hit = _cache.get("board")
    if hit and now - hit[0] < ttl:
        return hit[1]

    today = date.today()
    params = {"dateFrom": str(today - timedelta(days=1)),
              "dateTo": str(today + timedelta(days=1))}
    try:
        r = requests.get(BASE, headers={"X-Auth-Token": API_KEY},
                         params=params, timeout=8)
        if r.status_code == 429:
            payload = {"available": True, "live": [], "finished": [],
                       "upcoming": [], "note": "Rate limited — try again shortly."}
            _cache["board"] = (now, payload)
            return payload
        r.raise_for_status()
        cats = _categorize(r.json().get("matches", []))
    except requests.RequestException as e:
        return {"available": True, "live": [], "finished": [], "upcoming": [],
                "note": f"Live feed unavailable: {type(e).__name__}"}

    payload = {"available": True, "fetched_at": int(now), **cats}
    _cache["board"] = (now, payload)
    return payload


if __name__ == "__main__":
    # Offline unit check of categorization (no network / key needed).
    sample = [
        {"status": "IN_PLAY", "utcDate": "2025-06-29T15:00:00Z",
         "competition": {"name": "World Cup"},
         "homeTeam": {"name": "Brazil"}, "awayTeam": {"name": "Spain"},
         "score": {"fullTime": {"home": 1, "away": 1}}},
        {"status": "FINISHED", "utcDate": "2025-06-29T12:00:00Z",
         "competition": {"name": "Euro"},
         "homeTeam": {"name": "France"}, "awayTeam": {"name": "Italy"},
         "score": {"fullTime": {"home": 2, "away": 0}}},
        {"status": "TIMED", "utcDate": "2025-06-29T19:00:00Z",
         "competition": {"name": "Copa"},
         "homeTeam": {"name": "Argentina"}, "awayTeam": {"name": "Chile"},
         "score": {"fullTime": {"home": None, "away": None}}},
    ]
    cats = _categorize(sample)
    assert len(cats["live"]) == 1 and cats["live"][0]["home"] == "Brazil"
    assert len(cats["finished"]) == 1 and cats["finished"][0]["home_score"] == 2
    assert len(cats["upcoming"]) == 1 and cats["upcoming"][0]["home"] == "Argentina"
    print("categorize OK:", {k: len(v) for k, v in cats.items()})
    print("available() with no key ->", available())
