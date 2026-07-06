"""
Data loader — merged, weighted DataFrame of international matches.

Sources (merged + de-duplicated): seed_matches.csv, extra_results.csv, any
*.csv in backend/data/extra/, cache/matches.csv, or the football-data.org API.
Adds weight = competition_weight * recency_weight.
Columns: date, home_team, away_team, home_score, away_score, tournament,
         neutral, weight
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_HERE = Path(__file__).parent
CACHE_PATH = _HERE / "cache" / "matches.csv"
SEED_PATH = _HERE / "seed_matches.csv"
EXTRA_PATH = _HERE / "extra_results.csv"
DROPIN_DIR = _HERE / "extra"

FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
FOOTBALL_DATA_URL = "https://api.football-data.org/v4/competitions/WC/matches"

COMPETITION_WEIGHT = {
    "fifa world cup": 1.15, "world cup": 1.15, "world cup qualification": 0.75,
    "uefa euro": 1.05, "copa america": 1.05, "african cup of nations": 1.0,
    "afc asian cup": 1.0, "nations league": 0.9, "confederations cup": 0.95,
    "friendly": 0.45,
}
DEFAULT_COMP_WEIGHT = 0.7
RECENCY_HALF_LIFE_DAYS = 365 * 3


def _comp_weight(tournament: str) -> float:
    t = str(tournament).strip().lower()
    for key, w in COMPETITION_WEIGHT.items():
        if key in t:
            return w
    return DEFAULT_COMP_WEIGHT


def _read_csv(path: Path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, comment="#", parse_dates=["date"])
        return df if not df.empty else None
    except Exception as exc:
        print(f"[loader] could not read {path.name}: {exc}")
        return None


def _add_weights(df, as_of=None):
    df = df.copy()
    if "tournament" not in df:
        df["tournament"] = "Unknown"
    if "neutral" not in df:
        df["neutral"] = True
    comp = df["tournament"].map(_comp_weight)
    as_of = as_of or df["date"].max()
    age_days = (as_of - df["date"]).dt.days.clip(lower=0).fillna(0)
    recency = np.power(0.5, age_days / RECENCY_HALF_LIFE_DAYS)
    df["weight"] = (comp * recency).astype(float)
    return df


def _fetch_football_data():
    if not FOOTBALL_DATA_KEY:
        return None
    try:
        import requests
        r = requests.get(FOOTBALL_DATA_URL,
                         headers={"X-Auth-Token": FOOTBALL_DATA_KEY}, timeout=10)
        r.raise_for_status()
        rows = []
        for m in r.json().get("matches", []):
            if m.get("status") != "FINISHED":
                continue
            rows.append({
                "date": m["utcDate"][:10],
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_score": m["score"]["fullTime"]["home"],
                "away_score": m["score"]["fullTime"]["away"],
                "tournament": "FIFA World Cup", "neutral": True,
            })
        if rows:
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(CACHE_PATH, index=False)
            return df
    except Exception as exc:
        print(f"[loader] football-data.org fetch failed: {exc}")
    return None


def load_matches(verbose: bool = True):
    frames, sources = [], []
    for path, label in [(SEED_PATH, "seed"), (EXTRA_PATH, "extra")]:
        d = _read_csv(path)
        if d is not None:
            frames.append(d); sources.append(f"{label}:{len(d)}")
    if DROPIN_DIR.exists():
        for p in sorted(DROPIN_DIR.glob("*.csv")):
            d = _read_csv(p)
            if d is not None:
                frames.append(d); sources.append(f"{p.name}:{len(d)}")
    d = _read_csv(CACHE_PATH)
    if d is not None:
        frames.append(d); sources.append(f"cache:{len(d)}")
    elif not frames:
        d = _fetch_football_data()
        if d is not None:
            frames.append(d); sources.append(f"api:{len(d)}")
    if not frames:
        raise RuntimeError("No match data available")
    df = pd.concat(frames, ignore_index=True)
    needed = ["date", "home_team", "away_team", "home_score", "away_score"]
    df = df.dropna(subset=needed)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"])
    df = df.sort_values("date").reset_index(drop=True)
    df = _add_weights(df)
    if verbose:
        print(f"[loader] merged {len(df)} matches ({', '.join(sources)}); "
              f"{df['home_team'].nunique()} home teams, "
              f"{df['tournament'].nunique()} competitions")
    return df


def all_teams(df):
    return sorted(set(df["home_team"]) | set(df["away_team"]))


def team_record(df, team, n_last=10):
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    return df[mask].sort_values("date").tail(n_last).copy()


if __name__ == "__main__":
    m = load_matches()
    print(m.tail(6)[["date", "home_team", "away_team", "home_score",
                     "away_score", "tournament", "weight"]].to_string(index=False))
    print(f"\nTeams ({len(all_teams(m))}): {', '.join(all_teams(m)[:20])} ...")
