"""
One-command data upgrade. Run from the project root ON YOUR MACHINE:

    python scripts/download_data.py

Downloads the full open dataset of international results (martj42,
~48,000 matches since 1872), filters to the modern era, and drops it where
the loader picks it up automatically (backend/data/extra/). If a
FOOTBALL_DATA_API_KEY is set in .env it also refreshes World Cup 2026
results. Then retrains:  python train.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_URL = ("https://raw.githubusercontent.com/martj42/"
               "international_results/master/results.csv")
OUT = ROOT / "backend" / "data" / "extra" / "international_full.csv"
CUTOFF = os.getenv("DATA_CUTOFF", "2006-01-01")   # modern era; edit if you want more


def fetch_results() -> pd.DataFrame:
    print(f"[data] downloading {RESULTS_URL} ...")
    r = requests.get(RESULTS_URL, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["date"])
    print(f"[data] got {len(df):,} matches ({df['date'].min().date()} → "
          f"{df['date'].max().date()})")
    df = df[df["date"] >= pd.Timestamp(CUTOFF)].copy()
    df = df.rename(columns={"tournament": "tournament"})
    out = df[["date", "home_team", "away_team", "home_score", "away_score",
              "tournament", "neutral"]].dropna(
        subset=["home_score", "away_score"])
    print(f"[data] kept {len(out):,} matches since {CUTOFF}")
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = fetch_results()
    out.to_csv(OUT, index=False)
    print(f"[data] wrote {OUT.relative_to(ROOT)}")

    # Optional: refresh WC 2026 from football-data.org (free key in .env)
    try:
        from backend.data.loader import _fetch_football_data
        d = _fetch_football_data()
        if d is not None:
            print(f"[data] refreshed {len(d)} WC matches from football-data.org")
        else:
            print("[data] football-data.org skipped (no key in .env) — "
                  "bundled WC2026 CSV will be used")
    except Exception as exc:
        print(f"[data] football-data.org refresh failed: {exc}")

    print("\nNext:  python train.py   (retrains on the full dataset)")


if __name__ == "__main__":
    main()
