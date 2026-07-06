"""
Feature engineering — single-pass, leak-free, and reusable at inference time.

`FeatureBuilder` maintains running state (Elo, rolling form, head-to-head,
last-played dates) that is updated match by match in chronological order.
Features for a match are always computed BEFORE that match's result is applied,
so there is no look-ahead leakage. After training, the fitted builder is stored
inside the model so live predictions use the exact same real features
(previously inference used dummy 0.5s — a bug that made half the blend noise).

Features per match:
  elo_home, elo_away, elo_diff
  form_{home,away}_{wins,gd,att,def}   — rolling last-8 window
  h2h_home_winrate                      — last 5 meetings
  rest_home, rest_away                  — days since last match (capped 21)
  neutral
Target: 0 = home win, 1 = draw, 2 = away win.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from backend.models.elo import EloModel

FORM_WINDOW = 8
H2H_WINDOW = 5
DEFAULT_REST = 10.0
REST_CAP = 21.0

FEATURE_COLS = [
    "elo_home", "elo_away", "elo_diff",
    "form_home_wins", "form_home_gd", "form_home_att", "form_home_def",
    "form_away_wins", "form_away_gd", "form_away_att", "form_away_def",
    "h2h_home_winrate", "rest_home", "rest_away", "neutral",
]


class FeatureBuilder:
    """Running feature state. `features()` reads, `update()` writes."""

    def __init__(self, form_window: int = FORM_WINDOW,
                 h2h_window: int = H2H_WINDOW):
        self.elo = EloModel()
        self.form_window = form_window
        self.h2h_window = h2h_window
        self.recent: dict[str, deque] = {}      # (gf, ga) tuples
        self.h2h: dict[tuple, deque] = {}       # winner name or "draw"
        self.last_date: dict[str, pd.Timestamp] = {}

    # -- reads ---------------------------------------------------------------

    def _form(self, team: str) -> dict:
        rec = self.recent.get(team)
        if not rec:
            return {"wins": 0.34, "gd": 0.0, "att": 1.3, "def": 1.3}
        gf = np.array([r[0] for r in rec], dtype=float)
        ga = np.array([r[1] for r in rec], dtype=float)
        return {"wins": float(np.mean(gf > ga)), "gd": float(np.mean(gf - ga)),
                "att": float(np.mean(gf)), "def": float(np.mean(ga))}

    def _rest(self, team: str, date) -> float:
        if date is None or team not in self.last_date:
            return DEFAULT_REST
        days = (pd.Timestamp(date) - self.last_date[team]).days
        return float(np.clip(days, 1.0, REST_CAP))

    def features(self, home: str, away: str, neutral: bool = True,
                 date=None) -> dict:
        eh, ea = self.elo.ratings[home], self.elo.ratings[away]
        fh, fa = self._form(home), self._form(away)
        key = tuple(sorted((home, away)))
        meetings = self.h2h.get(key)
        if meetings:
            h2h = sum(1.0 if w == home else 0.5 if w == "draw" else 0.0
                      for w in meetings) / len(meetings)
        else:
            h2h = 0.5
        return {
            "elo_home": eh, "elo_away": ea, "elo_diff": eh - ea,
            "form_home_wins": fh["wins"], "form_home_gd": fh["gd"],
            "form_home_att": fh["att"], "form_home_def": fh["def"],
            "form_away_wins": fa["wins"], "form_away_gd": fa["gd"],
            "form_away_att": fa["att"], "form_away_def": fa["def"],
            "h2h_home_winrate": h2h,
            "rest_home": self._rest(home, date),
            "rest_away": self._rest(away, date),
            "neutral": float(neutral),
        }

    # -- writes --------------------------------------------------------------

    def update(self, home: str, away: str, hs: int, as_: int,
               neutral: bool, date=None) -> None:
        self.elo.update(home, away, hs, as_, neutral)
        self.recent.setdefault(home, deque(maxlen=self.form_window)).append((hs, as_))
        self.recent.setdefault(away, deque(maxlen=self.form_window)).append((as_, hs))
        key = tuple(sorted((home, away)))
        self.h2h.setdefault(key, deque(maxlen=self.h2h_window)).append(
            home if hs > as_ else away if as_ > hs else "draw")
        if date is not None:
            d = pd.Timestamp(date)
            self.last_date[home] = d
            self.last_date[away] = d


def build_features(df: pd.DataFrame,
                   return_builder: bool = False):
    """Single chronological pass. O(n) — handles 50k+ matches comfortably."""
    df = df.sort_values("date").reset_index(drop=True)
    fb = FeatureBuilder()
    rows, targets = [], []

    for r in df.itertuples(index=False):
        neutral = bool(getattr(r, "neutral", True))
        rows.append(fb.features(r.home_team, r.away_team, neutral, r.date))
        hs, as_ = int(r.home_score), int(r.away_score)
        targets.append(0 if hs > as_ else 1 if hs == as_ else 2)
        fb.update(r.home_team, r.away_team, hs, as_, neutral, r.date)

    X = pd.DataFrame(rows)[FEATURE_COLS]
    y = np.array(targets)
    return (X, y, fb) if return_builder else (X, y)


if __name__ == "__main__":
    from backend.data.loader import load_matches
    df = load_matches()
    X, y = build_features(df)
    print(X.describe().round(2))
    print(f"\nTarget distribution: {np.bincount(y)} (home_win/draw/away_win)")
