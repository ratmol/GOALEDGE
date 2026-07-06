"""
Elo baseline — fast, robust cold-start rating used as a sanity check and as a
feature for the main model. Supports home advantage and margin-of-victory.
"""
from __future__ import annotations

import math


class _Ratings(dict):
    """dict returning a base rating for unseen teams (picklable)."""

    def __init__(self, base: float = 1500.0):
        super().__init__()
        self.base = base

    def __missing__(self, key):
        return self.base


class EloModel:
    def __init__(self, k: float = 30.0, home_adv: float = 65.0,
                 base: float = 1500.0):
        self.k = k
        self.home_adv = home_adv
        self.ratings: dict[str, float] = _Ratings(base)

    def expected(self, home: str, away: str, neutral: bool = False) -> float:
        ha = 0 if neutral else self.home_adv
        diff = (self.ratings[home] + ha) - self.ratings[away]
        return 1.0 / (1.0 + 10 ** (-diff / 400))

    def update(self, home: str, away: str, home_goals: int, away_goals: int,
               neutral: bool = False) -> None:
        exp_home = self.expected(home, away, neutral)
        score = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        margin = abs(home_goals - away_goals)
        mult = math.log(max(margin, 1) + 1)  # margin-of-victory multiplier
        delta = self.k * mult * (score - exp_home)
        self.ratings[home] = self.ratings[home] + delta
        self.ratings[away] = self.ratings[away] - delta

    def win_draw_loss(self, home: str, away: str, neutral: bool = False) -> dict:
        """Rough WDL split from the Elo win expectation (draw band heuristic)."""
        e = self.expected(home, away, neutral)
        draw = 0.27 * (1 - abs(e - 0.5) * 2)  # more even -> more draw mass
        win = e * (1 - draw)
        loss = (1 - e) * (1 - draw)
        return {"win": win, "draw": draw, "loss": loss}


if __name__ == "__main__":
    m = EloModel()
    m.update("ARG", "MEX", 2, 0, neutral=True)
    print(m.win_draw_loss("ARG", "MEX", neutral=True))
