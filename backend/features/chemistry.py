"""
Team chemistry features.

Public data has no biometric/chemistry signal, so we proxy chemistry with
lineup-based stability metrics that research links to coordination:

  - lineup_continuity : how many of the starting XI also started last match
  - minutes_together  : avg shared on-pitch minutes among the current XI (season)
  - avg_partnership   : avg appearances of the most-used CB and CM pairings
  - squad_stability   : 1 - (rolling std of lineup changes over last K matches)
  - core_age_spread   : age cohesion of the spine (GK-CB-CM-ST)

All are normalized to 0..1 so they slot straight into the feature matrix.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LineupRecord:
    match_id: str
    team: str
    starters: list[str]          # player ids
    shared_minutes: dict | None = None  # (id,id) -> minutes together this season


def lineup_continuity(curr: list[str], prev: list[str]) -> float:
    if not prev:
        return 0.5  # neutral prior when no history
    return len(set(curr) & set(prev)) / 11.0


def squad_stability(recent_lineups: list[list[str]]) -> float:
    """1.0 = identical XI every match; lower = more rotation."""
    if len(recent_lineups) < 2:
        return 0.5
    base = set(recent_lineups[-1])
    overlaps = [len(base & set(xi)) / 11.0 for xi in recent_lineups[:-1]]
    return sum(overlaps) / len(overlaps)


def minutes_together_score(xi: list[str], shared: dict | None) -> float:
    """Average shared minutes among the XI, scaled to ~0..1 (cap 3000 min)."""
    if not shared:
        return 0.5
    pairs = [shared.get((a, b), shared.get((b, a), 0))
             for i, a in enumerate(xi) for b in xi[i + 1:]]
    if not pairs:
        return 0.5
    return min(1.0, (sum(pairs) / len(pairs)) / 3000.0)


def chemistry_features(curr: LineupRecord,
                       prev_starters: list[str],
                       recent_lineups: list[list[str]]) -> dict:
    return {
        "chem_lineup_continuity": lineup_continuity(curr.starters, prev_starters),
        "chem_squad_stability": squad_stability(recent_lineups + [curr.starters]),
        "chem_minutes_together": minutes_together_score(
            curr.starters, curr.shared_minutes),
    }


if __name__ == "__main__":
    prev = [f"p{i}" for i in range(11)]
    curr = LineupRecord("m2", "ARG", [f"p{i}" for i in range(2, 13)])
    print(chemistry_features(curr, prev, [prev]))
