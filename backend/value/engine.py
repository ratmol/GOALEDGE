"""
Value engine — the gambler's half of GoalEdge.

A prediction is worthless to a bettor without a price. This module compares
model probabilities against bookmaker odds and answers the only questions that
matter: is there positive expected value, how big is the edge after removing
the bookmaker's margin, and how much should be staked (fractional Kelly).

Core identities (decimal odds o, model probability p):
  implied prob        q_raw = 1 / o
  overround           sum(q_raw) - 1                  (the book's margin)
  fair prob           q = q_raw / sum(q_raw)          (proportional de-vig)
  edge                p - q
  EV per 1 unit stake p * (o - 1) - (1 - p)
  Kelly fraction      f* = (p * o - 1) / (o - 1), scaled by KELLY_FRACTION
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))   # quarter Kelly
EDGE_THRESHOLD = float(os.getenv("EDGE_THRESHOLD", "0.03"))   # min 3% edge
MAX_STAKE_PCT = float(os.getenv("MAX_STAKE_PCT", "0.05"))     # 5% bankroll cap
MIN_ODDS = 1.01


@dataclass
class BetAssessment:
    outcome: str
    odds: float
    model_prob: float
    implied_prob: float          # raw 1/odds
    fair_prob: float             # margin removed
    edge: float                  # model_prob - fair_prob
    ev_per_unit: float
    kelly_full: float
    kelly_staked: float          # fractional Kelly, capped
    stake: float                 # currency, given bankroll
    value_bet: bool
    note: str = ""


def _kelly(p: float, o: float) -> float:
    if o <= 1.0:
        return 0.0
    return max((p * o - 1.0) / (o - 1.0), 0.0)


def assess_market(model_probs: dict[str, float], odds: dict[str, float],
                  bankroll: float = 100.0,
                  kelly_fraction: float = KELLY_FRACTION,
                  edge_threshold: float = EDGE_THRESHOLD) -> dict:
    """
    Compare model probabilities vs bookmaker odds for any n-way market.
    model_probs / odds share keys, e.g. {"home": .., "draw": .., "away": ..}
    or {"yes": .., "no": ..} for BTTS / over-under lines.
    """
    keys = [k for k in model_probs if k in odds and odds[k] and odds[k] > MIN_ODDS]
    if not keys:
        return {"error": "no overlapping outcomes with valid odds"}

    q_raw = {k: 1.0 / float(odds[k]) for k in keys}
    complete = len(keys) == len(model_probs)   # full market -> can de-vig
    booksum = sum(q_raw.values()) if complete else 1.0
    overround = (sum(q_raw.values()) - 1.0) if complete else None

    out: list[BetAssessment] = []
    for k in keys:
        o = float(odds[k])
        p = float(model_probs[k])
        fair = q_raw[k] / booksum
        edge = p - fair
        ev = p * (o - 1.0) - (1.0 - p)
        kf = _kelly(p, o)
        staked_frac = min(kf * kelly_fraction, MAX_STAKE_PCT)
        is_value = edge >= edge_threshold and ev > 0
        note = ""
        if is_value and p < 0.10:
            note = "long shot — model edges at low probs are least reliable"
        if is_value and edge > 0.15:
            note = ("edge > 15% is usually a model blind spot (injury, "
                    "lineup, motivation) — verify the news before trusting it")
        out.append(BetAssessment(
            outcome=k, odds=o, model_prob=round(p, 4),
            implied_prob=round(q_raw[k], 4), fair_prob=round(fair, 4),
            edge=round(edge, 4), ev_per_unit=round(ev, 4),
            kelly_full=round(kf, 4), kelly_staked=round(staked_frac, 4),
            stake=round(staked_frac * bankroll if is_value else 0.0, 2),
            value_bet=is_value, note=note))

    out.sort(key=lambda b: b.ev_per_unit, reverse=True)
    best = next((b for b in out if b.value_bet), None)
    return {
        "overround_pct": (round(overround * 100, 2)
                          if overround is not None else None),
        "kelly_fraction_used": kelly_fraction,
        "edge_threshold": edge_threshold,
        "bankroll": bankroll,
        "assessments": [asdict(b) for b in out],
        "best_bet": asdict(best) if best else None,
        "verdict": (f"VALUE: {best.outcome} @ {best.odds} "
                    f"(edge {best.edge:+.1%}, stake {best.stake})"
                    if best else
                    "NO BET — no outcome clears the edge threshold. "
                    "Passing is a position."),
    }


def assess_1x2(model_probs: dict, odds_home: float, odds_draw: float,
               odds_away: float, home: str = "home", away: str = "away",
               bankroll: float = 100.0,
               kelly_fraction: float = KELLY_FRACTION) -> dict:
    """Convenience wrapper for the match-odds (1X2) market."""
    res = assess_market(
        {"home": model_probs["win"], "draw": model_probs["draw"],
         "away": model_probs["loss"]},
        {"home": odds_home, "draw": odds_draw, "away": odds_away},
        bankroll=bankroll, kelly_fraction=kelly_fraction)
    res["labels"] = {"home": home, "draw": "draw", "away": away}
    return res
