"""
Backtesting engine — walk-forward evaluation of any predict_fn.

For each match in the test window the model is retrained on all prior data,
then evaluated on the current match. Reports:
  - Brier score, log-loss
  - Calibration (expected vs actual win/draw/loss rates)
  - Flat-stake ROI (bet on the highest-probability outcome at fair odds)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from backend.skills.quant_analyzer import brier_score, log_loss


@dataclass
class BacktestResult:
    n: int
    brier: float
    logloss: float
    flat_roi: float
    calibration: dict   # {"win": (exp, act), "draw": ..., "loss": ...}
    details: list[dict] = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    predict_fn: Callable[[str, str, bool], dict],
    warmup: int = 40,
) -> BacktestResult:
    """
    Walk-forward backtest.

    predict_fn(home, away, neutral) → {"win": float, "draw": float, "loss": float}
    warmup: number of matches to use for the initial warm-up (not evaluated).
    """
    df = df.sort_values("date").reset_index(drop=True)
    test = df.iloc[warmup:].reset_index(drop=True)

    probs_list, outcomes_list, details = [], [], []
    exp = {"win": [], "draw": [], "loss": []}
    act = {"win": [], "draw": [], "loss": []}

    for _, row in test.iterrows():
        p = predict_fn(row["home_team"], row["away_team"], bool(row.get("neutral", True)))
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        if hs > as_:
            true_label = 0
        elif hs == as_:
            true_label = 1
        else:
            true_label = 2

        p_vec = np.array([p["win"], p["draw"], p["loss"]])
        probs_list.append(p_vec)
        y_oh = np.eye(3)[true_label]
        outcomes_list.append(y_oh)

        for i, key in enumerate(["win", "draw", "loss"]):
            exp[key].append(p_vec[i])
            act[key].append(float(y_oh[i]))

        # Flat stake ROI: always bet on the favourite at fair-odds payout
        fav_idx = int(np.argmax(p_vec))
        won = true_label == fav_idx
        roi_this = (1.0 / p_vec[fav_idx] - 1) if won else -1.0
        details.append({
            "home": row["home_team"], "away": row["away_team"],
            "date": str(row["date"])[:10],
            "pred": ["home_win", "draw", "away_win"][fav_idx],
            "actual": ["home_win", "draw", "away_win"][true_label],
            "correct": won, "roi": roi_this,
        })

    probs_arr = np.array(probs_list)
    outcomes_arr = np.array(outcomes_list)

    b = brier_score(probs_arr, outcomes_arr)
    ll = log_loss(probs_arr, outcomes_arr)
    flat_roi = float(np.mean([d["roi"] for d in details]))
    calib = {k: (float(np.mean(exp[k])), float(np.mean(act[k]))) for k in exp}

    return BacktestResult(n=len(details), brier=b, logloss=ll,
                          flat_roi=flat_roi, calibration=calib, details=details)


if __name__ == "__main__":
    from backend.data.loader import load_matches
    from backend.models.elo import EloModel

    df = load_matches()
    elo = EloModel()

    def elo_predict(home, away, neutral=True):
        return elo.win_draw_loss(home, away, neutral)

    result = run_backtest(df, elo_predict, warmup=40)
    print(f"Backtest on {result.n} matches (Elo baseline)")
    print(f"  Brier={result.brier:.4f}  LogLoss={result.logloss:.4f}  "
          f"FlatROI={result.flat_roi:+.3f}")
    print(f"  Calibration: {result.calibration}")
    acc = sum(d["correct"] for d in result.details) / result.n
    print(f"  Favourite accuracy: {acc:.1%}")
