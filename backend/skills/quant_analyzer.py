"""
Quant Analyzer — iterative, threshold-driven model refinement.

Loops through improvement actions until the validation Brier score is at or below
TARGET_BRIER, or until MAX_QUANT_ITERATIONS is reached. Returns the best model.

This is a working skeleton: the `_fit_and_score` and action hooks are wired to
clear extension points so the data/feature modules can plug in.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

TARGET_BRIER = float(os.getenv("TARGET_BRIER", "0.20"))
MAX_ITERS = int(os.getenv("MAX_QUANT_ITERATIONS", "15"))


@dataclass
class Iteration:
    step: int
    action: str
    brier: float
    log_loss: float


@dataclass
class QuantResult:
    best_brier: float
    reached_threshold: bool
    history: list[Iteration] = field(default_factory=list)
    model: object = None


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass Brier score. probs: (n,3), outcomes: one-hot (n,3)."""
    return float(np.mean(np.sum((probs - outcomes) ** 2, axis=1)))


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(np.sum(outcomes * np.log(p), axis=1)))


class QuantAnalyzer:
    """Refinement loop. Pass in a `fit_score_fn(state) -> (model, probs, y_onehot)`
    that fits the current configuration and returns validation predictions."""

    # ordered improvement actions; each mutates `self.state` then returns a label
    ACTIONS: list[str] = [
        "rolling_form_windows",
        "recovery_features",
        "team_chemistry_features",
        "retune_xgboost",
        "recalibrate_probabilities",
        "blend_poisson_xgb",
    ]

    def __init__(self, fit_score_fn: Callable, target_brier: float = TARGET_BRIER,
                 max_iters: int = MAX_ITERS):
        self.fit_score_fn = fit_score_fn
        self.target = target_brier
        self.max_iters = max_iters
        self.state: dict = {"enabled_actions": []}

    def run(self) -> QuantResult:
        result = QuantResult(best_brier=float("inf"), reached_threshold=False)
        best_model = None

        for step in range(self.max_iters):
            # Apply the next improvement action (if any remain).
            if step < len(self.ACTIONS):
                action = self.ACTIONS[step]
                self.state["enabled_actions"].append(action)
            else:
                action = "extra_grid_search"

            model, probs, y = self.fit_score_fn(self.state)
            b, ll = brier_score(probs, y), log_loss(probs, y)
            result.history.append(Iteration(step, action, b, ll))

            if b < result.best_brier:
                result.best_brier = b
                best_model = model

            print(f"[quant] iter {step:02d} | {action:<26} "
                  f"| brier={b:.4f} | logloss={ll:.4f}")

            if b <= self.target:
                result.reached_threshold = True
                print(f"[quant] threshold {self.target} reached — stopping.")
                break

        result.model = best_model
        if not result.reached_threshold:
            print(f"[quant] threshold not reached; returning best "
                  f"(brier={result.best_brier:.4f}).")
        return result


if __name__ == "__main__":
    # Demo with a synthetic fit function that improves each iteration.
    rng = np.random.default_rng(0)
    n = 400

    def demo_fit(state):
        # Fewer remaining error as more actions enable -> brier improves.
        noise = max(0.02, 0.30 - 0.03 * len(state["enabled_actions"]))
        true = rng.integers(0, 3, n)
        y = np.eye(3)[true]
        probs = y + rng.normal(0, noise, (n, 3))
        probs = np.clip(probs, 1e-6, None)
        probs /= probs.sum(axis=1, keepdims=True)
        return ("model_obj", probs, y)

    QuantAnalyzer(demo_fit).run()
