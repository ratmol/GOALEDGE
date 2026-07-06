"""
Phase-1 model: Dixon-Coles Poisson + calibrated XGBoost, blended for WDL probs.

What changed in v3 (and why a gambler should care):
  * Dixon-Coles low-score correction (rho) — plain Poisson systematically
    underprices draws and 1-0/0-1 games, exactly the lines where books make
    money off naive models.
  * Weighted MLE — recency + competition weights (friendlies count less than
    World Cup games).
  * CHRONOLOGICAL validation split — the old random split leaked future Elo
    into the past and overstated accuracy. Real betting is out-of-time.
  * Real inference features — predict() previously fed the XGB half dummy
    0.5s with an UNFITTED Elo (all teams 1500). The fitted FeatureBuilder is
    now stored inside the model.
  * Blend alpha tuned by grid search on the validation window, not magic
    constants.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from backend.data.loader import load_matches
from backend.features.engineer import FEATURE_COLS, build_features
from backend.skills.quant_analyzer import brier_score, log_loss

MODEL_PATH = Path(__file__).parent / "phase1.pkl"
VAL_FRACTION = 0.2          # most recent 20% of matches = validation window


# ---------------------------------------------------------------------------
# Dixon-Coles Poisson model
# ---------------------------------------------------------------------------

def _dc_tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray,
            rho: float) -> np.ndarray:
    """Dixon-Coles low-score adjustment factor tau(x, y)."""
    tau = np.ones_like(lam)
    tau = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1 + lam * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1 + mu * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
    return np.maximum(tau, 1e-10)


class PoissonModel:
    """Weighted Dixon-Coles: attack/defence strengths + home adv + rho."""

    def __init__(self, max_goals: int = 8):
        self.max_goals = max_goals
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.home_adv: float = 0.2
        self.intercept: float = float(np.log(1.3))
        self.rho: float = 0.0

    def fit(self, df: pd.DataFrame) -> None:
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        t2i = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = np.array([t2i[t] for t in df["home_team"]])
        ai = np.array([t2i[t] for t in df["away_team"]])
        hs = df["home_score"].to_numpy(dtype=float)
        as_ = df["away_score"].to_numpy(dtype=float)
        ha = np.where(df["neutral"].astype(bool), 0.0, 1.0)
        w = (df["weight"].to_numpy(dtype=float) if "weight" in df
             else np.ones(len(df)))

        def nll(params: np.ndarray) -> float:
            atk, dfn = params[:n], params[n:2 * n]
            home_adv, intercept, rho = params[2 * n], params[2 * n + 1], params[2 * n + 2]
            lam = np.exp(intercept + atk[hi] - dfn[ai] + ha * home_adv)
            mu = np.exp(intercept + atk[ai] - dfn[hi])
            eps = 1e-12
            ll = (hs * np.log(lam + eps) - lam + as_ * np.log(mu + eps) - mu
                  + np.log(_dc_tau(hs, as_, lam, mu, rho)))
            return -float(np.sum(w * ll))

        x0 = np.zeros(2 * n + 3)
        x0[2 * n + 1] = np.log(1.3)
        bounds = ([(-3, 3)] * (2 * n)) + [(0, 1), (-1, 3), (-0.2, 0.2)]
        res = minimize(nll, x0, bounds=bounds, method="L-BFGS-B",
                       options={"maxiter": 800, "ftol": 1e-9})
        p = res.x
        for t, i in t2i.items():
            self.attack[t], self.defence[t] = p[i], p[n + i]
        self.home_adv, self.intercept, self.rho = p[2 * n], p[2 * n + 1], p[2 * n + 2]

    def score_matrix(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        ha = 0.0 if neutral else self.home_adv
        lam = float(np.exp(self.intercept + self.attack.get(home, 0.0)
                           - self.defence.get(away, 0.0) + ha))
        mu = float(np.exp(self.intercept + self.attack.get(away, 0.0)
                          - self.defence.get(home, 0.0)))
        g = self.max_goals
        ph = poisson.pmf(np.arange(g + 1), lam)
        pa = poisson.pmf(np.arange(g + 1), mu)
        m = np.outer(ph, pa)
        # apply DC correction to the 2x2 low-score block
        xs, ys = np.meshgrid(np.arange(2), np.arange(2), indexing="ij")
        m[:2, :2] *= _dc_tau(xs.astype(float), ys.astype(float),
                             np.full((2, 2), lam), np.full((2, 2), mu), self.rho)
        return m / m.sum()

    def predict_proba(self, home: str, away: str, neutral: bool = True) -> dict:
        m = self.score_matrix(home, away, neutral)
        return {"win": float(np.tril(m, -1).sum()),
                "draw": float(np.trace(m)),
                "loss": float(np.triu(m, 1).sum())}

    def predict_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([
            list(self.predict_proba(r.home_team, r.away_team,
                                    bool(getattr(r, "neutral", True))).values())
            for r in df.itertuples(index=False)])


# ---------------------------------------------------------------------------
# XGBoost model
# ---------------------------------------------------------------------------

class XGBModel:
    def __init__(self):
        base = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            eval_metric="mlogloss", random_state=42, n_jobs=-1)
        self.clf = CalibratedClassifierCV(base, method="isotonic", cv=3)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.clf.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X)


# ---------------------------------------------------------------------------
# Blended Phase-1 model
# ---------------------------------------------------------------------------

class Phase1Model:
    def __init__(self, blend_alpha: float | None = None):
        self.poisson = PoissonModel()
        self.xgb = XGBModel()
        self.alpha = blend_alpha            # None -> tuned on validation
        self.feature_cols: list[str] = list(FEATURE_COLS)
        self.builder = None                 # fitted FeatureBuilder for inference
        self.metrics: dict = {}
        self._trained = False

    def train(self, df: pd.DataFrame | None = None, verbose: bool = True) -> dict:
        if df is None:
            df = load_matches(verbose=False)
        df = df.sort_values("date").reset_index(drop=True)

        X, y, builder = build_features(df, return_builder=True)
        self.builder = builder
        self.feature_cols = list(X.columns)

        cut = max(int(len(df) * (1 - VAL_FRACTION)), 1)
        df_tr, df_val = df.iloc[:cut], df.iloc[cut:]
        X_tr, X_val = X.iloc[:cut], X.iloc[cut:]
        y_tr, y_val = y[:cut], y[cut:]

        if verbose:
            print(f"[phase1] train {len(df_tr)} / validate {len(df_val)} "
                  f"(chronological split at {df_tr['date'].max().date()})")

        self.poisson.fit(df_tr)
        self.xgb.fit(X_tr, y_tr)

        metrics: dict = {"val_samples": int(len(y_val))}
        if len(df_val) >= 10:
            p_pois = self.poisson.predict_matrix(df_val)
            p_xgb = self.xgb.predict_proba(X_val)
            y_oh = np.eye(3)[y_val]

            if self.alpha is None:
                grid = np.linspace(0.0, 1.0, 21)
                briers = [brier_score(a * p_pois + (1 - a) * p_xgb, y_oh)
                          for a in grid]
                self.alpha = float(grid[int(np.argmin(briers))])

            p_blend = self.alpha * p_pois + (1 - self.alpha) * p_xgb
            metrics.update({
                "brier_poisson": brier_score(p_pois, y_oh),
                "brier_xgb": brier_score(p_xgb, y_oh),
                "brier_blend": brier_score(p_blend, y_oh),
                "logloss_blend": log_loss(p_blend, y_oh),
                "alpha_poisson_weight": self.alpha,
                "dc_rho": round(self.poisson.rho, 4),
            })
        else:
            self.alpha = self.alpha if self.alpha is not None else 0.5

        # Refit on ALL data with the tuned alpha so live predictions use
        # everything we know (validation numbers above remain out-of-time).
        self.poisson.fit(df)
        self.xgb.fit(X, y)

        self.metrics = metrics
        self._trained = True
        if verbose:
            for k, v in metrics.items():
                print(f"[phase1]   {k} = {v:.4f}" if isinstance(v, float)
                      else f"[phase1]   {k} = {v}")
        return metrics

    def predict(self, home: str, away: str, neutral: bool = True,
                rest_home: float | None = None,
                rest_away: float | None = None) -> dict:
        if not self._trained:
            raise RuntimeError("Call .train() first")

        p_pois = self.poisson.predict_proba(home, away, neutral)
        p_pois_arr = np.array([[p_pois["win"], p_pois["draw"], p_pois["loss"]]])

        feats = self.builder.features(home, away, neutral)
        if rest_home is not None:
            feats["rest_home"] = float(rest_home)
        if rest_away is not None:
            feats["rest_away"] = float(rest_away)
        X_row = pd.DataFrame([feats])[self.feature_cols]
        p_xgb_arr = self.xgb.predict_proba(X_row)

        p = (self.alpha * p_pois_arr + (1 - self.alpha) * p_xgb_arr)[0]
        p = p / p.sum()
        return {"win": float(p[0]), "draw": float(p[1]), "loss": float(p[2])}

    def score_matrix(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        return self.poisson.score_matrix(home, away, neutral)

    def save(self, path: Path = MODEL_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "Phase1Model":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Entry point (kept for train.py / API compatibility)
# ---------------------------------------------------------------------------

def run_quant_loop(df: pd.DataFrame | None = None) -> Phase1Model:
    """Train + tune (alpha grid on out-of-time window), save, return."""
    m = Phase1Model()
    m.train(df)
    m.save()
    b = m.metrics.get("brier_blend")
    if b is not None:
        print(f"\n[phase1] out-of-time Brier={b:.4f}  "
              f"(alpha={m.alpha:.2f}, rho={m.poisson.rho:+.3f})")
    return m


if __name__ == "__main__":
    m = run_quant_loop()
    for h, a in [("Argentina", "France"), ("Spain", "Portugal"),
                 ("Norway", "Morocco")]:
        print(f"[phase1] {h} vs {a}:", m.predict(h, a, neutral=True))
