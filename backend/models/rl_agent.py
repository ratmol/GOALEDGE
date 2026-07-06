"""
RL Agent (Phase 2 stub) — learns an optimal betting/selection policy.

Uses stable-baselines3 PPO on a custom Gymnasium environment.
The environment feeds the agent a match prediction vector (WDL probs + model
confidence) and asks it to allocate a fraction of the bankroll (0–max_bet).

This file is fully importable and runnable today; it trains on a synthetic
reward signal until real bankroll tracking data is available.

Integration point: replace `_synthetic_probs()` with Phase1Model.predict()
once the trained model is loaded.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

RL_MODEL_PATH = Path(__file__).parent / "rl_agent.zip"

MAX_BET_FRACTION = float(os.getenv("MAX_BET_FRACTION", "0.10"))
STARTING_BANKROLL = float(os.getenv("STARTING_BANKROLL", "1000.0"))


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class BettingEnv(gym.Env):
    """
    One episode = one tournament (configurable number of matches).
    Observation: [win_prob, draw_prob, loss_prob, bankroll_ratio, confidence]
    Action:      continuous in [0, MAX_BET_FRACTION] — fraction to wager
    Reward:      net gain / loss normalized by bankroll at episode start.
    """
    metadata = {"render_modes": []}

    def __init__(self, n_matches: int = 64, predict_fn: Any = None):
        super().__init__()
        self.n_matches = n_matches
        self.predict_fn = predict_fn or self._synthetic_probs

        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([MAX_BET_FRACTION], dtype=np.float32),
            dtype=np.float32,
        )
        self._bankroll = STARTING_BANKROLL
        self._step = 0
        self._init_bankroll = STARTING_BANKROLL

    def _synthetic_probs(self) -> tuple[np.ndarray, int]:
        """Synthetic WDL probs + true outcome (replace with real model later)."""
        rng = np.random
        probs = rng.dirichlet([2, 1, 1]).astype(np.float32)
        outcome = rng.choice([0, 1, 2], p=probs)
        return probs, outcome

    def _get_obs(self, probs: np.ndarray) -> np.ndarray:
        confidence = float(np.max(probs) - 1 / 3) / (2 / 3)
        bankroll_ratio = min(self._bankroll / self._init_bankroll, 2.0) / 2.0
        return np.array([*probs, bankroll_ratio, confidence], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._bankroll = STARTING_BANKROLL
        self._init_bankroll = STARTING_BANKROLL
        self._step = 0
        probs, self._pending_outcome = self._synthetic_probs()
        self._pending_probs = probs
        return self._get_obs(probs), {}

    def step(self, action: np.ndarray):
        bet_fraction = float(np.clip(action[0], 0, MAX_BET_FRACTION))
        wager = self._bankroll * bet_fraction
        probs = self._pending_probs
        outcome = self._pending_outcome

        # Simplified Kelly-inspired payoff: bet on the favourite
        favourite = int(np.argmax(probs))
        if outcome == favourite:
            odds = 1.0 / (probs[favourite] + 1e-9)
            reward = wager * (odds - 1) / self._init_bankroll
            self._bankroll += wager * (odds - 1)
        else:
            reward = -wager / self._init_bankroll
            self._bankroll -= wager

        self._bankroll = max(self._bankroll, 0.0)
        self._step += 1
        terminated = self._step >= self.n_matches or self._bankroll <= 0

        probs, self._pending_outcome = self._synthetic_probs()
        self._pending_probs = probs
        obs = self._get_obs(probs)
        return obs, reward, terminated, False, {}


# ---------------------------------------------------------------------------
# Train / load helpers
# ---------------------------------------------------------------------------

def train_rl_agent(total_timesteps: int = 50_000,
                   save_path: Path = RL_MODEL_PATH) -> PPO:
    env = BettingEnv()
    check_env(env, warn=True)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(str(save_path))
    print(f"[rl] agent saved to {save_path}")
    return model


def load_rl_agent(path: Path = RL_MODEL_PATH) -> PPO | None:
    if not path.exists():
        return None
    return PPO.load(str(path))


def evaluate_agent(model: PPO, n_episodes: int = 20) -> dict:
    env = BettingEnv()
    roi_list = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        roi_list.append((env._bankroll - STARTING_BANKROLL) / STARTING_BANKROLL)
    return {
        "mean_roi": float(np.mean(roi_list)),
        "std_roi": float(np.std(roi_list)),
        "win_rate": float(np.mean([r > 0 for r in roi_list])),
    }


if __name__ == "__main__":
    print("[rl] training agent (50k steps)...")
    agent = train_rl_agent(total_timesteps=50_000)
    stats = evaluate_agent(agent)
    print(f"[rl] eval over 20 episodes: {stats}")
