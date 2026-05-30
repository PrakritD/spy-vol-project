"""Reinforcement-learning sizing policy via PPO.

A `gymnasium.Env` wraps the daily walk-forward predictions + VXX returns. PPO
trains on the in-sample window (pre-2025-12) and is evaluated deterministically
on the test window. The policy outputs a continuous position size in [-1, +1];
the env converts that to long/short VXX exposure and computes P&L with the same
cost model the rule-based LFS execution uses.

Observation space (6-dim float):
    0: p_hat - 0.5            (ensemble probability, centred)
    1: vix_zscore             (rolling 20d z-score of VIX)
    2: gex_net_norm           (gex_net divided by trailing 252d std, clipped)
    3: term_9d_30d - 1.0      (curve inversion signal, centred at 1.0)
    4: recent_5d_pnl          (sum of last 5 days' net P&L)
    5: current_drawdown       (current equity / running peak - 1, in [-1, 0])

Action space: continuous size in [-1, +1]. Positive = long VXX, negative = short.

Reward: daily_net_pnl - 0.1 * abs(delta_size) - 0.5 * max(0, drawdown_change)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RLConfig:
    cost_bps_long: float = 5.0
    cost_bps_short: float = 10.0     # double for borrow + slippage on shorts
    turnover_penalty: float = 0.1
    drawdown_penalty: float = 0.5
    short_size_cap: float = 0.5
    max_position: float = 1.0


def _build_obs_panel(
    panel: pd.DataFrame,
    preds: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per-day prediction + features into the env's observation panel."""
    p = preds[["date", "p_hat"]].copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()

    pf = panel.copy()
    pf["date"] = pd.to_datetime(pf["date"]).dt.normalize()

    # Normalise GEX by its training-period std (computed from full history;
    # acceptable here since the RL env uses obs values as features only,
    # not for prediction-time leakage).
    gex_std = pf["gex_net_lag1"].std()
    pf["gex_net_norm"] = (pf["gex_net_lag1"] / (gex_std if gex_std > 0 else 1.0)).clip(-3, 3)

    cols = ["date", "vix_zscore_lag1", "gex_net_norm", "term_9d_30d_lag1", "vxx_close"]
    obs = p.merge(pf[cols], on="date", how="left").sort_values("date").reset_index(drop=True)
    obs["vxx_ret"] = obs["vxx_close"].pct_change().shift(-1)  # forward 1-day return
    obs = obs.dropna(subset=["vxx_ret", "p_hat", "vix_zscore_lag1"]).reset_index(drop=True)
    return obs


class SpyVolEnv:
    """Minimal gym-compatible env.

    Imports gymnasium lazily so the module loads even when SB3 isn't installed.
    """

    def __init__(self, obs_panel: pd.DataFrame, cfg: RLConfig | None = None):
        from gymnasium import spaces

        self.obs_panel = obs_panel.reset_index(drop=True)
        self.cfg = cfg or RLConfig()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # generous bounds — features are real-valued
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=np.float32)

        self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.t = 0
        self.prev_size = 0.0
        self.equity = 1.0
        self.peak = 1.0
        self.drawdown = 0.0
        self.recent_pnl = [0.0] * 5
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        if self.t >= len(self.obs_panel):
            return np.zeros(6, dtype=np.float32)
        row = self.obs_panel.iloc[self.t]
        return np.array([
            float(row["p_hat"]) - 0.5,
            float(row["vix_zscore_lag1"]),
            float(row["gex_net_norm"]) if pd.notna(row["gex_net_norm"]) else 0.0,
            float(row["term_9d_30d_lag1"]) - 1.0,
            float(np.sum(self.recent_pnl)),
            float(self.drawdown),
        ], dtype=np.float32)

    def step(self, action):
        if self.t >= len(self.obs_panel):
            # past terminal — should not happen if SB3 honours `done`
            return self._obs(), 0.0, True, True, {}

        size = float(np.clip(action[0], -1.0, 1.0))
        # Apply short-size cap asymmetrically
        if size < 0:
            size = max(size, -self.cfg.short_size_cap)
        else:
            size = min(size, self.cfg.max_position)

        row = self.obs_panel.iloc[self.t]
        vxx_ret = float(row["vxx_ret"])
        gross_pnl = size * vxx_ret

        # Side-aware cost on turnover
        delta = abs(size - self.prev_size)
        cost_bps = self.cfg.cost_bps_short if size < 0 else self.cfg.cost_bps_long
        cost = delta * cost_bps / 1e4

        net_pnl = gross_pnl - cost
        self.equity *= (1.0 + net_pnl)
        new_peak = max(self.peak, self.equity)
        new_dd = self.equity / new_peak - 1.0
        dd_change = new_dd - self.drawdown   # negative when drawdown deepens

        # Reward = net P&L - turnover penalty - drawdown-deepening penalty
        reward = net_pnl \
                 - self.cfg.turnover_penalty * delta * 0.01 \
                 - self.cfg.drawdown_penalty * max(0.0, -dd_change)

        # Advance state
        self.prev_size = size
        self.recent_pnl = self.recent_pnl[1:] + [net_pnl]
        self.peak = new_peak
        self.drawdown = new_dd
        self.t += 1

        terminated = (self.t >= len(self.obs_panel))
        truncated = False
        info = {"net_pnl": net_pnl, "size": size, "equity": self.equity, "drawdown": new_dd}
        return self._obs(), float(reward), terminated, truncated, info


def train_policy(
    train_obs_panel: pd.DataFrame,
    total_timesteps: int = 30_000,
    seed: int = 13,
    verbose: int = 0,
):
    """Train a PPO policy on the training window. Returns a fitted SB3 PPO model."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as e:
        raise ImportError("stable-baselines3 required; pip install stable-baselines3") from e

    def _make_env():
        return Monitor(_GymEnvWrapper(SpyVolEnv(train_obs_panel)))

    venv = DummyVecEnv([_make_env])
    model = PPO(
        "MlpPolicy", venv,
        learning_rate=3e-4,
        n_steps=512, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, vf_coef=0.5,
        verbose=verbose,
        seed=seed,
        policy_kwargs={"net_arch": [32, 32]},  # small policy — overfitting guard
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model


def evaluate_policy(
    model,
    test_obs_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Run the trained policy deterministically on the test window. Returns a
    per-day P&L frame compatible with `backtest.metrics.trader_summary`."""
    env = SpyVolEnv(test_obs_panel)
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        rows.append({
            "date": test_obs_panel.iloc[env.t - 1]["date"],
            "p_hat": float(test_obs_panel.iloc[env.t - 1]["p_hat"]),
            "size": info["size"],
            "net_pnl": info["net_pnl"],
            "gross_pnl": info["size"] * test_obs_panel.iloc[env.t - 1]["vxx_ret"],
            "cost": info["size"] * test_obs_panel.iloc[env.t - 1]["vxx_ret"] - info["net_pnl"],
            "equity": info["equity"],
        })
        obs = next_obs
        done = terminated or truncated

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# gymnasium-compatible wrapper. SB3 needs the canonical gymnasium API
# (Env class inheriting gym.Env). We define it here lazily.
# ---------------------------------------------------------------------------

def _GymEnvWrapper(env: SpyVolEnv):
    """Lazy adapter so SB3 sees a proper gymnasium.Env subclass."""
    import gymnasium as gym

    class _Wrapped(gym.Env):
        metadata = {"render_modes": []}
        def __init__(self, inner):
            super().__init__()
            self._inner = inner
            self.action_space = inner.action_space
            self.observation_space = inner.observation_space
        def reset(self, *, seed=None, options=None):
            return self._inner.reset(seed=seed, options=options)
        def step(self, action):
            return self._inner.step(action)
        def render(self):
            return None

    return _Wrapped(env)


# ---------------------------------------------------------------------------
# Orchestration: train on in-sample, evaluate on test, return comparable
# P&L frame plus headline metrics.
# ---------------------------------------------------------------------------

def run_rl_pipeline(
    panel: pd.DataFrame,
    preds: pd.DataFrame,
    test_start: str = "2025-12-01",
    test_end: str = "2026-04-30",
    total_timesteps: int = 30_000,
    seed: int = 13,
) -> dict:
    """Train PPO on pre-test data, evaluate on the test window."""
    obs_all = _build_obs_panel(panel, preds)
    obs_all["date"] = pd.to_datetime(obs_all["date"]).dt.normalize()

    test_start_ts = pd.Timestamp(test_start).normalize()
    test_end_ts = pd.Timestamp(test_end).normalize()
    train_obs = obs_all[obs_all["date"] < test_start_ts].reset_index(drop=True)
    test_obs = obs_all[(obs_all["date"] >= test_start_ts) & (obs_all["date"] <= test_end_ts)].reset_index(drop=True)

    if len(train_obs) < 50:
        raise ValueError(f"insufficient RL training data: {len(train_obs)} days")
    if len(test_obs) < 20:
        raise ValueError(f"insufficient RL test data: {len(test_obs)} days")

    print(f"  RL train: {len(train_obs)} days; test: {len(test_obs)} days; PPO {total_timesteps} steps")
    model = train_policy(train_obs, total_timesteps=total_timesteps, seed=seed)
    pnl = evaluate_policy(model, test_obs)
    return {"model": model, "pnl": pnl, "n_train": len(train_obs), "n_test": len(test_obs)}
