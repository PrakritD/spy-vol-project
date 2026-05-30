"""Confidence-scaled position sizing.

Two rules, both producing size ∈ [0, 1] of unit notional:

    linear:           size = clip(2 * p_hat - 1, 0, 1)
                      Trades when p_hat > 0.5; full size at p_hat ≥ 1.

    fractional_kelly: size = clip(frac * (p_hat - (1-p_hat)/b), 0, 1)
                      where b is the realised win/loss payoff ratio
                      estimated from training-fold P&L.
                      Default frac = 0.5 (half-Kelly, the standard
                      defensive cut).

The Kelly variant requires a per-fold `b` estimate. Compute it from
training-fold realised P&L using `estimate_kelly_b(train_pnl)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


SizingFn = Callable[[np.ndarray], np.ndarray]


def linear_sizing(p_hat: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * np.asarray(p_hat) - 1.0, 0.0, 1.0)


def kelly_sizing(p_hat: np.ndarray, b: float, frac: float = 0.5) -> np.ndarray:
    """Half-Kelly (default) given win-prob p_hat and payoff ratio b.

    f* = p - (1-p)/b  is full Kelly for a binary win/lose bet.
    We multiply by `frac` (default 0.5) to avoid the well-known full-Kelly
    drawdown problem and clip to [0, 1] for long-only sizing.
    """
    p = np.asarray(p_hat, dtype=float)
    if not np.isfinite(b) or b <= 0:
        return np.zeros_like(p)
    f_star = p - (1.0 - p) / b
    return np.clip(frac * f_star, 0.0, 1.0)


def make_kelly_fn(b: float, frac: float = 0.5) -> SizingFn:
    """Bind a particular b to produce a SizingFn(p_hat) -> size."""
    def _fn(p_hat: np.ndarray) -> np.ndarray:
        return kelly_sizing(p_hat, b=b, frac=frac)
    return _fn


def estimate_kelly_b(returns: pd.Series, min_obs: int = 30) -> float:
    """Average win / average loss magnitude from a P&L series.

    Returns NaN (caller should default to small size) if insufficient data
    or if there are no winning or no losing days.
    """
    r = pd.Series(returns).dropna()
    r = r[r != 0]
    if len(r) < min_obs:
        return float("nan")
    wins = r[r > 0]
    losses = r[r < 0]
    if wins.empty or losses.empty:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


@dataclass(frozen=True)
class SizingSpec:
    """Pair a name with a sizing function for execution-layer dispatch."""
    name: str
    fn: SizingFn

    def apply(self, p_hat: pd.Series | np.ndarray) -> pd.Series:
        out = self.fn(np.asarray(p_hat, dtype=float))
        if isinstance(p_hat, pd.Series):
            return pd.Series(out, index=p_hat.index, name=f"size_{self.name}")
        return pd.Series(out, name=f"size_{self.name}")
