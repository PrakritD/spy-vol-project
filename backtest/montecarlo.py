"""Monte Carlo stress test of a strategy return series.

Stationary block bootstrap with block size ~ N^(1/3). For each of n_paths
resampled return series, recompute final return / Sharpe / max drawdown.
Output a distribution of strategy outcomes, plus the probability the
strategy ends profitable (i.e. P(equity[-1] > 1)).

Use case: turn a single observed Sharpe number into a Monte Carlo
confidence band, the only honest claim available given autocorrelated
daily returns and small N_eff.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonteCarloResult:
    n_paths: int
    block_size: int
    sharpe_observed: float
    sharpe_mean: float
    sharpe_p5: float
    sharpe_p50: float
    sharpe_p95: float
    max_dd_observed: float
    max_dd_mean: float
    max_dd_p5: float
    max_dd_p95: float
    p_profitable: float
    equity_paths: np.ndarray | None = None    # (n_paths, N) — optional, for fan chart

    def headline_row(self) -> dict:
        """Compact dict for a summary table."""
        return {
            "n_paths": self.n_paths,
            "sharpe_obs": self.sharpe_observed,
            "sharpe_mc_p5": self.sharpe_p5,
            "sharpe_mc_p50": self.sharpe_p50,
            "sharpe_mc_p95": self.sharpe_p95,
            "max_dd_obs": self.max_dd_observed,
            "max_dd_mc_p95": self.max_dd_p95,
            "p_profitable": self.p_profitable,
        }


def _block_resample(returns: np.ndarray, block_size: int, n_paths: int,
                     seed: int) -> np.ndarray:
    """Stationary block bootstrap: (n_paths, N) resampled return matrix."""
    n = len(returns)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, max(1, n - block_size + 1), size=(n_paths, n_blocks))
    paths = np.empty((n_paths, n), dtype=float)
    for i in range(n_paths):
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts[i]])[:n]
        paths[i] = returns[idx]
    return paths


def _sharpe(r: np.ndarray, periods: int = 252) -> float:
    if r.size < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(periods))


def _max_drawdown_from_returns(r: np.ndarray) -> float:
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def simulate(
    strategy_returns: pd.Series | np.ndarray,
    n_paths: int = 10_000,
    block_size: int | None = None,
    return_equity_paths: bool = False,
    seed: int = 13,
) -> MonteCarloResult:
    """Run the block-bootstrap Monte Carlo.

    Args:
        strategy_returns: daily net P&L returns.
        n_paths:          number of resampled paths (default 10k).
        block_size:       block size in days. Default = round(N^(1/3)), with
                          a floor of 5 to respect daily-vol autocorrelation.
        return_equity_paths: if True, retain all equity curves for plotting.
                          Increases memory by O(n_paths * N).
        seed:             RNG seed for reproducibility.
    """
    r = np.asarray(pd.Series(strategy_returns).dropna(), dtype=float)
    n = len(r)
    if n < 30:
        raise ValueError(f"too few returns for Monte Carlo (n={n}, need >= 30)")
    if block_size is None:
        block_size = max(5, int(round(n ** (1 / 3))))

    paths = _block_resample(r, block_size, n_paths, seed)
    sharpes = np.apply_along_axis(_sharpe, 1, paths)
    max_dds = np.apply_along_axis(_max_drawdown_from_returns, 1, paths)
    final_equity = np.prod(1.0 + paths, axis=1)
    p_profit = float((final_equity > 1.0).mean())

    return MonteCarloResult(
        n_paths=n_paths,
        block_size=block_size,
        sharpe_observed=_sharpe(r),
        sharpe_mean=float(sharpes.mean()),
        sharpe_p5=float(np.percentile(sharpes, 5)),
        sharpe_p50=float(np.percentile(sharpes, 50)),
        sharpe_p95=float(np.percentile(sharpes, 95)),
        max_dd_observed=_max_drawdown_from_returns(r),
        max_dd_mean=float(max_dds.mean()),
        max_dd_p5=float(np.percentile(max_dds, 5)),
        max_dd_p95=float(np.percentile(max_dds, 95)),
        p_profitable=p_profit,
        equity_paths=(np.cumprod(1.0 + paths, axis=1) if return_equity_paths else None),
    )


def fan_chart(
    mc: MonteCarloResult,
    out_path: Path,
    title: str = "Monte Carlo equity-path fan chart",
) -> Path:
    """Plot 5/50/95 percentile fan + a sample of individual paths."""
    if mc.equity_paths is None:
        raise ValueError("simulate(...) was called with return_equity_paths=False")
    import matplotlib.pyplot as plt

    eq = mc.equity_paths
    p5 = np.percentile(eq, 5, axis=0)
    p25 = np.percentile(eq, 25, axis=0)
    p50 = np.percentile(eq, 50, axis=0)
    p75 = np.percentile(eq, 75, axis=0)
    p95 = np.percentile(eq, 95, axis=0)

    fig, ax = plt.subplots(figsize=(11, 5))
    days = np.arange(eq.shape[1])
    ax.fill_between(days, p5, p95, alpha=0.20, color="#1f77b4", label="5–95% band")
    ax.fill_between(days, p25, p75, alpha=0.35, color="#1f77b4", label="25–75% band")
    ax.plot(days, p50, color="black", linewidth=1.5, label="median")
    ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.5)
    # overlay a small sample of paths
    sample_idx = np.random.default_rng(0).choice(eq.shape[0], size=min(20, eq.shape[0]), replace=False)
    for i in sample_idx:
        ax.plot(days, eq[i], color="#1f77b4", alpha=0.10, linewidth=0.7)
    ax.set_xlabel("Day")
    ax.set_ylabel("Equity (start=1.0)")
    ax.set_title(f"{title}\n"
                  f"P(profitable) = {mc.p_profitable:.1%}  |  "
                  f"Sharpe MC (p5, p50, p95) = ({mc.sharpe_p5:.2f}, {mc.sharpe_p50:.2f}, {mc.sharpe_p95:.2f})")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
