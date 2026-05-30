"""Backtest evaluation metrics.

Three flavours, used together in the report:
    Classification:  AUC, Brier, base rate, log-loss.
    Trader / risk:   Sharpe, Sortino, Calmar, CAGR, max DD + duration,
                     downside dev, VaR / CVaR, skew, kurtosis,
                     hit rate, profit factor, turnover, monthly returns.
    Inference:       Probabilistic Sharpe Ratio (Bailey & López de Prado),
                     block-bootstrap Sharpe CI (autocorrelation-aware),
                     information ratio vs configurable benchmark.

Daily return series from this kind of strategy are autocorrelated (the
signal itself comes from autocorrelated features). Raw Sharpe SE is
biased — use the inference metrics for any significance claim.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss


TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Quant-classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(preds: pd.DataFrame) -> dict[str, float]:
    df = preds.dropna(subset=["y_true", "p_hat"])
    if df.empty:
        return {"auc": float("nan"), "brier": float("nan"), "base_rate": float("nan")}
    y, p = df["y_true"].astype(int).to_numpy(), df["p_hat"].clip(1e-6, 1 - 1e-6).to_numpy()
    base = float(y.mean())
    # Base-rate log-loss is what an "always predict base-rate" model achieves.
    base_logloss = -(base * np.log(max(base, 1e-6)) + (1 - base) * np.log(max(1 - base, 1e-6)))
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "logloss_base": float(base_logloss),
        "base_rate": base,
        "n_obs": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Trader / risk-adjusted metrics
# ---------------------------------------------------------------------------

def sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(returns).dropna()
    if r.std() == 0 or len(r) < 2:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, target: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(returns).dropna()
    downside = r[r < target]
    if downside.empty:
        return float("inf") if r.mean() > target else float("nan")
    dd_std = np.sqrt((downside ** 2).mean())
    if dd_std == 0:
        return float("nan")
    return float((r.mean() - target) / dd_std * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    e = pd.Series(equity).dropna()
    if e.empty:
        return float("nan")
    peak = e.cummax()
    return float((e / peak - 1.0).min())


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest stretch of bars between successive equity peaks."""
    e = pd.Series(equity).dropna().reset_index(drop=True)
    if e.empty:
        return 0
    peak = e.cummax()
    is_at_peak = (e == peak)
    # length of longest run of ~is_at_peak
    longest = current = 0
    for at_peak in is_at_peak:
        current = 0 if at_peak else current + 1
        longest = max(longest, current)
    return int(longest)


def calmar(returns: pd.Series, equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised return / |max drawdown|."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    ann_ret = (1.0 + r.mean()) ** periods_per_year - 1.0
    dd = abs(max_drawdown(equity))
    if dd == 0 or not np.isfinite(dd):
        return float("nan")
    return float(ann_ret / dd)


def hit_rate(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    nonzero = r[r != 0]
    if nonzero.empty:
        return float("nan")
    return float((nonzero > 0).mean())


def avg_win_loss(returns: pd.Series) -> tuple[float, float]:
    r = pd.Series(returns).dropna()
    wins = r[r > 0]
    losses = r[r < 0]
    avg_w = float(wins.mean()) if not wins.empty else float("nan")
    avg_l = float(losses.mean()) if not losses.empty else float("nan")
    return avg_w, avg_l


def profit_factor(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    gain = r[r > 0].sum()
    loss = abs(r[r < 0].sum())
    if loss == 0:
        return float("inf") if gain > 0 else float("nan")
    return float(gain / loss)


def time_in_market(size: pd.Series) -> float:
    s = pd.Series(size).dropna()
    if s.empty:
        return float("nan")
    return float((s.abs() > 1e-12).mean())


def turnover(size: pd.Series) -> float:
    """Average absolute daily change in position size."""
    s = pd.Series(size).dropna()
    if len(s) < 2:
        return float("nan")
    return float(s.diff().abs().mean())


def cagr(equity: pd.Series, n_days: int | None = None,
         periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate from an equity curve.

    Convention: equity is `(1 + r).cumprod()` — implicit starting capital of
    1.0 BEFORE the first return. So the total growth factor over the entire
    period is `equity.iloc[-1]` itself (not equity[-1]/equity[0], which would
    drop the first day's return).

    n_days = number of trading periods spanned (defaults to len(equity)).
    """
    e = pd.Series(equity).dropna()
    if e.empty:
        return float("nan")
    n = n_days if n_days is not None else len(e)
    if n <= 0:
        return float("nan")
    total_growth = float(e.iloc[-1])
    if total_growth <= 0:
        return float("nan")
    return total_growth ** (periods_per_year / n) - 1.0


def annualised_vol(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std() * np.sqrt(periods_per_year))


def downside_deviation(returns: pd.Series, target: float = 0.0,
                        periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(returns).dropna()
    short = np.minimum(r - target, 0.0)
    if len(short) == 0:
        return float("nan")
    return float(np.sqrt((short ** 2).mean()) * np.sqrt(periods_per_year))


def value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical VaR at level alpha (returns the absolute loss as positive)."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    return float(-np.percentile(r, 100 * alpha))


def conditional_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Expected loss conditional on being in the worst-alpha tail."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    cutoff = np.percentile(r, 100 * alpha)
    tail = r[r <= cutoff]
    if tail.empty:
        return float("nan")
    return float(-tail.mean())


def skew_kurt(returns: pd.Series) -> tuple[float, float]:
    """Excess skewness and kurtosis (Fisher: gaussian = 0)."""
    r = pd.Series(returns).dropna()
    if len(r) < 4:
        return float("nan"), float("nan")
    return float(stats.skew(r)), float(stats.kurtosis(r, fisher=True))


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0,
                                periods_per_year: int = TRADING_DAYS) -> float:
    """Bailey & López de Prado (2014) Probabilistic Sharpe Ratio.

    Returns P(true SR > sr_benchmark) accounting for skew/kurtosis-induced
    bias in the finite-sample Sharpe estimator. PSR > 0.95 = statistically
    significant edge at 95% confidence.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 4:
        return float("nan")
    sr_hat = sharpe(r, periods_per_year=periods_per_year)
    if not np.isfinite(sr_hat):
        return float("nan")
    # Convert annualised SR back to per-period for the moment-based formula.
    sr_per = sr_hat / np.sqrt(periods_per_year)
    sk, ku = skew_kurt(r)
    n = len(r)
    # Excess kurtosis (Fisher) -> add 3 to recover Pearson definition used by BLdP.
    pearson_kurt = ku + 3.0
    denom = np.sqrt(1.0 - sk * sr_per + ((pearson_kurt - 1.0) / 4.0) * sr_per ** 2)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    z = (sr_per - sr_benchmark / np.sqrt(periods_per_year)) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def _block_indices(n: int, block_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
    return idx[:n]


def block_bootstrap_sharpe_ci(returns: pd.Series, n_boot: int = 1000,
                                block_size: int | None = None,
                                alpha: float = 0.05,
                                periods_per_year: int = TRADING_DAYS,
                                seed: int = 13) -> tuple[float, float]:
    """(low, high) confidence interval on annualised Sharpe via stationary block bootstrap.

    Default block_size = max(8, round(N**(1/3))) — captures autocorrelation
    decay typical of daily strategy returns. Critical for strategies built on
    autocorrelated features like ours (VIX rho_1 = 0.92).
    """
    r = pd.Series(returns).dropna().to_numpy()
    n = len(r)
    if n < 30:
        return float("nan"), float("nan")
    if block_size is None:
        block_size = max(8, int(round(n ** (1.0 / 3.0))))
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = _block_indices(n, block_size, seed=seed + b)
        sample = r[idx]
        sd = sample.std()
        samples[b] = (sample.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series,
                       periods_per_year: int = TRADING_DAYS) -> float:
    """Sharpe of (strategy - benchmark) excess returns. Aligned on index."""
    r = pd.Series(returns).dropna()
    b = pd.Series(benchmark_returns).dropna()
    aligned = pd.concat([r.rename("r"), b.rename("b")], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    excess = aligned["r"] - aligned["b"]
    if excess.std() == 0:
        return float("nan")
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def monthly_returns(pnl_df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Compounded net P&L per (year, month). Returns wide YxM table."""
    df = pnl_df[[date_col, "net_pnl"]].dropna().copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    monthly = (
        df.groupby(["year", "month"])["net_pnl"]
          .apply(lambda r: (1.0 + r).prod() - 1.0)
          .unstack("month")
          .reindex(columns=range(1, 13))
    )
    return monthly


# ---------------------------------------------------------------------------
# One-call summaries
# ---------------------------------------------------------------------------

def summarize(pnl_df: pd.DataFrame) -> dict[str, float]:
    """Backwards-compatible: returns the original short summary."""
    return {
        "sharpe": sharpe(pnl_df["net_pnl"]),
        "sharpe_gross": sharpe(pnl_df["gross_pnl"]),
        "max_drawdown": max_drawdown(pnl_df["equity"]),
        "hit_rate": hit_rate(pnl_df["net_pnl"]),
        "total_return": float(pnl_df["equity"].iloc[-1] - 1.0),
        "n_days": int(len(pnl_df)),
        "n_flips": int(pnl_df["cost"].gt(0).sum()),
    }


def trader_summary(pnl_df: pd.DataFrame, size_col: str = "size",
                    benchmark_returns: pd.Series | None = None) -> dict[str, float]:
    """Full trader-flavoured stat block, with autocorrelation-aware inference.

    Args:
        pnl_df: output of `backtest.execution.backtest` — must have columns
                net_pnl, gross_pnl, equity, cost, and optionally `size_col`.
        benchmark_returns: optional daily return series for information ratio
                (e.g. VXX buy-and-hold returns or zero-return cash baseline).
    """
    r = pnl_df["net_pnl"]
    avg_w, avg_l = avg_win_loss(r)
    sk, ku = skew_kurt(r)
    sr_lo, sr_hi = block_bootstrap_sharpe_ci(r)

    out = {
        # --- return ---
        "total_return": float(pnl_df["equity"].iloc[-1] - 1.0),
        "cagr": cagr(pnl_df["equity"]),
        "ann_vol": annualised_vol(r),
        "downside_dev": downside_deviation(r),
        # --- risk-adjusted ---
        "sharpe_net": sharpe(r),
        "sharpe_gross": sharpe(pnl_df["gross_pnl"]),
        "sharpe_ci_lo": sr_lo,           # block-bootstrap 95% CI low
        "sharpe_ci_hi": sr_hi,           # block-bootstrap 95% CI high
        "psr_vs_zero": probabilistic_sharpe_ratio(r, sr_benchmark=0.0),
        "sortino": sortino(r),
        "calmar": calmar(r, pnl_df["equity"]),
        # --- drawdown / tail risk ---
        "max_drawdown": max_drawdown(pnl_df["equity"]),
        "max_drawdown_duration": max_drawdown_duration(pnl_df["equity"]),
        "var_95": value_at_risk(r, alpha=0.05),
        "cvar_95": conditional_var(r, alpha=0.05),
        "var_99": value_at_risk(r, alpha=0.01),
        "cvar_99": conditional_var(r, alpha=0.01),
        # --- distributional ---
        "skew": sk,
        "excess_kurt": ku,
        # --- trade quality ---
        "hit_rate": hit_rate(r),
        "avg_win": avg_w,
        "avg_loss": avg_l,
        "profit_factor": profit_factor(r),
        # --- bookkeeping ---
        "total_return_gross": float((1 + pnl_df["gross_pnl"].fillna(0)).prod() - 1.0),
        "n_days": int(len(pnl_df)),
        "total_cost_bps": float(pnl_df["cost"].sum() * 1e4),
    }
    if size_col in pnl_df.columns:
        out["time_in_market"] = time_in_market(pnl_df[size_col])
        out["turnover"] = turnover(pnl_df[size_col])
    if benchmark_returns is not None:
        out["info_ratio_vs_bench"] = information_ratio(r, benchmark_returns)
    return out
