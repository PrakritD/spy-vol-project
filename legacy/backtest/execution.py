"""VXX execution model with confidence-scaled sizing and regime slippage.

Signal rule:
    size_t = sizing_fn(p_hat_t) ∈ [0, 1] of unit notional (long-only).

Fill model:
    Signal at close(t) -> size effective from open(t+1) until next change.
    Return on day t+1 = size(t) * vxx_open_to_open_return(t+1, t+2)
        approximated as close-to-close to keep yfinance-only data sufficient.
        TODO when VXX intraday prices land: switch to open-to-open.

Costs:
    Per-day cost = |Δsize_t| * (base_bps + extra_bps_high_vol * vol_indicator_t)
    where vol_indicator_t = 1 if vix_zscore exceeds high_vol_z, else 0.

This produces a continuous turnover charge proportional to position change,
which is the realistic model for partial-resizing strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.sizing import SizingSpec, linear_sizing


@dataclass(frozen=True)
class ExecConfig:
    base_bps_per_side: float = 5.0
    extra_bps_high_vol: float = 5.0
    high_vol_zscore: float = 1.5
    threshold: float | None = None  # legacy long-flat compat (None = use sizing_fn)


def _flat_sizing_from_threshold(p_hat: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(p_hat) >= threshold).astype(float)


def backtest(
    preds: pd.DataFrame,
    vxx_prices: pd.DataFrame,
    cfg: ExecConfig | None = None,
    sizing: SizingSpec | None = None,
    vix_zscore: pd.Series | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Apply sizing rule, produce a daily P&L frame.

    preds       columns: date, p_hat (and optionally y_true)
    vxx_prices  columns: date, close (or vxx_close)
    sizing      SizingSpec; if None and cfg.threshold is set, uses long-flat.
                Default = linear-confidence sizing.
    vix_zscore  optional Series indexed by date; enables regime-conditional
                slippage. If None, uses base cost only.
    """
    cfg = cfg or ExecConfig()
    if sizing is None:
        if cfg.threshold is not None:
            sizing = SizingSpec(
                name=f"flat_{cfg.threshold:.2f}",
                fn=lambda p: _flat_sizing_from_threshold(p, cfg.threshold),
            )
        else:
            sizing = SizingSpec(name="linear", fn=linear_sizing)

    preds = preds[[date_col, "p_hat"]].copy()
    preds[date_col] = pd.to_datetime(preds[date_col]).dt.normalize()

    px = vxx_prices.copy()
    px[date_col] = pd.to_datetime(px[date_col]).dt.normalize()
    close_col = "vxx_close" if "vxx_close" in px.columns else "close"
    px = px[[date_col, close_col]].rename(columns={close_col: "vxx_close"})

    df = preds.merge(px, on=date_col, how="left").sort_values(date_col).reset_index(drop=True)
    df["size"] = sizing.fn(df["p_hat"].to_numpy())
    df["sizing_name"] = sizing.name
    df["vxx_ret_next"] = df["vxx_close"].pct_change().shift(-1)
    df["gross_pnl"] = df["size"] * df["vxx_ret_next"]

    # Turnover cost: |Δsize| × (base + extra in high-vol regime)
    delta_size = df["size"].diff().abs().fillna(df["size"].iloc[0])
    if vix_zscore is not None:
        z = vix_zscore.copy()
        z.index = pd.to_datetime(z.index).normalize()
        df["vix_z"] = df[date_col].map(z).astype(float)
    else:
        df["vix_z"] = np.nan
    high_vol_flag = (df["vix_z"] > cfg.high_vol_zscore).fillna(False).astype(int)
    cost_bps = cfg.base_bps_per_side + cfg.extra_bps_high_vol * high_vol_flag
    df["cost"] = delta_size * cost_bps / 1e4

    df["net_pnl"] = df["gross_pnl"].fillna(0) - df["cost"].fillna(0)
    df["equity"] = (1.0 + df["net_pnl"]).cumprod()
    return df


# Legacy helper, retained for old callers (test_pipeline_smoke).
def signal_to_position(p_hat: pd.Series, threshold: float = 0.5) -> pd.Series:
    return (p_hat >= threshold).astype(int)
