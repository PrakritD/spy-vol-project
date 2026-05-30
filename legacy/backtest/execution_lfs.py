"""Long-flat-short VXX execution.

Parallel to `backtest/execution.backtest()` but allows the strategy to be
short on the inverse signal. Long-only VXX bleeds through VIX-futures
contango (~30-50% per year); going short on low p_hat days harvests that
decay.

Position mapping:
    p_hat > p_long_threshold     -> long  VXX, size = +sizing_long(p_hat)
    p_hat < p_short_threshold    -> short VXX, size = -asymmetry * sizing_short(1 - p_hat)
    otherwise                    -> flat

Asymmetric sizing (default asymmetry=0.5) protects against vol-spike tails:
short VXX has unbounded downside when VIX squeezes. Short-side cost is
doubled to model borrow + slippage.

The function returns the same P&L frame schema as `execution.backtest()`
so downstream metrics + figures work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.sizing import SizingSpec, linear_sizing


@dataclass(frozen=True)
class LFSConfig:
    p_long_threshold: float = 0.55
    p_short_threshold: float = 0.45
    asymmetry: float = 0.5                 # short_size = asymmetry * long_size at same |p - 0.5|
    short_size_cap: float = 0.5            # hard cap on short notional
    base_bps_per_side: float = 5.0
    short_bps_multiplier: float = 2.0      # short cost = mult * long cost (borrow + slippage)
    extra_bps_high_vol: float = 5.0
    high_vol_zscore: float = 1.5
    vvix_vix_kill_switch: float | None = None   # if set, force-flatten shorts when vvix/vix > threshold


def _sign_and_magnitude(
    p_hat: np.ndarray, cfg: LFSConfig, sizing_long: SizingSpec, sizing_short: SizingSpec | None,
) -> np.ndarray:
    """Map p_hat -> signed size in [-cfg.short_size_cap, +1]."""
    p = np.asarray(p_hat, dtype=float)
    sizing_short = sizing_short or sizing_long
    long_mag = np.clip(sizing_long.fn(p), 0.0, 1.0)
    short_mag = np.clip(sizing_short.fn(1.0 - p), 0.0, 1.0) * cfg.asymmetry
    short_mag = np.clip(short_mag, 0.0, cfg.short_size_cap)

    sizes = np.zeros_like(p)
    long_mask = p > cfg.p_long_threshold
    short_mask = p < cfg.p_short_threshold
    sizes[long_mask] = +long_mag[long_mask]
    sizes[short_mask] = -short_mag[short_mask]
    return sizes


def backtest_lfs(
    preds: pd.DataFrame,
    vxx_prices: pd.DataFrame,
    cfg: LFSConfig | None = None,
    sizing_long: SizingSpec | None = None,
    sizing_short: SizingSpec | None = None,
    vix_zscore: pd.Series | None = None,
    vvix_vix: pd.Series | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Long-flat-short VXX backtest. Returns a daily P&L frame compatible with
    `backtest.metrics.trader_summary`."""
    cfg = cfg or LFSConfig()
    sizing_long = sizing_long or SizingSpec(name="linear", fn=linear_sizing)

    preds = preds[[date_col, "p_hat"]].copy()
    preds[date_col] = pd.to_datetime(preds[date_col]).dt.normalize()

    px = vxx_prices.copy()
    px[date_col] = pd.to_datetime(px[date_col]).dt.normalize()
    close_col = "vxx_close" if "vxx_close" in px.columns else "close"
    px = px[[date_col, close_col]].rename(columns={close_col: "vxx_close"})

    df = preds.merge(px, on=date_col, how="left").sort_values(date_col).reset_index(drop=True)

    df["size"] = _sign_and_magnitude(df["p_hat"].to_numpy(), cfg, sizing_long, sizing_short)

    # Apply VVIX/VIX kill switch on shorts only.
    if cfg.vvix_vix_kill_switch is not None and vvix_vix is not None:
        vv = vvix_vix.copy()
        vv.index = pd.to_datetime(vv.index).normalize()
        df["vvix_vix"] = df[date_col].map(vv).astype(float)
        kill = (df["vvix_vix"] > cfg.vvix_vix_kill_switch).fillna(False)
        df.loc[kill & (df["size"] < 0), "size"] = 0.0

    df["sizing_name"] = f"lfs_{sizing_long.name}"
    df["vxx_ret_next"] = df["vxx_close"].pct_change().shift(-1)
    df["gross_pnl"] = df["size"] * df["vxx_ret_next"]

    # Side-aware cost: shorts pay more.
    delta_size = df["size"].diff().abs().fillna(df["size"].abs().iloc[0])
    if vix_zscore is not None:
        z = vix_zscore.copy()
        z.index = pd.to_datetime(z.index).normalize()
        df["vix_z"] = df[date_col].map(z).astype(float)
    else:
        df["vix_z"] = np.nan
    high_vol_flag = (df["vix_z"] > cfg.high_vol_zscore).fillna(False).astype(int)

    # Cost basis: when going short, the |Δsize| × short_bps_multiplier
    is_short = (df["size"] < 0).astype(int)
    cost_bps = cfg.base_bps_per_side * (1.0 + (cfg.short_bps_multiplier - 1.0) * is_short) \
               + cfg.extra_bps_high_vol * high_vol_flag
    df["cost"] = delta_size * cost_bps / 1e4

    df["net_pnl"] = df["gross_pnl"].fillna(0) - df["cost"].fillna(0)
    df["equity"] = (1.0 + df["net_pnl"]).cumprod()
    return df
