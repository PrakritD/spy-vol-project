"""Realised-vol target construction.

RV_t = sqrt(252 * sum_{i in day t} log_return_i^2), computed on bar_minutes-bar
intraday SPY returns.

Label: y_t = 1 if RV_{t+1} > rolling_mean(RV[t-window+1 : t+1])  (no t+1 leak).
Returns a daily frame with columns: date, rv, rv_rolling_mean, y_next, rv_next.

The rolling mean ends AT date t. y_t compares RV at t+1 to that mean. The label
column 'y_next' is aligned to the *prediction* date t (so a model takes features
at t and predicts y_next[t]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetConfig:
    bar_minutes: int = 5
    rolling_window_days: int = 21
    trading_minutes_per_day: int = 390
    # Forward horizon for the target. 1 = "RV tomorrow vs trailing mean";
    # 5 = "average RV over next 5 days vs trailing mean". Longer horizons
    # average out daily noise and tend to have higher SNR at the cost of
    # building autocorrelation into the label sequence (consecutive labels
    # share forward-window days).
    forward_horizon_days: int = 1


def daily_rv(bars: pd.DataFrame, cfg: TargetConfig | None = None) -> pd.DataFrame:
    """Compute one RV per trading date from intraday bars.

    Input: columns ts (UTC), price. Trades are bucketed externally; this assumes
    one row per bar. Output: date, rv.
    """
    cfg = cfg or TargetConfig()
    df = bars.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = df["ts"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    df["log_ret"] = np.log(df["price"]).groupby(df["date"]).diff()
    df["log_ret2"] = df["log_ret"] ** 2
    rv = df.groupby("date", sort=True)["log_ret2"].sum().pipe(lambda s: np.sqrt(252.0 * s))
    return rv.rename("rv").reset_index()


def daily_yang_zhang_rv(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Single-day Yang-Zhang-style RV from daily OHLC.

    Combines an overnight return variance with a Garman-Klass intraday
    variance — captures both the close-to-open jump and intraday range.
    Drop-in replacement for `daily_rv` when intraday bars aren't available.

    Input columns: date, open, high, low, close (daily bars).
    Output: date, rv (annualised, sqrt(252) scaling).

    Reference: Yang & Zhang (2000); Garman-Klass (1980). This is the standard
    OHLC-only estimator used when tick data is too expensive — the standard
    daily microstructure proxy in the realised-vol literature.
    """
    df = ohlc.copy().sort_values("date").reset_index(drop=True)
    log_o = np.log(df["open"].astype(float))
    log_h = np.log(df["high"].astype(float))
    log_l = np.log(df["low"].astype(float))
    log_c = np.log(df["close"].astype(float))
    log_c_prev = log_c.shift(1)

    overnight = (log_o - log_c_prev) ** 2
    gk = 0.5 * (log_h - log_l) ** 2 - (2.0 * np.log(2.0) - 1.0) * (log_c - log_o) ** 2
    per_day_var = overnight + gk
    rv = np.sqrt(252.0 * per_day_var.clip(lower=0.0))
    return pd.DataFrame({"date": df["date"], "rv": rv.values})


def build_target(rv_daily: pd.DataFrame, cfg: TargetConfig | None = None) -> pd.DataFrame:
    """Construct rolling-mean comparison label.

    rv_daily must have columns date, rv (already aggregated to one row per day).

    With forward_horizon_days = h:
        rv_forward(t) = mean(RV[t+1 .. t+h])
        y_next(t)     = 1{rv_forward(t) > rv_rolling_mean(t)}
    The `rv_next` column is rv_forward(t) for backward compatibility.
    """
    cfg = cfg or TargetConfig()
    out = rv_daily.copy().sort_values("date").reset_index(drop=True)
    win = cfg.rolling_window_days
    h = max(1, cfg.forward_horizon_days)

    # Trailing mean ending AT t (inclusive). Same shape as before.
    out["rv_rolling_mean"] = out["rv"].rolling(win, min_periods=win).mean()

    # Forward h-day mean: mean(RV[t+1..t+h]).
    # Compute via shift(-1) then rolling, then shift back to align with t.
    if h == 1:
        out["rv_next"] = out["rv"].shift(-1)
    else:
        fwd = out["rv"].shift(-1).rolling(h, min_periods=h).mean()
        # `rolling(h)` at position k carries info from k-h+1..k. After shift(-1),
        # position t carries info from t-h+2..t+1 — i.e. RV[t+1..t+h] shifted
        # back by h-1. Final alignment: shift(-(h-1)).
        out["rv_next"] = fwd.shift(-(h - 1))

    out["y_next"] = (out["rv_next"] > out["rv_rolling_mean"]).astype("Int64")
    mask = out["rv_rolling_mean"].isna() | out["rv_next"].isna()
    out.loc[mask, "y_next"] = pd.NA
    return out
