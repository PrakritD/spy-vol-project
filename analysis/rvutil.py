"""Yang-Zhang OHLC realised-volatility, vendored into analysis/ so the deliverables
(FINDINGS deep-history + STRATEGY) have ZERO dependency on the `features/` tree. The
21-month OPRA sub-study still imports `features/` directly.

Single-day Yang-Zhang-style RV from daily OHLC: an overnight close-to-open variance plus a
Garman-Klass intraday-range variance. The standard OHLC-only estimator used when tick data is
too expensive. Reference: Yang & Zhang (2000); Garman-Klass (1980).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_yang_zhang_rv(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Input columns: date, open, high, low, close (daily bars).
    Output: date, rv (annualised, sqrt(252) scaling)."""
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
