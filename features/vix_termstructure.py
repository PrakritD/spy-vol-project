"""VIX family features: level, z-score, term structure ratios.

Input is a daily wide frame indexed by date with columns:
    vix, vix9d, vix3m, vvix
Output has the same date index plus lagged + derived features. All features
are lagged by one day at the join step downstream — this module operates on
contemporaneous data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VixConfig:
    zscore_window_days: int = 20


def compute(df: pd.DataFrame, cfg: VixConfig | None = None) -> pd.DataFrame:
    cfg = cfg or VixConfig()
    required = {"vix", "vix9d", "vix3m", "vvix"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing VIX columns: {sorted(missing)}")

    out = pd.DataFrame(index=df.index)
    out["vix_level"] = df["vix"]
    out["vix_log"] = np.log(df["vix"])
    out["vix_chg_1d"] = df["vix"].diff()
    out["vix_chg_5d"] = df["vix"].diff(5)

    win = cfg.zscore_window_days
    mu = df["vix"].rolling(win, min_periods=win).mean()
    sd = df["vix"].rolling(win, min_periods=win).std()
    out["vix_zscore"] = (df["vix"] - mu) / sd

    out["term_9d_30d"] = df["vix9d"] / df["vix"]
    out["term_30d_90d"] = df["vix"] / df["vix3m"]
    out["vvix_vix"] = df["vvix"] / df["vix"]
    return out
