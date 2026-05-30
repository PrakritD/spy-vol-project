"""Invariants that protect against future-leak bugs.

The target-construction test is the critical one: modifying RV values strictly
in the future must not change earlier labels. If this breaks, the whole
backtest is fiction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.rv_target import TargetConfig, build_target, daily_rv
from features.gex import GexConfig, run as gex_run


def _synthetic_rv_daily(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    rv = pd.Series(rng.lognormal(mean=-1.5, sigma=0.4, size=n), index=dates)
    return pd.DataFrame({"date": dates, "rv": rv.values})


def test_target_no_future_leak():
    rv = _synthetic_rv_daily(n=60)
    cfg = TargetConfig(rolling_window_days=21)
    base = build_target(rv, cfg)

    # Modify RV strictly after date index 40. Labels at t <= 38 must not change.
    rv_perturbed = rv.copy()
    rv_perturbed.loc[rv_perturbed.index >= 41, "rv"] *= 5.0
    perturbed = build_target(rv_perturbed, cfg)

    # y_next at t=38 depends on rv_rolling_mean computed on rv[18..38] and rv[39]
    # Neither uses indices >= 41, so y_next[38] must match.
    for t in range(0, 39):
        a, b = base.loc[t, "y_next"], perturbed.loc[t, "y_next"]
        # Both may be NA before the rolling window fills; equality of NA is OK.
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == b, f"label at index {t} changed under future perturbation"


def test_target_label_alignment():
    """y_next at row t must reflect RV at t+1, not t."""
    rv = _synthetic_rv_daily(n=40)
    out = build_target(rv, TargetConfig(rolling_window_days=21))
    # For each valid row, y_next == (rv_next > rv_rolling_mean) and rv_next == rv[t+1]
    for t in range(21, len(out) - 1):
        rolling = out.loc[t, "rv_rolling_mean"]
        next_rv = out.loc[t, "rv_next"]
        expected_next = rv.loc[t + 1, "rv"]
        assert next_rv == pytest.approx(expected_next), f"rv_next mismatch at {t}"
        if not pd.isna(out.loc[t, "y_next"]):
            assert (next_rv > rolling) == bool(out.loc[t, "y_next"])


def test_rv_from_bars_is_daily_aggregated():
    """daily_rv produces one row per ET trading date."""
    # 3 days, 78 5-min bars per day -> daily aggregation produces 3 rows
    rng = np.random.default_rng(7)
    rows = []
    for d in pd.bdate_range("2023-03-01", periods=3):
        for bar_idx in range(78):
            ts = (pd.Timestamp(d).tz_localize("America/New_York")
                  + pd.Timedelta(minutes=30 + 5 * bar_idx)).tz_convert("UTC")
            rows.append({"ts": ts, "price": 400 * np.exp(rng.normal(0, 0.001))})
    bars = pd.DataFrame(rows)
    out = daily_rv(bars)
    assert len(out) == 3
    assert (out["rv"] > 0).all()


def test_gex_aggregation_signs():
    """Calls contribute positive GEX, puts negative under our dealer-convention."""
    # Single date, two contracts equally weighted: a call and a put at same strike+expiry+IV.
    date = pd.Timestamp("2024-01-15")
    expiry = pd.Timestamp("2024-02-16")
    chain = pd.DataFrame({
        "date": [date, date],
        "expiry": [expiry, expiry],
        "strike": [450.0, 450.0],
        "option_type": ["C", "P"],
        # Approximate ATM call & put prices for S=450, K=450, T~32/365, r=0.05, sigma=0.18
        "price": [5.0, 4.5],
        "open_interest": [1000, 1000],
        "spot": [450.0, 450.0],
        "r": [0.05, 0.05],
        "q": [0.015, 0.015],
    })
    out = gex_run(chain, GexConfig())
    assert len(out) == 1
    row = out.iloc[0]
    assert row["gex_calls"] > 0
    assert row["gex_puts"] > 0  # raw gamma is positive; sign is applied in gex_net
    assert row["gex_net"] == pytest.approx(row["gex_calls"] - row["gex_puts"])
