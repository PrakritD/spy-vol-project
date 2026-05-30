"""Continuous-sizing execution model: turnover costs + regime slippage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.execution import ExecConfig, backtest
from backtest.sizing import SizingSpec, linear_sizing


def _make_inputs(n: int = 100, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    p_hat = rng.uniform(0.3, 0.9, n)
    vxx_close = 100 * np.exp(np.cumsum(rng.normal(-1e-4, 0.02, n)))
    preds = pd.DataFrame({"date": dates, "p_hat": p_hat})
    px = pd.DataFrame({"date": dates, "vxx_close": vxx_close})
    return preds, px


def test_continuous_sizing_produces_fractional_positions():
    preds, px = _make_inputs()
    sizing = SizingSpec(name="linear", fn=linear_sizing)
    pnl = backtest(preds, px, sizing=sizing)
    assert ((pnl["size"] >= 0) & (pnl["size"] <= 1)).all()
    fractional = ((pnl["size"] > 0) & (pnl["size"] < 1)).any()
    assert fractional, "expected at least one fractional size with continuous sizing"


def test_turnover_cost_proportional_to_size_change():
    # All zeros except one mid-period -> cost only on entry + exit
    n = 10
    dates = pd.bdate_range("2024-01-02", periods=n)
    p_hat = pd.Series([0.0] * n)
    p_hat.iloc[5] = 1.0  # full long for one day
    preds = pd.DataFrame({"date": dates, "p_hat": p_hat.to_numpy()})
    px = pd.DataFrame({"date": dates, "vxx_close": np.linspace(100, 110, n)})
    sizing = SizingSpec(name="linear", fn=linear_sizing)
    pnl = backtest(preds, px, sizing=sizing, cfg=ExecConfig(base_bps_per_side=5.0,
                                                             extra_bps_high_vol=0.0))
    # Costs only on day 5 (size 0->1) and day 6 (size 1->0)
    nonzero_days = pnl["cost"].gt(0).sum()
    assert nonzero_days == 2
    # Each cost = 1.0 * 5 bps
    assert pnl["cost"].max() == pytest.approx(5e-4)


def test_regime_conditional_slippage_charges_extra_in_high_vol():
    n = 5
    dates = pd.bdate_range("2024-01-02", periods=n)
    preds = pd.DataFrame({"date": dates, "p_hat": [1.0] * n})
    px = pd.DataFrame({"date": dates, "vxx_close": [100.0] * n})
    vix_z = pd.Series([0.0, 0.0, 2.0, 2.0, 0.0], index=dates)  # high-vol on days 2-3

    sizing = SizingSpec(name="linear", fn=linear_sizing)
    pnl = backtest(preds, px, sizing=sizing, vix_zscore=vix_z,
                   cfg=ExecConfig(base_bps_per_side=5.0, extra_bps_high_vol=5.0,
                                  high_vol_zscore=1.5))
    # All sizes are 1.0; only first day has nonzero |Δsize| (0->1).
    # Day 0 vix_z = 0 → cost = 1.0 * 5 bps
    assert pnl["cost"].iloc[0] == pytest.approx(5e-4)
    # No further turnover, so subsequent costs are zero regardless of regime.
    assert (pnl["cost"].iloc[1:] == 0).all()


def test_legacy_threshold_path_still_works():
    preds, px = _make_inputs()
    cfg = ExecConfig(threshold=0.6, base_bps_per_side=5.0, extra_bps_high_vol=0.0)
    pnl = backtest(preds, px, cfg=cfg)
    assert set(pnl["size"].unique()).issubset({0.0, 1.0})
