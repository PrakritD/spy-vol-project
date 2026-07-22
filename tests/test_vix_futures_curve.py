"""No-lookahead gate for the constant-maturity VIX futures construction.

Perturbs raw settle prices strictly in the future tail of a synthetic multi-contract panel
and asserts every earlier roll weight and index return is unchanged (mirrors
test_strategy.py::test_no_lookahead_end_to_end for the new front/second-contract, days-to-
expiry-weighted construction in analysis/vix_futures_curve.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import vix_futures_curve as V  # noqa: E402


def _synthetic_contracts(n_contracts: int = 6, life_days: int = 90, spacing_days: int = 30,
                         seed: int = 0) -> pd.DataFrame:
    """Staggered contracts (like real VX months): each trades for `life_days` before its own
    expiry, with expiries `spacing_days` apart, so several contracts overlap at any date."""
    rng = np.random.default_rng(seed)
    start0 = pd.Timestamp("2020-01-06")
    frames = []
    for i in range(n_contracts):
        expiry = start0 + pd.Timedelta(days=spacing_days * (i + 1))
        trade_dates = pd.bdate_range(expiry - pd.Timedelta(days=life_days), expiry)
        n = len(trade_dates)
        settle = np.clip(15 + np.cumsum(rng.normal(0, 0.3, n)), 8, 40)
        frames.append(pd.DataFrame({
            "trade_date": trade_dates, "contract_code": f"C{i}",
            "expiry_date": expiry, "settle": settle,
        }))
    return pd.concat(frames, ignore_index=True)


def test_build_curve_no_lookahead():
    panel = _synthetic_contracts()
    curve0 = V.build_curve(panel)

    cutoff = panel["trade_date"].max() - pd.Timedelta(days=20)
    perturbed = panel.copy()
    fut = perturbed["trade_date"] > cutoff
    rng = np.random.default_rng(7)
    perturbed.loc[fut, "settle"] = perturbed.loc[fut, "settle"] * rng.uniform(0.5, 1.5, fut.sum())
    curve1 = V.build_curve(perturbed)

    safe_end = cutoff - pd.Timedelta(days=5)   # margin beyond the 1-day shift(1) dependence
    c0 = curve0[curve0["date"] <= safe_end].reset_index(drop=True)
    c1 = curve1[curve1["date"] <= safe_end].reset_index(drop=True)
    assert len(c0) > 20 and len(c0) == len(c1)

    np.testing.assert_array_equal(c0["dte1"].to_numpy(), c1["dte1"].to_numpy())
    np.testing.assert_array_equal(c0["dte2"].to_numpy(), c1["dte2"].to_numpy())
    np.testing.assert_allclose(c0["w1"].to_numpy(), c1["w1"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(c0["index_ret"].to_numpy(), c1["index_ret"].to_numpy(),
                               atol=1e-12, equal_nan=True)


def test_weights_sum_to_one_and_bounded():
    """w1+w2 == 1 always; both in [0, 1] by the clip in build_curve."""
    curve = V.build_curve(_synthetic_contracts())
    np.testing.assert_allclose(curve["w1"] + curve["w2"], 1.0, atol=1e-12)
    assert (curve["w1"] >= 0).all() and (curve["w1"] <= 1).all()


def test_front_is_nearer_expiry_than_second():
    curve = V.build_curve(_synthetic_contracts())
    assert (curve["dte1"] <= curve["dte2"]).all()
