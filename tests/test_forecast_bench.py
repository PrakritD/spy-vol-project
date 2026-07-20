"""forecast_bench.py leakage tests: perturbing the raw panel strictly in the FUTURE tail
must not change any walk-forward prediction in the safe early region, for every model
family the benchmark reports (hard rule 1: every model gets this test before its
results are read). Synthetic panel, so this runs data-free like the rest of the suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import forecast_bench as F  # noqa: E402
import strategy_two_sleeve as S  # noqa: E402


def _synthetic_panel(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Same generator as test_strategy.py's `_synthetic_panel` (kept self-contained per
    this repo's per-test-file convention rather than cross-importing test modules)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=n)
    spy = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    vix = np.clip(15 + np.cumsum(rng.normal(0, 0.5, n)) % 25 + rng.normal(0, 2, n), 9, 80)
    vix3m = vix * rng.uniform(0.92, 1.08, n)
    vix9d = vix * rng.uniform(0.9, 1.1, n)
    vvix = 80 + rng.normal(0, 10, n).cumsum() % 40 + 60
    vixy = 100 * np.exp(np.cumsum(rng.normal(-0.0025, 0.03, n)))
    gex = rng.normal(0.5, 1.0, n)
    return pd.DataFrame({
        "date": dates,
        "open": spy * rng.uniform(0.99, 1.01, n), "high": spy * rng.uniform(1.0, 1.02, n),
        "low": spy * rng.uniform(0.98, 1.0, n), "close": spy, "spy_close": spy,
        "spy_adj": spy, "spy_vol": rng.uniform(5e7, 1.5e8, n),
        "rv": np.abs(rng.normal(0.15, 0.05, n)) + 0.05,
        "vixy_adj": vixy, "vixy_vol": rng.uniform(1e6, 5e6, n),
        "gex": gex, "dix": rng.uniform(0.38, 0.46, n),
        "vix": vix, "vix3m": vix3m, "vix9d": vix9d, "vvix": vvix,
        "dgs3mo": np.full(n, 2.0),
    })


def _perturb_tail(raw: pd.DataFrame, n: int, k: int = 100, seed: int = 99) -> pd.DataFrame:
    perturbed = raw.copy()
    fut = perturbed.index >= (n - k)
    rng = np.random.default_rng(seed)
    for col in ["vix", "vix3m", "vix9d", "vvix", "gex", "dix", "vixy_adj", "spy_adj", "spy_vol", "rv"]:
        perturbed.loc[fut, col] = perturbed.loc[fut, col] * rng.uniform(0.5, 1.5, fut.sum())
    return perturbed


def _panels(n=800, seed=0):
    raw = _synthetic_panel(n=n, seed=seed)
    d0 = F.add_forecast_columns(S.build_signals(raw))
    d1 = F.add_forecast_columns(S.build_signals(_perturb_tail(raw, n)))
    return d0, d1


def test_no_lookahead_wf_ols():
    """HAR and HAR+VIX share this function; HAR+VIX is the superset feature set."""
    n = 800
    d0, d1 = _panels(n=n)
    pred0 = F.wf_ols(d0[F.VIX_FEATS].to_numpy(), d0["target"].to_numpy())
    pred1 = F.wf_ols(d1[F.VIX_FEATS].to_numpy(), d1["target"].to_numpy())
    c = n - 200
    assert c > 100
    np.testing.assert_array_equal(pred0[:c], pred1[:c])


def test_no_lookahead_wf_mlp():
    n = 800
    d0, d1 = _panels(n=n)
    pred0, _ = F.wf_mlp(d0[F.VIX_FEATS].to_numpy(), d0["target"].to_numpy())
    pred1, _ = F.wf_mlp(d1[F.VIX_FEATS].to_numpy(), d1["target"].to_numpy())
    c = n - 200
    assert c > 100
    np.testing.assert_allclose(pred0[:c], pred1[:c], rtol=1e-10, atol=1e-12)


def test_no_lookahead_wf_qgb():
    n = 800
    d0, d1 = _panels(n=n)
    # small n_estimators just to keep this leakage check fast; the property under test
    # (no future leakage) does not depend on model size.
    crps0, med0 = F.wf_qgb(d0[F.VIX_FEATS].to_numpy(), d0["target"].to_numpy(), n_estimators=15)
    crps1, med1 = F.wf_qgb(d1[F.VIX_FEATS].to_numpy(), d1["target"].to_numpy(), n_estimators=15)
    c = n - 200
    assert c > 100
    np.testing.assert_allclose(med0[:c], med1[:c], rtol=1e-10, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(crps0[:c], crps1[:c], rtol=1e-10, atol=1e-12, equal_nan=True)
