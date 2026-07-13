"""Strategy tests: data-free (synthetic panels), so CI runs without the gitignored data.

The headline guarantee of the whole project is no look-ahead: a position formed for day t may
use only information available at the close of t-1. `test_no_lookahead_end_to_end` enforces it
by perturbing a raw input strictly in the FUTURE and asserting every earlier position and
cumulative P&L is byte-identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import strategy_two_sleeve as S  # noqa: E402


def _synthetic_panel(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """A raw daily panel with the columns load_panel() produces, from a seeded RNG."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=n)
    spy = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    vix = np.clip(15 + np.cumsum(rng.normal(0, 0.5, n)) % 25 + rng.normal(0, 2, n), 9, 80)
    vix3m = vix * rng.uniform(0.92, 1.08, n)          # crosses contango/backwardation both ways
    vix9d = vix * rng.uniform(0.9, 1.1, n)
    vvix = 80 + rng.normal(0, 10, n).cumsum() % 40 + 60
    vixy = 100 * np.exp(np.cumsum(rng.normal(-0.0025, 0.03, n)))   # decaying like VIXY
    gex = rng.normal(0.5, 1.0, n)                      # ~9% negative, like SqueezeMetrics gex
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


def _run_carry(panel: pd.DataFrame):
    """build_signals -> contango-filtered carry positions -> excess P&L, on a given panel."""
    d = S.build_signals(panel)
    need = ["vixy_ret", "spy_ret", "vixy_vol21", "t_30_90", "t_9_30", "vix_z", "vvix_z",
            "gex_neg", "amihud_z", "rf_d"]
    d = d.dropna(subset=need).reset_index(drop=True)
    pos = S.carry_positions(d)
    pnl = S.sleeve_excess(pos, d["vixy_ret"].to_numpy(), d["rf_d"].to_numpy(),
                          S.CostCfg().vixy_bps, S.CostCfg().borrow_ann)
    return d, pos, pnl


def test_no_lookahead_end_to_end():
    """Perturb every raw input only in the far-future TAIL; positions & cumulative P&L in the
    safe early region must be byte-identical. (A position at t uses only info <= t-1, so future
    data cannot leak back.) We perturb the last 100 raw rows and compare all but the last 200
    kept rows, leaving a margin well beyond any backward rolling-window warmup offset."""
    n = 800
    panel = _synthetic_panel(n=n)
    d0, pos0, pnl0 = _run_carry(panel)

    perturbed = panel.copy()
    fut = perturbed.index >= (n - 100)                 # perturb only the far-future tail
    rng = np.random.default_rng(99)
    for col in ["vix", "vix3m", "vix9d", "vvix", "gex", "dix", "vixy_adj", "spy_adj", "spy_vol", "rv"]:
        perturbed.loc[fut, col] = perturbed.loc[fut, col] * rng.uniform(0.5, 1.5, fut.sum())
    d1, pos1, pnl1 = _run_carry(perturbed)

    c = min(len(pos0), len(pos1)) - 200                # compare only the safe early region
    assert c > 100
    np.testing.assert_array_equal(pos0[:c], pos1[:c])                      # positions unchanged
    np.testing.assert_allclose(np.nancumsum(pnl0[:c]), np.nancumsum(pnl1[:c]), atol=1e-12)


def test_contango_flag_uses_lagged_data():
    """contango_flag at row t must reflect VIX/VIX3M at t-1, not t."""
    d = S.build_signals(_synthetic_panel())
    flag = S.contango_flag(d)
    # t_30_90 is (vix/vix3m).shift(1); the flag is (t_30_90 < 1) -> equals the t-1 raw contango
    raw_contango_prev = (d["vix"] / d["vix3m"] < 1.0).shift(1)
    ok = ~raw_contango_prev.isna()
    np.testing.assert_array_equal(flag[ok.to_numpy()], raw_contango_prev[ok].astype(float).to_numpy())


def test_costs_only_reduce_returns():
    """Adding spread + borrow can only lower the excess return stream, never raise it."""
    d, pos, _ = _run_carry(_synthetic_panel())
    vret, rf = d["vixy_ret"].to_numpy(), d["rf_d"].to_numpy()
    free = S.sleeve_excess(pos, vret, rf, 0.0, 0.0)
    costed = S.sleeve_excess(pos, vret, rf, 20.0, 0.05)
    assert np.nansum(costed) <= np.nansum(free) + 1e-12


def test_metrics_sanity():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, 2000)                      # positive-drift series
    m = S.metrics(r, None, "synthetic")
    assert m["sharpe"] > 0 and m["maxdd"] <= 0 and np.isfinite(m["calmar"])
    assert S.metrics(np.zeros(100), None, "flat")["sharpe"] != S.metrics(np.zeros(100), None, "flat")["sharpe"] \
        or not np.isfinite(S.metrics(np.zeros(100), None, "flat").get("sharpe", np.nan))  # degenerate handled


def test_deflated_sharpe_is_a_probability():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0004, 0.012, 1500)
    out = S.deflated_sharpe(r, n_trials=10, sr_trials_std=0.02)
    assert 0.0 <= out["dsr"] <= 1.0 and 0.0 <= out["psr_vs0"] <= 1.0


def test_timing_sleeve_is_lagged_and_flat_in_warmup():
    d = S.build_signals(_synthetic_panel(n=900))
    need = ["spy_ret", "vixy_ret", "vixy_vol21", "t_30_90", "t_9_30", "vix_z", "vvix_z",
            "gex_neg", "amihud_z", "rf_d"]
    d = d.dropna(subset=need).reset_index(drop=True)
    pos, p = S.timing_positions(d)
    assert np.all(pos[:S.TRAIN0] == 0.0)                    # no position before the initial train window


_NEED = ["vixy_ret", "spy_ret", "vixy_vol21", "t_30_90", "t_9_30", "vix_z", "vvix_z",
         "gex_neg", "amihud_z", "rf_d"]


def test_timing_sleeve_is_causal():
    """The walk-forward logistic timing sleeve shares the ML machinery, so it gets the same
    guarantee: perturb raw inputs only in the far-future tail, earlier positions are identical."""
    n = 1100
    panel = _synthetic_panel(n=n)
    d0 = S.build_signals(panel).dropna(subset=_NEED).reset_index(drop=True)
    pos0, _ = S.timing_positions(d0)

    perturbed = panel.copy()
    fut = perturbed.index >= (n - 100)
    rng = np.random.default_rng(321)
    for col in ["vix", "vix3m", "vix9d", "vvix", "gex", "dix", "vixy_adj", "spy_adj", "spy_vol", "rv"]:
        perturbed.loc[fut, col] = perturbed.loc[fut, col] * rng.uniform(0.5, 1.5, fut.sum())
    d1 = S.build_signals(perturbed).dropna(subset=_NEED).reset_index(drop=True)
    pos1, _ = S.timing_positions(d1)

    c = min(len(pos0), len(pos1)) - 200
    assert c > S.TRAIN0
    np.testing.assert_array_equal(pos0[:c], pos1[:c])


def test_metrics_golden_values():
    """Pin the metric formulas to closed-form values on a deterministic series: alternating
    +1.1%/-0.9% has mean 0.001, population sd exactly 0.01, and a -0.9% max drawdown."""
    r = np.tile([0.011, -0.009], 500)
    m = S.metrics(r, None, "golden")
    np.testing.assert_allclose(m["sharpe"], 0.001 / 0.01 * np.sqrt(S.ANN), rtol=1e-9)
    np.testing.assert_allclose(m["maxdd"], -0.009, rtol=1e-9)
    np.testing.assert_allclose(m["cagr"], (1.011 * 0.991) ** (S.ANN / 2) - 1, rtol=1e-9)
    np.testing.assert_allclose(m["calmar"], m["cagr"] / 0.009, rtol=1e-9)
    np.testing.assert_allclose(m["ann_vol"], 0.01 * np.sqrt(S.ANN), rtol=1e-9)


def test_deflated_sharpe_golden_values():
    """With no selection haircut (sr_trials_std=0) DSR must equal PSR-vs-0, and on the symmetric
    two-point series PSR has a closed form; more trials must always mean a lower DSR."""
    from scipy.stats import norm
    r = np.tile([0.011, -0.009], 500)
    ds0 = S.deflated_sharpe(r, n_trials=2, sr_trials_std=0.0)
    assert ds0["dsr"] == ds0["psr_vs0"]
    np.testing.assert_allclose(ds0["psr_vs0"], norm.cdf(0.1 * np.sqrt(999)), rtol=1e-3)
    ds_few = S.deflated_sharpe(r, n_trials=5, sr_trials_std=0.02)
    ds_many = S.deflated_sharpe(r, n_trials=100, sr_trials_std=0.02)
    assert ds_many["dsr"] < ds_few["dsr"] < ds0["dsr"]


def test_hac_tstat_tracks_classic_t_when_iid():
    """On an iid series the Newey-West t must sit near the classic t (no autocorrelation to
    correct), and it must never be negative for a positive-mean series."""
    rng = np.random.default_rng(5)
    x = rng.normal(0.0005, 0.01, 3000)
    t_hac = S.hac_tstat(x)
    t_classic = x.mean() / (x.std() / np.sqrt(len(x)))
    assert 0.8 < t_hac / t_classic < 1.2
    assert t_hac > 0


def test_pinned_headline_on_synthetic_panel():
    """Regression pin: the carry pipeline on the seeded synthetic panel must keep producing the
    same numbers. Catches silent drift in signals, costs, or metric math from any refactor."""
    d, pos, pnl = _run_carry(_synthetic_panel(n=800))
    m = S.metrics(pnl, None, "pin")
    assert m["n"] == 739
    assert int((pos != 0).sum()) == 359
    np.testing.assert_allclose(m["sharpe"], 0.7927492070, atol=1e-8)
    np.testing.assert_allclose(m["maxdd"], -0.0800484157, atol=1e-8)


def test_ml_sizing_is_causal_and_lagged():
    """The walk-forward Ridge sizing layer must not leak the future: perturb raw inputs only in
    the far-future tail and assert earlier ML positions are byte-identical. Causal by
    construction (expanding walk-forward + train-only scaling + an expanding exposure scale)."""
    n = 1100
    panel = _synthetic_panel(n=n)
    d0 = S.build_signals(panel).dropna(subset=_NEED).reset_index(drop=True)
    pos0, _ = S.ml_size_positions(d0)

    perturbed = panel.copy()
    fut = perturbed.index >= (n - 100)
    rng = np.random.default_rng(123)
    for col in ["vix", "vix3m", "vix9d", "vvix", "gex", "dix", "vixy_adj", "spy_adj", "spy_vol", "rv"]:
        perturbed.loc[fut, col] = perturbed.loc[fut, col] * rng.uniform(0.5, 1.5, fut.sum())
    d1 = S.build_signals(perturbed).dropna(subset=_NEED).reset_index(drop=True)
    pos1, _ = S.ml_size_positions(d1)

    c = min(len(pos0), len(pos1)) - 200
    assert c > S.TRAIN0                                     # compared region includes live ML positions
    np.testing.assert_array_equal(pos0[:c], pos1[:c])       # earlier positions unchanged by future data
    assert np.any(pos0[:c] != 0.0)                          # and the layer is actually exercised, not all-flat
