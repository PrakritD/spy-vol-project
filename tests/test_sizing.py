"""Sizing-rule invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.sizing import (
    SizingSpec,
    estimate_kelly_b,
    kelly_sizing,
    linear_sizing,
    make_kelly_fn,
)


def test_linear_sizing_clipped_to_unit_interval():
    p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    s = linear_sizing(p)
    assert (s >= 0).all() and (s <= 1).all()
    assert s[0] == 0.0
    assert s[2] == 0.0          # threshold at 0.5 by construction
    assert s[-1] == pytest.approx(1.0)


def test_linear_sizing_monotone():
    p = np.linspace(0, 1, 100)
    s = linear_sizing(p)
    assert (np.diff(s) >= -1e-12).all()


def test_kelly_sizing_zero_when_b_zero_or_negative():
    p = np.array([0.5, 0.7, 0.9])
    assert (kelly_sizing(p, b=0.0) == 0).all()
    assert (kelly_sizing(p, b=-1.0) == 0).all()


def test_kelly_sizing_higher_p_higher_size():
    # b = 1 (even-money), half-Kelly → size = 0.5 * (2p - 1)
    p = np.array([0.4, 0.5, 0.6, 0.7])
    s = kelly_sizing(p, b=1.0, frac=0.5)
    assert s[0] == 0.0       # negative Kelly clipped
    assert s[1] == 0.0
    assert s[2] == pytest.approx(0.5 * (2 * 0.6 - 1))
    assert s[3] == pytest.approx(0.5 * (2 * 0.7 - 1))


def test_estimate_kelly_b_reasonable_on_synthetic():
    rng = np.random.default_rng(42)
    # Wins ~ +2bps, losses ~ -1bp. Expected b ~ 2.0
    n = 500
    wins = rng.uniform(0.0001, 0.0005, n // 2)
    losses = -rng.uniform(0.00005, 0.00025, n // 2)
    r = pd.Series(np.concatenate([wins, losses]))
    b = estimate_kelly_b(r)
    assert 1.0 < b < 4.0


def test_estimate_kelly_b_returns_nan_when_insufficient():
    assert np.isnan(estimate_kelly_b(pd.Series([0.01, -0.01]), min_obs=30))


def test_sizing_spec_apply_preserves_index():
    p = pd.Series([0.4, 0.6, 0.8], index=pd.bdate_range("2024-01-02", periods=3))
    spec = SizingSpec(name="linear", fn=linear_sizing)
    out = spec.apply(p)
    assert (out.index == p.index).all()
    assert out.name == "size_linear"


def test_make_kelly_fn_binds_params():
    fn = make_kelly_fn(b=2.0, frac=0.25)
    p = np.array([0.6, 0.8])
    direct = kelly_sizing(p, b=2.0, frac=0.25)
    assert np.allclose(fn(p), direct)
