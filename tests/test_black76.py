"""Validation for the Black-76 pricing primitive: put-call parity and known boundary limits.
No look-ahead gate applies here (it is a closed-form pricing formula, not a fitted/causal
model over a time series), so this is a correctness suite, not a leakage test."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import black76 as B  # noqa: E402


def test_put_call_parity():
    f, k, t, r = 20.0, 22.0, 30 / 365, 0.03
    for sigma in (0.5, 0.8, 1.2, 2.0):     # VIX-option-like vol-of-vol levels
        c = B.call_price(f, k, t, r, sigma)
        p = B.put_price(f, k, t, r, sigma)
        parity_rhs = np.exp(-r * t) * (f - k)
        np.testing.assert_allclose(c - p, parity_rhs, atol=1e-10)


def test_call_to_intrinsic_as_vol_to_zero():
    f, k, t, r = 20.0, 15.0, 30 / 365, 0.03
    c = B.call_price(f, k, t, r, sigma=1e-6)
    intrinsic = np.exp(-r * t) * max(f - k, 0.0)
    np.testing.assert_allclose(c, intrinsic, atol=1e-6)


def test_put_to_intrinsic_as_vol_to_zero():
    f, k, t, r = 15.0, 20.0, 30 / 365, 0.03
    p = B.put_price(f, k, t, r, sigma=1e-6)
    intrinsic = np.exp(-r * t) * max(k - f, 0.0)
    np.testing.assert_allclose(p, intrinsic, atol=1e-6)


def test_deep_otm_call_near_zero_deep_itm_call_near_intrinsic():
    t, r, sigma = 30 / 365, 0.03, 0.9
    otm = B.call_price(f=15.0, k=60.0, t=t, r=r, sigma=sigma)
    assert otm < 0.05
    itm = B.call_price(f=60.0, k=15.0, t=t, r=r, sigma=sigma)
    intrinsic = np.exp(-r * t) * (60.0 - 15.0)
    assert abs(itm - intrinsic) < 0.5           # small time value only, deep ITM


def test_call_delta_bounded_0_1_and_put_delta_bounded_neg1_0():
    f, k, t, r, sigma = 20.0, 20.0, 30 / 365, 0.03, 0.9
    cd = B.call_delta(f, k, t, r, sigma)
    pd_ = B.put_delta(f, k, t, r, sigma)
    assert 0.0 < cd < 1.0
    assert -1.0 < pd_ < 0.0
    # call_delta - put_delta = exp(-r*T) exactly (both scaled by the same discount factor)
    np.testing.assert_allclose(cd - pd_, np.exp(-r * t), atol=1e-10)


def test_vega_and_gamma_positive():
    f, k, t, r, sigma = 20.0, 22.0, 30 / 365, 0.03, 0.9
    assert B.vega(f, k, t, r, sigma) > 0
    assert B.gamma(f, k, t, r, sigma) > 0


def test_vectorized_inputs():
    f = np.array([15.0, 20.0, 25.0])
    k = np.full(3, 20.0)
    t = np.full(3, 30 / 365)
    r = np.full(3, 0.03)
    sigma = np.full(3, 0.9)
    c = B.call_price(f, k, t, r, sigma)
    assert c.shape == (3,)
    assert (np.diff(c) > 0).all()    # call price increasing in forward
