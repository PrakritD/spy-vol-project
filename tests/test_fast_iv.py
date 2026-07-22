"""Validates the vectorized Newton-Raphson IV solver (features/fast_iv.py) against
features.gex's existing scalar Brent solver, on synthetic contracts (data-free CI). This is a
performance rewrite of the SAME equation, not a new model, so the bar is numerical agreement
with the validated scalar version, not a leakage test (no causal/time-series structure here).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.gex import GexConfig, compute_contract_greeks  # noqa: E402
from features.fast_iv import compute_contract_greeks_fast  # noqa: E402


def _synthetic_contracts(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Moneyness/DTE ranges matching this repo's actual filter bands (delta 0.05-0.95,
    DTE 7-60, see configs/features.yaml) -- not the full theoretical option space. Far
    OTM/ITM, near-zero-DTE contracts are numerically miserable for ANY IV solver (near-zero
    time value, tiny vega) and get filtered out downstream by both GEX and skew regardless
    of which solver is used, so a solver-agreement test has no business generating them."""
    rng = np.random.default_rng(seed)
    spot = rng.uniform(300, 700, n)
    moneyness = rng.uniform(0.85, 1.15, n)
    strike = np.round(spot * moneyness, 0)
    dte_days = rng.integers(7, 60, n)
    date = pd.Timestamp("2025-01-02")
    expiry = date + pd.to_timedelta(dte_days, unit="D")
    true_sigma = rng.uniform(0.10, 0.60, n)
    r = np.full(n, 0.04)
    q = np.full(n, 0.015)
    is_call = rng.random(n) < 0.5
    T = dte_days / 365.0

    from features.gex import bs_price
    price = np.array([bs_price(spot[i], strike[i], T[i], r[i], true_sigma[i], q[i], is_call[i])
                      for i in range(n)])

    return pd.DataFrame({
        "date": date, "expiry": expiry, "strike": strike,
        "option_type": np.where(is_call, "C", "P"),
        "price": price, "open_interest": rng.integers(1, 5000, n),
        "spot": spot, "r": r, "q": q,
    })


def test_fast_iv_matches_scalar_on_valid_rows():
    df = _synthetic_contracts()
    scalar = compute_contract_greeks(df, GexConfig())
    fast = compute_contract_greeks_fast(df)

    both_valid = scalar["iv"].notna() & fast["iv"].notna()
    assert both_valid.mean() > 0.95   # near-total agreement on which rows are solvable

    np.testing.assert_allclose(scalar.loc[both_valid, "iv"], fast.loc[both_valid, "iv"], atol=1e-5)
    np.testing.assert_allclose(scalar.loc[both_valid, "delta"], fast.loc[both_valid, "delta"], atol=1e-5)
    np.testing.assert_allclose(scalar.loc[both_valid, "gamma"], fast.loc[both_valid, "gamma"], atol=1e-5)


def test_fast_iv_recovers_true_vol():
    """Prices were generated from a known sigma; the solver should recover it."""
    df = _synthetic_contracts(seed=1)
    fast = compute_contract_greeks_fast(df)
    # re-derive the true sigma the same way _synthetic_contracts did (same seed/order)
    rng = np.random.default_rng(1)
    n = 500
    rng.uniform(300, 700, n)              # spot (unused here, just advancing the stream)
    rng.uniform(0.85, 1.15, n)             # moneyness
    rng.integers(7, 60, n)                 # dte_days
    true_sigma = rng.uniform(0.10, 0.60, n)
    valid = fast["iv"].notna()
    np.testing.assert_allclose(fast.loc[valid, "iv"], true_sigma[valid.to_numpy()], atol=1e-4)
