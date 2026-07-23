"""Data-free unit tests for analysis/risk_tearsheet.py.

Covers the Cornish-Fisher expansion against its known closed-form reduction (skew=0,
kurtosis=0 collapses to the plain Gaussian z-score) and the VaR/ES ordering invariant
(ES is always at least as extreme a loss as VaR at the same confidence level), both on
synthetic data and on the repo's own committed daily-return series when available.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import risk_tearsheet as RT  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
EQUITY_PATH = REPO_ROOT / "analysis" / "strategy_equity.parquet"


def test_cornish_fisher_reduces_to_gaussian_at_zero_skew_kurtosis():
    z = RT.Z_99
    z_cf = RT.cornish_fisher_z(z, skew=0.0, kurt=0.0)
    np.testing.assert_allclose(z_cf, z, atol=1e-12)


def test_cornish_fisher_var_es_matches_gaussian_var_when_skew_kurt_zero():
    rng = np.random.default_rng(0)
    # A large symmetric mesokurtic-by-construction sample: use the Gaussian quantile
    # formula directly rather than relying on a finite sample's skew/kurt hitting
    # exactly zero (it won't). This isolates the z-score formula, not sampling noise.
    r = rng.normal(loc=0.0002, scale=0.01, size=200_000)
    skew, kurt = RT.sample_skew_kurt(r)
    assert abs(skew) < 0.02
    assert abs(kurt) < 0.05
    var_cf, _ = RT.cornish_fisher_var_es(r, skew=0.0, kurt=0.0)
    var_gaussian = -(r.mean() - RT.Z_99 * r.std(ddof=0))
    np.testing.assert_allclose(var_cf, var_gaussian, rtol=1e-9)


def test_es_at_least_as_extreme_as_var_synthetic():
    rng = np.random.default_rng(1)
    # A fat-left-tailed synthetic series (mixture of normals), the shape this
    # strategy's own returns exhibit (short-vol premium, occasional large losses).
    n = 5000
    base = rng.normal(0.0005, 0.006, n)
    shock_idx = rng.random(n) < 0.02
    base[shock_idx] -= rng.exponential(0.05, shock_idx.sum())
    skew, kurt = RT.sample_skew_kurt(base)

    var_h, es_h = RT.historical_var_es(base)
    var_cf, es_cf = RT.cornish_fisher_var_es(base, skew, kurt)

    assert es_h >= var_h - 1e-12
    assert es_cf >= var_cf - 1e-12


def test_es_at_least_as_extreme_as_var_on_committed_equity_curve():
    if not EQUITY_PATH.exists():
        return  # data artifact not present in this checkout; skip rather than fail
    import pandas as pd

    eq = pd.read_parquet(EQUITY_PATH).sort_values("date").reset_index(drop=True)
    e = eq["carry"].to_numpy(float)
    r = e / np.concatenate(([1.0], e[:-1])) - 1.0

    skew, kurt = RT.sample_skew_kurt(r)
    var_h, es_h = RT.historical_var_es(r)
    var_cf, es_cf = RT.cornish_fisher_var_es(r, skew, kurt)

    assert es_h >= var_h - 1e-12
    assert es_cf >= var_cf - 1e-12
    # Sanity: this strategy has a documented left tail, so both loss estimates
    # should be strictly positive numbers (a real potential daily loss).
    assert var_h > 0
    assert var_cf > 0


def test_rolling_beta_matches_full_sample_ols_slope_on_synthetic_data():
    rng = np.random.default_rng(2)
    n = 400
    mkt = rng.normal(0, 0.01, n)
    true_beta = 0.6
    strat = true_beta * mkt + rng.normal(0, 0.002, n)

    beta = RT.rolling_beta(strat, mkt, window=n)
    # Closed-form cov/var ratio should recover the generating beta closely at n=400.
    np.testing.assert_allclose(beta[-1], true_beta, atol=0.05)
