"""Trader-flavoured metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    annualised_vol,
    avg_win_loss,
    block_bootstrap_sharpe_ci,
    cagr,
    calmar,
    conditional_var,
    downside_deviation,
    hit_rate,
    information_ratio,
    max_drawdown,
    max_drawdown_duration,
    monthly_returns,
    probabilistic_sharpe_ratio,
    profit_factor,
    sharpe,
    skew_kurt,
    sortino,
    time_in_market,
    trader_summary,
    turnover,
    value_at_risk,
)


def _equity_from_returns(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0)).cumprod()


def test_sharpe_zero_when_constant_returns():
    r = pd.Series([0.001] * 50)
    assert np.isnan(sharpe(r))


def test_sharpe_positive_for_positive_drift():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 252))
    assert sharpe(r) > 0


def test_sortino_higher_than_sharpe_when_downside_skewed():
    rng = np.random.default_rng(1)
    base = rng.normal(0.0008, 0.005, 500)
    # Inject extra small upside spikes — increases mean without raising downside std
    base[::20] += 0.005
    r = pd.Series(base)
    assert sortino(r) > sharpe(r) > 0


def test_max_drawdown_bounds():
    e = pd.Series([1.0, 1.1, 1.05, 0.9, 1.0])
    assert max_drawdown(e) == pytest.approx(0.9 / 1.1 - 1)


def test_max_drawdown_duration_counts_bars_below_peak():
    # Equity 1.0, 1.1 are peaks; 1.0, 0.95 are off-peak (2 bars); 1.2 makes new peak.
    e = pd.Series([1.0, 1.1, 1.0, 0.95, 1.2])
    assert max_drawdown_duration(e) == 2


def test_calmar_finite_with_drift():
    r = pd.Series(np.random.default_rng(2).normal(0.0006, 0.01, 500))
    e = _equity_from_returns(r)
    c = calmar(r, e)
    assert np.isfinite(c)


def test_hit_rate_excludes_zeros():
    r = pd.Series([0.01, 0, -0.005, 0, 0.02])
    assert hit_rate(r) == pytest.approx(2 / 3)


def test_avg_win_loss_signs():
    r = pd.Series([0.02, -0.01, 0.03, -0.005])
    avg_w, avg_l = avg_win_loss(r)
    assert avg_w == pytest.approx(0.025)
    assert avg_l == pytest.approx(-0.0075)


def test_profit_factor_inf_when_no_losses():
    r = pd.Series([0.01, 0.02, 0.0])
    assert profit_factor(r) == float("inf")


def test_time_in_market_fraction():
    s = pd.Series([0.0, 0.5, 1.0, 0.0])
    assert time_in_market(s) == pytest.approx(0.5)


def test_turnover_average_abs_change():
    s = pd.Series([0.0, 0.5, 0.5, 0.0, 1.0])
    expected = np.mean([0.5, 0.0, 0.5, 1.0])
    assert turnover(s) == pytest.approx(expected)


def test_monthly_returns_shape_and_values():
    dates = pd.date_range("2024-01-01", "2024-03-31", freq="B")
    pnl_df = pd.DataFrame({"date": dates, "net_pnl": [0.001] * len(dates)})
    table = monthly_returns(pnl_df)
    assert set(table.index) == {2024}
    assert set(table.columns) == set(range(1, 13))
    # Jan, Feb, Mar should be > 0; Apr+ should be NaN
    assert (table.loc[2024, [1, 2, 3]] > 0).all()
    assert table.loc[2024, [4, 5]].isna().all()


def test_cagr_matches_total_return_over_1_year():
    # Exactly 252 trading days of +0.001/day -> CAGR ~ (1.001)^252 - 1
    r = pd.Series([0.001] * 252)
    e = _equity_from_returns(r)
    expected = (1.001) ** 252 - 1
    assert cagr(e) == pytest.approx(expected, rel=1e-6)


def test_annualised_vol_scales_with_sqrt_252():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 252))
    assert annualised_vol(r) == pytest.approx(r.std() * np.sqrt(252))


def test_downside_deviation_excludes_upside():
    # All-positive returns -> downside deviation is 0
    r = pd.Series([0.01, 0.02, 0.005])
    assert downside_deviation(r) == pytest.approx(0.0)


def test_value_at_risk_returns_positive_loss_magnitude():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.02, 1000))
    var = value_at_risk(r, alpha=0.05)
    # VaR should be positive (it's a loss magnitude) and roughly ~1.65 sigma
    assert var > 0
    assert var == pytest.approx(0.02 * 1.645, rel=0.2)


def test_cvar_worse_than_var():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.02, 1000))
    var = value_at_risk(r, alpha=0.05)
    cvar = conditional_var(r, alpha=0.05)
    assert cvar >= var, "CVaR (expected shortfall) must be >= VaR by definition"


def test_skew_kurt_gaussian_close_to_zero():
    r = pd.Series(np.random.default_rng(7).normal(0, 1, 5000))
    sk, ku = skew_kurt(r)
    assert abs(sk) < 0.1
    assert abs(ku) < 0.2


def test_psr_high_for_clearly_positive_strategy():
    # Strong daily edge -> PSR -> 1
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.005, 500))
    assert probabilistic_sharpe_ratio(r) > 0.99


def test_psr_low_for_clearly_negative_strategy():
    # Persistent negative drift -> PSR -> 0 (strategy worse than zero SR)
    r = pd.Series(np.random.default_rng(1).normal(-0.001, 0.005, 500))
    assert probabilistic_sharpe_ratio(r) < 0.05


def test_psr_is_a_valid_probability():
    # Whatever the data, PSR must lie in [0, 1].
    rng = np.random.default_rng(2)
    for drift in [-0.001, 0.0, 0.0005]:
        r = pd.Series(rng.normal(drift, 0.01, 300))
        p = probabilistic_sharpe_ratio(r)
        assert 0.0 <= p <= 1.0, f"PSR {p} out of [0,1] for drift={drift}"


def test_block_bootstrap_ci_brackets_point_sharpe():
    rng = np.random.default_rng(8)
    r = pd.Series(rng.normal(0.0008, 0.01, 500))
    point = sharpe(r)
    lo, hi = block_bootstrap_sharpe_ci(r, n_boot=300, seed=8)
    # Point estimate should lie inside the 95% CI for an unbiased estimator.
    assert lo - 1e-6 <= point <= hi + 1e-6
    assert hi > lo


def test_information_ratio_zero_when_strategy_equals_benchmark():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 252))
    assert information_ratio(r, r.copy()) != information_ratio(r, r.copy())  # nan
    # Add tiny offset so excess returns aren't degenerate
    bench = r - 1e-4
    ir = information_ratio(r, bench)
    assert np.isfinite(ir)


def test_trader_summary_keys_present():
    rng = np.random.default_rng(5)
    n = 252
    pnl_df = pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=n),
        "size": rng.uniform(0, 1, n),
        "gross_pnl": rng.normal(0.0005, 0.01, n),
        "net_pnl": rng.normal(0.0004, 0.01, n),
        "cost": rng.uniform(0, 1e-4, n),
    })
    pnl_df["equity"] = _equity_from_returns(pnl_df["net_pnl"])
    s = trader_summary(pnl_df)
    expected = {
        "sharpe_net", "sharpe_gross", "sortino", "calmar",
        "max_drawdown", "max_drawdown_duration", "hit_rate",
        "avg_win", "avg_loss", "profit_factor", "total_return",
        "n_days", "total_cost_bps", "time_in_market", "turnover",
        # New risk + inference metrics
        "cagr", "ann_vol", "downside_dev", "var_95", "cvar_95",
        "var_99", "cvar_99", "skew", "excess_kurt",
        "sharpe_ci_lo", "sharpe_ci_hi", "psr_vs_zero",
    }
    assert expected.issubset(s.keys())
    assert s["n_days"] == n
