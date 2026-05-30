"""Smoke tests for the live-readiness layer.

Verifies that `predict_today` produces a well-formed prediction dict
without crashing, on a small synthetic panel. Does NOT verify that the
classifier actually has any edge — that's what the walk-forward backtest
is for.

These tests run on a temporary fake panel so they don't depend on the
expensive features pipeline being run first. The fake panel has the
columns `predict_today` requires: VIX-lagged features, GEX-lagged features
(optional), rv triple, rv_next, y_next.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_fake_panel(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Build a feature panel with the schema predict_today expects."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)

    # Latent log-vol with positive autocorrelation.
    eps = rng.standard_normal(n) * 0.4
    log_rv = np.zeros(n)
    log_rv[0] = -2.0
    for i in range(1, n):
        log_rv[i] = 0.7 * log_rv[i - 1] + 0.3 * (-2.0) + eps[i]
    rv = np.exp(log_rv)
    rv_5d = pd.Series(rv).rolling(5, min_periods=5).mean().fillna(rv.mean()).values
    rv_21d = pd.Series(rv).rolling(21, min_periods=21).mean().fillna(rv.mean()).values
    rv_next = np.r_[rv[1:], rv[-1]]
    y = (rv_next > rv_21d).astype(int)

    vix_z = log_rv + rng.standard_normal(n) * 0.3
    # vix_level: always positive (clip to a realistic floor of 8)
    vix_level = np.clip(vix_z * 5 + 22, 8.0, 80.0)
    term_9_30 = np.clip(0.95 + 0.05 * (log_rv - log_rv.mean()), 0.7, 1.4)
    term_30_90 = np.clip(0.92 + 0.04 * (log_rv - log_rv.mean()), 0.7, 1.4)
    gex_net = -1e9 - 5e8 * (log_rv - log_rv.mean()) + rng.standard_normal(n) * 3e8

    return pd.DataFrame({
        "date": dates,
        "vix_level_lag1": vix_level,
        "vix_log_lag1": np.log(vix_level),
        "vix_chg_1d_lag1": np.r_[0, np.diff(vix_level)],
        "vix_chg_5d_lag1": pd.Series(vix_level).diff(5).fillna(0).values,
        "vix_zscore_lag1": vix_z,
        "term_9d_30d_lag1": term_9_30,
        "term_30d_90d_lag1": term_30_90,
        "vvix_vix_lag1": np.clip(5.5 + rng.standard_normal(n) * 0.3, 3.0, 10.0),
        "gex_net_lag1": gex_net,
        "gex_calls_lag1": np.abs(gex_net) + rng.standard_normal(n) * 1e8,
        "gex_puts_lag1": np.abs(gex_net) + rng.standard_normal(n) * 1e8,
        "n_contracts_lag1": rng.integers(200, 800, n).astype(float),
        "rv": rv,
        "rv_5d_mean": rv_5d,
        "rv_rolling_mean": rv_21d,
        "rv_next": rv_next,
        "y_next": y,
        "spy_close": 500 + np.cumsum(rng.normal(0.5, 5, n)),
        "vxx_close": 50 + np.cumsum(rng.normal(-0.05, 0.8, n)),
    })


@pytest.fixture
def fake_panel_path(tmp_path: Path) -> Path:
    p = tmp_path / "features_panel.parquet"
    _make_fake_panel().to_parquet(p)
    return p


def test_predict_today_returns_wellformed_dict(fake_panel_path: Path):
    """Sanity check on the main entry point."""
    from live.predict_today import predict_today

    result = predict_today(
        model_type="logistic_vix_only",
        threshold=0.55,
        panel_path=fake_panel_path,
    )

    # Required keys
    for k in ("date", "p_hat", "size", "action", "model", "threshold",
              "n_train_rows", "note", "generated_at_utc"):
        assert k in result, f"missing key: {k}"

    # Value-level sanity
    assert 0.0 <= result["p_hat"] <= 1.0
    assert 0.0 <= result["size"] <= 1.0
    assert result["action"] in ("long", "flat")
    assert result["model"] == "logistic_vix_only"
    assert isinstance(result["n_train_rows"], int) and result["n_train_rows"] > 0


def test_predict_today_threshold_pushes_size_to_zero(fake_panel_path: Path):
    """If threshold is 0.99, the result should always be flat."""
    from live.predict_today import predict_today

    result = predict_today(
        model_type="logistic_vix_only",
        threshold=0.99,
        panel_path=fake_panel_path,
    )
    assert result["size"] == 0.0
    assert result["action"] == "flat"


def test_predict_today_rejects_unknown_model(fake_panel_path: Path):
    """Unknown model_type should raise ValueError with the available list."""
    from live.predict_today import predict_today

    with pytest.raises(ValueError, match="not in experiment.yaml"):
        predict_today(
            model_type="not_a_real_model",
            panel_path=fake_panel_path,
        )


def test_predict_today_rejects_missing_panel(tmp_path: Path):
    """Missing panel file should raise FileNotFoundError with a helpful message."""
    from live.predict_today import predict_today

    with pytest.raises(FileNotFoundError, match="missing"):
        predict_today(
            model_type="logistic_vix_only",
            panel_path=tmp_path / "does_not_exist.parquet",
        )
