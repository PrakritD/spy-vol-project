"""End-to-end plumbing smoke test on synthetic data.

Validates that walk-forward harness -> execution -> metrics wires together,
and that a model with informative features beats a random baseline on AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.execution import ExecConfig, backtest
from backtest.metrics import classification_metrics, summarize
from backtest.walk_forward import WalkForwardConfig, run as walk_forward_run
from models.logistic import LogisticModel


def _synthetic_panel(n: int = 600, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    # informative features
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    f3 = rng.standard_normal(n)
    # noisy logit driven by f1+f2
    logit = 1.2 * f1 + 0.8 * f2 + 0.3 * rng.standard_normal(n)
    p_true = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < p_true).astype(int)
    vxx_ret = rng.normal(loc=-2e-4, scale=0.025, size=n) + 0.02 * (y - 0.5)
    vxx_close = 100 * np.exp(np.cumsum(vxx_ret))
    return pd.DataFrame({
        "date": dates,
        "f1": f1, "f2": f2, "f3": f3,
        "y_next": y,
        "vxx_close": vxx_close,
    })


def test_walk_forward_runs_and_beats_random():
    panel = _synthetic_panel()
    preds = walk_forward_run(
        panel,
        feature_cols=["f1", "f2", "f3"],
        target_col="y_next",
        date_col="date",
        model_factory=LogisticModel,
        cfg=WalkForwardConfig(initial_train_months=6, refit_freq_months=1),
    )
    assert not preds.empty
    metrics = classification_metrics(preds)
    assert metrics["auc"] > 0.6, f"AUC suspiciously low: {metrics['auc']}"


def _structured_panel(n: int = 400, seed: int = 2) -> pd.DataFrame:
    """Synthetic panel that exposes the real-data columns each model expects.

    HAR-X needs rv, rv_5d_mean, rv_rolling_mean, rv_next, and a few VIX lag
    columns. Interactions model needs gex/vix lag columns. We build a tiny
    realistic schema with an informative signal so AUC > 0.5 is achievable.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n)
    # latent vol with positive autocorrelation
    eps = rng.standard_normal(n) * 0.4
    log_rv = np.zeros(n)
    log_rv[0] = -2.0
    for i in range(1, n):
        log_rv[i] = 0.7 * log_rv[i - 1] + 0.3 * (-2.0) + eps[i]
    rv = np.exp(log_rv)
    rv_5d = pd.Series(rv).rolling(5, min_periods=5).mean().fillna(rv.mean()).values
    rv_21d = pd.Series(rv).rolling(21, min_periods=21).mean().fillna(rv.mean()).values
    rv_next = np.roll(rv, -1)
    rv_next[-1] = rv[-1]
    y = (rv_next > rv_21d).astype(int)
    # surrogate VIX-like features correlated with log_rv
    vix_zscore = log_rv + rng.standard_normal(n) * 0.3
    term_9d_30d = 0.95 + 0.05 * (log_rv - log_rv.mean())
    term_30d_90d = 0.92 + 0.04 * (log_rv - log_rv.mean())
    vix_chg_1d = np.r_[0, np.diff(vix_zscore)]
    vix_chg_5d = pd.Series(vix_zscore).diff(5).fillna(0).values
    gex_net = -1e9 - 5e8 * (log_rv - log_rv.mean()) + rng.standard_normal(n) * 3e8
    return pd.DataFrame({
        "date": dates,
        "rv": rv, "rv_5d_mean": rv_5d, "rv_rolling_mean": rv_21d, "rv_next": rv_next,
        "vix_zscore_lag1": vix_zscore,
        "term_9d_30d_lag1": term_9d_30d,
        "term_30d_90d_lag1": term_30d_90d,
        "vix_chg_1d_lag1": vix_chg_1d,
        "vix_chg_5d_lag1": vix_chg_5d,
        "gex_net_lag1": gex_net,
        "y_next": y,
    })


def _assert_predict_proba_well_formed(model, X, y):
    model.fit(X, y)
    p = model.predict_proba(X)
    assert isinstance(p, np.ndarray), f"predict_proba must return ndarray, got {type(p)}"
    assert p.shape == (len(X),), f"shape {p.shape} != ({len(X)},)"
    assert np.all(np.isfinite(p)), "non-finite probabilities"
    assert np.all((p >= 0.0) & (p <= 1.0)), f"probs out of [0,1]: min={p.min()} max={p.max()}"


def test_har_x_conforms_to_protocol():
    from models.har_x import HARXClassifier
    panel = _structured_panel()
    cols = ["rv", "rv_5d_mean", "rv_rolling_mean", "rv_next",
            "vix_zscore_lag1", "term_9d_30d_lag1", "term_30d_90d_lag1", "gex_net_lag1"]
    _assert_predict_proba_well_formed(HARXClassifier(), panel[cols], panel["y_next"])


def test_logistic_interactions_conforms_to_protocol():
    from models.logistic_interactions import LogisticInteractionsModel
    panel = _structured_panel()
    cols = ["vix_zscore_lag1", "term_9d_30d_lag1", "vix_chg_5d_lag1",
            "vix_chg_1d_lag1", "gex_net_lag1"]
    _assert_predict_proba_well_formed(LogisticInteractionsModel(), panel[cols], panel["y_next"])


def test_mlp_small_conforms_to_protocol():
    from models.mlp_small import SmallMLPModel
    panel = _structured_panel()
    cols = ["vix_zscore_lag1", "term_9d_30d_lag1", "term_30d_90d_lag1",
            "vix_chg_1d_lag1", "vix_chg_5d_lag1", "gex_net_lag1"]
    _assert_predict_proba_well_formed(SmallMLPModel(), panel[cols], panel["y_next"])


def test_bayesian_head_conforms_to_protocol():
    from models.bayesian_head import BayesianHeadModel
    panel = _structured_panel().iloc[:200]   # GP is O(N^3); keep small
    cols = ["vix_zscore_lag1", "term_9d_30d_lag1", "gex_net_lag1"]
    _assert_predict_proba_well_formed(BayesianHeadModel(), panel[cols], panel["y_next"])


def test_factory_instantiates_all_models():
    from models import available_models, make_model
    for t in available_models():
        m = make_model(t)
        assert hasattr(m, "fit") and hasattr(m, "predict_proba"), f"{t} missing protocol"
        assert isinstance(m.name, str)


def test_har_x_lookahead_guard():
    """If rv_next is in X at predict time, the model must NOT use it."""
    from models.har_x import HARXClassifier
    panel = _structured_panel()
    cols = ["rv", "rv_5d_mean", "rv_rolling_mean", "rv_next",
            "vix_zscore_lag1", "term_9d_30d_lag1", "term_30d_90d_lag1", "gex_net_lag1"]
    m = HARXClassifier().fit(panel[cols].iloc[:300], panel["y_next"].iloc[:300])
    p_with_rv_next = m.predict_proba(panel[cols].iloc[300:])
    # Corrupt rv_next; predictions should be unchanged.
    corrupted = panel[cols].iloc[300:].copy()
    corrupted["rv_next"] = corrupted["rv_next"].values[::-1]
    p_corrupted = m.predict_proba(corrupted)
    assert np.allclose(p_with_rv_next, p_corrupted), \
        "HAR-X predict_proba should NOT depend on rv_next (lookahead guard failed)"


def test_execution_produces_equity_curve():
    panel = _synthetic_panel()
    preds = walk_forward_run(
        panel,
        feature_cols=["f1", "f2", "f3"],
        target_col="y_next",
        date_col="date",
        model_factory=LogisticModel,
        cfg=WalkForwardConfig(initial_train_months=6, refit_freq_months=1),
    )
    pnl = backtest(preds, panel[["date", "vxx_close"]], ExecConfig(threshold=0.55))
    summary = summarize(pnl)
    assert pnl["equity"].iloc[-1] > 0  # not blown up
    assert summary["n_days"] > 0
    assert np.isfinite(summary["sharpe"]) or pnl["net_pnl"].std() == 0
