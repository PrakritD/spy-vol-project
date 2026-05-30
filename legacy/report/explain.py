"""Feature importance per model.

Re-fits each model spec from `configs/experiment.yaml` on the full features
panel (no walk-forward — this is descriptive, not predictive) and produces
a feature-importance ranking, written to `report/_build/`:

    importance_<model>.png      one bar chart per model
    feature_importance.csv      tall frame: (model, feature, importance)

Importance per model class:
    linear models       -> scaled coefficient magnitude (with sign)
    har_x               -> OLS coefficient × in-training feature stdev
    xgb_calibrated      -> SHAP TreeExplainer mean |SHAP| (falls back to
                           XGBoost's built-in `gain` if shap isn't installed)
    mlp_small           -> sklearn permutation_importance
    bayesian_head       -> sklearn permutation_importance

Run via:
    python -m report.explain
    make explain
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt

from models.factory import make_model

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = REPO_ROOT / "data" / "processed" / "features_panel.parquet"
CFG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
BUILD = REPO_ROOT / "report" / "_build"
CSV_OUT = BUILD / "feature_importance.csv"

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#17becf", "#bcbd22",
]


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _resolve_features(spec: dict, groups: dict) -> list[str]:
    cols: list[str] = []
    for g in spec.get("feature_groups", []):
        cols.extend(groups.get(g, []))
    return list(dict.fromkeys(cols))


def _importance_linear(model, feature_cols: list[str]) -> pd.Series:
    """For LogisticModel / LogisticInteractionsModel."""
    pipe = model._pipe
    lr = pipe.named_steps["lr"]
    # If the model engineered interaction features, the column count won't match
    # feature_cols. Use the names directly from the fitted scaler if available.
    n_coef = lr.coef_.shape[1]
    if n_coef == len(feature_cols):
        names = feature_cols
    else:
        # logistic_interactions: append the engineered names
        from models.logistic_interactions import _INTERACTIONS
        extras = []
        for a, b in _INTERACTIONS:
            if b is None:
                extras.append(f"{a}_sq")
            else:
                extras.append(f"{a}_x_{b}")
        names = feature_cols + extras
        if len(names) != n_coef:
            names = [f"feat_{i}" for i in range(n_coef)]
    coefs = lr.coef_.ravel()
    return pd.Series(coefs, index=names, name="coef")


def _importance_har_x(model, feature_cols: list[str], X: pd.DataFrame) -> pd.Series:
    """Coefficient × training feature stdev — the standardised effect size."""
    reg = model._regressor
    if reg is None:
        return pd.Series(dtype=float)
    cols = model._feature_cols
    # log-transform RV columns to match what the regressor was fit on
    Xt = X[cols].copy()
    from models.har_x import HARXClassifier
    for c in HARXClassifier.LOG_RV_COLS:
        if c in Xt.columns:
            Xt[c] = np.log(Xt[c].clip(lower=1e-6))
    sd = Xt.std()
    coef = pd.Series(reg.coef_, index=cols, name="coef")
    return (coef * sd).rename("std_effect")


def _importance_xgb(model, X: pd.DataFrame, y: pd.Series, feature_cols: list[str]) -> pd.Series:
    """SHAP mean |SHAP| per feature; fall back to XGB built-in gain if shap missing."""
    # `model._calibrator` wraps a fitted XGBClassifier via FrozenEstimator
    # (sklearn 1.6+). Walk through to the raw booster.
    impute = model._pipe.named_steps["impute"]
    Xi = impute.transform(X)
    def _unwrap_xgb(m):
        """Drill through CalibratedClassifierCV → FrozenEstimator → XGBClassifier."""
        # CalibratedClassifierCV in sklearn 1.6+ wraps prefit models via FrozenEstimator
        if hasattr(m, "_calibrator"):
            for cc in m._calibrator.calibrated_classifiers_:
                est = cc.estimator
                # FrozenEstimator stores its wrapped model on `.estimator`
                while hasattr(est, "estimator") and not hasattr(est, "feature_importances_"):
                    est = est.estimator
                return est
        return m._pipe.named_steps.get("xgb")

    booster = _unwrap_xgb(model)
    try:
        import shap
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(Xi)
        mean_abs = np.abs(shap_values).mean(axis=0)
        return pd.Series(mean_abs, index=feature_cols, name="mean_abs_shap")
    except (ImportError, Exception):
        # Fallback: XGBoost gain importance from the raw booster
        if hasattr(booster, "feature_importances_"):
            return pd.Series(booster.feature_importances_,
                              index=feature_cols, name="xgb_gain")
        return pd.Series([np.nan] * len(feature_cols),
                          index=feature_cols, name="unavailable")


def _importance_permutation(model, X: pd.DataFrame, y: pd.Series,
                              feature_cols: list[str]) -> pd.Series:
    """sklearn permutation_importance with AUC scoring.

    The model wrapper inherits from BaseEstimator + ClassifierMixin so
    sklearn 1.6+'s introspection (`__sklearn_tags__`, `is_classifier`) works.
    """
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.inspection import permutation_importance

    class _Wrap(ClassifierMixin, BaseEstimator):
        def __init__(self, inner):
            self.inner = inner
            self.classes_ = np.array([0, 1])

        def fit(self, X_, y_):
            return self  # already fitted upstream

        def predict_proba(self, X_):
            p = self.inner.predict_proba(X_)
            return np.column_stack([1 - p, p])

        def predict(self, X_):
            return (self.predict_proba(X_)[:, 1] >= 0.5).astype(int)

    w = _Wrap(model)
    res = permutation_importance(w, X, y.astype(int), scoring="roc_auc",
                                  n_repeats=8, random_state=13, n_jobs=1)
    return pd.Series(res.importances_mean, index=feature_cols,
                      name="perm_importance_auc_drop")


def _plot_importance(importance: pd.Series, model_name: str, color: str) -> Path:
    df = importance.dropna().abs().sort_values(ascending=True)
    signs = importance.dropna().loc[df.index].pipe(lambda s: np.sign(s))
    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.32 * len(df) + 1.5)))
    bars = ax.barh(df.index, df.values, color=[color if s >= 0 else "#888888" for s in signs],
                    alpha=0.9)
    ax.set_xlabel("|importance|")
    ax.set_title(f"Feature importance — {model_name}\n(grey = negative coefficient where applicable)")
    for b, v in zip(bars, df.values):
        ax.text(b.get_width() * 1.01, b.get_y() + b.get_height()/2,
                f"{v:.3g}", va="center", fontsize=8)
    out = BUILD / f"importance_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def explain_all() -> pd.DataFrame:
    _setup_style()
    BUILD.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CFG_PATH.read_text())
    groups = cfg.get("feature_groups", {})
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    long_rows: list[dict] = []
    for i, spec in enumerate(cfg["models"]):
        name = spec["name"]
        feat = _resolve_features(spec, groups)
        missing = [c for c in feat if c not in panel.columns]
        if missing:
            print(f"[skip] {name}: missing columns {missing}")
            continue
        sub = panel.dropna(subset=feat + ["y_next"]).reset_index(drop=True)
        if len(sub) < 50:
            print(f"[skip] {name}: only {len(sub)} clean rows")
            continue
        X, y = sub[feat], sub["y_next"].astype(int)
        print(f"[fit ] {name:<24s} {len(X)} rows, {len(feat)} features")

        hyper = spec.get("hyperparams", {}) or {}
        model = make_model(spec["type"], **hyper).fit(X, y)

        t = spec["type"]
        if t == "logistic":
            imp = _importance_linear(model, feat)
        elif t == "logistic_interactions":
            imp = _importance_linear(model, feat)
        elif t == "har_x":
            imp = _importance_har_x(model, feat, X)
        elif t == "xgb_calibrated":
            imp = _importance_xgb(model, X, y, feat)
        elif t in ("mlp_small", "bayesian_head"):
            imp = _importance_permutation(model, X, y, feat)
        else:
            print(f"[skip] {name}: unknown importance method for type {t}")
            continue

        png = _plot_importance(imp, name, _PALETTE[i % len(_PALETTE)])
        print(f"       wrote {png.relative_to(REPO_ROOT)}")

        for feat_name, val in imp.items():
            long_rows.append({"model": name, "feature": feat_name,
                              "importance": float(val) if pd.notna(val) else np.nan,
                              "abs_importance": float(abs(val)) if pd.notna(val) else np.nan})

    out_df = pd.DataFrame(long_rows)
    out_df.to_csv(CSV_OUT, index=False)
    print(f"\nwrote {CSV_OUT.relative_to(REPO_ROOT)} ({len(out_df)} rows)")
    return out_df


def main():
    explain_all()


if __name__ == "__main__":
    main()
