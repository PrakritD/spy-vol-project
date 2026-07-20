"""FORECASTING.md — walk-forward probabilistic volatility forecasting benchmark.

The one place in the repo where ML is set up to legitimately beat a classical baseline
(every ML component in STRATEGY.md is a null). Target: next-day log realized volatility
(Yang-Zhang RV, the `rv` column already used everywhere else in the repo). Baselines are
HAR (Corsi 2009: daily/weekly/monthly lagged log-RV) and HAR+VIX-family (adds VIX level,
the VIX/VIX3M term-structure ratio, and VVIX) — reusing `build_signals`'s existing
`har_d/har_w/har_m/vix_l/t_30_90` columns verbatim (already `shift(1)`'d there) plus one
added `vvix_l` column with the identical convention. Challengers are quantile gradient
boosting (genuinely probabilistic: CRPS from a pinball-loss quantile grid) and a small
MLP (point forecast, scored like the linear baselines).

Protocol identical to `ml_size_positions`/`timing_positions` in strategy_two_sleeve.py:
expanding walk-forward, monthly refit, train-only standardization, 5-day embargo between
train end and prediction. CRPS for the point-forecast models (HAR, HAR+VIX, MLP) uses a
CAUSAL ROLLING-VARIANCE sigma (rolling std of the last 63 realized residuals, computed
strictly before day i) rather than one sigma fixed for a whole refit block, so the score
adapts to volatility clustering instead of assuming constant residual variance for three
weeks at a time.

Inference: `dm_test`/`cw_test`/`gaussian_crps` imported verbatim from
`phase1_deep_history.py` (the repo's one hand-rolled DM/CW implementation; statsmodels is
absent from the `trading` env). HAR vs HAR+VIX is a nested comparison (VIX augments HAR)
and gets both CW (point-forecast squared-error, the test's original use) and DM (on the
CRPS loss differential). HAR+VIX vs the two challengers is non-nested (different model
families) and gets DM only, on CRPS. Reported overall and per era (pre2020/2020-21/2022+,
the repo's standard regime split) exactly like FINDINGS.md's decomposition.

Honest-outcome contract: this script and FORECASTING.md report whichever of ML-wins or
ML-nulls is actually true. The GBM CRPS is a finite-quantile-grid approximation (11
quantiles spanning 0.05-0.95); it truncates the tails beyond that range, understating true
CRPS somewhat for both models equally, so it is fine for a head-to-head comparison but is
not a calibrated absolute CRPS estimate.

Run: python analysis/forecast_bench.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

REPO = __file__.rsplit("/analysis/", 1)[0]
sys.path.insert(0, REPO + "/analysis")
from strategy_two_sleeve import build_signals, load_panel  # noqa: E402
from phase1_deep_history import _nw_var, cw_test, dm_test, gaussian_crps  # noqa: E402

TRAIN0 = 504          # ~2y initial train, matches ml_size_positions/timing_positions
REFIT_EVERY = 21      # monthly refit
EMBARGO = 5           # purge/embargo days between train end and prediction
SIGMA_WINDOW = 63     # trailing residuals used for the rolling-variance CRPS sigma
SIGMA_MIN_PERIODS = 21
QUANTILES = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
GBM_N_ESTIMATORS = 60
GBM_MAX_DEPTH = 2
MLP_HIDDEN = (8,)
SEED = 7

HAR_FEATS = ["har_d", "har_w", "har_m"]
VIX_FEATS = ["har_d", "har_w", "har_m", "vix_l", "t_30_90", "vvix_l"]


# ------------------------------------------------------------------ data ----
def add_forecast_columns(d: pd.DataFrame) -> pd.DataFrame:
    """Add `vvix_l` (VVIX level, same shift(1) convention as build_signals) and the
    target (today's log-RV; relative to the shift(1)'d predictors below this is a
    next-day-RV forecast) to an already `build_signals`-processed panel. Split out from
    `build_panel` so tests can exercise it on a synthetic panel without touching real
    data files."""
    d = d.copy()
    d["vvix_l"] = d["vvix"].shift(1)
    d["target"] = np.log(d["rv"].clip(lower=1e-6))
    return d


def build_panel() -> pd.DataFrame:
    """Reuse the flagship's own panel + HAR/VIX feature construction (all already
    shift(1)'d there) so this benchmark shares one definition of the data with
    STRATEGY.md/FINDINGS.md rather than re-deriving it."""
    return add_forecast_columns(build_signals(load_panel()))


# ------------------------------------------------------------------ scoring ----
def _rolling_sigma(y: np.ndarray, pred: np.ndarray,
                    window: int = SIGMA_WINDOW, min_periods: int = SIGMA_MIN_PERIODS) -> np.ndarray:
    """Causal rolling std of PAST residuals only (`shift(1)` before `.rolling`, so
    sigma[i] never uses resid[i])."""
    resid = pd.Series(y - pred)
    return resid.shift(1).rolling(window, min_periods=min_periods).std().to_numpy()


def _pinball(y: np.ndarray, q_hat: np.ndarray, tau: np.ndarray) -> np.ndarray:
    diff = y - q_hat
    return np.where(diff >= 0, tau * diff, (tau - 1) * diff)


# ------------------------------------------------------------------ walk-forward models ----
def wf_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Expanding walk-forward OLS point forecast. Used for both HAR and HAR+VIX (same
    function, different X)."""
    n = len(y)
    pred = np.full(n, np.nan)
    beta = mu = sd = None
    valid = ~np.isnan(X).any(1) & ~np.isnan(y)
    for i in range(TRAIN0, n):
        if (i - TRAIN0) % REFIT_EVERY == 0:
            tr = valid[:i - EMBARGO]
            Xtr, ytr = X[:i - EMBARGO][tr], y[:i - EMBARGO][tr]
            if len(ytr) < 100:
                beta = None
                continue
            mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
            A = np.column_stack([np.ones(len(ytr)), (Xtr - mu) / sd])
            beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        if beta is None or not valid[i]:
            continue
        xi = np.concatenate([[1.0], (X[i] - mu) / sd])
        pred[i] = xi @ beta
    return pred


def wf_mlp(X: np.ndarray, y: np.ndarray, hidden=MLP_HIDDEN, seed: int = SEED) -> tuple[np.ndarray, int]:
    """Small feedforward MLP point forecast on the same tabular HAR+VIX features (not a
    recurrent/temporal architecture; see the stage-03 checkpoint for why). Same
    walk-forward protocol as wf_ols. Returns (pred, param_count of the final fit)."""
    n = len(y)
    pred = np.full(n, np.nan)
    mdl = mu = sd = None
    valid = ~np.isnan(X).any(1) & ~np.isnan(y)
    for i in range(TRAIN0, n):
        if (i - TRAIN0) % REFIT_EVERY == 0:
            tr = valid[:i - EMBARGO]
            Xtr, ytr = X[:i - EMBARGO][tr], y[:i - EMBARGO][tr]
            if len(ytr) < 100:
                mdl = None
                continue
            mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
            mdl = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", alpha=1e-3,
                                max_iter=2000, random_state=seed)
            mdl.fit((Xtr - mu) / sd, ytr)
        if mdl is None or not valid[i]:
            continue
        pred[i] = mdl.predict(((X[i] - mu) / sd)[None, :])[0]
    param_count = int(sum(c.size for c in mdl.coefs_) + sum(b.size for b in mdl.intercepts_)) if mdl else 0
    return pred, param_count


def wf_qgb(X: np.ndarray, y: np.ndarray, quantiles: np.ndarray = QUANTILES,
           n_estimators: int = GBM_N_ESTIMATORS, max_depth: int = GBM_MAX_DEPTH,
           seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward quantile gradient boosting: one GradientBoostingRegressor per
    quantile, same refit cadence as the linear baselines. Returns (crps, median_pred);
    CRPS is `2 * mean(pinball loss over the quantile grid)`, the standard finite-grid
    CRPS approximation (Gneiting & Raftery 2007)."""
    n = len(y)
    crps = np.full(n, np.nan)
    med_pred = np.full(n, np.nan)
    mdls = mu = sd = None
    valid = ~np.isnan(X).any(1) & ~np.isnan(y)
    mid_idx = int(np.argmin(np.abs(quantiles - 0.5)))
    for i in range(TRAIN0, n):
        if (i - TRAIN0) % REFIT_EVERY == 0:
            tr = valid[:i - EMBARGO]
            Xtr, ytr = X[:i - EMBARGO][tr], y[:i - EMBARGO][tr]
            if len(ytr) < 100:
                mdls = None
                continue
            mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
            Xs = (Xtr - mu) / sd
            mdls = [GradientBoostingRegressor(loss="quantile", alpha=float(q),
                                               n_estimators=n_estimators, max_depth=max_depth,
                                               random_state=seed).fit(Xs, ytr)
                    for q in quantiles]
        if mdls is None or not valid[i]:
            continue
        xi = ((X[i] - mu) / sd)[None, :]
        qhat = np.sort(np.array([m.predict(xi)[0] for m in mdls]))  # guard quantile crossing
        crps[i] = 2 * _pinball(y[i], qhat, quantiles).mean()
        med_pred[i] = qhat[mid_idx]
    return crps, med_pred


# ------------------------------------------------------------------ evaluation ----
def crps_ci(crps_arr: np.ndarray, mask: np.ndarray, z: float = 1.96) -> tuple[float, float, float]:
    """Newey-West mean + CI for a CRPS series (reuses phase1_deep_history's `_nw_var`,
    the repo's one hand-rolled long-run-variance estimator)."""
    x = crps_arr[mask]
    n = len(x)
    lag = max(1, int(round(n ** (1 / 3))))
    se = np.sqrt(_nw_var(x - x.mean(), lag) / n)
    m = float(x.mean())
    return m, m - z * se, m + z * se


def summarize(crps_arr: np.ndarray, mask: np.ndarray) -> dict:
    m, lo, hi = crps_ci(crps_arr, mask)
    return {"n": int(mask.sum()), "mean_crps": m, "crps_ci95": [lo, hi]}


def cmp_dm(crps0: np.ndarray, crps1: np.ndarray, mask: np.ndarray) -> dict:
    d_mean, stat, p = dm_test(crps0[mask], crps1[mask])
    return {"n": int(mask.sum()), "dcrps": float(d_mean), "dm_stat": float(stat), "dm_p": float(p)}


def cmp_cw(y: np.ndarray, pred0: np.ndarray, pred1: np.ndarray, mask: np.ndarray) -> dict:
    f_mean, stat, p = cw_test(y[mask], pred0[mask], pred1[mask])
    return {"cw_stat": float(stat), "cw_p": float(p)}


def main() -> int:
    d = build_panel()
    y = d["target"].to_numpy()
    era = d["era"].to_numpy()
    Xhar = d[HAR_FEATS].to_numpy()
    Xvix = d[VIX_FEATS].to_numpy()

    print(f"panel: n={len(d)}  train0={TRAIN0}  refit_every={REFIT_EVERY}d  embargo={EMBARGO}d")

    pred_har = wf_ols(Xhar, y)
    crps_har = gaussian_crps(y, pred_har, _rolling_sigma(y, pred_har))

    pred_vix = wf_ols(Xvix, y)
    crps_vix = gaussian_crps(y, pred_vix, _rolling_sigma(y, pred_vix))

    pred_mlp, mlp_params = wf_mlp(Xvix, y)
    crps_mlp = gaussian_crps(y, pred_mlp, _rolling_sigma(y, pred_mlp))

    crps_gbm, pred_gbm = wf_qgb(Xvix, y)

    # common evaluation window: every model must have a finite CRPS that day, so every
    # pairwise test runs on identically the same sample.
    common = np.isfinite(crps_har) & np.isfinite(crps_vix) & np.isfinite(crps_mlp) & np.isfinite(crps_gbm)
    print(f"common evaluation window: {int(common.sum())} days "
          f"({d['date'].iloc[np.where(common)[0][0]].date()} -> {d['date'].iloc[np.where(common)[0][-1]].date()})")

    results: dict = {
        "protocol": {"train0": TRAIN0, "refit_every": REFIT_EVERY, "embargo": EMBARGO,
                     "sigma_window": SIGMA_WINDOW, "quantile_grid": QUANTILES.tolist()},
        "models": {
            "har": {**summarize(crps_har, common), "features": HAR_FEATS},
            "har_vix": {**summarize(crps_vix, common), "features": VIX_FEATS},
            "mlp": {**summarize(crps_mlp, common), "features": VIX_FEATS,
                    "hidden_layer_sizes": list(MLP_HIDDEN), "param_count": mlp_params},
            "qgb": {**summarize(crps_gbm, common), "features": VIX_FEATS,
                    "n_estimators": GBM_N_ESTIMATORS, "max_depth": GBM_MAX_DEPTH,
                    "n_quantile_models": len(QUANTILES)},
        },
        "comparisons": {},
        "multiplicity": {
            "n_comparisons": 12,
            "note": ("3 model-comparison pairs (HAR vs HAR+VIX nested-CW+DM, HAR+VIX vs "
                     "QGB non-nested-DM, HAR+VIX vs MLP non-nested-DM), each run overall "
                     "plus 3 regime blocks = 12 tests. No formal multiplicity correction "
                     "(e.g. Bonferroni) is applied; p-values are reported raw and this "
                     "count is stated so a reader can judge for themselves."),
        },
    }

    per_model_crps = {"har": crps_har, "har_vix": crps_vix, "mlp": crps_mlp, "qgb": crps_gbm}
    blocks = {"overall": np.ones(len(d), dtype=bool),
              "pre2020": era == "pre2020", "2020-21": era == "2020-21", "2022+": era == "2022+"}
    results["models_by_era"] = {}
    for name, blk in blocks.items():
        m = common & blk
        results["comparisons"][name] = {
            "har_vs_har_vix": {**cmp_dm(crps_har, crps_vix, m), **cmp_cw(y, pred_har, pred_vix, m)},
            "har_vix_vs_qgb": cmp_dm(crps_vix, crps_gbm, m),
            "har_vix_vs_mlp": cmp_dm(crps_vix, crps_mlp, m),
        }
        results["models_by_era"][name] = {mn: summarize(mc, m) for mn, mc in per_model_crps.items()}
        print(f"[{name:8s} n={int(m.sum()):5d}] "
              f"HAR {results['comparisons'][name]['har_vs_har_vix']} | "
              f"QGB {results['comparisons'][name]['har_vix_vs_qgb']} | "
              f"MLP {results['comparisons'][name]['har_vix_vs_mlp']}")

    out_path = f"{REPO}/analysis/forecast_bench_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
