"""Phase 0 go/no-go (spec 2026-05-29 §11).

Question: does dealer gamma carry next-day RV-regime information INCREMENTAL to a
HAR-X + VIX model, on the 21-month signed-gamma window already on disk?

Method (no new data):
- Target: log RV[t] (one-step), predictors strictly <= t-1 (no lookahead).
- Contamination-fixed binary regime: y_bin[t] = 1[RV[t] > mean(RV[t-21..t-1])]
  (baseline ends t-1, excludes RV[t]; this is the fix for v1's placebo-beats-model bug).
- Nested models, expanding walk-forward, Gaussian predictive density:
    M0  HAR-X + VIX            (the bar)
    M1  + gamma LEVEL
    M2  + gamma SIGN bucket    (the "2-state signed-GEX bucket" the spec names)
    M3  + sign-interacted HAR  (regime-switching dynamics)
- Headline metric: Diebold-Mariano test on the CRPS differential (M0 - Mk),
  Newey-West HAC + Harvey small-sample correction. Positive & significant => gamma helps.
- Secondary: OOS RMSE, binary AUC, regime-conditional RV, gamma VIF.

Honest caveat baked into the verdict: N~265 OOS, one calm regime, mostly short-gamma.
A positive is encouraging; a strong null is a caution, not proof of absence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

REPO = __file__.rsplit("/analysis/", 1)[0]
PANEL = f"{REPO}/data/processed/features_panel.parquet"
GSCALE = 1e9  # scale gamma to ~unit


def gaussian_crps(y, mu, sigma):
    """Closed-form CRPS of N(mu, sigma) at observation y (lower is better)."""
    sigma = np.maximum(sigma, 1e-9)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def ols_fit_predict(Xtr, ytr, Xte):
    """OLS via lstsq; returns test point preds and the train residual std."""
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    resid = ytr - Xtr @ beta
    sigma = resid.std(ddof=Xtr.shape[1])
    return Xte @ beta, sigma, beta


def newey_west_var(d, lag):
    """Long-run variance of the mean of d (Newey-West)."""
    d = d - d.mean()
    n = len(d)
    g0 = (d @ d) / n
    s = g0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        gk = (d[k:] @ d[:-k]) / n
        s += 2 * w * gk
    return s / n  # variance of the mean


def dm_test(loss0, loss1, h=1):
    """DM on loss differential d = loss0 - loss1 (positive => model1 better).
    Newey-West HAC with Harvey-Leybourne-Newbold small-sample correction."""
    d = loss0 - loss1
    n = len(d)
    lag = max(1, int(round(n ** (1 / 3))))
    var = newey_west_var(d, lag)
    stat = d.mean() / np.sqrt(var)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)  # HLN correction
    stat *= corr
    p = 2 * (1 - norm.cdf(abs(stat)))
    return d.mean(), stat, p


def vif(X):
    """Variance inflation factor per column of a standardized design (no const)."""
    out = []
    for j in range(X.shape[1]):
        y = X[:, j]
        Z = np.delete(X, j, axis=1)
        Z = np.column_stack([np.ones(len(Z)), Z])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        r2 = 1 - ((y - Z @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out.append(1 / max(1 - r2, 1e-9))
    return out


def main():
    df = pd.read_parquet(PANEL).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    lrv = np.log(df["rv"])

    # HAR terms (log), all shifted to use info <= t-1
    df["har_d"] = lrv.shift(1)
    df["har_w"] = lrv.rolling(5).mean().shift(1)
    df["har_m"] = lrv.rolling(22).mean().shift(1)
    # contamination-fixed regime baseline (ends t-1, excludes rv[t])
    base = df["rv"].rolling(21).mean().shift(1)
    df["y_bin"] = (df["rv"] > base).astype(float)
    df["y"] = lrv  # continuous target = log RV[t]

    df["g_level"] = df["gex_net_lag1"] / GSCALE
    df["g_sign"] = (df["gex_net_lag1"] < 0).astype(float)  # 1 = short-gamma (negative)

    vixf = ["vix_level_lag1", "vix_zscore_lag1", "term_9d_30d_lag1",
            "term_30d_90d_lag1", "vvix_vix_lag1"]
    base_f = ["har_d", "har_w", "har_m"] + vixf

    d = df.dropna(subset=base_f + ["g_level", "y", "y_bin"]).reset_index(drop=True)
    d = d[d["gex_net_lag1"].notna()].reset_index(drop=True)
    print(f"modeling rows: {len(d)}  ({d['date'].min().date()} -> {d['date'].max().date()})")
    print(f"gamma: %short(neg)={d['g_sign'].mean()*100:.1f}%  base rate y_bin={d['y_bin'].mean():.3f}")

    specs = {
        "M0_HARX_VIX": base_f,
        "M1_+gammaLevel": base_f + ["g_level"],
        "M2_+gammaSign": base_f + ["g_sign"],
        "M3_+signXHAR": base_f + ["g_sign", "gsxhar_d", "gsxhar_w", "gsxhar_m"],
    }
    d["gsxhar_d"] = d["g_sign"] * d["har_d"]
    d["gsxhar_w"] = d["g_sign"] * d["har_w"]
    d["gsxhar_m"] = d["g_sign"] * d["har_m"]

    y = d["y"].to_numpy()
    ybin = d["y_bin"].to_numpy()
    n = len(d)
    train0 = 150
    oos = range(train0, n)

    preds = {k: np.full(n, np.nan) for k in specs}
    sigmas = {k: np.full(n, np.nan) for k in specs}
    for name, feats in specs.items():
        Xall = d[feats].to_numpy()
        for i in oos:
            Xtr, ytr = Xall[:i], y[:i]
            mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1
            Xtr_s = np.column_stack([np.ones(i), (Xtr - mu) / sd])
            Xte_s = np.concatenate([[1.0], (Xall[i] - mu) / sd])
            yhat, sig, _ = ols_fit_predict(Xtr_s, ytr, Xte_s[None, :])
            preds[name][i] = yhat[0]
            sigmas[name][i] = sig

    idx = np.array(list(oos))
    yo = y[idx]; ybino = ybin[idx]
    crps = {k: gaussian_crps(yo, preds[k][idx], sigmas[k][idx]) for k in specs}
    rmse = {k: np.sqrt(np.mean((yo - preds[k][idx]) ** 2)) for k in specs}
    # AUC: P(regime up) = P(logRV > log baseline). Use predicted mu vs OOS baseline proxy:
    # rank by predicted level relative to each model's own rolling mean of preds (proxy).
    auc = {}
    for k in specs:
        score = preds[k][idx]  # higher predicted log-RV => more likely "up" regime
        try:
            auc[k] = roc_auc_score(ybino, score)
        except Exception:
            auc[k] = np.nan

    print("\n=== OOS forecast quality (lower CRPS/RMSE better; N_oos=%d) ===" % len(idx))
    print(f"{'model':16s} {'meanCRPS':>10s} {'RMSE':>8s} {'AUC':>6s}")
    for k in specs:
        print(f"{k:16s} {crps[k].mean():10.4f} {rmse[k]:8.4f} {auc[k]:6.3f}")

    print("\n=== Diebold-Mariano on CRPS differential vs M0 (positive & p<.05 => gamma helps) ===")
    for k in specs:
        if k == "M0_HARX_VIX":
            continue
        dbar, stat, p = dm_test(crps["M0_HARX_VIX"], crps[k])
        flag = "  <-- gamma helps" if (dbar > 0 and p < 0.05) else ("  (favors M0)" if dbar < 0 else "")
        print(f"{k:16s} dCRPS={dbar:+.5f}  DM={stat:+.2f}  p={p:.3f}{flag}")

    # regime-conditional RV (descriptive: does gamma sign modulate RV?)
    neg = d["g_sign"] == 1
    rv_neg, rv_pos = np.log(d["rv"][neg]), np.log(d["rv"][~neg])
    from scipy.stats import ttest_ind
    t, pt = ttest_ind(rv_neg, rv_pos, equal_var=False)
    print("\n=== regime-conditional log-RV (descriptive) ===")
    print(f"short-gamma(neg) mean logRV={rv_neg.mean():.3f} (n={neg.sum()}) | "
          f"long-gamma(pos) mean logRV={rv_pos.mean():.3f} (n={(~neg).sum()})")
    print(f"Welch t={t:+.2f} p={pt:.3f}  "
          f"({'higher RV under short-gamma (mechanism-consistent)' if rv_neg.mean()>rv_pos.mean() else 'NOT mechanism-consistent'})")

    # VIF of gamma level vs VIX block
    Xs = d[["g_level"] + vixf].to_numpy()
    Xs = (Xs - Xs.mean(0)) / Xs.std(0)
    v = vif(Xs)
    print("\n=== VIF (gamma level vs VIX block) ===")
    for nm, val in zip(["g_level"] + vixf, v):
        print(f"  {nm:18s} VIF={val:.2f}")

    print("\n=== VERDICT ===")
    best_dm = max(
        ((k,) + dm_test(crps["M0_HARX_VIX"], crps[k]) for k in specs if k != "M0_HARX_VIX"),
        key=lambda r: r[1],
    )
    k, dbar, stat, p = best_dm
    if dbar > 0 and p < 0.05:
        print(f"GO (tentative): {k} beats HAR-X+VIX, dCRPS={dbar:+.5f}, DM p={p:.3f}.")
    elif dbar > 0:
        print(f"WEAK/INCONCLUSIVE: best gamma model ({k}) improves CRPS (dCRPS={dbar:+.5f}) "
              f"but not significantly (DM p={p:.3f}).")
    else:
        print(f"NO-GO signal: no gamma model beats HAR-X+VIX incrementally on CRPS "
              f"(best {k}: dCRPS={dbar:+.5f}, p={p:.3f}).")
    print("CAVEAT: N_oos=%d, one calm regime (2024-08->2026-03), %.0f%% short-gamma. "
          "Underpowered: treat a null as a caution, not proof of absence. Deep-history "
          "(2011->) level test + intraday remain the real evidence." % (len(idx), d['g_sign'].mean()*100))


if __name__ == "__main__":
    main()
