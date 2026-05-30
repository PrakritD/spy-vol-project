"""Phase 0.5 — reframe exploration (signal-finding stage).

Phase 0 showed dealer gamma is redundant with VIX for forecasting RV *level*.
VIX *is* the price of forward variance, so that was the wrong target. Here we test
whether gamma predicts what VIX does NOT price: the PATH, DYNAMICS, and TAILS of returns.

PRE-REGISTERED targets (stated a priori from the dealer-hedging mechanism, 2026-05-29).
All tested incremental to a VIX/HAR baseline, all reported, Bonferroni-corrected across
the family, any hit treated as a HYPOTHESIS to confirm on deep-history/intraday data:

  T1  Intraday-range compression (pinning): +gamma -> tighter intraday range
      (Parkinson high-low vol), incremental to VIX+HAR(rv-level). [continuous, DM-on-CRPS]
  T2  Return mean-reversion vs trend: +gamma -> reversal, -gamma -> momentum.
      [in-sample HAC coefficient test on the lag-return x gamma interaction + conditional rr]
  T3  Downside/jump tails: -gamma (short gamma) -> more large DOWN moves,
      incremental to VIX+VVIX. [binary, OOS log-loss/AUC + block-bootstrap]
  T4  Regime-direction classification (the Phase-0 AUC thread): gamma added to a
      VIX/HAR logistic for the contamination-fixed RV-regime. [binary, OOS log-loss/AUC + bootstrap]

Same 21-month signed window as Phase 0. Underpowered + one calm regime: hits are leads, not proof.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

REPO = __file__.rsplit("/analysis/", 1)[0]
PANEL = f"{REPO}/data/processed/features_panel.parquet"
SPY = f"{REPO}/data/raw/yfinance/SPY.parquet"
GSCALE = 1e9
TRAIN0 = 150


def gaussian_crps(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-9)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def newey_west_var(d, lag):
    d = d - d.mean(); n = len(d); s = (d @ d) / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return s / n


def dm_test(loss0, loss1, h=1):
    d = loss0 - loss1; n = len(d); lag = max(1, int(round(n ** (1 / 3))))
    var = newey_west_var(d, lag)
    stat = d.mean() / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return d.mean(), stat, 2 * (1 - norm.cdf(abs(stat)))


def wf_ols_crps(d, feats, target):
    y = d[target].to_numpy(); X = d[feats].to_numpy(); n = len(d)
    crps = np.full(n, np.nan)
    for i in range(TRAIN0, n):
        Xtr, ytr = X[:i], y[:i]
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
        A = np.column_stack([np.ones(i), (Xtr - mu) / sd])
        beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        sig = (ytr - A @ beta).std(ddof=A.shape[1])
        xte = np.concatenate([[1.0], (X[i] - mu) / sd])
        crps[i] = gaussian_crps(y[i], xte @ beta, sig)
    return crps[TRAIN0:], y[TRAIN0:]


def wf_logit(d, feats, target):
    y = d[target].to_numpy(); X = d[feats].to_numpy(); n = len(d)
    p = np.full(n, np.nan)
    for i in range(TRAIN0, n):
        ytr = y[:i]
        if len(np.unique(ytr)) < 2:
            p[i] = ytr.mean(); continue
        Xtr = X[:i]; mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit((Xtr - mu) / sd, ytr)
        p[i] = clf.predict_proba(((X[i] - mu) / sd)[None, :])[0, 1]
    return p[TRAIN0:], y[TRAIN0:]


def block_boot_pval(y, p0, p1, metric, n_boot=2000, seed=7):
    """Block-bootstrap p-value for metric(model1) - metric(model0) != 0."""
    n = len(y); bs = max(5, int(round(n ** (1 / 3)))); rng = np.random.default_rng(seed)
    def m(yy, pp):
        if metric == "auc":
            return roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan
        return -log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6), labels=[0, 1])  # higher=better
    obs = m(y, p1) - m(y, p0)
    nb = int(np.ceil(n / bs)); diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - bs + 1), size=nb)
        idx = np.concatenate([np.arange(s, s + bs) for s in starts])[:n]
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        diffs.append(m(yy, p1[idx]) - m(yy, p0[idx]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return obs, p


def main():
    df = pd.read_parquet(PANEL).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    spy = pd.read_parquet(SPY); spy["date"] = pd.to_datetime(spy["date"])
    df = df.merge(spy[["date", "open", "high", "low", "close"]], on="date", how="left", suffixes=("", "_spy"))

    lrv = np.log(df["rv"])
    df["har_d"] = lrv.shift(1); df["har_w"] = lrv.rolling(5).mean().shift(1); df["har_m"] = lrv.rolling(22).mean().shift(1)
    # path / return building blocks (day t)
    c = df["close"]; h = df["high"]; lo = df["low"]
    df["ret"] = np.log(c / c.shift(1))
    df["parkinson"] = (np.log(h / lo) ** 2) / (4 * np.log(2))           # intraday range variance
    df["lpark"] = 0.5 * np.log(df["parkinson"].clip(lower=1e-10))       # T1 target: log intraday-range vol
    df["down_move"] = (df["ret"] < -0.01).astype(float)                # T3 target: large down day
    base_reg = df["rv"].rolling(21).mean().shift(1)
    df["y_bin"] = (df["rv"] > base_reg).astype(float)                  # T4 target (contamination-fixed)

    df["g_level"] = df["gex_net_lag1"] / GSCALE
    df["g_sign"] = (df["gex_net_lag1"] < 0).astype(float)              # 1 = short gamma (negative)
    df["pos_gamma"] = (df["gex_net_lag1"] >= 0).astype(float)
    vixf = ["vix_level_lag1", "vix_zscore_lag1", "term_9d_30d_lag1", "term_30d_90d_lag1", "vvix_vix_lag1"]

    need = ["har_d", "har_w", "har_m", "lpark", "ret", "y_bin", "g_level"] + vixf
    d = df.dropna(subset=need).reset_index(drop=True)
    d = d[d["gex_net_lag1"].notna()].reset_index(drop=True)
    print(f"rows: {len(d)} ({d['date'].min().date()}->{d['date'].max().date()}) | %short={d['g_sign'].mean()*100:.0f}%")
    gamma_add = ["g_sign", "g_level"]
    results = []  # (name, stat_desc, pval)

    # ---- T1 intraday-range compression (continuous, DM-on-CRPS) ----
    base = ["har_d", "har_w", "har_m"] + vixf
    c0, y1 = wf_ols_crps(d, base, "lpark")
    c1, _ = wf_ols_crps(d, base + gamma_add, "lpark")
    db, st, p = dm_test(c0, c1)
    print(f"\n[T1 intraday-range] meanCRPS base={c0.mean():.4f} +gamma={c1.mean():.4f}  dCRPS={db:+.5f} DM={st:+.2f} p={p:.3f}")
    results.append(("T1 intraday-range", f"dCRPS={db:+.5f}", p))

    # ---- T2 mean-reversion vs trend (in-sample HAC interaction) ----
    d["ret_lag"] = d["ret"].shift(1)
    dd = d.dropna(subset=["ret_lag"]).reset_index(drop=True)
    X = np.column_stack([np.ones(len(dd)), dd["ret_lag"], dd["ret_lag"] * dd["pos_gamma"], dd["pos_gamma"]])
    yv = dd["ret"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    # HAC (Newey-West) SE on the interaction coef (index 2)
    n = len(dd); lag = max(1, int(round(n ** (1 / 3))))
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        Xe = X * resid[:, None]
        G = Xe[k:].T @ Xe[:-k]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    b_int, se_int = beta[2], np.sqrt(cov[2, 2])
    t_int = b_int / se_int; p_int = 2 * (1 - norm.cdf(abs(t_int)))
    rr = dd["ret"] * dd["ret_lag"]
    t_rr, p_rr = ttest_ind(rr[dd["pos_gamma"] == 1], rr[dd["pos_gamma"] == 0], equal_var=False)
    print(f"\n[T2 reversion] lagret x pos_gamma coef={b_int:+.4f} (HAC t={t_int:+.2f} p={p_int:.3f})  "
          f"[neg coef => +gamma adds reversal]")
    print(f"   ret*ret_lag: pos-gamma mean={rr[dd['pos_gamma']==1].mean():+.2e} vs neg={rr[dd['pos_gamma']==0].mean():+.2e} (t={t_rr:+.2f} p={p_rr:.3f})")
    results.append(("T2 mean-reversion (interaction)", f"coef={b_int:+.4f} t={t_int:+.2f}", p_int))

    # ---- T3 downside/jump tails (binary OOS) ----
    tail_base = ["vix_level_lag1", "vix_zscore_lag1", "vvix_vix_lag1", "har_d"]
    p0, yb = wf_logit(d, tail_base, "down_move")
    p1, _ = wf_logit(d, tail_base + gamma_add, "down_move")
    a_obs, a_p = block_boot_pval(yb, p0, p1, "auc")
    l_obs, l_p = block_boot_pval(yb, p0, p1, "logloss")
    print(f"\n[T3 downside tail] base rate={yb.mean():.3f}  dAUC={a_obs:+.3f} (p={a_p:.3f})  dLogLik={l_obs:+.4f} (p={l_p:.3f})")
    results.append(("T3 downside-tail (AUC)", f"dAUC={a_obs:+.3f}", a_p))

    # ---- T4 regime-direction classification (binary OOS; the AUC thread) ----
    reg_base = ["har_d", "har_w", "har_m"] + vixf
    p0, yb = wf_logit(d, reg_base, "y_bin")
    p1, _ = wf_logit(d, reg_base + gamma_add, "y_bin")
    a_obs, a_p = block_boot_pval(yb, p0, p1, "auc")
    l_obs, l_p = block_boot_pval(yb, p0, p1, "logloss")
    print(f"\n[T4 regime-direction] base rate={yb.mean():.3f}  dAUC={a_obs:+.3f} (p={a_p:.3f})  dLogLik={l_obs:+.4f} (p={l_p:.3f})")
    results.append(("T4 regime-direction (AUC)", f"dAUC={a_obs:+.3f}", a_p))

    # ---- family-wise summary ----
    m = len(results); alpha = 0.05; bonf = alpha / m
    print(f"\n=== PRE-REGISTERED FAMILY ({m} tests; Bonferroni alpha={bonf:.4f}) ===")
    print(f"{'target':34s} {'effect':26s} {'p':>7s}  {'Bonf hit?':>9s}")
    for nm, eff, p in results:
        print(f"{nm:34s} {eff:26s} {p:7.3f}  {'YES' if p < bonf else 'no':>9s}")
    hits = [r for r in results if r[2] < bonf]
    raw_hits = [r for r in results if r[2] < alpha]
    print("\n=== VERDICT ===")
    if hits:
        print("LEAD(S) survive Bonferroni:", ", ".join(h[0] for h in hits))
    elif raw_hits:
        print("Suggestive (raw p<.05 but NOT Bonferroni-robust):", ", ".join(h[0] for h in raw_hits),
              "-> hypothesis for deep-history/intraday, not a result.")
    else:
        print("No target shows gamma adding value beyond VIX/HAR on this window.")
    print("CAVEAT: 21 calm months, ~72% short-gamma, N_oos~265. These are LEADS to confirm on "
          "deep-history (2011->) and intraday data, never standalone results.")


if __name__ == "__main__":
    main()
