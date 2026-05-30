"""Phase 1 — deep-history test (2011-2026): the powered, multi-regime version.

The 21-month owned window nulled (Phase 0/0.5/0.5b) but was underpowered and one calm
regime. This re-runs the core question on ~15 years of FREE data through 2011/2015/2018/
2020/2022 — where the negative-gamma (amplification) regime actually appears (9.1% of days).

Data (all free, verified, download-and-gitignore):
  SqueezeMetrics GEX/DIX (2011-05->), CBOE VIX (1990->), yfinance SPY OHLC + VIX3M/VIX9D/VVIX.

Same discipline as before: contamination-fixed target (baseline ends t-1), no-lookahead
(all predictors <= t-1), expanding walk-forward, nested Diebold-Mariano on CRPS, block-
bootstrap for binary, reported OVERALL and PER REGIME BLOCK (never pooled across the 0DTE
break). Gamma enters as level percentile + negative-gamma indicator + DIX, incremental to
a full VIX/HAR baseline.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

REPO = __file__.rsplit("/analysis/", 1)[0]
sys.path.insert(0, REPO + "/analysis")
from rvutil import daily_yang_zhang_rv  # noqa: E402  (vendored; no v1 dependency)

TRAIN0 = 504        # ~2y initial train
REFIT_EVERY = 5     # refit cadence (days) for speed; predictions still daily OOS


def gaussian_crps(y, mu, s):
    s = np.maximum(s, 1e-9); z = (y - mu) / s
    return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def _nw_var(d, lag):
    d = d - d.mean(); n = len(d); s = (d @ d) / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return s / max(n, 1)


def dm_test(l0, l1, h=1):
    d = l0 - l1; n = len(d)
    if n < 10:
        return d.mean(), np.nan, np.nan
    lag = max(1, int(round(n ** (1 / 3))))
    st = d.mean() / np.sqrt(_nw_var(d, lag)) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return d.mean(), st, 2 * (1 - norm.cdf(abs(st)))


def wf_ols_crps(X, y, refit=REFIT_EVERY):
    n = len(y); crps = np.full(n, np.nan); pred = np.full(n, np.nan)
    beta = sig = mu = sd = None
    for i in range(TRAIN0, n):
        if (i - TRAIN0) % refit == 0:
            Xtr, ytr = X[:i], y[:i]; mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
            A = np.column_stack([np.ones(i), (Xtr - mu) / sd]); beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
            sig = (ytr - A @ beta).std(ddof=A.shape[1])
        xte = np.concatenate([[1.0], (X[i] - mu) / sd]); pred[i] = xte @ beta
        crps[i] = gaussian_crps(y[i], pred[i], sig)
    return crps, pred


def wf_logit(X, y, refit=REFIT_EVERY):
    n = len(y); p = np.full(n, np.nan); clf = mu = sd = None
    for i in range(TRAIN0, n):
        if (i - TRAIN0) % refit == 0:
            ytr = y[:i]
            if len(np.unique(ytr)) < 2:
                clf = None; base = ytr.mean()
            else:
                Xtr = X[:i]; mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
                clf = LogisticRegression(C=1.0, max_iter=2000).fit((Xtr - mu) / sd, ytr)
        p[i] = base if clf is None else clf.predict_proba(((X[i] - mu) / sd)[None, :])[0, 1]
    return p


def boot_p(y, p0, p1, metric, nb=2000, seed=7):
    n = len(y); bs = max(5, int(round(n ** (1 / 3)))); rng = np.random.default_rng(seed)
    def m(yy, pp):
        if metric == "auc":
            return roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan
        return -log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6), labels=[0, 1])
    obs = m(y, p1) - m(y, p0); k = int(np.ceil(n / bs)); diffs = []
    for _ in range(nb):
        st = rng.integers(0, max(1, n - bs + 1), size=k)
        idx = np.concatenate([np.arange(s, s + bs) for s in st])[:n]
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(m(y[idx], p1[idx]) - m(y[idx], p0[idx]))
    diffs = np.array(diffs)
    return obs, (2 * min((diffs <= 0).mean(), (diffs >= 0).mean()) if len(diffs) else np.nan)


def load_panel():
    spy = pd.read_parquet(f"{REPO}/data/raw/deep/SPY.parquet")[["date", "open", "high", "low", "close"]]
    spy["date"] = pd.to_datetime(spy["date"])
    rv = daily_yang_zhang_rv(spy)
    rv["ret"] = np.log(spy["close"] / spy["close"].shift(1)).values
    sq = pd.read_csv(f"{REPO}/data/raw/squeeze_dix.csv"); sq["date"] = pd.to_datetime(sq["date"])
    vix = pd.read_csv(f"{REPO}/data/raw/cboe_vix.csv"); vix["date"] = pd.to_datetime(vix["DATE"])
    vix = vix[["date", "CLOSE"]].rename(columns={"CLOSE": "vix"})
    def yf(name):
        d = pd.read_parquet(f"{REPO}/data/raw/deep/{name}.parquet")[["date", "close"]]
        d["date"] = pd.to_datetime(d["date"]); return d.rename(columns={"close": name.lower()})
    df = rv.merge(sq[["date", "gex", "dix"]], on="date", how="inner").merge(vix, on="date", how="inner")
    for nm in ["VIX3M", "VIX9D", "VVIX"]:
        df = df.merge(yf(nm), on="date", how="left")
    return df.sort_values("date").reset_index(drop=True)


def main():
    df = load_panel()
    print(f"deep panel: {len(df)} rows  {df['date'].min().date()} -> {df['date'].max().date()}")

    lrv = np.log(df["rv"].clip(lower=1e-6))
    # HAR (info <= t-1)
    df["har_d"] = lrv.shift(1); df["har_w"] = lrv.rolling(5).mean().shift(1); df["har_m"] = lrv.rolling(22).mean().shift(1)
    # VIX family (info <= t-1)
    df["vix_l"] = df["vix"].shift(1)
    df["vix_z"] = ((df["vix"] - df["vix"].rolling(20).mean()) / df["vix"].rolling(20).std()).shift(1)
    df["t_9_30"] = (df["vix9d"] / df["vix"]).shift(1)
    df["t_30_90"] = (df["vix"] / df["vix3m"]).shift(1)
    df["vvix_vix"] = (df["vvix"] / df["vix"]).shift(1)
    # Gamma (info <= t-1)
    df["gex_pct"] = df["gex"].rolling(252, min_periods=60).apply(lambda a: (a[-1] > a).mean(), raw=True).shift(1)
    df["gex_neg"] = (df["gex"] < 0).astype(float).shift(1)
    df["dix_l"] = df["dix"].shift(1)
    # contamination-fixed target: RV[t] vs trailing mean ending t-1
    base = df["rv"].rolling(21).mean().shift(1)
    df["y_bin"] = (df["rv"] > base).astype(float)
    df["y"] = lrv

    vixf = ["vix_l", "vix_z", "t_9_30", "t_30_90", "vvix_vix"]
    gam = ["gex_pct", "gex_neg", "dix_l"]
    need = ["har_d", "har_w", "har_m", "y", "y_bin"] + vixf + gam
    d = df.dropna(subset=need).reset_index(drop=True)
    print(f"modeling rows: {len(d)}  {d['date'].min().date()} -> {d['date'].max().date()} | %neg-gamma={d['gex_neg'].mean()*100:.1f}%")

    # ---- descriptive: mechanism by gamma regime, overall + per block ----
    d["era"] = np.where(d["date"] < "2020-01-01", "pre2020",
                np.where(d["date"] < "2022-01-01", "2020-21", "2022+"))
    print("\n=== mechanism (mean log-RV by gamma sign) ===")
    for era in ["ALL", "pre2020", "2020-21", "2022+"]:
        s = d if era == "ALL" else d[d["era"] == era]
        neg, pos = s.loc[s["gex_neg"] == 1, "y"], s.loc[s["gex_neg"] == 0, "y"]
        if len(neg) > 5 and len(pos) > 5:
            t, p = ttest_ind(neg, pos, equal_var=False)
            print(f"  {era:8s} short-γ logRV={neg.mean():+.3f} (n={len(neg)}) vs long-γ={pos.mean():+.3f} (n={len(pos)})  t={t:+.1f} p={p:.1e}")

    # ---- incremental skill: baseline HAR+VIX vs +gamma ----
    base_f = ["har_d", "har_w", "har_m"] + vixf
    Xb, Xg = d[base_f].to_numpy(), d[base_f + gam].to_numpy()
    y = d["y"].to_numpy(); yb = d["y_bin"].to_numpy()
    c0, _ = wf_ols_crps(Xb, y); c1, _ = wf_ols_crps(Xg, y)
    p0 = wf_logit(Xb, yb); p1 = wf_logit(Xg, yb)
    oos = np.arange(TRAIN0, len(d))
    era_oos = d["era"].to_numpy()[oos]
    dates_oos = d["date"].to_numpy()[oos]

    print(f"\n=== incremental skill of gamma over VIX/HAR (OOS {pd.Timestamp(dates_oos[0]).date()} -> {pd.Timestamp(dates_oos[-1]).date()}, n={len(oos)}) ===")
    print(f"{'block':10s} {'n':>5s} {'dCRPS':>9s} {'DM':>6s} {'p':>6s}   {'dAUC':>7s} {'p':>6s}   verdict")
    for blk in ["ALL", "pre2020", "2020-21", "2022+"]:
        sel = np.ones(len(oos), bool) if blk == "ALL" else (era_oos == blk)
        cc0, cc1 = c0[oos][sel], c1[oos][sel]
        db, dm, pdm = dm_test(cc0, cc1)
        a_obs, a_p = boot_p(yb[oos][sel], p0[oos][sel], p1[oos][sel], "auc")
        helps = (db > 0 and pdm < 0.05) or (a_obs > 0 and (a_p is not None and a_p < 0.05))
        hurts = (db < 0 and pdm < 0.05)
        v = "GAMMA HELPS" if helps else ("hurts(overfit)" if hurts else "null")
        print(f"{blk:10s} {sel.sum():5d} {db:+9.5f} {dm:+6.2f} {pdm:6.3f}   {a_obs:+7.3f} {a_p if a_p is not None else float('nan'):6.3f}   {v}")

    print("\n=== VERDICT ===")
    db_all, dm_all, p_all = dm_test(c0[oos], c1[oos])
    if db_all > 0 and p_all < 0.05:
        print(f"Gamma adds incremental skill over VIX/HAR on deep history (dCRPS={db_all:+.5f}, DM p={p_all:.3f}).")
    else:
        print(f"NULL on deep history: gamma adds no incremental skill over VIX/HAR (dCRPS={db_all:+.5f}, DM p={p_all:.3f}).")
    print("Per-block results above are the real test — look for a stress-regime (2020-21 / 2022+) effect "
          "that the calm 21-month window could not see. Multiple blocks => read with multiplicity in mind.")
    print(f"NOTE: ~15y, real regimes, %neg-gamma={d['gex_neg'].mean()*100:.1f}%. This is the powered test; "
          "a null here is far more conclusive than the 21-month null.")


if __name__ == "__main__":
    main()
