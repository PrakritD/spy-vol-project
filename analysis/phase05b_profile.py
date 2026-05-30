"""Phase 0.5b — the LAST test on the 21-month window: gamma PROFILE SHAPE (F3).

Phase 0 (level) and Phase 0.5 (path/dynamics/tails/regime) both nulled. The one
genuinely distinct, most-novel module still untested on this window is the SHAPE of
the by-strike gamma profile (call/put walls, pinning channel, concentration, skew) —
not the scalar net level.

PRE-REGISTERED (a priori from the pinning mechanism): a tight, high-concentration gamma
channel with spot centered between walls -> price PINS -> smaller next-day move / range.
Profile features tested INCREMENTAL to [HAR + VIX + NET gamma], so they must add beyond
the net level already shown to be a VIX echo. All reported, Bonferroni-corrected.

Targets (pinning's most direct predictions):
  P1  log intraday range (Parkinson)      -- pinning compresses range   [DM-on-CRPS]
  P2  log |next-day return| (move size)    -- pinning shrinks the move    [DM-on-CRPS]
  P3  RV-regime direction (y_bin)          -- secondary                   [logit + bootstrap]

Underpowered, one calm regime: any hit is a LEAD for deep-history/intraday, not a result.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

REPO = __file__.rsplit("/analysis/", 1)[0]
sys.path.insert(0, REPO)
from features.gex import GexConfig, compute_contract_greeks, filter_contracts  # noqa: E402

PANEL = f"{REPO}/data/processed/features_panel.parquet"
OPT = f"{REPO}/data/processed/options_panel.parquet"
SPY = f"{REPO}/data/raw/yfinance/SPY.parquet"
CACHE = f"{REPO}/data/interim/contract_greeks_filtered.parquet"
GSCALE = 1e9
TRAIN0 = 150


# ---------- shared helpers (mirror phase05) ----------
def gaussian_crps(y, mu, s):
    s = np.maximum(s, 1e-9); z = (y - mu) / s
    return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def _nw_var(d, lag):
    d = d - d.mean(); n = len(d); s = (d @ d) / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return s / n


def dm_test(l0, l1, h=1):
    d = l0 - l1; n = len(d); lag = max(1, int(round(n ** (1 / 3))))
    st = d.mean() / np.sqrt(_nw_var(d, lag)) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return d.mean(), st, 2 * (1 - norm.cdf(abs(st)))


def wf_ols_crps(d, feats, tgt):
    y = d[tgt].to_numpy(); X = d[feats].to_numpy(); n = len(d); crps = np.full(n, np.nan)
    for i in range(TRAIN0, n):
        Xtr, ytr = X[:i], y[:i]; mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
        A = np.column_stack([np.ones(i), (Xtr - mu) / sd]); b, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        sig = (ytr - A @ b).std(ddof=A.shape[1]); xte = np.concatenate([[1.0], (X[i] - mu) / sd])
        crps[i] = gaussian_crps(y[i], xte @ b, sig)
    return crps[TRAIN0:], y[TRAIN0:]


def wf_logit(d, feats, tgt):
    y = d[tgt].to_numpy(); X = d[feats].to_numpy(); n = len(d); p = np.full(n, np.nan)
    for i in range(TRAIN0, n):
        ytr = y[:i]
        if len(np.unique(ytr)) < 2:
            p[i] = ytr.mean(); continue
        Xtr = X[:i]; mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
        clf = LogisticRegression(C=1.0, max_iter=2000).fit((Xtr - mu) / sd, ytr)
        p[i] = clf.predict_proba(((X[i] - mu) / sd)[None, :])[0, 1]
    return p[TRAIN0:], y[TRAIN0:]


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
    return obs, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())


# ---------- greeks (cached) ----------
def get_greeks():
    import os
    if os.path.exists(CACHE):
        print("loading cached greeks…")
        return pd.read_parquet(CACHE)
    print("computing contract greeks (IV inversion; cached after first run)…")
    op = pd.read_parquet(OPT)
    op["date"] = pd.to_datetime(op["date"]); op["expiry"] = pd.to_datetime(op["expiry"])
    dte = (op["expiry"] - op["date"]).dt.days
    op = op[(dte >= 5) & (dte <= 65)].reset_index(drop=True)  # prefilter to cut IV-inversion cost
    print(f"  contracts after DTE prefilter: {len(op)}")
    cfg = GexConfig()
    g = compute_contract_greeks(op, cfg)
    g = filter_contracts(g, cfg)
    g["contract_gex"] = g["gamma"] * g["open_interest"] * g["spot"] ** 2 * cfg.multiplier * 0.01
    keep = ["date", "strike", "option_type", "spot", "contract_gex"]
    g = g[keep].copy()
    g.to_parquet(CACHE)
    print(f"  cached {len(g)} filtered contract-days -> {CACHE}")
    return g


def build_profile(g):
    rows = []
    for dt, day in g.groupby("date", sort=True):
        spot = day["spot"].iloc[0]
        calls = day[day["option_type"].str.upper() == "C"]
        puts = day[day["option_type"].str.upper() == "P"]
        tot = day["contract_gex"].sum()
        if tot <= 0 or calls.empty or puts.empty:
            continue
        cw = calls.loc[calls["contract_gex"].idxmax(), "strike"]
        pw = puts.loc[puts["contract_gex"].idxmax(), "strike"]
        herf = ((day["contract_gex"] / tot) ** 2).sum()
        com = (day["strike"] * day["contract_gex"]).sum() / tot  # gamma center of mass
        width = (cw - pw) / spot
        rows.append({
            "date": dt,
            "dist_call_wall": (cw - spot) / spot,
            "dist_put_wall": (spot - pw) / spot,
            "channel_width": width,
            "spot_in_channel": (spot - pw) / (cw - pw) if cw != pw else 0.5,
            "gamma_herf": herf,
            "gamma_com_rel": com / spot - 1.0,
        })
    return pd.DataFrame(rows)


def main():
    g = get_greeks()
    prof = build_profile(g)
    print(f"profile days: {len(prof)} ({prof['date'].min().date()}->{prof['date'].max().date()})")

    df = pd.read_parquet(PANEL).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    spy = pd.read_parquet(SPY); spy["date"] = pd.to_datetime(spy["date"])
    df = df.merge(spy[["date", "open", "high", "low", "close"]], on="date", how="left")
    df = df.merge(prof, on="date", how="left")

    lrv = np.log(df["rv"])
    df["har_d"] = lrv.shift(1); df["har_w"] = lrv.rolling(5).mean().shift(1); df["har_m"] = lrv.rolling(22).mean().shift(1)
    df["lpark"] = 0.5 * np.log(((np.log(df["high"] / df["low"]) ** 2) / (4 * np.log(2))).clip(lower=1e-10))
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["labsret"] = np.log(df["ret"].abs().clip(lower=1e-5))
    base_reg = df["rv"].rolling(21).mean().shift(1)
    df["y_bin"] = (df["rv"] > base_reg).astype(float)
    df["g_level"] = df["gex_net_lag1"] / GSCALE

    prof_cols = ["dist_call_wall", "dist_put_wall", "channel_width", "spot_in_channel", "gamma_herf", "gamma_com_rel"]
    # lag the profile by 1 day (no lookahead: profile known at close t-1 predicts day t)
    for c in prof_cols:
        df[c] = df[c].shift(1)

    vixf = ["vix_level_lag1", "vix_zscore_lag1", "term_9d_30d_lag1", "term_30d_90d_lag1", "vvix_vix_lag1"]
    need = ["har_d", "har_w", "har_m", "lpark", "labsret", "y_bin", "g_level"] + vixf + prof_cols
    d = df.dropna(subset=need).reset_index(drop=True)
    d = d[d["gex_net_lag1"].notna()].reset_index(drop=True)
    print(f"modeling rows (profile available, lagged): {len(d)} ({d['date'].min().date()}->{d['date'].max().date()})")

    base = ["har_d", "har_w", "har_m"] + vixf + ["g_level"]   # includes NET gamma; profile must add beyond it
    results = []

    for tgt, label in [("lpark", "P1 intraday-range"), ("labsret", "P2 move-size")]:
        c0, _ = wf_ols_crps(d, base, tgt)
        c1, _ = wf_ols_crps(d, base + prof_cols, tgt)
        db, st, p = dm_test(c0, c1)
        print(f"[{label}] meanCRPS base={c0.mean():.4f} +profile={c1.mean():.4f}  dCRPS={db:+.5f} DM={st:+.2f} p={p:.3f}")
        results.append((label, f"dCRPS={db:+.5f}", p, db))  # 4th elem = signed effect (>0 => profile helps)

    p0, yb = wf_logit(d, base, "y_bin")
    p1, _ = wf_logit(d, base + prof_cols, "y_bin")
    a_obs, a_p = boot_p(yb, p0, p1, "auc")
    l_obs, l_p = boot_p(yb, p0, p1, "logloss")
    print(f"[P3 regime-direction] dAUC={a_obs:+.3f} (p={a_p:.3f})  dLogLik={l_obs:+.4f} (p={l_p:.3f})")
    results.append(("P3 regime-direction (AUC)", f"dAUC={a_obs:+.3f}", a_p, a_obs))  # signed: >0 => helps

    m = len(results); bonf = 0.05 / m
    print(f"\n=== F3 PROFILE FAMILY ({m} tests; Bonferroni alpha={bonf:.4f}) ===")
    print(f"{'target':28s} {'effect':22s} {'p':>7s}  {'verdict':>14s}")
    for nm, eff, p, sgn in results:
        if p < bonf:
            v = "HELPS" if sgn > 0 else "HURTS(overfit)"
        else:
            v = "no"
        print(f"{nm:28s} {eff:22s} {p:7.3f}  {v:>14s}")
    # a LEAD requires the effect to be POSITIVE (gamma helps) AND significant — sign matters.
    hits = [r for r in results if r[2] < bonf and r[3] > 0]
    raw = [r for r in results if r[2] < 0.05 and r[3] > 0]
    harmful = [r for r in results if r[2] < bonf and r[3] < 0]
    print("\n=== VERDICT ===")
    if hits:
        print("PROFILE LEAD (positive & Bonferroni-robust):", ", ".join(h[0] for h in hits), "-> confirm on deep/intraday.")
    elif raw:
        print("Suggestive only (positive, raw p<.05, not Bonferroni):", ", ".join(h[0] for h in raw))
    else:
        print("NULL: gamma profile SHAPE adds NO positive signal beyond VIX/HAR/net-gamma on this window.")
    if harmful:
        print("Note: profile features significantly DEGRADE", ", ".join(h[0] for h in harmful),
              "(small-N overfit — exactly the F3 risk the spec flagged).")
    print("=> If null, the 21-month window is EXHAUSTED (level, path, dynamics, tails, regime, shape "
          "all tested). Remaining real tests: intraday (timescale) or deep-history (power) — or write up the null.")


if __name__ == "__main__":
    main()
