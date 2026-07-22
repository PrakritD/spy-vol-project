"""Does 25-delta put-call skew carry next-day RV information incremental to HAR+VIX?

Same kill-switch-first question phase0_gonogo.py asks of dealer gamma, asked here of the
options-market's own risk-reversal (25-delta put IV minus 25-delta call IV, features/skew.py),
on the same OPRA statistics pull (data already paid for, ~2024-08 -> 2026-04).

Method (identical protocol to phase0_gonogo.py, so the two are directly comparable):
- Target: log RV[t] (one-step), predictors strictly <= t-1 (no lookahead, already shift(1)-ed
  in features/assemble.py).
- Nested models, expanding walk-forward, Gaussian predictive density:
    M0  HAR-X + VIX            (the bar; identical feature set to phase0's M0)
    M1  + skew_25d LEVEL       (put IV - call IV, raw vol points)
    M2  + skew_25d_norm        (skew_25d / atm_iv, level-normalized)
- Headline metric: Diebold-Mariano test on the CRPS differential (M0 - Mk),
  Newey-West HAC + Harvey small-sample correction. Positive & significant => skew helps.

Caveat baked into the verdict, same as phase0's: short window (~21 months on disk), one
mostly-calm regime. A positive is encouraging; a null is a caution, not proof of absence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = __file__.rsplit("/analysis/", 1)[0]
PANEL = f"{REPO}/data/processed/features_panel.parquet"


def gaussian_crps(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-9)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def ols_fit_predict(Xtr, ytr, Xte):
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    resid = ytr - Xtr @ beta
    sigma = resid.std(ddof=Xtr.shape[1])
    return Xte @ beta, sigma, beta


def newey_west_var(d, lag):
    d = d - d.mean()
    n = len(d)
    g0 = (d @ d) / n
    s = g0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        gk = (d[k:] @ d[:-k]) / n
        s += 2 * w * gk
    return s / n


def dm_test(loss0, loss1, h=1):
    d = loss0 - loss1
    n = len(d)
    lag = max(1, int(round(n ** (1 / 3))))
    var = newey_west_var(d, lag)
    stat = d.mean() / np.sqrt(var)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= corr
    p = 2 * (1 - norm.cdf(abs(stat)))
    return d.mean(), stat, p


def main():
    df = pd.read_parquet(PANEL).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    lrv = np.log(df["rv"])

    df["har_d"] = lrv.shift(1)
    df["har_w"] = lrv.rolling(5).mean().shift(1)
    df["har_m"] = lrv.rolling(22).mean().shift(1)
    df["y"] = lrv

    vixf = ["vix_level_lag1", "vix_zscore_lag1", "term_9d_30d_lag1",
            "term_30d_90d_lag1", "vvix_vix_lag1"]
    base_f = ["har_d", "har_w", "har_m"] + vixf

    d = df.dropna(subset=base_f + ["y", "skew_25d_lag1", "skew_25d_norm_lag1"]).reset_index(drop=True)
    print(f"modeling rows: {len(d)}  ({d['date'].min().date()} -> {d['date'].max().date()})")
    print(f"skew_25d_lag1: mean={d['skew_25d_lag1'].mean():+.4f} std={d['skew_25d_lag1'].std():.4f}")

    specs = {
        "M0_HARX_VIX": base_f,
        "M1_+skew25d": base_f + ["skew_25d_lag1"],
        "M2_+skew25d_norm": base_f + ["skew_25d_norm_lag1"],
    }

    y = d["y"].to_numpy()
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
    yo = y[idx]
    crps = {k: gaussian_crps(yo, preds[k][idx], sigmas[k][idx]) for k in specs}
    rmse = {k: np.sqrt(np.mean((yo - preds[k][idx]) ** 2)) for k in specs}

    print(f"\n=== OOS forecast quality (lower CRPS/RMSE better; N_oos={len(idx)}) ===")
    print(f"{'model':20s} {'meanCRPS':>10s} {'RMSE':>8s}")
    for k in specs:
        print(f"{k:20s} {crps[k].mean():10.4f} {rmse[k]:8.4f}")

    print("\n=== Diebold-Mariano on CRPS differential vs M0 (positive & p<.05 => skew helps) ===")
    dm_results = {}
    for k in specs:
        if k == "M0_HARX_VIX":
            continue
        dbar, stat, p = dm_test(crps["M0_HARX_VIX"], crps[k])
        dm_results[k] = {"dcrps": float(dbar), "dm_stat": float(stat), "dm_p": float(p)}
        flag = "  <-- skew helps" if (dbar > 0 and p < 0.05) else ("  (favors M0)" if dbar < 0 else "")
        print(f"{k:20s} dCRPS={dbar:+.5f}  DM={stat:+.2f}  p={p:.3f}{flag}")

    print("\n=== VERDICT ===")
    best = max(dm_results.items(), key=lambda kv: kv[1]["dcrps"])
    k, r = best
    if r["dcrps"] > 0 and r["dm_p"] < 0.05:
        print(f"GO (tentative): {k} beats HAR-X+VIX, dCRPS={r['dcrps']:+.5f}, DM p={r['dm_p']:.3f}.")
    elif r["dcrps"] > 0:
        print(f"WEAK/INCONCLUSIVE: best skew model ({k}) improves CRPS (dCRPS={r['dcrps']:+.5f}) "
              f"but not significantly (DM p={r['dm_p']:.3f}).")
    else:
        print("NULL: no skew formulation beats HAR-X+VIX on CRPS in this window.")

    out = {
        "window": [str(d["date"].min().date()), str(d["date"].max().date())],
        "n_total": int(n), "n_oos": int(len(idx)), "train0": train0,
        "skew_25d_lag1_vol_pts": {
            "mean": float(d["skew_25d_lag1"].mean()), "std": float(d["skew_25d_lag1"].std()),
            "min": float(d["skew_25d_lag1"].min()), "max": float(d["skew_25d_lag1"].max()),
            "frac_positive": float((d["skew_25d_lag1"] > 0).mean()),
        },
        "mean_crps": {k: float(crps[k].mean()) for k in specs},
        "rmse": {k: float(rmse[k]) for k in specs},
        "dm_vs_m0": dm_results,
    }
    import json
    with open(f"{REPO}/analysis/phase_skew_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nsaved analysis/phase_skew_results.json")


if __name__ == "__main__":
    main()
