"""Robustness on the deep-history incremental result: WHAT drives the +skill?

Critical confound: SqueezeMetrics ships gex (gamma) AND dix (Dark Index = a
short-volume/flow signal, NOT gamma). The headline "+gamma" model used both.
Decompose so we credit the right signal, and check the increment survives a
richer VIX baseline (so it isn't just 'VIX moved today' in disguise).
"""
from __future__ import annotations
import json

import numpy as np
from phase1_deep_history import load_panel, wf_ols_crps, wf_logit, dm_test, cw_test, boot_p, TRAIN0

REPO = __file__.rsplit("/analysis/", 1)[0]


def build(df):
    lrv = np.log(df["rv"].clip(lower=1e-6))
    df["har_d"] = lrv.shift(1); df["har_w"] = lrv.rolling(5).mean().shift(1); df["har_m"] = lrv.rolling(22).mean().shift(1)
    df["vix_l"] = df["vix"].shift(1)
    df["vix_z"] = ((df["vix"] - df["vix"].rolling(20).mean()) / df["vix"].rolling(20).std()).shift(1)
    df["t_9_30"] = (df["vix9d"] / df["vix"]).shift(1)
    df["t_30_90"] = (df["vix"] / df["vix3m"]).shift(1)
    df["vvix_vix"] = (df["vvix"] / df["vix"]).shift(1)
    df["dvix"] = (df["vix"] - df["vix"].shift(1)).shift(1)          # richer VIX: yesterday's VIX change
    df["gex_pct"] = df["gex"].rolling(252, min_periods=60).apply(lambda a: (a[-1] > a).mean(), raw=True).shift(1)
    df["gex_neg"] = (df["gex"] < 0).astype(float).shift(1)
    df["dix_l"] = df["dix"].shift(1)
    df["y"] = lrv
    df["y_bin"] = (df["rv"] > df["rv"].rolling(21).mean().shift(1)).astype(float)
    return df


def eval_add(d, base_f, add_f, tag):
    Xb = d[base_f].to_numpy(); Xa = d[base_f + add_f].to_numpy()
    y = d["y"].to_numpy(); yb = d["y_bin"].to_numpy(); oos = np.arange(TRAIN0, len(d))
    c0, pred0 = wf_ols_crps(Xb, y); c1, pred1 = wf_ols_crps(Xa, y)
    db, dm, pdm = dm_test(c0[oos], c1[oos])
    _, cw, pcw = cw_test(y[oos], pred0[oos], pred1[oos])
    p0 = wf_logit(Xb, yb); p1 = wf_logit(Xa, yb)
    a, ap = boot_p(yb[oos], p0[oos], p1[oos], "auc")
    print(f"{tag:32s} dCRPS={db:+.5f} DM={dm:+.2f} p(2s)={pdm:.3f} CW={cw:+.2f} p(1s)={pcw:.3f}   dAUC={a:+.3f} p={ap:.3f}")
    return {"tag": tag, "dcrps": float(db), "dm": float(dm), "dm_p": float(pdm),
            "cw": float(cw), "cw_p": float(pcw), "dauc": float(a), "dauc_p": float(ap)}


def r2_ols(y, X):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return 1.0 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()


def vix_echo_decomposition(d, base_f, gam_f):
    """In-sample variance decomposition (not a forecast; a redundancy check): how much of
    gamma's own explanatory power for log-RV is already contained in the VIX/HAR baseline.
    echo_frac = 1 - (R2_full - R2_base) / R2_gamma_alone."""
    y = d["y"].to_numpy()
    r2_gamma = r2_ols(y, d[gam_f].to_numpy())
    r2_base = r2_ols(y, d[base_f].to_numpy())
    r2_full = r2_ols(y, d[base_f + gam_f].to_numpy())
    incr = r2_full - r2_base
    echo_frac = 1.0 - incr / r2_gamma
    print(f"\n=== VIX-echo decomposition (in-sample R2, n={len(d)}) ===")
    print(f"  R2(gamma alone)={r2_gamma:.4f}  R2(VIX/HAR)={r2_base:.4f}  R2(full)={r2_full:.4f}")
    print(f"  incremental R2 over VIX/HAR = {incr:.4f}  ->  gamma is {echo_frac*100:.1f}% a VIX echo, "
          f"{(1-echo_frac)*100:.1f}% incremental.")
    return {"r2_gamma_alone": float(r2_gamma), "r2_vix_har": float(r2_base), "r2_full": float(r2_full),
            "incremental_r2": float(incr), "vix_echo_fraction": float(echo_frac)}


def main():
    d = build(load_panel())
    vixf = ["vix_l", "vix_z", "t_9_30", "t_30_90", "vvix_vix"]
    base = ["har_d", "har_w", "har_m"] + vixf
    base_rich = base + ["dvix"]
    d = d.dropna(subset=base_rich + ["gex_pct", "gex_neg", "dix_l", "y", "y_bin"]).reset_index(drop=True)
    print(f"rows {len(d)}  {d['date'].min().date()}->{d['date'].max().date()}  (OOS n={len(d)-TRAIN0})\n")

    print("=== decompose the increment over VIX/HAR (full OOS) ===")
    results = {}
    results["gamma_only"] = eval_add(d, base, ["gex_pct", "gex_neg"], "+ gamma only (gex_pct,gex_neg)")
    results["dix_only"] = eval_add(d, base, ["dix_l"], "+ DIX only (dix)")
    results["gamma_plus_dix"] = eval_add(d, base, ["gex_pct", "gex_neg", "dix_l"], "+ gamma + DIX (headline)")
    print("\n=== does the gamma increment survive a richer VIX baseline (+dVIX)? ===")
    results["gamma_only_rich_vix"] = eval_add(d, base_rich, ["gex_pct", "gex_neg"], "+ gamma only, over VIX+dVIX")
    results["gamma_plus_dix_rich_vix"] = eval_add(d, base_rich, ["gex_pct", "gex_neg", "dix_l"], "+ gamma+DIX, over VIX+dVIX")
    print("\n(If 'gamma only' is ~null and 'DIX only' carries it -> credit DIX/flow, not gamma.\n"
          " If gamma-only survives over VIX+dVIX -> a real, if small, gamma increment.)")
    print("NOTE: DIX-only DM (p=0.97, null) and CW (p=0.007, significant) disagree; the two tests are not "
          "resolved here and DIX-only is reported with both, not collapsed to a single verdict.")

    results["vix_echo"] = vix_echo_decomposition(d, base, ["gex_pct", "gex_neg"])
    with open(f"{REPO}/analysis/phase1_robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved analysis/phase1_robustness_results.json")


if __name__ == "__main__":
    main()
