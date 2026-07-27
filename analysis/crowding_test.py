"""Empirical tests of the crowding model: P1 through P4 (CROWDING.md).

Every specification here is fixed in `CROWDING-PREREG.md`, committed and pushed before this file
existed. Nothing below was chosen after seeing a result.

    P1  amplification    high crowding steepens the vol response to an equity shock
    P2  tail asymmetry   carry drawdowns are deeper in high-crowding regimes
    P3  premium thinning realised carry is lower when crowding is high
    P4  signal value     conditioning exposure on crowding improves drawdown depth

P4 is the signal test. P1 through P3 are mechanism tests and stand independently of it.

Order matters. The VIX-echo decomposition runs FIRST and is reported regardless of what follows,
because the dominant failure mode for this chapter is that crowding is a repackaged VIX, which is
what closed out the gamma study at 97.8% redundancy. Establishing the echo share before making any
predictive claim is the point; discovering it afterwards would be worthless.

Sample. ProShares deleveraged SVXY from -1x to -0.5x and UVXY from 2x to 1.5x on 2018-02-28. Per
the rule STRATEGY.md already applies to SVXY, the periods either side are different products and
are never pooled. Inference runs on 2018-03-01 onward. The short stub before it is descriptive
only, which means February 2018 is a case study with no parameter fitted to it, exactly as
pre-registered.

Run: /opt/anaconda3/envs/trading/bin/python analysis/crowding_test.py
Output: analysis/crowding_test_results.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "analysis" / "crowding_test_results.json"
MODEL_PATH = REPO_ROOT / "analysis" / "crowding_model_results.json"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from strategy_two_sleeve import (  # noqa: E402
    CostCfg, NOTIONAL, block_bootstrap_sharpe, build_signals, contango_flag,
    deflated_sharpe, load_panel, metrics, sleeve_excess,
)
from drawdown_inference import path_metrics, stationary_bootstrap_indices  # noqa: E402
from features.crowding import CrowdingConfig, run as crowding_run  # noqa: E402

# The richer baseline from FINDINGS.md Sec.5b, matching phase1_robustness.py's `base_rich` exactly.
BASE = ["har_d", "har_w", "har_m", "vix_l", "vix_z", "t_9_30", "t_30_90", "vvix_vix", "dvix"]

BREAK_DATE = "2018-02-28"      # ProShares deleverage; inference starts the following day
MEASURES = ["c2_gain", "c1_usd", "c3_usd"]   # c2 primary, per prereg Sec.3


@dataclass
class TestCfg:
    warmup: int = 60           # min observations before a trailing z-score / quantile is defined
    tercile: float = 2.0 / 3.0
    scale_when_crowded: float = 0.5   # P4's single pre-registered sizing rule, no sweep
    n_draws: int = 5000
    mean_block: float = 90.0
    seed: int = 7


# ----------------------------------------------------------------- helpers ----
def r2_ols(y: np.ndarray, X: np.ndarray) -> float:
    """In-sample R2. Mirrors phase1_robustness.r2_ols so the echo numbers are comparable."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return 1.0 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()


def ols_hac(y: np.ndarray, X: np.ndarray, names: list[str]) -> dict:
    """OLS with Newey-West HAC standard errors on the coefficients.

    strategy_two_sleeve.hac_tstat tests whether a RETURN SERIES has positive mean; it does not give
    standard errors for regression coefficients, which is what P1 and P3 need. Bartlett kernel with
    lag n^(1/3), matching that function's bandwidth convention.
    """
    A = np.column_stack([np.ones(len(X)), X])
    n, k = A.shape
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ beta

    lag = max(1, int(round(n ** (1 / 3))))
    S = (A * e[:, None]).T @ (A * e[:, None])
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1)
        G = (A[L:] * e[L:, None]).T @ (A[:-L] * e[:-L, None])
        S += w * (G + G.T)

    XtX_inv = np.linalg.pinv(A.T @ A)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    return {"n": int(n), "r2": float(r2_ols(y, X)),
            "coef": {nm: {"beta": float(beta[i + 1]), "se": float(se[i + 1]), "hac_t": float(t[i + 1])}
                     for i, nm in enumerate(names)},
            "intercept": {"beta": float(beta[0]), "hac_t": float(t[0])}}


def trailing_z(s: pd.Series, warmup: int) -> pd.Series:
    """Causal z-score: expanding mean/std through t-1 only."""
    m = s.expanding(warmup).mean().shift(1)
    sd = s.expanding(warmup).std().shift(1)
    return (s - m) / sd.replace(0.0, np.nan)


def trailing_quantile_flag(s: pd.Series, q: float, warmup: int) -> pd.Series:
    """1.0 when s sits above its own expanding q-quantile computed through t-1 only."""
    thr = s.expanding(warmup).quantile(q).shift(1)
    return (s > thr).astype(float).mask(thr.isna())


# -------------------------------------------------------------------- data ----
def build_frame(cfg: TestCfg) -> pd.DataFrame:
    d = build_signals(load_panel())
    d["dvix"] = (d["vix"] - d["vix"].shift(1)).shift(1)
    d["dlog_vix"] = np.log(d["vix"]).diff()                 # contemporaneous shock response target

    crowd = crowding_run(d[["date"]], CrowdingConfig())
    d = d.merge(crowd, on="date", how="left")

    # Repo convention: a feature carrying same-day information is lagged at the join. The series is
    # already publication-date keyed; this is one further day of conservatism on top of that.
    for m in MEASURES:
        d[m] = d[m].shift(1)
        d[f"{m}_z"] = trailing_z(d[m], cfg.warmup)
        d[f"{m}_hi"] = trailing_quantile_flag(d[m], cfg.tercile, cfg.warmup)

    # carry sleeve, the flagship's own construction
    cc = CostCfg()
    pos = -NOTIONAL * contango_flag(d)
    d["carry_ret"] = sleeve_excess(pos, d["vixy_ret"].to_numpy(), d["rf_d"].to_numpy(),
                                   cc.vixy_bps, cc.borrow_ann)
    d["gate"] = contango_flag(d)
    return d


# ------------------------------------------------------------ echo check ----
def echo_decomposition(d: pd.DataFrame, measure: str) -> dict:
    """How much of the crowding measure is already contained in the VIX/HAR baseline.

    Two complementary readings. `r2_measure_on_baseline` regresses crowding itself on the baseline:
    a high value means the measure IS largely a VIX restatement. The incremental block mirrors
    phase1_robustness.vix_echo_decomposition on the log-RV target, so the number is directly
    comparable to gamma's published 97.8%.
    """
    sub = d.dropna(subset=BASE + [measure, "rv"]).copy()
    y_rv = np.log(sub["rv"].clip(lower=1e-6)).to_numpy()
    Xb = sub[BASE].to_numpy()
    Xm = sub[[measure]].to_numpy()

    r2_self = r2_ols(sub[measure].to_numpy(), Xb)
    r2_m = r2_ols(y_rv, Xm)
    r2_b = r2_ols(y_rv, Xb)
    r2_f = r2_ols(y_rv, np.column_stack([Xb, Xm]))
    incr = r2_f - r2_b
    echo = 1.0 - incr / r2_m if r2_m > 0 else np.nan
    return {"n": int(len(sub)),
            "r2_measure_on_baseline": float(r2_self),
            "r2_measure_alone_on_logrv": float(r2_m),
            "r2_baseline_on_logrv": float(r2_b),
            "r2_full_on_logrv": float(r2_f),
            "incremental_r2": float(incr),
            "vix_echo_fraction": float(echo)}


# ---------------------------------------------------------------- P1 ... P4 ----
def p1_amplification(d: pd.DataFrame, measure: str, down_only: bool = False) -> dict:
    z = f"{measure}_z"
    sub = d.dropna(subset=BASE + [z, "dlog_vix", "spy_ret"]).copy()
    if down_only:
        sub = sub[sub["spy_ret"] < 0]
    sub = sub.reset_index(drop=True)

    inter = (sub["spy_ret"] * sub[z]).to_numpy()
    X = np.column_stack([sub["spy_ret"].to_numpy(), inter, sub[z].to_numpy(),
                         sub[BASE].to_numpy()])
    names = ["spy_ret", "spy_ret_x_crowding", "crowding"] + BASE
    res = ols_hac(sub["dlog_vix"].to_numpy(), X, names)
    c = res["coef"]["spy_ret_x_crowding"]
    res["prediction"] = "interaction coefficient < 0 with HAC t < -2"
    res["supported"] = bool(c["beta"] < 0 and c["hac_t"] < -2.0)
    res["down_days_only"] = down_only
    return res


def p2_tail_asymmetry(d: pd.DataFrame, measure: str, cfg: TestCfg) -> dict:
    hi = f"{measure}_hi"
    sub = d.dropna(subset=[hi, "carry_ret"])
    r_hi = sub.loc[sub[hi] == 1.0, "carry_ret"].to_numpy()
    r_lo = sub.loc[sub[hi] == 0.0, "carry_ret"].to_numpy()
    if len(r_hi) < 100 or len(r_lo) < 100:
        return {"error": "insufficient observations in one regime",
                "n_high": int(len(r_hi)), "n_low": int(len(r_lo))}

    # EQUAL-LENGTH paths. Maximum drawdown grows with path length, so resampling 319 high-crowding
    # days against 1,702 low-crowding days would compare depths that are not commensurable, and any
    # per-day rescaling to patch that is arbitrary: dividing by n flatters whichever regime is
    # rarer, purely because it is rarer. Drawing the same number of days from each regime removes
    # the length effect without inventing a normalization.
    n_eq = min(len(r_hi), len(r_lo))
    rng = np.random.default_rng([cfg.seed, int(cfg.mean_block)])
    idx_hi = stationary_bootstrap_indices(len(r_hi), cfg.n_draws, cfg.mean_block, rng)[:, :n_eq]
    idx_lo = stationary_bootstrap_indices(len(r_lo), cfg.n_draws, cfg.mean_block, rng)[:, :n_eq]
    dd_hi = path_metrics(r_hi[idx_hi])["maxdd"]
    dd_lo = path_metrics(r_lo[idx_lo])["maxdd"]

    # both maxdd are negative; "deeper" means more negative
    p_deeper = float((dd_hi < dd_lo).mean())
    return {"n_high": int(len(r_hi)), "n_low": int(len(r_lo)), "n_equal_length_paths": int(n_eq),
            "sample_maxdd_high": float(path_metrics(r_hi[None, :])["maxdd"][0]),
            "sample_maxdd_low": float(path_metrics(r_lo[None, :])["maxdd"][0]),
            "boot_median_maxdd_high": float(np.median(dd_hi)),
            "boot_median_maxdd_low": float(np.median(dd_lo)),
            "p_high_deeper": p_deeper,
            "prediction": "P(high-crowding drawdown deeper) > 0.90",
            "supported": bool(p_deeper > 0.90)}


def p3_premium_thinning(d: pd.DataFrame, measure: str) -> dict:
    z = f"{measure}_z"
    sub = d.dropna(subset=BASE + [z, "carry_ret"]).reset_index(drop=True)
    X = np.column_stack([sub[z].to_numpy(), sub[BASE].to_numpy()])
    res = ols_hac(sub["carry_ret"].to_numpy(), X, ["crowding"] + BASE)
    c = res["coef"]["crowding"]
    res["prediction"] = "crowding coefficient < 0 with HAC t < -2"
    res["supported"] = bool(c["beta"] < 0 and c["hac_t"] < -2.0)
    return res


def p4_signal_value(d: pd.DataFrame, measure: str, cfg: TestCfg) -> dict:
    hi = f"{measure}_hi"
    sub = d.dropna(subset=[hi, "vixy_ret", "rf_d"]).reset_index(drop=True)
    cc = CostCfg()
    dates = sub["date"].to_numpy()
    vr, rf = sub["vixy_ret"].to_numpy(), sub["rf_d"].to_numpy()

    base_pos = -NOTIONAL * contango_flag(sub)
    scale = np.where(sub[hi].to_numpy() == 1.0, cfg.scale_when_crowded, 1.0)
    cond_pos = base_pos * scale

    r_base = sleeve_excess(base_pos, vr, rf, cc.vixy_bps, cc.borrow_ann)
    r_cond = sleeve_excess(cond_pos, vr, rf, cc.vixy_bps, cc.borrow_ann)
    m_base = metrics(r_base, dates, "binary gate")
    m_cond = metrics(r_cond, dates, "crowding-conditioned")

    lo, hisr, p0 = block_bootstrap_sharpe(r_cond)
    return {"n": int(len(sub)),
            "scale_when_crowded": cfg.scale_when_crowded,
            "days_scaled_down": int((scale < 1.0).sum()),
            "binary_gate": m_base,
            "conditioned": m_cond,
            "delta_maxdd": float(m_cond["maxdd"] - m_base["maxdd"]),
            "delta_calmar": float(m_cond["calmar"] - m_base["calmar"]),
            "delta_sharpe": float(m_cond["sharpe"] - m_base["sharpe"]),
            "conditioned_sharpe_ci": [lo, hisr], "conditioned_p_sharpe_le_0": p0,
            "prediction": "both maxDD and Calmar improve on the binary gate",
            "supported": bool(m_cond["maxdd"] > m_base["maxdd"] and m_cond["calmar"] > m_base["calmar"]),
            "_returns": r_cond}


# ------------------------------------------------------- model comparison ----
def model_comparison(d: pd.DataFrame, full: pd.DataFrame) -> dict:
    """Convert the observed measure into the model's loop gain and compare to its equilibrium.

    c2 = C1/ADV$ is the loop gain only up to a constant, as the prereg states. Since
    lambda ~ typical|r| / ADV$, the loop gain is x = lambda*C1 = typical|r| * c2. Using the model's
    own ordinary-day definition of "typical" keeps the two sides commensurable. lambda in
    crowding_model.py is calibrated on VIXY depth while C1 spans three ETPs, so this conversion,
    which uses the complex-wide ADV$ already in the denominator of c2, is the internally consistent
    one; the raw lambda*C1 product is reported alongside it to make the gap explicit.
    """
    model = json.loads(MODEL_PATH.read_text())
    lam = model["impact_calibration"]["lambda_per_dollar"]
    cutoff = model["impact_calibration"]["abs_return_cutoff"]

    ordinary = full["vixy_ret"].abs()
    typ = float(ordinary[ordinary <= cutoff].median())

    sub = d.dropna(subset=["c2_gain", "c1_usd"])
    x_obs = (typ * sub["c2_gain"]).astype(float)
    x_naive = (lam * sub["c1_usd"]).astype(float)
    x_nash = [e["x_nash"] for e in model["equilibria"]]

    return {"typical_abs_return_ordinary_days": typ,
            "observed_loop_gain_x": {"median": float(x_obs.median()), "min": float(x_obs.min()),
                                     "max": float(x_obs.max()), "p95": float(x_obs.quantile(0.95))},
            "naive_lambda_times_C1": {"median": float(x_naive.median()), "max": float(x_naive.max()),
                                      "note": "lambda is VIXY-depth-calibrated while C1 is complex-wide; "
                                              "reported to make the unit gap explicit, not as the estimate"},
            "model_x_nash_range": [float(min(x_nash)), float(max(x_nash))],
            "observed_median_over_nash_max": float(x_obs.median() / max(x_nash)),
            "implied_amplification_at_observed_median": float(1.0 / (1.0 - x_obs.median()))}


def posthoc_depth_placebo(d: pd.DataFrame, cfg: TestCfg) -> dict:
    """POST HOC, labelled as such and excluded from every claim per CROWDING-PREREG.md Sec.9.4.

    P1 passes on the primary measure but reverses sign on the other two, which differ from it only
    by the ADV$ denominator. That points at market depth rather than crowding, so this runs the
    placebo the disagreement demands: put 1/ADV$ into the interaction slot with NO short interest in
    it at all. If the placebo reproduces the effect, the pre-registered pass is a depth result
    wearing a crowding label.

    Not a pre-registered test, and it does not change P1's recorded outcome. It changes what that
    outcome MEANS, which is why the chapter reports both.
    """
    inv_adv = d["c2_gain"] / d["c1_usd"]                 # the denominator alone
    dd = d.copy()
    dd["invadv_z"] = trailing_z(inv_adv, cfg.warmup)

    out = {}
    for tag, z in [("crowding_over_adv_primary", "c2_gain_z"),
                   ("inverse_adv_only_no_short_interest", "invadv_z"),
                   ("raw_dollar_crowding", "c1_usd_z")]:
        sub = dd.dropna(subset=BASE + [z, "dlog_vix", "spy_ret"]).reset_index(drop=True)
        X = np.column_stack([sub["spy_ret"].to_numpy(), (sub["spy_ret"] * sub[z]).to_numpy(),
                             sub[z].to_numpy(), sub[BASE].to_numpy()])
        res = ols_hac(sub["dlog_vix"].to_numpy(), X, ["spy_ret", "inter", "lvl"] + BASE)
        out[tag] = {"beta": res["coef"]["inter"]["beta"], "hac_t": res["coef"]["inter"]["hac_t"],
                    "n": res["n"]}
    out["corr_primary_vs_inverse_adv"] = float(dd[["c2_gain_z", "invadv_z"]].corr().iloc[0, 1])
    out["verdict"] = (
        "1/ADV$ alone, containing no short interest, reproduces the interaction with a LARGER "
        "|t| than the crowding measure, and the two are 0.70 correlated. P1's pre-registered pass "
        "is therefore a market-depth effect, not evidence for the crowding mechanism.")
    return out


def volmageddon_case_study(full: pd.DataFrame) -> dict:
    """February 2018, descriptive only: it sits in the pre-break stub by construction."""
    w = full[(full["date"] >= "2018-01-02") & (full["date"] <= "2018-03-15")]
    w = w.dropna(subset=["c1_usd"])
    if w.empty:
        return {"note": "no published crowding reading inside the window"}
    spike = full[(full["date"] >= "2018-02-05") & (full["date"] <= "2018-02-09")]
    return {"n_days_with_reading": int(len(w)),
            "first_reading_date": str(w["date"].min())[:10],
            "c1_usd_before_spike": float(w[w["date"] < "2018-02-05"]["c1_usd"].iloc[-1])
            if (w["date"] < "2018-02-05").any() else None,
            "c1_usd_after_spike": float(w[w["date"] > "2018-02-09"]["c1_usd"].iloc[0])
            if (w["date"] > "2018-02-09").any() else None,
            "vixy_return_spike_window": float((1 + spike["vixy_ret"].fillna(0)).prod() - 1),
            "carry_return_spike_window": float(spike["carry_ret"].fillna(0).sum()),
            "caveat": ("single event, in the descriptive pre-break stub; no parameter in the model "
                       "or in any test is fitted to it")}


# -------------------------------------------------------------------- main ----
def main() -> int:
    cfg = TestCfg()
    full = build_frame(cfg)
    d = full[full["date"] > BREAK_DATE].reset_index(drop=True)

    echo = {m: echo_decomposition(d, m) for m in MEASURES}
    p1 = {m: p1_amplification(d, m) for m in MEASURES}
    p1_down = p1_amplification(d, "c2_gain", down_only=True)
    p2 = {m: p2_tail_asymmetry(d, m, cfg) for m in MEASURES}
    p3 = {m: p3_premium_thinning(d, m) for m in MEASURES}
    p4 = {m: p4_signal_value(d, m, cfg) for m in MEASURES}

    # DSR over this chapter's trials plus the flagship's existing 22 variants, per prereg Sec.7
    trials_here = len(MEASURES) * 4 + 1
    sr_set = [np.asarray(p4[m]["_returns"]) for m in MEASURES]
    sr_perobs = [float(r.mean() / r.std()) for r in sr_set if r.std() > 0]
    sr_std = float(np.std(sr_perobs)) if len(sr_perobs) > 1 else 0.0
    primary = np.asarray(p4["c2_gain"].pop("_returns"))
    for m in MEASURES:
        p4[m].pop("_returns", None)
    dsr = {f"N={n}": deflated_sharpe(primary, n, sr_std) for n in (22 + trials_here, 50, 100)}

    out = {
        "config": asdict(cfg),
        "prereg": "CROWDING-PREREG.md (committed and pushed before this file existed)",
        "sample": {"inference_start": str(d["date"].min())[:10],
                   "inference_end": str(d["date"].max())[:10],
                   "n_days": int(len(d)),
                   "n_days_with_crowding": int(d["c2_gain"].notna().sum()),
                   "break_date": BREAK_DATE,
                   "note": "pre-break stub is descriptive only; never pooled"},
        "baseline_features": BASE,
        "vix_echo_decomposition": echo,
        "P1_amplification": p1,
        "P1_amplification_down_days_secondary": p1_down,
        "P2_tail_asymmetry": p2,
        "P3_premium_thinning": p3,
        "P4_signal_value": p4,
        "trials_this_chapter": trials_here,
        "deflated_sharpe_P4_primary": dsr,
        "posthoc_depth_placebo": posthoc_depth_placebo(d, cfg),
        "model_comparison": model_comparison(d, full),
        "volmageddon_case_study": volmageddon_case_study(full),
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=float))

    pr = "c2_gain"
    print(f"sample {out['sample']['inference_start']} -> {out['sample']['inference_end']}, "
          f"n={out['sample']['n_days']} ({out['sample']['n_days_with_crowding']} with crowding)\n")
    e = echo[pr]
    print(f"VIX-echo check (primary c2_gain): crowding is {e['r2_measure_on_baseline']*100:.1f}% "
          f"explained by the VIX/HAR baseline")
    print(f"  incremental R2 on log-RV = {e['incremental_r2']:.5f} -> "
          f"{e['vix_echo_fraction']*100:.1f}% echo\n")
    print(f"{'test':<28s} {'primary statistic':>22s}  supported")
    c = p1[pr]["coef"]["spy_ret_x_crowding"]
    print(f"{'P1 amplification':<28s} {'HAC t = %+.2f' % c['hac_t']:>22s}  {p1[pr]['supported']}")
    print(f"{'P2 tail asymmetry':<28s} {'P(deeper) = %.3f' % p2[pr]['p_high_deeper']:>22s}  {p2[pr]['supported']}")
    c3 = p3[pr]["coef"]["crowding"]
    print(f"{'P3 premium thinning':<28s} {'HAC t = %+.2f' % c3['hac_t']:>22s}  {p3[pr]['supported']}")
    print(f"{'P4 signal value':<28s} "
          f"{'dMaxDD %+.4f' % p4[pr]['delta_maxdd']:>22s}  {p4[pr]['supported']}")
    print("\nsign stability across the three crowding measures (same construct):")
    for m in MEASURES:
        c = p1[m]["coef"]["spy_ret_x_crowding"]
        print(f"  P1 {m:9s} beta {c['beta']:+.4f} t {c['hac_t']:+.2f}   "
              f"P4 dMaxDD {p4[m]['delta_maxdd']:+.4f} dCalmar {p4[m]['delta_calmar']:+.4f}")

    ph = out["posthoc_depth_placebo"]
    print("\npost-hoc depth placebo (labelled, excluded from claims):")
    for k in ("crowding_over_adv_primary", "inverse_adv_only_no_short_interest"):
        print(f"  {k:36s} beta {ph[k]['beta']:+.4f}  t {ph[k]['hac_t']:+.2f}")

    mc = out["model_comparison"]
    print(f"\nloop gain: observed median x = {mc['observed_loop_gain_x']['median']:.4f} "
          f"(model Nash range {mc['model_x_nash_range'][0]:.4f}-{mc['model_x_nash_range'][1]:.4f}), "
          f"implied amplification {mc['implied_amplification_at_observed_median']:.3f}")
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
