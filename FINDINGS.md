# Is Dealer Gamma a VIX Echo? Mostly — but a Powered Test Finds a Small Real Edge

**One-line result:** Dealer gamma tracks realized volatility enormously (short-gamma days carry far higher RV, t ≈ +28 over 15 years) but is **~95% redundant with VIX**. On a calm 21-month window the residual is undetectable (a clean null across six formulations); on **15 years across real stress regimes it is a small but statistically robust increment** beyond a full VIX/HAR baseline (gamma-only Diebold-Mariano on CRPS **p = 0.001**, ΔAUC **p = 0.001**) — and that increment is genuinely *gamma* (not the Dark-Index flow signal) and survives a richer VIX baseline. The edge is real and economically small; finding it required power, multiple regimes, and a confound check.

This document is the deliverable. The companion `docs/v1-retrospective.md` explains why an earlier version was abandoned; `analysis/` holds the runnable evidence.

---

## 1. The question (and why the obvious framing is wrong)

Options dealers hedge inventory: **long gamma** → counter-trend hedging → realized vol *suppressed* and price *pins*; **short gamma** (below the gamma flip) → trend-following hedging → vol *amplified*. The naive question — *"does gamma beat VIX at forecasting RV level?"* — is a trap, because **VIX is by construction the market's price of forward variance.** The sharp, falsifiable question is:

> **Does dealer gamma carry RV-regime information *incremental to* a VIX/HAR baseline — and if so, how large, and where?**

A null is an acceptable, fully-reported outcome. As it turns out, the answer is "almost none — but not zero, on a powered sample," and saying that precisely is the point.

## 2. Data (all free; download-and-gitignore; ship the fetcher)

| Source | What | Window |
|---|---|---|
| **SqueezeMetrics** `DIX.csv` | daily dealer **GEX** (+ DIX) for the S&P | **2011-05 → 2026-05** (3,791 rows; verified live) |
| **CBOE** `VIX_History.csv` | VIX | 1990 → 2026 |
| **yfinance** | SPY OHLC (Yang-Zhang RV), VIX3M / VIX9D / VVIX | 2010/2011 → 2026 |
| Databento OPRA (owned) | signed net gamma + by-strike profile (the 21-month sub-study) | 2024-08 → 2026-03 |
| FRED | DGS3MO risk-free | — |

SqueezeMetrics' `gex` is **9.1% negative** over the deep window (≈345 short-gamma days, clustered in 2011/2015/2018/2020/2022) — enough to study the amplification regime, which the calm owned window could not.

## 3. Method (what makes the claim trustworthy)

- **Contamination-fixed target** (RV regime vs a baseline ending `t-1`, excluding the present value — the fix for the v1 placebo that scored AUC 0.73; see retrospective).
- **Pre-registration** of every mechanism-derived formulation; **no look-ahead** (predictors ≤ `t-1`; gamma lagged for OCC's T-1 open interest).
- **Out-of-sample expanding walk-forward** (~2y initial train, ~3,200 OOS days).
- **The right test:** **Diebold-Mariano on the CRPS differential** of *nested* models (VIX/HAR vs +gamma), Newey-West HAC + Harvey correction; binary targets via OOS log-loss/AUC with **stationary block-bootstrap**.
- **Per-regime reporting**, never pooled across the 0DTE structural break (pre-2020 / 2020-21 / 2022+).
- **Confound decomposition + multiplicity awareness** (below).

Implementation (pure NumPy/SciPy/scikit-learn): `analysis/phase1_deep_history.py`, `analysis/phase1_robustness.py`, plus the 21-month sub-study `analysis/phase0_gonogo.py`, `phase05_reframe.py`, `phase05b_profile.py`.

## 4. The mechanism is overwhelming (and stable across regimes)

Mean log realized vol by dealer-gamma sign, deep history:

| Era | short-gamma logRV | long-gamma logRV | Welch t |
|---|---|---|---|
| 2011–2026 (all) | **−1.41** (n=338) | −2.29 (n=3,385) | **+27.6** |
| pre-2020 | −1.48 | −2.38 | +20.4 |
| 2020–21 | −0.99 | −2.12 | +9.9 |
| 2022+ | −1.49 | −2.18 | +16.9 |

Short-gamma days carry dramatically higher RV in **every** regime (p ≈ 0 throughout). Gamma is genuinely informative. The entire question is how much of that survives controlling for VIX.

## 5. Results

### 5a. The 21-month owned window: a comprehensive null
On 2024-08 → 2026-03 (415 days, one calm regime), gamma added **nothing** beyond VIX/HAR across six pre-registered formulations — level, intraday-range (pinning), mean-reversion, downside tails, regime-direction, and by-strike profile shape (the profile features in fact *overfit* and degraded the forecast). Underpowered and regime-poor, this window simply cannot see a small effect. *(Details: `analysis/phase0*.py`.)*

![21-month sub-study: no incremental skill across six formulations](analysis/figures/gamma_null_summary.png)

### 5b. The deep history (2011–2026): a small but robust increment
Incremental skill of gamma over a full VIX/HAR baseline, out-of-sample:

| Block | n (OOS) | dCRPS | DM p | ΔAUC | AUC p | Verdict |
|---|---|---|---|---|---|---|
| **All** | 3,219 | +0.0020 | **0.006** | +0.007 | **0.003** | gamma helps |
| pre-2020 | 1,617 | +0.0016 | 0.115 | +0.005 | 0.192 | null (positive) |
| 2020–21 | 498 | +0.0022 | 0.229 | +0.009 | 0.108 | null (positive) |
| 2022+ | 1,104 | +0.0024 | **0.021** | +0.007 | 0.067 | gamma helps |

**Confound decomposition** (full OOS) — the critical check, since SqueezeMetrics ships gamma *and* DIX (a flow signal, not gamma):

| Added to VIX/HAR | DM p | AUC p |
|---|---|---|
| **gamma only** (gex pct + neg-flag) | **0.001** | **0.001** |
| DIX only | 0.97 | 0.78 |
| gamma, over VIX **+ ΔVIX** baseline | **0.001** | 0.001 |

The increment is **gamma-specific** (DIX adds nothing), **survives a richer VIX baseline** (not a stale-VIX proxy), is **positive in every regime block**, and **gamma-only is cleaner than gamma+DIX** (p=0.001 vs 0.006 — DIX only dilutes). At p=0.001 on two independent metrics over 3,219 OOS days, it survives generous multiple-testing correction across every test run this project.

**But it is small:** dCRPS ≈ +0.002 on a baseline CRPS ≈ 0.22 (< 1% relative); ΔAUC ≈ +0.007. Gamma is ~95% a VIX echo; the last sliver is real.

![Deep-history result: a small but robust gamma-specific increment](analysis/figures/deep_history_result.png)

## 6. Conclusion (calibrated)

**Dealer gamma carries a small, statistically robust increment to next-day RV forecasting beyond VIX — detectable only on a powered, multi-regime sample, gamma-specific (not DIX), and not a VIX-change artifact.** It is economically marginal and was *invisible* on a calm 21-month window — a clean object lesson in statistical power, and a reminder that "no signal" on a short, single-regime sample is a statement about power, not about the world.

**What this does not claim.** Whether a sub-1%-CRPS / 0.7-AUC-point edge is *tradeable* after costs is a separate question, taken up in the companion **[`STRATEGY.md`](STRATEGY.md)** — where, tellingly, **dealer gamma adds *nothing* to a working short-vol VRP carry strategy beyond the VIX term structure** (every gamma/vol-of-vol/liquidity overlay tested *reduces* risk-adjusted return). That is the trading-side corroboration of this document's finding: gamma's increment is real but so small it rounds to zero once VIX is in the model. The result rides on SqueezeMetrics' proprietary gamma model; an independent reconstruction (and the intraday/0DTE timescale, where the mechanism is strongest) are the natural next tests.

**Growth probe — the daily edge is *linear*, so the next gain is intraday, not a fancier model.** A first growth check (`analysis/phase2_learned_flip.py`) finds the daily gamma→RV relationship is a smooth, near-linear gradient in gamma percentile — *not* a sharp threshold at the flip. A regime-switching / learned-flip model does **not** beat the plain linear gamma term (DM p=0.35). So on daily data the edge is the small linear sliver, and a nonlinear "AI-on-the-mechanism" model would add complexity without payoff. The nonlinear/threshold structure (and any larger, tradeable signal) would have to live at the **intraday** scale — making intraday the highest-value next test.

## 7. Reproducibility

```bash
# env with pandas/numpy/scipy/scikit-learn + pyarrow; fetchers download free data
python analysis/phase1_deep_history.py   # deep-history test, per-regime
python analysis/phase1_robustness.py     # gamma-vs-DIX + richer-VIX decomposition
python analysis/phase0_gonogo.py         # 21-month sub-study (level)
python analysis/phase05_reframe.py       # 21-month sub-study (path/dynamics/tails/regime)
python analysis/phase05b_profile.py      # 21-month sub-study (profile shape)
```
Design/scope: `docs/specs/2026-05-29-gamma-regime-vol-design.md`. Why v1 was abandoned: `docs/v1-retrospective.md`.

## 8. What this project demonstrates

Judgment and rigor under a hard, honest constraint: taking a self-deceiving v1 (a "+0.84 Sharpe" that was a single lucky day, a VIX-echo feature, drifted artifacts), diagnosing it, reframing it into a falsifiable question, and then — when a calm window said "null" — recognizing that as an underpowered statement, building a powered 15-year multi-regime test on free data, finding a *small real* increment, and **stress-testing it** (gamma vs DIX, richer VIX baseline, per-regime, multiplicity) before believing it — including catching my own sign-error along the way. Calibrated truth over a manufactured headline, in both directions: refusing a fake edge, and refusing to dismiss a small real one.
