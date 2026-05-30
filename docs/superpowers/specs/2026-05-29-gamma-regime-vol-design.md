# Design Spec — Gamma-Regime Volatility: Mechanism & Signal (spy_vol v2)

**Date:** 2026-05-29
**Status:** Draft for review
**Author:** Prakrit Dayal (with Claude Code)
**Supersedes:** the v1 "SPY RV regime classifier → VXX long-flat strategy" framing.
**Companion:** `docs/v1-retrospective.md` (why v1 was inefficient).

---

## 0. Why this rewrite exists

A multi-agent audit of v1 (2026-05-29), each finding verified against on-disk artifacts:

1. **The only "profitable" v1 result is a single day** (`lfs_nogate_voltarget` Sharpe **+0.840 → −0.878** when 2026-01-16 is removed; equal-notional **constant-long beats it, +1.28**; signal-gated variants ~−2.0 — the ML *subtracts* value).
2. **Scalar GEX is a VIX echo** (`corr = −0.69`; CV AUC 0.621 → **0.572** when GEX is added).
3. **Contaminated target & inconsistent artifacts** (config h=21 vs docs' h=1; placebo `−rv_rolling_mean` AUC 0.727 beats every model).

Effort-allocation lessons are in `docs/v1-retrospective.md`. This rewrite turns each finding into a falsifiable research question.

## 1. Thesis (and the honest prior)

Dealer gamma modulates the **conditional dynamics and distribution** of S&P realized volatility: long-gamma dealers hedge counter-trend → RV **suppressed** (pinning, thin tails); short-gamma dealers (below the *gamma flip*) hedge trend-following → RV **amplified** (momentum, fat tails).

**Honest prior: scalar gamma is NOT orthogonal to VIX** (FlashAlpha's 8-yr SPY study: GEX→RV corr −0.36 raw, **−0.03 (p=0.18)** after controlling for VIX). So the question is sharper and falsifiable:

> **Does the *nonlinear / threshold / shape* structure of dealer gamma carry RV-regime information a linear VIX/HAR model misses — and does it survive honest small-sample inference?**

A well-characterised **null is an acceptable, even strong, outcome.** GEX is widely watched, so the simple linear signal is *expected* to be dead; any defensible edge lives in what is not commonly computed (the threshold, the profile shape, intraday charm/vanna).

## 1.1 Project shape — deliverable, scope, and "nothing is core"

- **The deliverable is the eventual TRADING STRATEGY.** **The scope of *this* plan is the SIGNAL only.** They are separate plans (§12).
- **Strategy deployment is its own phase with its own mathematics** — the stochastic-calculus risk/sizing layer (§9) belongs to the *strategy* phase, not the signal phase. Cleanly separated.
- **Nothing is "core."** Every module — HAR/VIX baselines, the learned-flip model, the gated-density model, profile features, later charm/vanna — is an **independently-valid peer** that works *adjacently*. **Robustness is an emergent property of convergence across the suite, not a single load-bearing model.**
- **No combiner** (§6): there is no single blended prediction that could become a hidden core. The "system" is the suite of diagnostics plus an explicit **agreement/disagreement map** — and disagreement is itself signal.

```
mechanism (gamma hedging, continuous-time)
  → SIGNAL phase  [THIS PLAN]: a suite of adjacent diagnostics; robustness = convergence
  → STRATEGY phase [SEPARATE PLAN]: own mathematics (stochastic-calculus risk layer, §9) → deployment
```

## 2. Data plan (all free; download-and-gitignore; ship the fetcher)

| Need | Primary (free) | Fallback | Schema / depth | Caveats |
|---|---|---|---|---|
| **Daily scalar GEX + DIX (deep, Phase 1)** | SqueezeMetrics `DIX.csv` (verified live: 3,791 rows, **2011-05 → 2026-05**) | net-GEX from owned OPRA (2024-08+) | `date,price,dix,gex`; SPX; **`gex` unsigned, ~always positive — a LEVEL, not a signed flip** | ToS bars redistribution → ship downloader, gitignore CSV. **Lag 1 trading day** (OCC OI is T-1). Used for the **level-regime** claim only. |
| **Signed net-GEX + by-strike profile (sign-flip claim; F3)** | owned OPRA OI (**2024-08 → 2026-04, ~435 rows**) | FlashAlpha intraday (Phase 2) | per-(strike,γ,OI) panel via `features/opra_panel.py` | No free by-strike *history*. The **sign-flip / amplification** claim is confined to this signed window. |
| **Intraday gamma (Phase 2)** | FlashAlpha free API (SPY minute GEX, **2018+**) | — | pre-computed GEX/greeks | 5 req/day free; the actual mechanism timescale; sets up charm/vanna. |
| **VIX family** | CBOE `VIX_History.csv` (^VIX 1990→) | yfinance `^VIX3M` (2006→), `^VIX9D` (2011→), `^VVIX` (2007→), `SPY` | daily OHLC/close | **`^VIX3M` not dead `^VXV`; `^VIX9D` not `^VXST`.** |
| **Vol-futures reference (strategy phase only)** | reconstruct SPVXSTR from CBOE VX settlement (2004→) + DGS3MO | Yahoo VXX (2018+, splice marked) | continuous 30-day CMT TR | **Not built in the signal phase.** |
| **Risk-free** | FRED `DGS3MO` (wired) | — | daily yield | public domain. |

**Binding window for Phase 1:** ~**2011-05 → 2026-05 (~3,780 daily obs)**.

## 3. Features & the no-lookahead contract

- **Gamma-state, deep (Phase 1, level claim):** `gex_level` (notional-normalized), **`gex_pct` (trailing percentile, scale-free)**, `gex_chg`, `dix`.
- **Gamma-state, signed (signed window only, flip claim):** signed `gex_net`, distance-to-flip, (F3) by-strike profile scalars.
- **VIX family:** level, log, Δ1d, Δ5d, z-score; `VIX9D/VIX`, `VIX/VIX3M`; `VVIX/VIX`.
- **RV / HAR:** Yang-Zhang daily RV; HAR lags.
- **No-lookahead contract (hard):** gamma at close of *t* is **lagged 1 trading day**. New test mirrors `test_target_no_future_leak`: perturb day-*t* OI/GEX, assert every gamma feature at *t* is unchanged, **run through `features/assemble.assemble`** (v1's flagship test never touched the join — W8). Verify SqueezeMetrics' as-of timestamp before use.

## 4. Target (contamination-fixed)

`y_t = 1[ RV_{[t+1,t+h]} > baseline ]`, **baseline = trailing RV mean ending at `t−1`** (excludes overlapping `RV_t`). Headline **h=1**; **h=5/h=21 robustness panel**. Continuous forward-log-RV retained for HAR + density models. Overlapping h>1 → **purged + embargoed walk-forward**.

## 5. The core analysis = this phase's deliverable (the signal)

Two claims, each on the data that supports it (the hybrid):
1. **Level claim (deep, 2011→):** does gamma *percentile/level* modulate the RV regime? Per regime block (§7).
2. **Sign-flip claim (signed window):** does the gamma *sign flip* (amplification vs suppression) matter, beyond level?
3. **Incremental value over VIX/HAR** (both claims): headline metric is a **Diebold-Mariano / Giacomini-White test on the CRPS differential** of nested `(VIX,HAR-X)` vs `(+gamma)`, with reported **VIF**. Never standalone-vs-standalone or raw correlation.
4. **Regime stability** across pre-2020 / 2020-21 / 2022+.

## 6. The module suite (adjacent peers — none core; no combiner)

Each module is independently fit, evaluated, and reported. The "system" is the suite + an **agreement/disagreement map** (where modules converge = robust; where they diverge = flagged signal). **No blended prediction.**

- **Baselines (peers, not strawmen):** HAR; HAR-X+VIX; GARCH(1,1). The bar everything is measured against.
- **Learned-flip smooth-transition (LSTAR):** `G = σ(k·(gex_pct − c))`; learn flip `c` (CI) and sharpness `k`; report whether learned `ĉ` matches the dealer-convention flip. *Guard:* `k` weakly identified → profile likelihood over a k-grid (no Hessian SE), block-bootstrap `(c,k)`; if `ĉ`'s CI spans most of the range, state the flip claim is unsupported. *Novelty:* genuine whitespace (gamma as transition variable + estimated flip). ~8–10 params.
- **Gamma-gated distributional RV:** 2-state hard gate → regime-switching HAR-on-log-RV → Normal/skew-t **predictive density**; eval CRPS + **blocked/adaptive conformal coverage**. *Guard:* shelve soft gating net / neural MDN until justified. *Novelty:* adjacent (cf. MoGU 2025). ~11–20 params.
- **Functional gamma-profile (signed window only):** 4–6 hand-built profile scalars (distance-to-flip, call/put-wall height/strike, |γ| Herfindahl, gamma-skew); fPCA (2–3 PCs) the only "learned" version. *Guard:* **no 1D-CNN/deep-sets** until multi-year by-strike data. *Novelty:* whitespace for the profile.

All encode the mechanism in their *structure*; the win is inductive bias, not parameter count.

## 7. Statistical rigor & guardrails (audit-fix checklist, baked in)

Purged + embargoed walk-forward · **deflated skill over every module & config tried** (pre-register the count) · **blocked/adaptive conformal coverage** reported · **never pool across the 0DTE break** — per-regime blocks (pre-2020 / 2020-21 / 2022+), EOD-gamma intraday-blindness stated up front as an honesty asset · honest N_eff per horizon · orthogonality only via the nested DM-on-CRPS test + VIF · per-module results + convergence map reported, never a single cherry-picked number.

## 8. Strategy = the eventual deliverable (SEPARATE phase; out of scope here)

The trading strategy is the project's end goal but is **explicitly out of scope for this plan** and gets its **own plan with its own mathematics** (§9). Once the signal is characterised, candidate forms (none chosen now): simple long/flat or long/short vol; a VIX/term-structure gate; a mean-reversion-vs-momentum regime filter; risk-controlled sizing via §9. Any deployment inherits the §7 fragility discipline (RV-first; P&L strictly secondary; reconstructed SPVXSTR with marked splices; top-k-day + constant-leg benchmarks beside any Sharpe). Kept separate to avoid v1's strategy-before-signal failure.

## 9. Strategy-phase mathematics — stochastic-calculus layer (directions; built in the strategy phase)

A first-class pillar of the **strategy** phase (not this one), because dealer hedging *is* a continuous-time process. Direction chosen once the signal is known; role = risk optimiser or strategy support:

- **A. Gamma-feedback SDE** — `dS = μ dt + σ_t S_t dW`, effective vol/feedback an explicit function of net dealer gamma; ties GEX to realized variance via Itô.
- **B. Gamma-flip as a barrier** — distance-to-flip as a diffusion; **first-passage-time / crossing probability** to the zero-gamma level; times regime transitions.
- **C. Stochastic optimal control (HJB)** — the **risk optimiser**: size exposure to a CVaR/utility objective under gamma-conditional vol dynamics; the rigorous replacement for v1's ad-hoc Kelly.

*Discipline:* small-N budget (closed-form / low-dimensional, not a deep neural SDE on 435 days); validated vs a simpler benchmark; never the headline before the signal is established. **Implementation deferred to the strategy phase.**

## 10. Repo hygiene (hard requirements)

Real git repo (retire nested `repo/`); **ship fetchers + `.gitignore` all raw data** (satisfies SqueezeMetrics ToS; fixes committed-binaries sin); **delete the leaked path** in `configs/experiment.yaml:39`; stop committing PNGs/`.DS_Store`/`.ruff_cache`; add CI; `torch` to optional extra; one pinned config with a hash stamped into every artifact.

## 11. Phase 0 — go/no-go (DO FIRST, ~1 day, existing data)

On existing `features_panel.parquet` (has *signed* OPRA `gex_net`), contamination-fixed target: does a 2-state signed-GEX bucket beat HAR-X+VIX incrementally (DM-on-CRPS)? **If not, stop before building the suite.** Underpowered at 21 months: a positive is encouraging, a strong null a caution — sets expectations before any pipeline investment.

## 12. Phasing (signal plan vs strategy plan are separate)

- **Phase 0:** go/no-go on existing data (§11).
- **Phase 1 — SIGNAL, daily-deep:** fetchers (SqueezeMetrics, CBOE VIX, yfinance, FRED) → 2011→2026 panel → hybrid target → the **module suite** (baselines + learned-flip + gated-density on the deep *level* claim; profile + *sign-flip* on the signed window) → DM orthogonality → per-regime blocks → **suite agreement/disagreement map** → repo hygiene + tests. *No strategy.*
- **Phase 2 — SIGNAL, intraday:** FlashAlpha intraday gamma (2018+) — the actual-timescale test; sets up charm/vanna.
- **Phase 3 — mechanism breadth:** charm, vanna, signed flow as adjacent modules.
- **Phase 4 — STRATEGY (separate plan):** stochastic-calculus layer (§9 A/B/C) + a deployment; its own mathematics and its own spec.

## 13. Success criteria (this plan = the signal)

- Clean, reproducible 2011→2026 pipeline (`make` from scratch; CI green; no committed raw data).
- Pre-registered, purged-WF, deflated evaluation; conformal coverage; per-regime blocks.
- An explicit, calibrated statement — **including a clear null if that's the answer** — of whether gamma *level* (deep) and *sign-flip* (signed window) recover RV-regime signal beyond VIX/HAR (DM-on-CRPS, with CI), plus learned `ĉ` vs the dealer convention.
- A **suite agreement/disagreement map** as a first-class deliverable (robustness = convergence; divergence = flagged).
- Every headline number reconciled to artifacts; v1's contradictions gone.

## 14. Key references

Barbon et al. (2021) dealer gamma; Corsi (2009) HAR-RV; Bailey & López de Prado (2014) Deflated Sharpe + purged/embargoed CV; Aviv et al. (2025) MoGU; STAR-GARCH (Livingston & Nur 2020); Adaptive Conformal Inference (arXiv 2410.13115); SqueezeMetrics GEX white paper; FlashAlpha 8-yr GEX-vs-VIX-control backtest. Strategy-phase: market-impact-of-hedging / gamma-feedback SDE; first-passage methods; stochastic optimal control / HJB.
