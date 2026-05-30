# Why v1 Was Inefficient — a Retrospective

v1 shipped a clean, leakage-disciplined pipeline. The problem was never the engineering; it was **where the effort went**. This note records the misallocations so v2 doesn't repeat them. (Findings below were verified against the on-disk artifacts.)

## The five inefficiencies

1. **Spent the whole data budget before testing the cheap baseline.** The entire ~$95 Databento credit went to OPRA options data to build GEX — *before* checking whether scalar GEX adds anything over free VIX. It doesn't: `corr(gex_net, vix_zscore) = −0.69`, and cross-validated AUC *falls* when GEX is added (0.621 → 0.572). A two-line baseline on free data would have de-risked the most expensive decision in the project. **Lesson: test the cheapest kill-switch first.**

2. **Built breadth the sample couldn't cash.** Six models (logistic, interactions, HAR-X, XGB, MLP, GP) were trained where the effective sample (N_eff ≈ a few hundred, autocorr ≈ 0.9) cannot statistically distinguish any of them — every Sharpe and AUC CI overlaps. Effort went into a model zoo instead of one sharp question. **Lesson: at small N, spend effort on the inductive bias and the inference, not the model count.**

3. **Optimised a strategy before establishing a signal.** A long-flat-short variant reported **+0.84 Sharpe** — but deleting one day (2026-01-16) flips it to **−0.88**, and an equal-notional constant-long beats it (+1.28). The "edge" was being structurally short a contango-bleeding ETN in a calm window; the ML signal *subtracted* value. **Lesson: don't build (or headline) a trading strategy until the signal is real.**

4. **Rigor as a consolation prize, not a multiplier.** v1's strongest card — block-bootstrap CIs, N_eff, pre-registration — was deployed to *defend a null* ("the rigor is the signal"). Rigor multiplies a sound experiment; it can't rescue an underpowered one. The honest move is to ask a **falsifiable** question the data can actually speak to. **Lesson: point the rigor at a sharp question, not at excusing a weak result.**

5. **Artifacts drifted out of sync — which destroys the very credibility the project sells.** The shipped config set a 21-day horizon while every doc described next-day; the headline AUC table was generated under yet a third target; the public GitHub README claimed a profitable strategy the local README said didn't exist; a zero-information placebo (`−rv_rolling_mean`, AUC 0.727) beat every model because the target's baseline contained the present value it was being compared to. **Lesson: one pinned config, every number reconciled to the artifact that produced it, or the honesty pitch collapses on contact.**

## The efficient pattern v2 adopts

- **Kill-switch first:** a 1-day go/no-go on data already on disk before any new build (`docs/superpowers/specs/2026-05-29-gamma-regime-vol-design.md` §11).
- **One falsifiable question:** does the *nonlinear/threshold/shape* structure of dealer gamma beat VIX/HAR — judged by a Diebold-Mariano test on CRPS — accepting a clean null as a real result.
- **Depth in the mechanism, not the model zoo:** a learned gamma-flip (LSTAR), a calibrated density, and a stochastic-calculus layer that treats dealer hedging as the continuous-time process it actually is.
- **Signal before strategy:** the output stays deployment-agnostic until it is established.
- **Honest scope from data reality:** ~15 years of *free* daily gamma (depth), explicit per-regime reporting across the 0DTE structural break, no committed raw data, no drifted numbers.
