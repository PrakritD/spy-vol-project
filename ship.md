# Ship Checklist — SPY Vol-Regime Classifier

Live tracker. Items get checked off as they complete.

## STATUS — `PHASE 2: PROFITABLE STRATEGY`

Last updated: 2026-05-16.

**Phase 1 complete (recruiter-ready baseline)** — pushed to `https://github.com/PrakritD/spy-vol-project` on `main`. 52 tests pass • ruff clean • 6 models trained • 27 figures • paper compiled • 3 notebooks • README + STATISTICAL_RIGOR.md polished + de-AI'd.

**Phase 2 in progress** — convert the long-only loss-making strategy into a profitable long-flat-short strategy with AI/quant layer on top. Develop locally in `spy_vol/`; push to GitHub after verification.

---

## Phase 2 — Profitable strategy + AI/quant breadth

Full plan: `/Users/prakrit/.claude/plans/is-there-a-different-velvety-hennessy.md`.

**Goal**: positive Sharpe on the last-5-month test window (2025-12 → 2026-04, ≈ 105 trading days), with daily refit on a 250-day rolling window. Headline target — at least one strategy variant has Sharpe CI excluding zero on Monte Carlo bootstrap.

### Order of execution (with failure modes & optimisation paths)

#### 1. Daily walk-forward + 5-month test window  *(~1 hr)*

- [x] Extend `backtest/walk_forward.WalkForwardConfig` with `refit_freq_days`, `test_start`, `test_end`
- [x] Add `_daily_schedule()` helper that loops over actual trading dates
- [x] Add `rolling_train_days` to `run()` (overrides `rolling_train_months` when set)
- [ ] Smoke test: run on a synthetic 200-row panel with daily refit, assert one prediction per test day
- [ ] Run on the real `features_panel.parquet` for all 6 models — produces `data/processed/walk_forward_preds_daily.parquet`

**Failure modes**:
- *Compute blowup*: 6 models × 105 daily refits × O(N³) GP fit could explode. GP is the worst offender at ~5 s/fit → 9 min. Mitigation: train GP only every 5 days instead of every 1, interpolate between refits. Set `refit_freq_days=5` for GP variant.
- *Cold-start*: first few daily test rows have only 250 training rows. Models with high VC dimension (MLP, GP) may underfit. Mitigation: warm-start from previous fold's weights for MLP; reduce GP RBF length-scale prior.
- *Drift*: rolling 250-day window may miss the long-memory of vol (Corsi HAR has 21-day component). Mitigation: option to override `rolling_train_days=None` for HAR-X so it uses expanding window.

**How to optimise compute**:
- Parallelise across the 6 models via `joblib.Parallel(n_jobs=6)`.
- Cache feature transformations (StandardScaler stats) across folds when the rolling window only slides by 1 day.
- For XGB, use `xgb_model=prev_booster` to warm-start each fold from the previous booster (incremental training).

#### 2. Bayesian model averaging ensemble  *(~2 hrs)*

- [ ] `models/ensemble.py` — `BayesianEnsemble.combine(preds_long_frame, eta=1.0, lookback=30)`
- [ ] Per-day weights $w_i^{(t)} \propto \exp(-\eta \cdot \overline{\text{logloss}_i^{[t-30, t-1]}})$
- [ ] Ensemble prediction = $\sum_i w_i^{(t)} \cdot \hat p_i^{(t)}$
- [ ] Smoke test: identical models → uniform weights; weights non-negative & sum to 1
- [ ] Add ensemble row to `walk_forward_preds_daily.parquet` as a 7th "model"

**Failure modes**:
- *Cold-start lookback*: first 30 days of test have insufficient history → fall back to uniform weights. Document this in the runner.
- *Degenerate weights*: if one model's log-loss is dramatically lower, eta=1.0 collapses to that model alone. Mitigation: also try eta=0.5 (more diversified) and eta=2.0 (more concentrated) — report all three.
- *Log-loss vs Sharpe disagreement*: a model with the best log-loss may not have the best P&L (calibration ≠ profit). Consider also a Sharpe-weighted ensemble variant.

**How to optimise**:
- Cache the per-day log-loss matrix; the weight update is then a 30-row exponential-weighted moving average (cheap).
- For Sharpe-weighted variant, use realised in-sample Sharpe over the lookback rather than log-loss — gives a different bias/variance trade-off.

#### 3. HMM regime detection + gating  *(~2 hrs)*

- [ ] `backtest/regime_hmm.py` — fit `hmmlearn.GaussianHMM(n_components=2)` on training-only `(vix_level, log_rv)` daily series
- [ ] Order states by mean VIX so state 0 = "low-vol", state 1 = "high-vol" (deterministic)
- [ ] Viterbi-decode each test date; emit `state_prob[t]` for state-1 (high-vol)
- [ ] Two gating modes: hard (trade only when state==1) and soft (size *= state_prob)
- [ ] Smoke tests: produces 2 distinct states; state assignment stable to seed
- [ ] Add `regime_state` and `regime_prob_highvol` columns to the per-day output

**Failure modes**:
- *Convergence failure*: HMM can land in degenerate local optima with both states identical. Mitigation: 5 random restarts, keep the best log-likelihood; assert the two states have means at least 1.5 σ apart.
- *Look-ahead*: fitting the HMM on the FULL panel including the test window leaks future information. Mitigation: re-fit the HMM at each test date on training-only history (expensive but correct), OR fit once on the in-sample window and roll forward via online Viterbi. Implement the latter.
- *Regime persistence*: 2 states may be insufficient; vol has multi-modal distribution. Mitigation: also try 3-state HMM as a comparison.
- *Bad VIX standardisation*: if `vix_level` isn't z-scored against training stats, the state assignment drifts over time. Mitigation: z-score on training stats per fold.

**How to optimise**:
- Use Forward-filtering Backward-smoothing rather than Viterbi for soft state probabilities (more information per day).
- Add transition-matrix prior from CAS literature: state-1 persistence ~0.85, state-0 persistence ~0.95.

#### 4. Vol-target + Mean-CVaR sizing  *(~3 hrs)*

- [ ] `backtest/sizing_advanced.py`:
  - [ ] `vol_target_sizing(p_hat, predicted_vol, target_ann_vol=0.10)` — inverse-vol scaling
  - [ ] `kelly_with_uncertainty(p_hat, p_var, b, shrinkage=1.0)` — half-Kelly × `(1 - shrinkage·√p_var)`
  - [ ] `mean_cvar_sizing(p_hat, training_returns, cvar_target=0.05)` — cvxpy LP over candidate sizes
- [ ] Predicted vol source: rolling 21-day RV of VXX (simple, robust) OR GARCH(1,1) fit per fold
- [ ] Smoke tests: vol-target shrinks position when vol up; CVaR sizing respects constraint on in-sample data
- [ ] Wire all three through `SizingSpec` for use by the LFS execution

**Failure modes**:
- *cvxpy unavailable / solver fails*: fallback to half-Kelly. Document the fallback in module docstring.
- *Vol forecast errors*: GARCH MLE can fail to converge on short windows; if so, use the simpler rolling-21d-RV estimator.
- *Mean-CVaR over-shrinks*: with 250-day training and a tight CVaR constraint (5%), sizes can collapse to ~0. Mitigation: parameterise cvar_target and report sensitivity.
- *Kelly with high p_var*: if posterior variance is large, position collapses to zero. Mitigation: clip variance shrinkage to ≤ 0.7 of half-Kelly.

**How to optimise**:
- Pre-compute the candidate-size grid; LP is then 105 trivially-small problems (cvxpy ~50 ms each).
- Use the closed-form analytic Mean-CVaR solution from Rockafellar-Uryasev (avoids cvxpy entirely): solve for α such that the conditional tail mean equals the constraint.

#### 5. Long-flat-short VXX execution  *(~2 hrs)*

- [ ] `backtest/execution_lfs.py` — `backtest_lfs(preds, vxx_prices, cfg, sizing_long, sizing_short=None, ...)`
- [ ] Position mapping:
  - `p̂ > p_long_threshold (0.55)` → long, size = `sizing_long(p̂)`
  - `p̂ < p_short_threshold (0.45)` → short, size = `-asymmetry * sizing(1 − p̂)` (default asymmetry=0.5)
  - else flat
- [ ] Cost model: `cost_bps_short = 2 × cost_bps_long` (borrow + slippage)
- [ ] Smoke test: inverse signal → opposite-sign P&L
- [ ] Apply to ensemble prediction → first LFS P&L stream

**Failure modes**:
- *Short squeeze risk*: vol-spike days deliver −20%+ on short-VXX. Single day can wipe out months. Mitigation: hard cap short size at 0.5 of unit notional regardless of confidence; add a "vol kill switch" — flatten shorts when VVIX/VIX > 8.
- *Asymmetric tax / borrow drift*: real-world short-VXX borrow can spike during stress. Mitigation: model the borrow cost as `2 × base_bps` baseline + `10 × base_bps` when high-vol regime.
- *VXX gap risk*: overnight gaps don't show up in close-to-close returns. Mitigation: switch to open-to-open when intraday VXX is available.
- *Threshold sensitivity*: 0.55/0.45 is arbitrary. Mitigation: sweep thresholds and pick the one maximising in-sample Sortino, or use a quantile-based threshold (top/bottom 30% of $\hat p$).

**How to optimise**:
- Add a `sizing_short` argument so the long and short sides can use different sizing rules (e.g. long = vol-target, short = Mean-CVaR).
- Add a hold-out period: don't flip from long to short same-day; require at least 1 day flat between sign changes (reduces whipsaw).

#### 6. Strategy runner + headline comparison  *(~2 hrs)*

- [ ] `backtest/runner_v2.py` — orchestrator
- [ ] Run daily walk-forward for 6 base models → ensemble → HMM-gated ensemble
- [ ] Apply each sizing rule × LFS execution → 4 strategy variants per ensemble:
  - `lfs_linear` (baseline LFS)
  - `lfs_voltarget`
  - `lfs_meancvar`
  - `lfs_kelly_uncert`
- [ ] Headline comparison table: each variant's Sharpe / Sortino / PSR / MaxDD / Calmar + Monte Carlo P(profit)
- [ ] Save figures to `report/_build/profitable/`: equity overlays, drawdown overlay, fan chart, regime timeline

**Failure modes**:
- *All variants still lose*: signal margin may not be enough even with LFS. Then we honest-report and pivot to either (a) higher-margin signal extraction (try a 5d horizon target), (b) different vehicle (VIX futures). Document the result explicitly.
- *Overfitting via post-hoc selection*: picking the best variant from 4 inflates significance by 4×. Mitigation: report all four side-by-side; the headline claim is the *best* with Bonferroni correction applied to its PSR.
- *Cost-sensitivity collapses Sharpe*: if going from 5 to 20 bps flips the sign, the result is fragile. Mitigation: report sensitivity sweep alongside headline.

**How to optimise**:
- Cache the daily walk-forward predictions once; the runner then iterates cheaply over sizing + execution variants without re-fitting models.
- Run all 4 variants in parallel via `concurrent.futures` (each is independent of the others).

#### 7. Monte Carlo bootstrap of strategy returns  *(~1.5 hrs)*

- [ ] `backtest/montecarlo.py` — `simulate(returns, n_paths=10000, block_size=None)`
- [ ] Stationary block bootstrap; default `block_size = int(N^(1/3))`
- [ ] Outputs:
  - Sharpe distribution percentiles
  - Max-drawdown distribution percentiles
  - P(equity[-1] > 1.0) — empirical probability of profit
  - Fan chart saved to `report/_build/montecarlo_fan.png`
- [ ] Smoke test: observed Sharpe at the median of the distribution; CI matches `metrics.block_bootstrap_sharpe_ci`

**Failure modes**:
- *Block size too small*: under-respects autocorrelation, narrow CI. Mitigation: test 3 block sizes (N^(1/3), N^(1/2), N^(2/3)) and report the most conservative.
- *Sample size too small*: 105 days × 10k bootstrap = stable Sharpe distribution; max-DD distribution noisier. Use 50k for max-DD specifically.
- *Path dependency*: equity curve from resampled returns may not preserve the original autocorrelation in drawdowns. Acceptable for distributional summary, document the limitation.

**How to optimise**:
- Vectorise the bootstrap: pre-allocate `(n_paths, N)` array of resampled returns, compute cumulative product in one numpy call.
- For Sharpe distribution only, skip the full equity-curve reconstruction.

#### 8. RL sizing policy (PPO)  *(~6 hrs)*

- [ ] `models/rl_sizing.py`:
  - [ ] `SpyVolEnv(gym.Env)` — observation space, action space, reward
  - [ ] Training loop: PPO via `stable_baselines3.PPO`, 100k timesteps on in-sample window
  - [ ] Inference wrapper: deterministic action mapping for the test window
- [ ] Observation: `(p_hat, vix_zscore, gex_z, term_9d_30d, recent_5d_pnl, recent_drawdown)` ∈ R^6
- [ ] Action: continuous `size ∈ [-1, 1]`
- [ ] Reward: daily net P&L − 0.1·|Δsize| (turnover) − 0.5·max(0, ΔDD) (drawdown penalty)
- [ ] Compare on test window vs rule-based sizing
- [ ] Smoke tests: env produces valid tuples; deterministic on seed

**Failure modes** (this is the highest-risk component):
- *Severe overfitting*: 340 in-sample training days with PPO is at the edge of viability. The agent can memorise the training path. Mitigation: train on bootstrap-resampled training data (each episode = different sample order); validate on a held-out portion of training, not just test.
- *Reward hacking*: agent learns to stay flat to avoid turnover/DD penalty. Mitigation: weight the P&L term 10× the penalty terms; verify the agent achieves non-zero gross P&L on training.
- *Sparse vol-spike rewards*: real signal value is concentrated in ~10 high-vol days; PPO struggles with rare-event learning. Mitigation: reweight rewards by recent realised vol (gives more credit to correct calls in stressed regimes).
- *Train-test distribution shift*: 2024-08 → 2025-11 may have different regime than 2025-12 → 2026-04. RL is brittle to distribution shift. Mitigation: include vol z-score in observation (lets agent infer regime); compare in-sample Sharpe to out-of-sample.
- *Compute time*: 100k PPO timesteps on this env ≈ 2-6 hrs on CPU. Mitigation: vectorised env (parallel rollouts) via `stable_baselines3.make_vec_env(..., n_envs=4)`; cuts compute ~3×.

**How to optimise**:
- Use SAC (Soft Actor-Critic) instead of PPO — sample-efficient on small data.
- Reward shaping: add a "Sharpe ratio" reward term (rolling 21-day Sharpe of the agent's recent P&L) — encodes risk-aware behaviour directly.
- Curriculum: start with no penalty terms, gradually introduce them.
- Imitation learning warm-start: pre-train policy to match the rule-based sizing, then fine-tune via PPO.
- Comparison protocol: also report the policy on the in-sample data — if it doesn't beat rule-based even in-sample, the architecture is wrong.

#### 9. Walkthrough notebook  *(~2 hrs)*

- [ ] `notebooks/04_profitable_strategy.ipynb`
- [ ] Sections:
  - Recap of Phase 1 (signal works, strategy doesn't, why)
  - The 4-piece fix (LFS, ensemble, regime gate, smarter sizing)
  - AI/quant layer (HMM, ensemble, RL, Monte Carlo)
  - Side-by-side equity curves
  - Monte Carlo fan chart on the final strategy
  - Where positive Sharpe comes from + generalisation honesty
- [ ] Execute end-to-end via nbconvert
- [ ] Embed key figures from `report/_build/profitable/`

**Failure modes**:
- *Notebook execution fails*: dependencies (hmmlearn, sb3, cvxpy) may not import in the kernel. Mitigation: explicit `pip install` cell at top with try/except imports.
- *Compute time*: full notebook execution could take 1+ hr if it re-runs the walk-forward. Mitigation: load cached `walk_forward_preds_daily.parquet` rather than refitting.

### Verification

- [ ] All existing 52 tests still pass
- [ ] No-lookahead invariant (`tests/test_no_lookahead.py`) still passes
- [ ] Daily walk-forward produces 105-day predictions for each of the 6 models
- [ ] Headline: at least one variant has positive Sharpe with block-bootstrap CI excluding zero; if none, explicit honest report
- [ ] Monte Carlo fan chart brackets observed Sharpe at the median
- [ ] `make backtest-daily && make profitable` regenerates everything from cached panel

### What to do AFTER results are in

If at least one variant is profitable:
1. **Refresh the GitHub repo** (the `repo/` folder). Copy new modules, runner, figures, notebook 04.
2. **Update README headline**: replace "every model has negative Sharpe" with the profitable variant numbers. Add a new §"Phase 2: Profitable strategy" pointing to notebook 04.
3. **Update STATISTICAL_RIGOR**: note the regime in which the strategy works (high-vol gating), the structural-bias caveat (the AUC source unchanged), and the cost-sensitivity bounds.
4. **Commit + push** as a new commit on main (the long-only Phase 1 numbers stay in the README appendix for the methodological honesty).

If no variant profits:
1. The HONEST report goes into a new §"Phase 2: What changed and didn't" in README. Explain LFS, ensemble, regime gating, sizing — and the structural reason none crossed the bleed threshold.
2. The notebook 04 still ships as a methodological walkthrough.
3. Pivot to either (a) different target (5d horizon, multi-class regime), (b) different vehicle (VIX futures direct), or (c) richer feature engineering (intraday once data is available).

### Stretch goals (only if 1–9 finish ahead)

- [ ] Cross-asset extension: apply the LFS + ensemble + HMM stack to NQ (Nasdaq-100)
- [ ] VIX futures direct: replace VXX with front-month + back-month VX with roll P&L attribution
- [ ] Online learning: replace daily refit with streaming SGD updates (lower compute, faster regime adaptation)

---

## Pragmatism guard — what's actually needed vs what's showcase

The 9 components serve **two** purposes that should stay separated. Don't conflate them.

### Pragmatic core (must-do for profitability)

These are the only steps that change *whether the strategy makes money*. Total ~6 hrs.

| # | Component | Why it's core |
|---|---|---|
| 1 | Daily walk-forward + 5mo test | The user's stated test window. Without it, no apples-to-apples comparison. |
| 5 | Long-flat-short execution | The actual profit fix. The classifier signal × VXX contango harvesting on the inverse side IS the strategy. |
| 7 | Monte Carlo bootstrap | The only honest way to claim "positive Sharpe" given autocorrelated daily returns. One Sharpe value is meaningless; the distribution matters. |
| 6 | Runner_v2 orchestrator | Glue. Without it, nothing runs end-to-end. |

If only these four ship, the project has a defensible profitable-or-not answer. Total compute: ~45 min after current state.

### Showcase layer (the quant breadth)

These add interview-grade depth but **do not change the core answer**. They demonstrate range.

| # | Component | Why it earns its keep |
|---|---|---|
| 2 | Bayesian ensemble | Standard practice in quant; ~2 hrs; usually +5% on Sharpe |
| 3 | HMM regime gating | Unsupervised-ML literacy; ~2 hrs; expected +10-30% Sharpe via gating |
| 4 | Vol-target + Mean-CVaR sizing | Risk-parity / convex-optimisation literacy; ~3 hrs; smooths equity curve |
| 8 | RL sizing (PPO) | AI-on-trader-side flagship; ~6 hrs; honest baseline-comparison value even if it loses |
| 9 | Walkthrough notebook | The recruiter-facing narrative; ~2 hrs; integrates everything |

**Decision rule**: do 1, 5, 6, 7 first. If the core result is positive-Sharpe, add 2, 3, 9 to make the recruiter pitch complete. Add 4 and 8 only if there's time and the core was clean.

### Over-engineering red flags — STOP if any of these happen

- Tuning thresholds (0.55/0.45) post-hoc to maximise OOS Sharpe → that's data snooping. The thresholds are part of the pre-registration; one tune sweep allowed, in-sample only.
- Adding more models. Six is already at the limit of what the sample supports. Adding a 7th = +1 Bonferroni penalty.
- Going below 30 trading days of test window. Statistical claims collapse.
- Spending >2 hrs on cvxpy convergence issues. Fall back to half-Kelly.
- Spending >6 hrs on PPO. If it isn't converging, document the negative result and ship.

---

## Phase 2 — files that will be updated when results land

After the profitable-strategy work ships (or honest-fails), the following docs and configs get refreshed. **Update these in `spy_vol/` first, then copy to `repo/` for the GitHub push.**

### Documentation

- **`repo/README.md`** — substantial revisions:
  - TL;DR table: add Phase-2 LFS variants alongside the Phase-1 long-only. If profitable, the *Phase-2 winner* becomes the headline; Phase-1 results move to an appendix table.
  - New §"Phase 2: making it profitable" — 1-page narrative of LFS + AI/quant layer with figures from `report/_build/profitable/`.
  - §"Why it doesn't profit (yet)" → renamed §"How the long-flat-short rewrite closes the gap" (if profitable) or expanded with what's structurally still missing (if not).
  - §"Future directions" — trimmed because some items now done.
  - §"Repo layout" — adds `backtest/runner_v2.py`, `backtest/execution_lfs.py`, `backtest/regime_hmm.py`, `backtest/sizing_advanced.py`, `backtest/montecarlo.py`, `models/ensemble.py`, `models/rl_sizing.py`, `notebooks/04_profitable_strategy.ipynb`.

- **`repo/STATISTICAL_RIGOR.md`** — append §9 "Phase 2 caveats":
  - LFS execution introduces short-vol tail risk. Document the asymmetric sizing rule and the kill-switch on VVIX/VIX > threshold.
  - Regime-gated results condition on HMM accuracy; report HMM training-only fit log-likelihood.
  - The 4 sizing variants are post-selected: report Bonferroni-corrected PSR (α/4 = 0.0125) on the headline winner.
  - PPO training is *in-sample*; the 5-month OOS test of the trained policy is the only valid comparison.

- **`repo/Outline.md`** — header note: "Phase 2 (long-flat-short + AI stack) ships in commit X. See README §"Phase 2"."

- **`repo/paper/spy_vol.tex`** — optional addendum: a §10 "Postscript on execution-vehicle mismatch" with the LFS result and Monte Carlo CI. ~1 hr to write. Recompile.

### Code (sync from `spy_vol/` → `repo/`)

- `repo/backtest/walk_forward.py` (modified — daily refit mode)
- `repo/backtest/execution_lfs.py` (new)
- `repo/backtest/sizing_advanced.py` (new)
- `repo/backtest/montecarlo.py` (new)
- `repo/backtest/regime_hmm.py` (new)
- `repo/backtest/runner_v2.py` (new)
- `repo/models/ensemble.py` (new)
- `repo/models/rl_sizing.py` (new — only if shipped)
- `repo/tests/test_lfs.py`, `test_sizing_advanced.py`, `test_montecarlo.py`, `test_regime_hmm.py`, `test_ensemble.py` (new)
- `repo/notebooks/04_profitable_strategy.ipynb` (new)

### Configs / build

- `repo/pyproject.toml` — add `hmmlearn`, `stable-baselines3`, `gymnasium`, `cvxpy` to `[dev]` extras
- `repo/Makefile` — add `make backtest-daily`, `make profitable`, `make rl` (if RL shipped)
- `repo/configs/experiment.yaml` — add ensemble + LFS variant entries

### Artifacts to add to the repo

- `repo/report/_build/profitable/equity_overlay.png` — Phase-1 long-only vs Phase-2 LFS variants
- `repo/report/_build/profitable/drawdown_overlay.png`
- `repo/report/_build/profitable/montecarlo_fan.png` — fan chart with 5%/50%/95% bands
- `repo/report/_build/profitable/regime_timeline.png` — HMM-decoded state over the test window
- `repo/report/_build/profitable/summary.csv` — strategy-variant comparison table

---

## Quant-perspective justification (for README §Phase 2 intro)

This is the framing to use when narrating Phase 2 in the README. Short, no fluff, no overselling.

> Phase 1 demonstrated the classifier captures real signal: AUC 0.60-0.67 on a 5-month OOS slice with autocorrelation-adjusted CIs that exclude 0.50. The trading-strategy translation — long-only VXX — was structurally a losing trade: VXX shed 52.8% over the test window through VIX-futures contango, and being long-only can't harvest that decay. The classifier was right; the execution choice was wrong.
>
> Phase 2 fixes the execution. The standard quant move: trade both sides of the signal. When the model predicts vol will rise, go long VXX (catch the spike). When the model predicts vol will fall, go short VXX (harvest the contango bleed on the way down). The classifier and the underlying pipeline are unchanged — only the execution arm flips from long-only to long-flat-short.
>
> Three quant-standard techniques layer on top:
>
> 1. **Bayesian model averaging** combines the six pre-registered models' probabilities daily, weighted by recent log-loss. Standard practice in quant; protects against any single model's regime-specific failures.
> 2. **Hidden Markov regime detection** on (VIX, RV) gates entries: trade only when the unsupervised state classifier confirms a high-vol regime. Reduces whipsaw in low-vol drift periods.
> 3. **Monte Carlo path simulation** (stationary block bootstrap, 10K resamples) produces a Sharpe distribution rather than a single point estimate — the only honest way to claim "positive Sharpe" on N_eff ≈ 100 daily observations.
>
> This is not novel research; it's standard quant infrastructure applied with discipline. The contribution is the rigor — pre-registered specs, autocorrelation-aware inference, regime-aware risk gating, all subject to the same lookahead invariant as Phase 1.

### Anti-overselling guardrails (apply when writing Phase 2 content)

- Don't claim Sharpe > 1.0 without showing the block-bootstrap CI excludes zero.
- Don't claim the strategy "works in all regimes"; report the regime breakdown.
- Don't hide cost sensitivity. Report Sharpe at 5/10/20/50 bps explicitly.
- Don't present RL as the "core" — it's a comparison piece. Rule-based sizing is the core.
- Don't use phrases like "alpha", "edge", "production-ready" unless backed by numbers in the same paragraph.
- Use plain numbers, not adjectives. "Sharpe 0.7" beats "strong risk-adjusted performance."

---

## Block A — Engine + headline result  *(critical path, ~30 min nominal)*

- [x] Build OPRA preprocessor (`features/opra_panel.py`) — 668k contract-day rows
- [x] Yang-Zhang RV target (`features/rv_target.daily_yang_zhang_rv`)
- [x] End-to-end `features/assemble.py` — wires VIX + GEX + RV target with shift(1) lag
- [x] Rebuild `data/processed/features_panel.parquet` w/ `rv_5d_mean` (HAR-X input)
- [x] 6 model classes implemented + factory + pre-registered config
  - [x] logistic
  - [x] logistic_interactions
  - [x] har_x (HAR-X w/ regression-to-binary wrap)
  - [x] xgb_calibrated (sklearn 1.6+ FrozenEstimator API)
  - [x] mlp_small
  - [x] bayesian_head (GP classifier)
- [x] Extended `backtest/metrics.py` — CAGR, ann vol, VaR/CVaR, PSR, block-bootstrap Sharpe CI, info ratio, skew/kurt
- [x] `backtest/runner.py` orchestrator + benchmarks (VXX BAH, cash)
- [x] `make backtest` wired through new runner
- [x] 48 tests pass (incl. no-lookahead invariant + new model conformance tests)
- [x] **Full backtest runs cleanly, no crashes** (XGB sklearn-1.6 FrozenEstimator fix landed)
- [x] Sanity-gate the headline numbers (all PASS):
  - [x] AUC per model in [0.61, 0.67] — within plausible range, NO leakage red flag
        (math-olympiad analysis already predicted target-construction bias would push above 0.56-0.62 prior)
  - [x] Sharpe net in [-1.7, -0.4] — all NEGATIVE; CIs include zero. Strategy structurally
        loses to VXX contango bleed even with real signal. Documented honestly in writeup.
  - [x] VXX BAH benchmark: -52.8% return, -0.08 Sharpe (75.8% ann vol) — cost model realistic
  - [x] Base rate inferable from y_true distribution; matches feature panel ~39% expected

## Block B — Visualization  *(DONE)*

- [x] `report/figures.py` — produce + save PNGs to `report/_build/`
  - [x] Equity curves: one line per model + 2 benchmarks
  - [x] Drawdown timeline (underwater)
  - [x] Calibration / reliability plots (per model, 8-bin)
  - [x] Monthly returns heatmap (one per model)
  - [x] AUC bar chart with block-bootstrap 95% CI
- [x] `make report` wired to figures.py

## Block C — Recruiter writeup  *(DONE)*

- [x] `README.md` — replace scaffold with recruiter-grade front door
  - [x] One-paragraph elevator pitch
  - [x] Headline AUC + Sharpe table (with live numbers)
  - [x] Architecture diagram (ASCII)
  - [x] How-to-reproduce (`make features && make backtest`)
  - [x] Cost-engineering story ($336 → $94.74 monthly-only filter inside free $100 credit)
  - [x] Link to statistical-rigor section
- [x] Drop-in headline numbers from the live backtest

## Block D — Statistical rigor section  *(DONE)*

- [x] `STATISTICAL_RIGOR.md` (top-level, 9 sections)
  - [x] Effective N derivation from autocorrelations
  - [x] Why naïve Sharpe SE is wrong with autocorrelated returns
  - [x] PSR vs nominal Sharpe — how much "edge" survives finite-sample skew/kurt correction
  - [x] Pre-registration commitment (features + hyperparams locked before CV)
  - [x] Honest limitations: 21-month window, GEX-VIX correlation -0.68, contemporaneous-RV-in-denominator target bias
  - [x] Effect-size CIs from block-bootstrap
  - [x] What the strategy losing money means (vehicle choice, not model failure)
  - [x] What would change with more data
  - [x] References

## Block H — Formal recruiter-grade README *(NEW — highest priority)*

User intent: a README that, when the repo is opened on GitHub, immediately reads
as a quant-trading-shop-quality showcase. Dense, high-SNR, every concept defined
from scratch, embedded figures, every section earns its keep.

Structure (18 sections, in order):

1. **Title + tagline** — one-line elevator
2. **TL;DR result table** — headline AUC/Sharpe/PSR with explicit caveats inline
3. **What this strategy bets on** *(the core thesis, explained from scratch)*
   - Vol persistence (vol clustering / GARCH)
   - Variance risk premium (VIX overprices RV; the spread is the trade)
   - Dealer gamma hedging — short gamma amplifies vol, long gamma dampens
   - Why VXX (and why VXX bleeds — contango decay explained from first principles)
4. **The data** — what was pulled, how, costs
   - OPRA options stats + definitions (Databento, $94.74)
   - Daily SPY/VIX/VIX9D/VIX3M/VVIX/VXX (yfinance, free)
   - 3-month T-bill (FRED, free)
   - The 2-stage filtered-pull architecture diagram
5. **Cost engineering case study** — $336 naïve → $94.74 monthly-only filter
6. **Feature engineering** — every feature explained with its mechanism
   - VIX family (8): level, log, 1d/5d change, 20-day z-score, term ratios, VVIX/VIX
   - GEX (4): gex_net, gex_calls, gex_puts, n_contracts — with the BS IV inversion math
   - Yang-Zhang RV target — overnight + GK intraday components
7. **Model architectures** — why six, what each captures
   - logistic (baseline, linear projection through scaled features)
   - logistic_interactions (pre-registered handcrafted)
   - HAR-X (Corsi 2009, the literature gold standard for vol forecasting)
   - xgb_calibrated (non-linear tree + isotonic on time-ordered holdout)
   - mlp_small (sample-size-honest neural sanity)
   - bayesian_head (GP for calibrated uncertainty → downstream sizing)
8. **Backtest methodology** — walk-forward, sizing, execution model
   - Why walk-forward (vs k-fold) — the temporal-leakage argument
   - Linear vs half-Kelly sizing
   - VXX next-open execution + turnover costs + regime-conditional slippage
9. **Results — embedded figures**
   - Equity curves (`report/_build/equity_curves.png`)
   - Drawdown timeline (`report/_build/drawdown.png`)
   - Calibration plots (`report/_build/calibration.png`)
   - AUC + CI bars (`report/_build/auc_bars.png`)
   - Sample monthly heatmap (one model)
10. **Why it doesn't profit (yet)** — VXX vehicle as the structural constraint
11. **Statistical honesty** — pointer to STATISTICAL_RIGOR.md + 1-paragraph summary
12. **Open analyses (TODO)** — what to do once tests are finalised
    - SHAP / coefficient importance per model
    - Regime-conditional performance (vol quartiles)
    - Cost sensitivity (5/10/20/50 bps)
    - Threshold optimization vs PSR-optimization
    - Drawdown attribution per signal
    - Stacking ensemble — does it add edge?
    - Half-Kelly vs linear sizing comparison
13. **Future direction — with more data access**
    - WRDS / OptionMetrics for 10-year GEX (the academic route)
    - VIX futures term-structure data (CBOE)
    - SPY intraday tbbo for microstructure features
    - Cross-asset extension (NQ, RTY, sector ETFs)
14. **Future direction — with better compute**
    - Bayesian hyperparameter search inside walk-forward folds
    - Larger NN architectures with proper regularization
    - Online learning / streaming model updates
    - GPU-accelerated GP for the Bayesian head
15. **Future direction — strategy extensions**
    - Short-VXX side (capture contango decay directly)
    - VIX futures roll-aware execution
    - Multi-instrument: UVXY long / SVXY short hedge
    - Volatility-target sizing (size into VIX z-score regime)
    - Multi-horizon target (1-day + 5-day + 21-day binary)
16. **Tech stack & infrastructure**
17. **Repo layout** (tree)
18. **Reproducibility** + **References** + **License**

Writing style commitments:
- High-SNR dense prose — no padding
- Define every term inline on first use (VXX, GEX, RV, PSR, etc.)
- Embed figures with captions
- Concrete numbers, not "good performance"
- Math notation in TeX where useful (target definition, BS Greeks, PSR)
- Cross-link to STATISTICAL_RIGOR.md and code files

Implementation steps:

- [x] Sketch structure in ship.md
- [x] Verify all 10 figures rendered (10 PNGs in `report/_build/`)
- [x] Write `README.md` v2 — replaced with formal 18-section recruiter-grade README
- [x] Verify all embedded figure paths resolve (relative paths to `report/_build/*.png`)
- [ ] Final pass: read README cold once and trim any padding (after Block E cleanup)

## Block E — Cleanup  *(IN PROGRESS)*

- [x] Quarantine `models/sequence_lstm.py` with a clear "UNUSED IN SHIPPED PIPELINE" header
- [x] Update `CLAUDE.md` — drop LSTM as a deployed model, document the 6 actual variants, update architecture diagram + ingest section
- [x] Update `Outline.md` — prefaced with a status note documenting what was built vs originally designed
- [x] Confirm `.gitignore` already covers `data/raw/`, `data/interim/`, `data/processed/`, `report/_build/`, `report/*.pdf`, `*.parquet`, `*.csv`. Verified — no action needed.
- [x] `data/interim/` files are gitignored (won't be committed)

## Block F — Final pass  *(DONE except git steps the user owns)*

- [ ] (Optional) Clean run from scratch: `make clean && make features && make backtest && make report && make explain && make sensitivity && make paper`
- [x] `pytest -q` — 52/52 passing
- [x] `ruff check .` — clean (per-file-ignores for notebooks documented in pyproject.toml)
- [x] No-lookahead test still passes (in the 52)
- [ ] **USER ACTION**: `git init && git add -A && git commit -m '...'` and push to GitHub
- [ ] **USER ACTION**: GitHub repo metadata (description, topics, pinned)

## Block I — Productionised notebooks  *(NEW — high recruiter ROI)*

User intent: three notebooks that read as a quant-shop interview deliverable.
Markdown-heavy with math in LaTeX, live code that produces every figure,
honest commentary on what worked / didn't.

- [ ] `notebooks/01_strategy_thesis_and_data.ipynb`
  - The vol-regime prediction problem, formal target definition
  - Vol clustering / GARCH visualised in our panel
  - Variance risk premium (VIX vs realised — the spread is the bet)
  - VXX contango decay explained from first principles + visualised
  - Autocorrelation analysis → effective N derivation
- [ ] `notebooks/02_signal_construction.ipynb`
  - VIX-family features one by one, with mechanism
  - GEX construction: BS IV inversion, gamma, greeks, aggregation
  - Yang-Zhang RV target derivation with full formula
  - The crucial GEX-VIX correlation finding (-0.68)
  - No-lookahead invariant visualised
- [ ] `notebooks/03_models_and_results.ipynb`
  - Six architectures explained one by one
  - Walk-forward methodology diagram
  - Headline results with block-bootstrap CIs
  - PSR formula + interpretation
  - The "signal works, strategy doesn't" decomposition
  - Future directions

Each notebook: ~10-15 cells, mix of markdown (explanation, math) and code
(live computation, figures). Productionised = runs top-to-bottom without
manual intervention; reads as a finished analysis, not a working draft.

## Block J — Recruiter deliverables  *(NEW — approved plan, ~10 hrs)*

User-selected high-ROI gaps after a survey of quant-shop / analytical-shop expectations.
See `/Users/prakrit/.claude/plans/is-there-a-different-velvety-hennessy.md` for full spec.

- [x] LICENSE (MIT) — repo root
- [x] Notebook 3: `notebooks/03_models_and_results.ipynb` — six architectures + walk-forward viz + headline results + honest assessment
- [x] `live/predict_today.py` — production-thinking stub. `(date, p_hat, size, action)`. `tests/test_live.py` WRITTEN (4 smoke tests) — NOT executed per user direction.
- [x] `report/explain.py` — feature importance per model (SHAP for trees w/ FrozenEstimator unwrap, coefficients for linear, permutation for MLP/GP w/ sklearn 1.6+ tags). All 6 PNGs in `report/_build/`.
- [x] `backtest/sensitivity.py` — cost × threshold sweep on long-flat sizing. All 6 PNGs (Sharpe + MaxDD heatmaps) in `report/_build/`.
- [x] `paper/spy_vol.tex` — 6-page LaTeX writeup compiles cleanly to `paper/spy_vol.pdf` (378 KB).
- [x] Makefile: `make explain`, `make sensitivity`, `make paper` targets wired.
- [x] README: top-of-doc "Where to look first" section + §9.6/9.7 reference the new artifacts inline (importance, sensitivity, paper PDF, notebooks)
- [x] `pyproject.toml`: added `shap>=0.44` to dev extras; included `live*` in package discovery.

## Block G — Stretch (optional)  *(~2 hrs if time)*

- [ ] `models/stacking.py` — ensemble (only if individual model AUCs span > 0.05)
- [ ] Sensitivity table: linear sizing vs half-Kelly side-by-side  *(superseded by Block J item)*
- [ ] Per-feature SHAP / coefficient importance plot  *(superseded by Block J item)*

---

## Decision gates — STOP and reconsider if...

1. **After Block A**: every model AUC < 0.52 → reframe writeup as "pipeline + statistical-honesty signal." Don't fake edge. The math-olympiad section already pre-warned about this.
2. **After Block A**: any model AUC > 0.70 → audit immediately. Suspects: HAR-X using rv_next, panel join error, target construction. Don't ship until resolved.
3. **End of day 1**: if A took > 4 hrs due to bugs, drop Block G entirely.

---

## What ships at the end

- `README.md` — recruiter front door with headline results
- `data/processed/backtest_summary.csv` — the numbers
- `report/_build/*.png` — the figures
- `STATISTICAL_RIGOR.md` — the math-olympiad honesty section
- Clean repo: tests pass, no dead code, reproducible via `make`
- The cost-engineering story documented
