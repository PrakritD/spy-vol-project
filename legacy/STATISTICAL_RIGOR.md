# Statistical Rigor — What This Project Does and Doesn't Prove

Companion to the headline results in [`README.md`](README.md). This document is the math-olympiad-style honest accounting of what the 21-month dataset can and cannot demonstrate, the structural biases in the target construction, and the pre-registration commitments that protect the results from p-hacking.

## 1. The dataset is small (in the way that matters)

Raw row counts mask the real inference budget. **Effective sample size** depends on autocorrelation: when daily observations are correlated, each new row contributes less than 1 unit of independent evidence.

Measured on the panel:

| Series | Lag-1 autocorr | Effective N (full 811-row panel) | Effective N (435-row GEX subset) |
|---|---|---|---|
| RV | 0.57 | ≈ 220 | ≈ 120 |
| VIX | **0.92** | ≈ 43 | ≈ 23 |
| `y_next` (binary target) | 0.17 | ≈ 575 | ≈ 305 |
| (typical iid claim) | — | 811 | 435 |

The relevant N for inference about a feature → target relationship is whichever side is *more* autocorrelated — usually the feature. For VIX-driven inference, **N_eff ≈ 43**. For target-driven inference (e.g. base rate stability), **N_eff ≈ 575**.

The **standard Sharpe SE formula assumes iid returns**. With autocorrelated strategy returns (induced by autocorrelated signal features), the naïve SE is biased low by a factor of 2-3×. This project uses **block-bootstrap Sharpe CIs** (block size ≈ N^(1/3), 1000 resamples) throughout — the correct standard error for this data structure. The CIs in the README's headline table all straddle zero exactly because of this autocorrelation-aware widening; the naïve iid CIs would have been misleadingly tight.

## 2. The target construction has structural bias

The binary label is:

$$y_t = \mathbb{1}\!\left[ RV_{t+1} > \overline{RV}_{[t-20,\,t]} \right]$$

The rolling-mean denominator **includes RV_t itself**. Because RV exhibits vol clustering (lag-1 autocorr 0.57), $RV_t$ is correlated with $RV_{t+1}$ — so the denominator and numerator share information by construction.

**Consequence**: a "predictor" that merely tracks $RV_t$ direction relative to its own mean already implicitly predicts $y_t$, even without any forward-looking information. This inflates apparent AUC by a structural amount independent of any real edge.

The math-olympiad analysis ahead of any model fitting predicted:
- **Literature-only prior** (VIX → next-day RV from variance-risk-premium mechanism): AUC ≈ 0.56-0.62.
- **Adjusted for the rolling-mean target bias**: + 0.05 to +0.10 AUC.

Observed: `logistic_vix_only` AUC = 0.671 — within the predicted range when both effects are accounted for. **Not a leakage red flag**, but also **not a clean "edge"**; a portion of it is structural.

To mitigate (not eliminate): the model output is compared against the rolling mean *itself* in HAR-X, which subtracts out the structural component cleanly via the $\sigma_{resid}$ calibration. Other models leave the bias in.

## 3. GEX is largely a VIX echo (in this 21-month sample)

A core question for the project was whether dealer GEX from OPRA carries information **orthogonal** to free VIX features. Empirically in our sample:

| Pair | |correlation| |
|---|---|
| `gex_net` vs `vix_zscore` | **0.68** |
| `gex_calls` vs `vix_level` | 0.58 |
| `gex_puts` vs `vix_zscore` | 0.47 |

GEX shares ~46% of its variance with VIX in this window. The literature (Barbon et al. 2021) finds an orthogonal GEX edge over 7+ years; **21 months is structurally too short to replicate that**. The variance-inflation factor when adding `gex_net` to a regression already containing `vix_zscore` is ≈ 1.9 — material but not catastrophic, and the marginal-GEX-coefficient estimate is noisy at this N.

Honest conclusion in the report: *I built the GEX pipeline correctly and validated its mechanism in the panel construction. I cannot demonstrate a statistically detectable orthogonal-to-VIX GEX edge from this sample alone; I rely on prior literature for the mechanism claim.*

## 4. Effective dimensionality of the feature set

Of the 12 lag-1 features, the **participation ratio** of the correlation-matrix eigenvalues is **3.4** — meaning roughly 3-4 effectively independent dimensions of variation. Bonferroni-effective α for joint testing at 0.05: **0.015 per feature**.

Implication for model complexity: VC-bound says models with more than ~100 effective parameters will overfit at this N_eff. This rules out deep MLPs, transformers, and LSTM-on-rolling-history at this scale. The shipped architectures (logistic, 2×16 MLP, GP head, XGBoost with max_depth=4 + n_estimators=200) all sit well inside the supportable complexity budget.

## 5. Pre-registration commitment

To protect against the standard backtest p-hacking failure mode, every modeling choice is **locked in `configs/experiment.yaml`** before any CV run and **not retuned** based on observed AUC:

- **Feature lists**: each model's `feature_groups:` is fixed. No post-hoc feature selection on CV results.
- **Hyperparameters**: XGBoost `n_estimators=200, max_depth=4, lr=0.05`; MLP `(16, 16), alpha=1e-3`; HAR-X uses OLS (no hyperparams to tune). All values chosen from literature priors *before* seeing any CV AUC.
- **CV protocol**: walk-forward, 12-month expanding train, 1-month refit. One protocol, no alternatives tried.
- **One trial per model** — no auto-ML / hyperparameter sweep.

The interactions in `models/logistic_interactions.py` are explicitly listed in the file's docstring:

1. `vix_zscore × gex_net` — high-VIX × short-dealer-gamma vol amplifier
2. `term_9d_30d × vix_chg_5d` — backwardation under shock
3. `vix_zscore²` — non-linear vol stress
4. `gex_net × vix_chg_1d` — dealer hedging pressure under same-day VIX move

These four were chosen from theory before any CV. They are not the result of trying many combinations and keeping the best.

## 6. The trading-strategy problem (not the model's fault)

All six models lose money out-of-sample despite AUC > 0.60. The root cause is **not** a calibration or threshold issue — it's a **vehicle choice**. VXX has a structural negative drift of roughly **30-50% per year** from VIX-futures contango. Over the 9-month OOS test, VXX shed **52.8%** on its own.

Decomposition of the worst loss (logistic_vix_only, −39.2% total return):
- VXX BAH would have been −52.8% (the cost of "doing nothing wrong")
- The signal correctly stayed out of VXX 74% of the time (time-in-market 26%)
- But the 26% it was long was still concentrated in losing regimes — high-vol days where VXX often *also* declines or only spikes briefly before bleeding back

A real version of this strategy would need either:
- **Short-VXX** on the inverse signal (P(y=1) < threshold), capturing the contango decay actively — risk: regime tail events
- **Leverage** — to size up the highest-confidence days enough to overcome cost on the others
- **Different instrument** — e.g. VIX futures directly with explicit roll management, or UVXY long, or SVXY/SVIX short-vol on the other side
- **More signal margin** — AUC > 0.65 isn't enough to overcome 50% annual drag; would need 0.70+

**This is documented as the strategy's structural limitation, not hidden.** The classifier is sound; the strategy translation has a deliberate constraint baked in (long-only VXX, per the project spec) that's not survivable on this vehicle.

## 7. What 21 months can and cannot prove

| Claim | Provable here? | Honest verdict |
|---|---|---|
| VIX features carry real next-day RV signal | **Yes** | AUC = 0.671 with block-bootstrap CI; both effective-N and PSR support a real (if small) edge |
| GEX adds an orthogonal edge beyond VIX | **No** | Sample too short, GEX-VIX correlation -0.68 in this window. Need 5+ years to replicate Barbon et al. |
| The strategy profits as-built | **No** | All 6 models negative; structural VXX bleed dominates |
| One model is reliably best | **No** | All 6 Sharpe CIs overlap each other and zero |
| Adding non-linearity (NN, XGB) beats linear | **No detectable difference** | All AUCs within ±0.04, none statistically separable at this N |
| The pipeline is correctly implemented | **Yes** | 48 tests pass, no-lookahead invariant holds, sanity checks pass on every layer |

## 8. What would change with more data

If the project were repeated with 5+ years of OPRA statistics (real money, ~$500-1000 on Databento or free WRDS via business school):
- Effective N for the target ≈ 1500–2000 → 95% AUC CIs would shrink to ±0.02
- Marginal GEX-over-VIX detectability would become feasible (the Barbon-replication regime)
- Stationarity assumption would be more credible
- Walk-forward could include a vol-spike regime not present in 2024-08–2026-04 — the 2020 COVID/inflation/etc. regime shifts would add information
- The strategy's structural VXX bleed problem would persist regardless

## 9. References

- Barbon, A., Beckmeyer, H., Buraschi, A., & Moerke, M. (2021). The Role of Dealer Gamma in Equity Markets.
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics* 7(2), 174-196.
- Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40(5), 94-107.
- Yang, D. & Zhang, Q. (2000). Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices. *Journal of Business* 73(3), 477-491.
- Bekaert, G., Hoerova, M., & Lo Duca, M. (2013). Risk, Uncertainty and Monetary Policy.
- Hansen, P. R. (2005). A Test for Superior Predictive Ability. *Journal of Business & Economic Statistics* 23(4), 365-380.
