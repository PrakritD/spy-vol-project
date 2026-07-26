# Pre-registration: crowding and the short-volatility unwind

**Locked 2026-07-27 00:23 AEST, before any analysis code was written.**

This file is committed to the repository deliberately. Its purpose is to fix every specification
below in advance of seeing a single result, so that the outcome reported in `CROWDING.md` cannot
be the product of choices made after the fact. Nothing here may be revised once results exist. If
a specification turns out to be unworkable, the revision and its reason get appended below with a
new timestamp, and the original stays visible.

The design is the same discipline used for the unpublished RL pilot, applied where a reader can
check it.

---

## 1. The claim under test

The strategy in `STRATEGY.md` earns its keep through drawdown depth, not Sharpe. The stated reason
is that the contango filter flattens the book before a volatility spike arrives. That is currently
a narrative. This chapter replaces it with a model that makes falsifiable predictions.

**Model.** Two participant classes hold short-volatility exposure. Leveraged and inverse volatility
ETPs rebalance mechanically and must buy vega when volatility rises. Discretionary carry traders
operate under margin constraints and are forced to cover into the same move. With linear price
impact `λ` on net vega demand and aggregate constrained exposure `K`, an exogenous shock `r₀`
resolves to

```
r = r₀ + λK·r    =>    r = r₀ / (1 − λK)    =>    amplification multiplier  m = 1/(1 − λK)
```

with critical crowding `K* = 1/λ`.

**Strategic layer.** Each trader chooses exposure `wᵢ` where `K = Σwᵢ + k_M`. Carry is a strategic
complement, because the crowd's own hedging suppresses realized volatility and makes the trade look
better as more capital enters. Tail risk is a strategic substitute, because higher `K` raises `m`
for everyone. Each trader internalises their own tail loss but not their marginal contribution to
`λK`. That congestion externality implies `K_Nash > K_opt`, with nothing in the private
optimisation preventing `K_Nash` from crossing `K*`.

---

## 2. Data decision, locked before any result

Three candidate crowding proxies were ranked by preference before any was tested. The ranking is
recorded here so that the choice cannot be attributed to which one produced a better answer.

1. **FINRA consolidated short interest** (preferred). Verified available 2026-07-27 via
   `api.finra.org/data/group/otcMarket/name/consolidatedShortInterest`.
2. VIX futures aggregate open interest, from the `open_interest` column already present in
   `data/raw/vix_futures/vix_futures_panel.parquet`.
3. ETP dollar volume relative to VIX futures dollar volume.

**Option 1 was reached and is therefore the measure used.** Options 2 and 3 are not run. Coverage
verified at lock time: VIXY, UVXY and SVXY each 206 biweekly observations from 2017-12-29 to
2026-07-15 with no gap exceeding 19 days; VXX 199 observations over the same span with one 120-day
gap. Field `daysToCoverQuantity` is floored at 1.00 on every VIXY record and is unusable; days to
cover is computed directly where needed.

### Publication lag

Short interest is reported against a settlement date and disseminated later. Keying the panel on
settlement date would be a lookahead error. FINRA's published schedule was read at lock time across
24 consecutive periods; the settlement-to-publication lag ranges from 9 to 12 calendar days and
never exceeds 12.

**Rule: a short interest observation enters the panel 15 calendar days after its settlement date,
snapped forward to the next trading day.** Fifteen exceeds every observed scheduled lag with margin.
On a biweekly series the extra staleness is immaterial; the protection against a lookahead is not.
A test asserts the rule never undershoots the FINRA schedule.

### Leverage weights

Verified empirically against `data/raw/deep/` rather than assumed, by regressing each ETP's daily
return on VIXY's across the 2018-02-28 ProShares deleverage:

| ETP | beta vs VIXY, pre 2018-02-28 | beta vs VIXY, post | weight used |
|---|---|---|---|
| VIXY | 1.000 | 1.000 | +1.0 |
| VXX | (1x by construction) | (1x) | +1.0 |
| UVXY | +1.988 | +1.477 | +1.5 |
| SVXY | −0.986 | −0.501 | −0.5 |

The inference window lies entirely after the break, so the post-break weights apply throughout.
Short interest in a long-volatility ETP is short-volatility exposure and enters positively. Short
interest in the inverse fund SVXY is long-volatility exposure and enters negatively.

---

## 3. Crowding measures

Let `SIᵢ,t` be short interest in shares and `Pᵢ,t` the price.

- **C1** = `Σᵢ Lᵢ · SIᵢ,t · Pᵢ,t` over VIXY, VXX, UVXY. Leverage-weighted dollar short interest.
- **C2** = `C1 / ADV$`, where `ADV$` is the 21-day rolling median dollar volume of the VIX complex,
  computed as in `analysis/capacity.py`. **This is the primary measure.**
- **C3** = C1 net of the SVXY term.

C2 is primary because the model specifies the loop gain `λK`, and `K/ADV` is that quantity up to a
constant. The primary is chosen by the theory, not by inspection of results. C1 and C3 are
robustness only and may not be used to support a claim that C2 does not support.

---

## 4. Sample and era split

ProShares deleveraged SVXY from −1x to −0.5x and UVXY from 2x to 1.5x effective 2018-02-28. Per the
rule `STRATEGY.md` already applies to SVXY, periods either side are different products in substance
and are never pooled.

- **Inference window: 2018-03-01 to 2026-05-29**, matching the flagship's pinned data end.
- The 2017-12-29 to 2018-02-27 stub is roughly two months and is reported descriptively only. No
  inferential claim may be drawn from it.
- **Volmageddon (February 2018) therefore falls in the descriptive window and is a case study, not
  an inference input.** It is a single event. No parameter is fitted to it.

---

## 5. Baseline

Every predictive test scores against the richer baseline used in `FINDINGS.md` §5b: lagged VIX
level, lagged change in VIX, and the three HAR terms (`har_d`, `har_w`, `har_m` from
`build_signals`). Bare VIX is not an acceptable baseline here.

The dominant failure mode for this chapter is that crowding is a VIX echo, which is what closed out
the gamma study at 97.8% redundancy. **The incremental R² decomposition, mirroring
`vix_echo_decomposition` in `analysis/phase1_robustness.py`, is computed and reported before any
predictive claim is made, not after.**

---

## 6. Predictions and falsification criteria

All crowding terms enter lagged, on top of the publication-date rule. HAC standard errors
throughout, using `hac_tstat` from `analysis/strategy_two_sleeve.py`.

### P1. Amplification

```
Δlog VIX_t = a + b·spy_ret_t + c·(spy_ret_t × C_{t−1}) + d·C_{t−1} + baseline + ε
```

Vol rises when equities fall, so `b < 0`. Amplification under crowding means the sensitivity
steepens, so the prediction is **`c < 0`**.

- Supported if `c < 0` with HAC `t < −2`.
- **Falsified otherwise.**
- Secondary specification, counted as a trial: the same regression restricted to down-days.

### P2. Tail asymmetry

Carry-sleeve drawdowns are deeper when crowding is high. Terciles of `C` formed on a trailing,
causal basis only. Paired stationary bootstrap, 5,000 draws, seed 7, mean block 90 days, matching
`analysis/drawdown_inference.py`.

- Supported if `P(high-crowding maxDD deeper than low-crowding maxDD) > 0.90`, matching the
  evidential bar the repo already applies to its 0.963 depth claim.
- **Falsified otherwise.**

### P3. Premium thinning

Carry sleeve daily excess return regressed on lagged `C` plus baseline.

- Supported if the coefficient is negative with HAC `t < −2`.
- **Falsified otherwise.**

### P4. Signal value

One specification, fixed here, with no sweep. Exposure is the existing binary contango gate
multiplied by 0.5 when trailing-causal `C` sits in its top tercile, and 1.0 otherwise. Evaluated
through `sleeve_excess` with the flagship `CostCfg`, then `metrics`, `deflated_sharpe`,
`block_bootstrap_sharpe`.

- Supported if **both** maxDD and Calmar improve on the binary gate, and the deflated Sharpe
  survives the trial count in §7.
- **Falsified otherwise.**

P4 is the signal test. P1 through P3 are mechanism tests and stand or fall independently of it.

---

## 7. Multiplicity

Trials introduced by this chapter: 3 crowding measures × 4 predictions, plus the one P1 secondary,
equals 13. Deflated Sharpe for P4 is reported over the flagship's existing 22 variants plus these,
and as a curve to N = 100, following the convention already established in `strategy_results.json`.

Separately, and independent of this chapter's outcome, a project-level multiplicity note is added
counting every signal family tested across `FINDINGS.md`, `STRATEGY.md` §4b through §4d, the three
unpublished pilots, and this chapter. Per-study deflated Sharpe does not account for search
conducted across studies on overlapping data, and the repo currently does not say so anywhere.

---

## 8. Known biases, recorded in advance

- **Motivated mechanism.** A confirmed loop would validate the shipped strategy. That makes this
  the highest confirmation-bias exposure in the project, which is why this file exists.
- **Reverse causality.** Large moves change short interest. The publication-date rule plus lagging
  is the defence; it is not a complete one, and the write-up says so.
- **Survivorship in the crowd.** Participants who were liquidated in February 2018 appear in no
  surviving short interest series. The measure understates crowding precisely when crowding matters
  most. This biases against finding the effect, so a positive result is conservative and a null is
  partly explained by it. Both directions get stated.
- **Single instability event.** One observation of the loop firing. `λ` is calibrated from
  ordinary-day depth only and never fitted to February 2018.

---

## 9. Commitments

1. All four predictions are reported at equal prominence whichever way they land.
2. **A null ships as a chapter.** This is pre-committed, not decided after the fact.
3. No specification in this file changes after results exist.
4. No additional crowding measure, sizing rule, or regression specification is added after results
   exist. If one is added anyway, it is labelled post hoc and excluded from every claim.
5. The flagship backtest, its fill convention, and its published numbers are not touched.
