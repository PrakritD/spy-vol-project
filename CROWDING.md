# Does Crowding in the Short-Volatility Trade Amplify Its Own Unwind?

## Abstract

`STRATEGY.md` earns its keep through drawdown depth, and explains that depth with a story: the
contango gate flattens the book before a volatility spike arrives. This chapter replaces the story
with a model that can fail, and then fails it.

The model treats short volatility as a congestion game. Leveraged and inverse volatility ETPs must
buy vega when volatility rises, discretionary carry traders are forced to cover into the same move,
and with linear price impact the two demands close a feedback loop with amplification
`m = 1/(1 - λK)` in aggregate constrained exposure `K`. Each trader internalises their own tail loss
but not their contribution to `λK`, so the Nash equilibrium sits strictly above the social optimum.
That wedge is structural, running 1.18 to 1.77 across the parameter sweep, and it follows from
`m'(x) > 0` rather than from any calibration.

Four predictions were pre-registered in [`CROWDING-PREREG.md`](CROWDING-PREREG.md), committed and
pushed before the test code existed. All four fail in substance on 2,073 days of FINRA short
interest. Tail asymmetry is falsified outright (P = 0.477, a coin flip) and so is premium thinning
(HAC t = −0.20). Amplification and sizing value pass on the pre-registered primary measure and
reverse sign on the other two measures of the same construct. A post-hoc placebo settles it: the
ADV denominator alone, containing no short interest at all, reproduces the amplification result at
HAC t = −2.59, a larger magnitude than the crowding measure achieves. What passed was market depth
wearing a crowding label.

The measure is not a VIX echo, which distinguishes it from the gamma null in `FINDINGS.md`. It is
only 25.1% explained by a VIX/HAR baseline, against gamma's 97.8%. It is simply close to
uninformative: 1.2% of log realized variance alone, and 0.013% incremental.

---

## 1. The question, and why it is worth asking

Every prior null in this repository killed a signal someone else proposed. This one starts from a
mechanism and asks whether the mechanism is there.

The mechanism is not speculative. Leveraged and inverse volatility ETPs rebalance to a constant
leverage target, so a rise in volatility mechanically obliges them to buy vega, in a size that is
public and at a time that is known. Discretionary short-volatility traders operate under margin
constraints and are forced to cover into the same move. February 2018 is the canonical case: an
inverse ETP complex whose rebalance demand exceeded what the underlying futures could absorb.

If that loop is material at observed exposure levels, three things follow. Shocks should be
amplified more when crowding is high, drawdowns should be deeper in crowded regimes, and the
premium should thin as more capital chases it. If it is not material, the strategy's drawdown edge
needs a different explanation, and the term-structure gate is doing something simpler than the
story suggests.

## 2. The model

Two participant classes hold short-volatility exposure: mechanical rebalancers with price-
insensitive, pre-announced demand, and discretionary traders under a binding risk budget. With
linear impact `λ` on net vega demand and aggregate constrained exposure `K`, a shock `r₀` resolves
to a fixed point:

```
r = r₀ + λK·r    =>    r = r₀ / (1 − λK)    =>    m(K) = 1 / (1 − λK)
```

Critical crowding is `K* = 1/λ`. Below it the loop is a convergent geometric series; at it, any
shock is self-amplifying.

The strategic layer is what makes it a game. Carry is a strategic complement, because the crowd's
own hedging suppresses realized volatility and makes the trade look better as more capital enters.
Tail risk is a strategic substitute, because higher `K` raises `m` for everyone. With payoff
`π = μx − (Θ/2)x² − x·m(x)·s·p` in normalized exposure `x = K/K*`, the atomistic trader solves
`μ − Θx − m(x)sp = 0` while a planner internalises the externality and solves
`μ − Θx − m(x)sp − x·m'(x)sp = 0`. Since `m'(x) = m(x)² > 0`, the planner's extra term is strictly
negative and `x_opt < x_Nash` whenever any exposure is held.

Impact is calibrated from ordinary-day depth only: `λ = 4.36 × 10⁻¹⁰` per dollar, the median of
`|VIXY return| / VIXY dollar volume` over 3,650 days, with 193 days above the 95th absolute-return
percentile excluded so that no crisis observation informs it. This gives `K* = $2.29bn` at the
baseline `κ = 1`, ranging $1.15bn to $4.58bn across the sweep. `analysis/crowding_model.py` never
loads a crowding series at all, so `K*` is a prediction rather than a fit.

**The model's own verdict is weaker than the story that motivated it.** The equilibrium never
crosses `K*` at any calibration tried: `x_Nash` lands at 3 to 11 percent of critical, with
amplification of 1.03 to 1.12. What the model supports is a persistent inefficiency of roughly 50
percent over-exposure, not spontaneous instability. Instability requires depth to collapse, not
crowding to build.

## 3. Data

FINRA consolidated short interest for VIXY, VXX, UVXY and SVXY, biweekly, settlement dates
2017-12-29 to 2026-04-30. Free. Short interest in a long-volatility ETP is short-volatility
exposure held by someone; short interest in the inverse fund SVXY is the opposite and enters
negatively. Funds are weighted by leverage, verified empirically rather than assumed by regressing
each ETP's daily return on VIXY's across the 2018-02-28 ProShares deleverage: UVXY +1.988 before
and +1.477 after, SVXY −0.986 and −0.501.

Two data properties determine whether any of this means anything.

**Publication lag.** FINRA reports against a settlement date and disseminates over a week later.
Keying the panel on settlement dates would let it see numbers nobody had. The published schedule
was read across 24 consecutive periods: the lag runs 9 to 12 calendar days and never exceeds 12.
Observations enter at settlement plus 15 calendar days, snapped to the next trading day, with a
further `shift(1)` at the join.

**Split adjustment.** FINRA reports the actual share count outstanding; stored prices are
back-adjusted for splits; these ETPs reverse-split relentlessly, UVXY 13 times and VIXY 6. The
naive product overstates notional by the cumulative split factor, which reaches 2,500 on UVXY
inside this window and produced a $3.1 trillion aggregate on a first pass. Splits are pulled to
disk and divided out, giving contemporaneous prices with true medians of $21.94 VIXY and $19.41
UVXY. The ratio's denominator was never affected, since the two adjustments cancel in volume times
price.

Three measures: `c1` is leverage-weighted dollar short interest across the long-volatility ETPs,
`c2 = c1/ADV$` is the primary and is the model's loop gain up to a constant, `c3` is `c1` net of
SVXY. Inference runs 2018-03-01 to 2026-05-29, n = 2,073, entirely after the deleverage so the two
leverage regimes are never pooled.

## 4. Method

Every specification was fixed in [`CROWDING-PREREG.md`](CROWDING-PREREG.md) and pushed to this
repository before `analysis/crowding_test.py` existed, so the ordering is checkable rather than
asserted. Predictions score against the richer baseline from `FINDINGS.md` §5b (HAR terms, VIX
level and z-score, two term-structure ratios, VVIX/VIX, and lagged ΔVIX), not bare VIX. HAC
standard errors throughout.

The VIX-echo decomposition runs first and is reported regardless of what follows, because the
dominant failure mode here is a repackaged VIX, which is what closed out the gamma study at 97.8%
redundancy. Establishing the echo share before any predictive claim is the point.

## 5. Results

### 5a. The measure is not a VIX echo, it is close to uninformative

Crowding is 25.1% explained by the VIX/HAR baseline. Gamma was 97.8%. This is a genuinely distinct
variable. It also carries almost nothing: it explains 1.2% of log realized variance alone, and adds
0.013% over the baseline. A different failure mode from every prior null in this repository, and a
cleaner one.

### 5b. All four predictions fail

| | Prediction | Primary statistic | Outcome |
|---|---|---|---|
| P1 | Amplification | interaction HAC t = **−2.22** | passes the letter, fails in substance |
| P2 | Tail asymmetry | P(deeper) = **0.477** | **falsified** |
| P3 | Premium thinning | HAC t = **−0.20** | **falsified** |
| P4 | Signal value | ΔmaxDD **+0.0130**, ΔCalmar **+0.0229** | primary-only, economically trivial |

P2 fails in the direction opposite to the prediction: high-crowding maximum drawdown is −0.093
against −0.131 in low-crowding regimes, so crowded drawdowns are *shallower*. P3's coefficient is
indistinguishable from zero on all three measures.

P1 and P4 pass only on `c2`. On `c1` and `c3` the amplification interaction is **+1.005 (t +4.74)**
and **+0.976 (t +4.57)**, the opposite sign and considerably more significant, while P4's conditioning
makes both maximum drawdown and Calmar worse. Three measures of one construct pointing in
contradictory directions is specification dependence, not evidence.

### 5c. The placebo that settles it

The three measures differ only by the ADV denominator, so the disagreement points at depth rather
than crowding. Putting `1/ADV$` into the interaction slot with no short interest in it at all
reproduces the effect at **HAC t = −2.59**, a larger magnitude than the crowding measure manages, at
0.70 correlation with it.

P1's pre-registered pass is a market-depth result. This test is post hoc, labelled as such, and
excluded from every claim under prereg §9.4. It does not change P1's recorded outcome. It changes
what that outcome means.

### 5d. Model against data

Observed loop gain has a median of 0.0168, against the model's Nash range of 0.0271 to 0.1085.
Realized crowding sits *below* the model's own equilibrium, implying amplification of 1.017. Theory
and data agree that the mechanism is weak at observed exposure levels.

### 5e. February 2018, and why the measure would not have helped

The aggregate does not exist before the spike: it needs all three long-volatility constituents and
VXX, the relaunched ETN, has no price history before 2018-01-25. The first aggregate reading is
2018-02-16.

On VIXY alone, the build was real and invisible in time. The last reading published before 5
February was 568,058 shares. Short interest had risen 64.7% to 935,475 by the 31 January
settlement, but FINRA did not publish that until 15 February, ten days after the crash. The
publication lag that makes this study causally sound also blinds the measure at the one moment it
would have mattered. That is a property of the data source, not of the specification, and it caps
what any real-time crowding signal built from public short interest could ever do.

VIXY returned +51.6% over 5 to 9 February. The carry sleeve returned −0.02%, because the
term-structure gate was already flat.

## 6. Search conducted across this project

Every study here reports its own deflated Sharpe or its own Diebold-Mariano p-value. None of them
accounts for search conducted *across* studies on overlapping data, and this repository has now
tested a large number of signal families against closely related targets on the same underlying:
dealer gamma in six formulations, dark-pool flow, put-call skew in two, walk-forward Ridge sizing,
a walk-forward logistic direction sleeve, gamma, VVIX, VIX-z and liquidity gates as construction
add-ons, vol targeting, continuous roll-yield sizing, term-structure PCA sizing, fitted-Q
reinforcement sizing, a cross-index universe study, an intraday gamma test, and now crowding in
three measures against four predictions. Most were nulls, which is why the accumulated selection
problem has stayed invisible.

The consequence is specific and it cuts against this chapter's own numbers. A per-study deflated
Sharpe treats its variant count as the selection set. The true selection set is the union across
every family tried, and it is far larger. Any positive result surviving at a per-study threshold is
therefore weaker than its own statistic reports, and the deflated Sharpe of 0.93 on P4 is not doing
real work here in any case: with three trials the across-variant dispersion estimate is degenerate
and the haircut lands near zero.

This note ships whether or not the chapter had found anything, and it applies retroactively to
`FINDINGS.md` and `STRATEGY.md` §4b through §4d.

## 7. Conclusion

The congestion externality is real as theory: the Nash-above-social wedge follows from `m'(x) > 0`
and holds across 135 parameter combinations. The amplification loop is not detectable in public
data at observed exposure levels, and the model itself predicts it should not be, since realized
crowding sits at a few percent of critical.

`STRATEGY.md`'s drawdown edge therefore does not get the mechanistic explanation this chapter set
out to give it. The term-structure gate flattens the book before spikes, and that remains the
description rather than the cause. Crowding, as measurable from free data, is not why.

The methodological result is more durable than the empirical one. A pre-registration published
before the tests, a placebo that dismantled the one pre-registered pass, a false positive caught in
a self-imposed normalization, and a no-lookahead gate rewritten after mutation testing showed the
first version could not fail, all did more work than a positive finding would have.

## 8. Reproduce

```bash
# env with pandas/numpy/scipy/pyarrow; free data, no licensed feeds
python -m ingest.short_interest_pull          # FINRA short interest + splits -> data/raw/short_interest/
python -m ingest.short_interest_pull --check  # validate coverage and the publication-lag rule
python analysis/crowding_model.py             # -> analysis/crowding_model_results.json
python analysis/crowding_test.py              # -> analysis/crowding_test_results.json
pytest tests/test_crowding.py -q              # no-lookahead, publication-lag and model gates
```

Every number above traces to one of those two JSON artifacts.

## 9. References

Brunnermeier, M. and Pedersen, L. (2009). Market Liquidity and Funding Liquidity. *Review of
Financial Studies* 22(6).

Kyle, A. (1985). Continuous Auctions and Insider Trading. *Econometrica* 53(6).

Politis, D. and Romano, J. (1994). The Stationary Bootstrap. *Journal of the American Statistical
Association* 89(428).

Bailey, D. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio
Management* 40(5).

Simon, D. and Campasano, J. (2014). The VIX Futures Basis: Evidence and Trading Strategies.
*Journal of Derivatives* 21(3).
