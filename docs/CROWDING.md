# Does Crowding in the Short-Volatility Trade Amplify Its Own Unwind?

## Abstract

`STRATEGY.md` attributes the strategy's drawdown edge to the contango gate flattening the book
before a volatility spike arrives. That is a description of what happens, not an account of why the
spike is violent when it comes. This chapter proposes one, derives predictions from it, and tests
them.

The model treats short volatility as a congestion game. Leveraged and inverse volatility ETPs must
buy vega when volatility rises, discretionary carry traders are forced to cover into the same move,
and with linear price impact the two demands close a feedback loop with amplification
`m = 1/(1 - λK)` in aggregate constrained exposure `K`. Each trader internalises their own tail loss
but not their contribution to `λK`, so within the model the Nash equilibrium sits above the social
optimum, by 1.18 to 1.77 across the parameter sweep. That wedge follows from `m'(x) > 0` rather than
from any particular calibration.

Four predictions were pre-registered in [`CROWDING-PREREG.md`](CROWDING-PREREG.md), committed and
pushed before the test code existed. On 2,073 days of FINRA short interest, two are falsified
outright: tail asymmetry (P = 0.477) and premium thinning (HAC t = −0.20). The other two,
amplification and sizing value, pass on the pre-registered primary measure and reverse sign on the
other two measures of the same construct, which is specification dependence rather than support. A
post-hoc diagnostic suggests why: the ADV denominator alone, with no short interest in it,
reproduces the amplification result at HAC t = −2.59. That diagnostic is labelled and excluded from
the recorded outcomes.

The measure is not a VIX echo, unlike gamma in `RESEARCH.md`: 25.1% of it is explained by a VIX/HAR
baseline, against gamma's 97.8%. It carries little information of its own, explaining 1.2% of log
realized variance alone and 0.013% incremental.

The chapter's result is a null.

---

## 1. The question

Most signal tests in this repository take a candidate predictor and ask whether it forecasts
anything. This one starts from a specific market mechanism and asks whether that mechanism is
detectable.

The mechanism is mechanical rather than behavioural. Leveraged and inverse volatility ETPs rebalance
to a constant leverage target, so a rise in volatility obliges them to buy vega, in a size that is
public and at a time that is known. Discretionary short-volatility traders operate under margin
constraints and are forced to cover into the same move. February 2018 is the standard example: an
inverse ETP complex whose rebalance demand exceeded what the underlying futures could absorb.

If that loop is material at observed exposure levels, shocks should be amplified more when crowding
is high, drawdowns should be deeper in crowded regimes, and the premium should thin as more capital
chases it. If it is not material, the strategy's drawdown edge needs a different account, and the
term-structure gate is doing something simpler.

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

The strategic layer is what distinguishes this from a plain feedback loop. Carry is a strategic
complement, because the crowd's
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

The model implies something weaker than the motivating account. The equilibrium never crosses `K*`
at any calibration tried: `x_Nash` lands at 3 to 11 percent of critical, with amplification of
1.03 to 1.12. What it supports is a persistent inefficiency of roughly 50 percent over-exposure.
Reaching the unstable region requires depth to collapse, which crowding alone does not achieve at
these levels.

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
asserted. Predictions score against the richer baseline from `RESEARCH.md` §5b (HAR terms, VIX
level and z-score, two term-structure ratios, VVIX/VIX, and lagged ΔVIX), not bare VIX. HAC
standard errors throughout.

The VIX-echo decomposition runs first and is reported regardless of what follows, because the
dominant failure mode here is a repackaged VIX, which is what closed out the gamma study at 97.8%
redundancy. The echo share is therefore established before any predictive claim, not after one.

## 5. Results

### 5a. The measure is distinct from VIX, and close to uninformative

Crowding is 25.1% explained by the VIX/HAR baseline, against 97.8% for gamma. It is a distinct
variable rather than a restatement of VIX. It also carries little: 1.2% of log realized variance
alone, and 0.013% incremental over the baseline. That is a different failure mode from the gamma
null, which failed by redundancy rather than by weakness.

### 5b. Two predictions falsified, two not robust

| | Prediction | Primary statistic | Outcome |
|---|---|---|---|
| P1 | Amplification | interaction HAC t = −2.22 | passes as pre-registered, not robust across measures |
| P2 | Tail asymmetry | P(deeper) = 0.477 | falsified |
| P3 | Premium thinning | HAC t = −0.20 | falsified |
| P4 | Signal value | ΔmaxDD +0.0130, ΔCalmar +0.0229 | passes on the primary only, effect economically small |

P2 fails in the direction opposite to the prediction: high-crowding maximum drawdown is −0.093
against −0.131 in low-crowding regimes, so crowded drawdowns are shallower. P3's coefficient is
indistinguishable from zero on all three measures.

P1 and P4 pass only on `c2`. On `c1` and `c3` the amplification interaction is +1.005 (t +4.74) and
+0.976 (t +4.57), the opposite sign and more significant, while P4's conditioning makes both
maximum drawdown and Calmar worse. P4's effect on the primary is small in any case: maximum drawdown
moves from −15.4% to −14.1% and Calmar from 0.390 to 0.413, on a rule that halves exposure on 319 of
2,021 days. Three measures of one construct pointing in different directions is specification
dependence, and this chapter treats it as such rather than as support.

### 5c. A post-hoc diagnostic

The three measures differ only by the ADV denominator, which points at depth rather than crowding.
Putting `1/ADV$` into the interaction slot, with no short interest in it at all, reproduces the
amplification result at HAC t = −2.59, a larger magnitude than the crowding measure manages, and the
two are 0.70 correlated.

This test was not pre-registered. Under prereg §9.4 it is labelled post hoc and excluded from the
recorded outcomes, so P1 stands as it is reported above. It is included because it offers a concrete
explanation for the sign instability in §5b, and because omitting it would leave the reader with a
pre-registered pass and no account of why the other two measures disagree.

### 5d. Model against data

Observed loop gain has a median of 0.0168, against the model's Nash range of 0.0271 to 0.1085.
Realized crowding sits *below* the model's own equilibrium, implying amplification of 1.017. Theory
and data agree that the mechanism is weak at observed exposure levels.

### 5e. February 2018

The aggregate does not exist before the spike: it needs all three long-volatility constituents, and
VXX, the relaunched ETN, has no price history before 2018-01-25. The first aggregate reading is
2018-02-16. This section is therefore on VIXY alone, and it is descriptive.

The build was real and not visible at the time. The last reading published before 5 February was
568,058 shares. Short interest had risen 64.7% to 935,475 by the 31 January settlement, but FINRA
did not publish that until 15 February, ten days after the crash. The publication lag that makes
this study causally sound also meant the measure was stale through the event. That is a property of
the data source rather than of the specification, and it constrains what a real-time signal built
from public short interest can do.

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
three measures against four predictions. Most returned nulls, so no single result has ever needed a
multiplicity correction badly enough to prompt one.

The consequence is specific and it cuts against this chapter's own numbers. A per-study deflated
Sharpe treats its variant count as the selection set. The true selection set is the union across
every family tried, and it is far larger. Any positive result surviving at a per-study threshold is
therefore weaker than its own statistic reports, and the deflated Sharpe of 0.93 on P4 is not doing
real work here in any case: with three trials the across-variant dispersion estimate is degenerate
and the haircut lands near zero.

This note ships whether or not the chapter had found anything, and it applies retroactively to
`RESEARCH.md` and `STRATEGY.md` §4b through §4d.

## 7. Conclusion

Within the model, the congestion externality holds: the Nash-above-social wedge follows from
`m'(x) > 0` and survives 135 parameter combinations. It is not detectable in public data at observed
exposure levels, and the model predicts it should not be, since realized crowding sits at a few
percent of critical.

`STRATEGY.md`'s drawdown edge does not get the mechanistic account this chapter set out to provide.
The term-structure gate flattens the book before spikes, and that remains a description of the
behaviour rather than an explanation of it.

What the null does and does not establish is worth separating. The measure covers ETP short
interest, which is one visible slice of aggregate short-volatility exposure and excludes futures,
options and over-the-counter positions entirely. It is biweekly, and published with a lag long
enough that it missed the one event it would most want to catch. So this rules out that the
publicly measurable slice of crowding predicts amplification, drawdown depth, or premium thinning
on this sample. It does not rule out the mechanism, and §2's model gives a reason to expect the
effect to be small at these exposure levels even if the mechanism is exactly as described.

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
