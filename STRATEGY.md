# Short-Volatility Carry on SPY

Index options are structurally expensive. Investors pay up for crash protection they are unwilling to sell, so implied volatility prints persistently above the volatility that actually realizes, and the front of the VIX futures curve sits in contango, each contract rolling down toward a lower spot as it nears expiry. A seller of that curve earns both halves of the gap at once: the spread of implied over realized, and the roll-down as the future converges. Together they are the variance risk premium. The premium is well-documented and dangerous in equal measure, because it accrues quietly for years and then surrenders that accumulation in the few days of a volatility spike.

That asymmetry drives the design. Selling volatility continuously means owning the tail, so this strategy sells only while the term structure is sloped in its favor: short VIXY whenever one-month implied trades below three-month (`VIX < VIX3M`), flat otherwise. Because the curve typically inverts *before* a spike rather than during it, the rule has the position closed by the time the damage lands. Over 2011–2026 (3,791 trading days, every short-vol blowup in the sample), net of costs and borrow, it compounds at **8.1%/yr against a −15.4% maximum drawdown** (excess of cash; roughly 9.9%/yr in total-return terms), under half of what SPY surrendered through Volmageddon and COVID. With fills at the next morning's open instead of the same close, the drawdown is −20.1%, still well under SPY's −33.8% (§5). Sharpe is 0.72 against SPY's 0.76, so the strategy does not win on risk-adjusted return; the drawdown difference is the part that holds up.

This is the strategy [`RESEARCH.md`](RESEARCH.md) set out to improve and did not. Every signal tested as an overlay, dealer gamma included, moved the metrics the wrong way. Runnable evidence: [`analysis/strategy_two_sleeve.py`](analysis/strategy_two_sleeve.py).

**Related work.** Shorting VIX futures or their ETPs in contango is a published trade, not a discovery made here. Simon and Campasano (2014) document the VIX futures basis as a tradeable signal; Cooper (2013) builds the ETP versions; Whaley (2013) quantifies the ETP roll decay; Alexander and Korovilas (2012) examine the products across regimes; Cheng (2019) prices the premium's time variation. I picked a known trade deliberately, so the work would go into testing it rather than claiming a discovery. What this document adds is the testing: a 2011–2026 window containing every modern short-vol blowup at full severity, explicit cost and borrow, selection-aware inference over every variant tried, and a record of the overlays that failed (§4b–4d).

---

## 1. The premium and the vehicle

The premium has two sources, and both are mechanical rather than predictive. The first is the gap between implied and realized variance: VIX prices the market's demand for protection, and that demand keeps implied richer than what subsequently realizes most of the time. The second is roll yield: when the futures curve slopes upward, a long position rolls *up* the curve into a cheaper contract each day and bleeds, which means the short side collects that bleed. Neither requires a forecast. Both pay simply for holding the position while the curve cooperates.

The vehicle is **short VIXY** (ProShares VIX Short-Term Futures ETF, inception Jan-2011), which rolls the front two VIX-futures months. From the negative roll yield it decays at roughly **−48%/yr**, losing essentially all of its value long-only over the window, so shorting it harvests the VRP and the roll directly. VIXY matters here for a second reason: it is a single, un-spliced, free series that lived through every disaster in the modern short-vol record. August 2015, Volmageddon (Feb-2018), the COVID crash (Mar-2020), and 2022 are all in the sample at full severity. The blowups that liquidated the XIV note are not smoothed over; they are exactly what the strategy has to survive. VIXY is also chronically hard to borrow, which is why the cost analysis in §5 treats borrow as the binding cost rather than an afterthought.

## 2. The rule

> **Short a fixed, modest notional of VIXY whenever `VIX < VIX3M`. Flatten otherwise.**

`VIX/VIX3M` is a widely documented term-structure signal, and no parameter in the rule was fit on this sample: the rule comes from the prior literature, and the boundary at `1.0` is the structural point where the curve crosses from contango into backwardation (`VIX = VIX3M`), not a fitted threshold. Both indices are end-of-day, so the signal is read from the prior close and the position is held over the next close-to-close day, which removes any look-ahead (the no-lookahead invariant is enforced by a future-perturbation test in `tests/test_strategy.py`). P&L is close-to-close, costs are 10 bps per unit of turnover, and the short pays a borrow fee that §5 stresses across a wide range.

## 3. Performance

Both the strategy and "SPY (excess)" are quoted excess of the risk-free rate (avg ≈ 1.6%/yr over the window); "SPY (total)" is the figure investors actually quote for buy-and-hold.

| | Sharpe | Sortino | Calmar | CAGR | Ann vol | maxDD |
|---|---|---|---|---|---|---|
| **Contango-filtered carry** | 0.72 | 0.78 | **0.53** | 8.1% | 11.9% | **−15.4%** |
| Buy-hold SPY (excess-of-rf) | 0.76 | 0.93 | 0.36 | 12.3% | 17.2% | −33.8% |
| Buy-hold SPY (total return) | 0.85 | 1.05 | 0.42 | 14.1% | 17.2% | −33.7% |
| 0.6x SPY (vol-matched, excess) | 0.76 | 0.93 | 0.35 | 7.6% | 10.3% | −21.3% |

The carry compounds 3.2x over the window on under half of SPY's drawdown, and it does so with a left tail (skew −1.31, kurtosis 6.1) that is the signature of the premium it harvests: shorting volatility means being paid to carry exactly that downside. The edge the data supports is depth: the maximum drawdown is under half of SPY's (−15% vs −34%), and in a paired block-bootstrap it is the shallower of the two in 96% of resamples (§5). The Calmar gap (0.53 vs 0.36) is economically large, but its bootstrap CI is wide and spans zero, so it is quoted as a magnitude, not a significance claim. On Sharpe and Sortino it runs just behind buy-and-hold SPY, which is the more efficient pure *return* engine; §4e places that comparison in its proper context.

![Headline dashboard: equity, drawdown, blowup-dodging, borrow sensitivity](analysis/figures/strategy_headline.png)

**It avoids the blowups, in-sample.** The filter flattens the short as the term structure inverts into a spike, then re-enters once the curve normalizes:

| Event | strategy P&L over window | long-VIXY move | time in-market |
|---|---|---|---|
| **Volmageddon 2018** | **−3.9%** | +75% | 54% |
| **COVID crash 2020** | **−5.6%** | +273% | 12% |
| 2022 bear | −2.1% | −25% | 94% |

A daily signal cannot react inside the intraday Feb-5-2018 spike, which is the −3.9% figure, but it avoids the far larger loss a constant short takes.

## 4. Where the profit comes from

### 4a. Attribution: the contango filter drives the result

Building the position one control at a time isolates which decision earns the risk-adjusted return:

| Construction | Sharpe | Calmar | maxDD |
|---|---|---|---|
| 1. constant short, no controls | 0.55 | 0.22 | −32.0% |
| 2. + causal vol-targeting (no filter) | 0.43 | 0.20 | −23.2% |
| 3. **+ contango filter (headline)** | **0.72** | **0.53** | **−15.4%** |
| &nbsp;&nbsp;&nbsp;alt: continuous roll-yield sizing | 0.61 | 0.36 | −32.4% |
| 4. + extra signal gates (gamma/vvix/vix_z/liq) | 0.42 | 0.24 | −14.4% |

A constant short already earns the premium (Sharpe 0.55), but it pays for it with a −32% drawdown. Vol-targeting alone is roughly neutral on VIXY, because the asset's payoff is driven by where it sits on the curve rather than by recent realized vol. The contango filter converts the raw premium into a managed one: it more than doubles Calmar (0.22 → 0.53) and halves the drawdown (−32% → −15%) by simply being absent during the regime that produces the losses. Layering further gates on top (row 4) only sacrifices carry without buying additional protection.

![Construction ladder and signal attribution](analysis/figures/strategy_research.png)

### 4b. Signal attribution: nothing improves on the filter

Adding any single risk signal *on top of* the contango filter moves the metrics the wrong way (Δ vs the filter-only headline):

| Add-on | ΔSharpe | ΔCalmar | verdict |
|---|---|---|---|
| + dealer gamma (reduce on neg-γ) | −0.05 | −0.07 | null, consistent with [`RESEARCH.md`](RESEARCH.md) |
| + vol-of-vol (VVIX z) | −0.10 | −0.13 | hurts |
| + VIX z-score | −0.09 | −0.05 | hurts (marginal maxDD help only) |
| + liquidity (Amihud) | −0.07 | −0.10 | hurts |

Dealer gamma adding essentially nothing here is the trading-side corroboration of the signal study: gamma's incremental information beyond VIX is real but tiny, and a position overlay is too coarse to use it. The term-structure signal already prices the regime these add-ons are trying to detect.

### 4c. A learned sizing layer was tested and does not beat the rule

The natural next question is whether the binary in/out gate leaves size on the table, so I built a walk-forward regularized-linear (Ridge) model that predicts next-day carry from the term structure, vol-of-vol, realized-vol lags, and gamma, then sizes the short to the predicted magnitude. It does not help. Sizing the magnitude *within* the contango gate returns Calmar 0.24 at a −18% drawdown, and letting the model *replace* the gate entirely returns Calmar 0.15 at −26%, against the rule's 0.53 and −15%; the learned variants' Deflated Sharpe (0.21–0.30) sits well below the rule's 0.64–0.79. The model is causal by construction (expanding walk-forward, train-only scaling, an expanding exposure normaliser) and held to the same no-lookahead test as the rest of the book. The term structure already prices what the model is trying to learn, so the untuned rule ships. For a case where a learned model does help, on forecasting rather than sizing, see [`RESEARCH.md`](RESEARCH.md) §6; that benchmark does not feed this strategy.

### 4d. A direction sleeve was tested and is a coin flip

A walk-forward logistic (expanding window, monthly refit, 5-day embargo, train-only scaling) predicting next-day SPY direction from DIX flow, dealer gamma, trend, momentum, VIX regime, and relative volume scores an out-of-sample AUC of **0.51**. The fit collapses to closet-long (69% long, 3% short, +0.76 correlated to SPY). DIX and the other daily signals do not predict next-day SPY direction, so the sleeve is reported as the null it is and excluded from the book.

### 4e. What it is, in portfolio terms

The carry is **+0.62 correlated to SPY**, which is the identity of the premium rather than a flaw in the strategy: selling volatility is being short tail risk, which loads on the same bad states as being long equity. That is why its Sharpe lands near SPY's, and why blending it into an equity book does not lift the book's Sharpe much. The differentiation is the drawdown profile, not diversification. The strategy is a capital-efficient way to hold a beta-like risk premium at under half the equity drawdown.

The regression version of that identity ([`analysis/factor_regression.py`](analysis/factor_regression.py)): SPY beta 0.43 (HAC t = 6.5), annualized alpha +3.0% at t = 1.2, not significant; adding the Fama-French five factors and momentum moves the alpha to +3.4% (t = 1.3) with no economically meaningful loadings beyond the market (SMB is +0.07, nominally significant at t = 2.7 but too small to matter). The strategy therefore claims no uncorrelated alpha. What the factor view adds is where the beta lives: 0.18 on SPY up-days, 0.42 on down-days, 0.71 while the book is in-market, 0.00 while flat, and +0.01 (t = 0.1) across SPY's worst decile of days, where the filter has already cut the exposure. One caveat stays in view: the book is still in-market on 81% of worst-decile days and takes a level loss there (mean −1.2%/day), so the tail protection is a vanished slope plus the episode record below, not immunity.

| SPY drawdown > 10% | SPY (total return) | strategy, same dates | time in-market |
|---|---|---|---|
| 2011 debt-ceiling | −18.4% | **−4.8%** | 36% |
| 2015–16 | −13.0% | −6.9% | 85% |
| Volmageddon 2018 | −10.1% | −3.8% | 60% |
| Q4 2018 | −19.4% | −11.8% | 61% |
| COVID 2020 | −33.7% | **−5.6%** | 17% |
| 2022 bear | −24.5% | −10.3% | 93% |
| 2025 | −18.8% | −3.4% | 60% |

## 5. Robustness

- **Borrow is the binding cost.** Turnover is light (~12 flips/yr), so the bid-ask spread barely matters: 5 to 30 bps moves Sharpe by 0.09. But the book is short, and therefore paying borrow, on ~92% of days, and VIXY is chronically hard to borrow, so borrow is a near-constant drag rather than a rare-stress one:

  | VIXY borrow (%/yr) | 0 | 3 (headline) | 5 | 8 | 12 | 18 | 25 |
  |---|---|---|---|---|---|---|---|
  | carry Sharpe | 0.76 | 0.72 | 0.69 | 0.64 | 0.58 | 0.49 | 0.38 |
  | carry Calmar | 0.58 | 0.53 | 0.50 | 0.44 | 0.37 | 0.28 | 0.19 |

  A VIX-conditioned borrow (base 5% plus a stress add-on, averaging ~6% on short days) leaves Sharpe 0.67, Calmar 0.48, maxDD −16%. The drawdown edge over SPY holds past 20%/yr borrow. The Sharpe never overtakes SPY's once any realistic borrow is charged.

- **Execution lag is measured, not assumed.** VIX-family indices print until 4:15pm ET while VIXY stops trading at 4:00pm, so the headline's same-close fill is optimistic. Repricing the fills ([`analysis/execution_lag.py`](analysis/execution_lag.py)):

  | fill | Sharpe | Calmar | maxDD | CAGR |
  |---|---|---|---|---|
  | t−1 close (headline) | 0.72 | 0.53 | −15.4% | 8.1% |
  | next open | 0.72 | 0.41 | −20.1% | 8.3% |
  | next close (+1 full day) | 0.52 | 0.20 | −28.5% | 5.7% |

  The realistic next-open fill leaves the return engine intact and pays in drawdown, because exits hold the short through one overnight gap and VIXY gaps hardest on exactly the nights the curve inverts. The Feb 1–15, 2018 window costs −1.8% at the close print and −8.9% under a full extra day of lag. The signal itself is stable at the 4:00pm cutoff: of ~12.5 flips per year, ~3.6 sit within 1% of the contango boundary. Roughly a quarter of the drawdown advantage is fill convention, so the depth claim is quoted against the next-open row too, where −20% remains well under SPY's −34%.

- **Selection-aware significance.** Deflated Sharpe is **0.64–0.79** across N = 22 variants tried in this document, comfortably above the 0.5 bar. The lower bound uses empirical Sharpe dispersion across the trial set, the upper the theory-grounded Bailey–López-de-Prado per-trial null. What happens under a larger hypothetical trial count, and the i.i.d. assumption this figure rests on, are in the [robustness appendix](docs/ROBUSTNESS-APPENDIX.md). This prices variants tried here, not the search conducted across the repo's other studies; [`docs/CROWDING.md`](docs/CROWDING.md) §6 states that project-level caveat.

- **Block-bootstrap 95% CI on Sharpe: [+0.25, +1.16]; P(Sharpe ≤ 0) = 0.001.** Significance against zero, with no multiple-testing adjustment. The DSR above is the selection-aware bar.

- **The drawdown edge, tested directly.** A paired stationary bootstrap (5,000 draws, 90-day mean blocks, the same resampled dates applied to strategy and SPY; [`analysis/drawdown_inference.py`](analysis/drawdown_inference.py)) puts the strategy's maximum drawdown shallower than SPY's in **96% of draws**, with P(strategy maxDD worse than −25%) = 8.4%. The ΔCalmar 95% CI is [−0.36, +0.48] and spans zero, so the Calmar gap is an economic magnitude, not a significance claim. Block length matters and is tabulated in the [appendix](docs/ROBUSTNESS-APPENDIX.md).

- **Gap risk, quantified.** A true overnight gap through the daily contango gate is the residual risk a once-a-day signal cannot hedge. A regime-conditional stationary block bootstrap of the carry sleeve's own return stream ([`analysis/gap_risk_mc.py`](analysis/gap_risk_mc.py); 5,000 draws, 90-day mean block, contiguous blocks keeping return and regime label paired so crisis clustering survives resampling) puts P(maxDD worse than −20%) at 31%, −25% at 8%, and −30% at 2%, against the realized −15% headline. A materially wider tail than the single realized path shows.

- **Sub-period stability.** No true holdout exists for a rule taken from the published literature, so these splits measure stability rather than out-of-sample skill:

  | Sub-period | carry Sharpe (HAC t) | carry Calmar | maxDD | SPY-excess Sharpe |
  |---|---|---|---|---|
  | 2011–2018 | +0.77 (t=2.3) | 0.70 | −12% | 0.73 |
  | 2019–2026 | +0.66 (t=2.1) | 0.59 | −13% | 0.80 |
  | 2018+ (post-XIV) | +0.49 (t=1.6) | 0.35 | −15% | 0.68 |

  The edge roughly halves after 2018, post-XIV and post-0DTE, but stays positive. A forward-looking reader should anchor on the recent-regime Sharpe of ~0.49, not the pooled 0.72. A rolling 756-day view tells the same story continuously: rolling Sharpe is positive in 99.9% of 3,036 windows (median 0.75), but rolling Calmar only clears SPY's in 54% of them, so the Calmar edge is a persistent tilt rather than a reliable win in every three-year stretch. Per-regime with HAC t-stats: pre-2020 +0.77 (t=2.44, significant); 2020–21 +0.82 (t=1.40, n=505); 2022+ +0.56 (t=1.33). The latter two point estimates collapse toward +0.58 and +0.41 minus their top three days and should not be read as robust alone.

- **Threshold stability and few-days fragility.** Across contango thresholds 0.97–1.05 the Sharpe spans 0.54–0.75, all positive, and the structural `1.00` gives 0.72 without being the grid maximum (1.05 gives 0.75), so it is not cherry-picked. Sharpe minus the single best day is 0.70, minus top-5 is 0.65, minus top-10 is 0.59. The result is not a handful of lucky sessions.

- **Capacity.** A flip trades the whole 0.2x book (`analysis/capacity.py`). VIXY's typical dollar ADV is ~$45.5M, most recently ~$80.6M:

  | Book size | Flip trade | % of typical ADV | % of recent ADV |
  |---|---|---|---|
  | $1M | $200K | 0.44% | 0.25% |
  | $10M | $2M | 4.39% | 2.48% |
  | $50M | $10M | 21.97% | 12.41% |

  At ~12.5 flips/yr a $1–10M book is a rounding error on VIXY's tape. A $50M book starts to move the market on flip days, and VIXY is a note rather than a future, so its own liquidity caps how much can run in this vehicle at all. Locate availability also tends to tighten when the trade most wants to be on, which is a qualitative risk free data cannot size. That, with the borrow drag, is why a futures-level implementation is the version that scales.

- **Data staleness and margin.** The contango flag is never computed from a stale print: VIX3M is present on every panel day in the current vintage, with zero forward-filled observations. Shorting VIXY draws elevated house margin, often 100% of notional, but at a 0.2x book the position is comfortably financeable. Borrow binds, not margin.

Four further studies live in the [**robustness appendix**](docs/ROBUSTNESS-APPENDIX.md): the same rule run on VXX, SVXY and UVXY; an ETF-free reconstruction on real VIX futures, which does not beat the ETP even paying zero borrow; a walk-forward PCA slope-sizing proof of concept on the futures curve; and Black-76 groundwork for a tail floor.

## 6. Limitations

- **Fill convention.** There is no intraday execution, and §5 quantifies what that costs: next-open fills deepen the maximum drawdown to −20%, and a full day of signal lag roughly halves the strategy. The daily signal cannot react within a crash day; the −3.9% Volmageddon figure reflects exactly that.
- **Borrow swings the result.** Net of realistic borrow the strategy keeps its drawdown advantage over SPY while giving up any Sharpe advantage, because the book pays borrow on nearly every day it is short.
- **It is short volatility.** The premium is real and so is the left tail (skew −1.31, kurt 6.1). The strategy is built to flatten *before* the gap, and a true overnight gap through the filter is the residual risk it cannot hedge with a daily signal.
- **Edge decay.** Post-2018 the Sharpe roughly halves; the recent-regime numbers (~0.49–0.56) are the right forward anchor, not the pooled 0.72.
- **Capacity and vehicle.** Results ride on VIXY's tradability and on `VIX/VIX3M` as the curve proxy. A futures-level, no-borrow implementation was tried (§5) but only a clean 2008–2013 free-data window is usable, and it did not clearly beat the ETP even there; extending it to the full sample needs a paid futures feed.

### 6a. Risk in standard vocabulary: VaR, ES, stress, beta

Everything above is stated in Sharpe/Calmar/drawdown terms. For a reader who thinks in VaR and Expected Shortfall, the same carry-sleeve return series ([`analysis/risk_tearsheet.py`](analysis/risk_tearsheet.py)) translates as: 99% 1-day historical VaR is **2.58%**, with Expected Shortfall (the average loss beyond that threshold) at **3.34%**. The sleeve's own recomputed skew and kurtosis (−1.31, 6.11) reconcile to §6's pooled headline numbers exactly. A Cornish-Fisher tail adjustment, which corrects a Gaussian quantile for sample skew and kurtosis, gives a smaller pair here: VaR **1.58%**, ES **2.37%**. That is not the adjustment working as usually advertised: decomposing the expansion shows the negative-skew terms (−0.96 from the linear skew term, −0.64 from the skew² term) dominate the positive kurtosis term (+1.43), pulling the adjusted quantile in *less* conservative than plain Gaussian, let alone historical. This is a known breakdown mode of the Cornish-Fisher expansion at large negative skew, not evidence the true tail is thinner, so the historical estimate, which needs no distributional assumption, is the one to anchor sizing on, not the parametric correction.

The co-drawdown table from §4e reformats directly into a stress-scenario table, unchanged in substance (source: `analysis/factor_regression_results.json`, not recomputed):

| Scenario | SPY move | Strategy move | % days in-market |
|---|---|---|---|
| 2011 US debt-ceiling downgrade | −18.4% | −4.8% | 36% |
| 2015–16 China slowdown / oil crash | −13.0% | −6.9% | 85% |
| Volmageddon 2018 | −10.1% | −3.8% | 60% |
| Q4 2018 rate-shock selloff | −19.4% | −11.8% | 61% |
| COVID 2020 | −33.7% | −5.6% | 17% |
| 2022 rate-hike bear market | −24.5% | −10.3% | 93% |
| 2025 tariff-shock selloff | −18.8% | −3.4% | 60% |

A 126-day (~6-month) rolling beta against SPY ranges from **+0.04 to +1.42** (mean +0.63, current +0.55), wider swings than the static full-sample CAPM beta of 0.43 (§4e) shows on its own. The two are not in conflict: the full-sample pooled beta recomputed here matches §4e's static CAPM beta of 0.43; the gap to the 0.63 rolling mean reflects real time-varying exposure; the mean of many short-window ratios is not the same statistic as one ratio computed on the pooled sample. The book's rolling beta rises around 2018 and again through 2022, consistent with §4e's finding that the beta concentrates on down-days and while the book is in-market: the static beta describes the average exposure, the rolling beta shows when that exposure is elevated.

![Risk tearsheet](analysis/figures/risk_tearsheet.png)

Also available as a standalone one-page PDF leave-behind: [`report/risk_tearsheet.pdf`](report/risk_tearsheet.pdf).

## 7. Where this goes next

The three highest-value free-data upgrades all target the *drawdown*, where the edge is, rather than the Sharpe the strategy is not built to win:

1. **Continuous, magnitude-scaled roll/slope sizing** to replace the binary switch, sizing on the *predicted magnitude* of the roll rather than its sign. A crude version (sizing directly on the VIX3M-VIX slope, `carry_rollyield` in `analysis/strategy_two_sleeve.py`, with a manually chosen 5% full-scale slope and 2x cap, neither fit on this sample nor drawn from the literature) already under-performs the binary gate; a properly walk-forward PCA slope score on the real futures curve looks considerably better in a narrow 2008-2013 test (§5), but it is one untuned specification on 931 observations with no multiplicity sweep, so validating it the way the headline strategy is validated (a variant sweep, a deflated Sharpe) is the next step before it earns any more weight than a proof of concept.
2. **Explicit forward-VRP conditioning**: size on model-free implied variance minus a Yang–Zhang realized-variance forecast, cutting exposure as the *ex-ante* premium collapses, which is the regime that precedes blowups.
3. **A convex left-tail floor** (a VIX-call ladder or SPX put-spread) sized as negative carry, to cap the one thing a daily term-structure gate provably cannot defend: the intraday Feb-2018-style spike. The pricing primitive exists now (§5, Black-76), but it needs real VIX-options quotes, not a realized-vol proxy, before it prices anything but a lower bound.

The dead-ends are equally clear and not worth relitigating: gamma/DIX timing (a VIX echo, see `RESEARCH.md`), crowding as a mechanistic account of the drawdown edge (falsified or not robust across measures, see [`docs/CROWDING.md`](docs/CROWDING.md)), naive vol-targeting (neutral on VIXY), fixed roll-yield thresholds (a textbook out-of-sample failure), and backwardation as a re-entry timer. Daily short-vol remains a beta-like premium, and the drawdown profile is its only durable differentiation.

## 8. Reproduce

```bash
# env with numpy/scipy/scikit-learn/pandas/pyarrow + matplotlib
python -m ingest.deep_pull               # fetch the free inputs; sha256 manifest + VIXY split check
python analysis/strategy_two_sleeve.py   # full backtest + tables -> strategy_results.json
python analysis/execution_lag.py         # fill-convention sensitivity -> execution_lag_results.json
python analysis/factor_regression.py     # CAPM/FF6, state betas, co-drawdowns -> factor_regression_results.json
python analysis/drawdown_inference.py    # paired bootstrap on the drawdown edge -> drawdown_inference_results.json
python analysis/gap_risk_mc.py           # regime-conditional gap-risk bootstrap -> gap_risk_mc_results.json
python analysis/capacity.py              # flip size vs VIXY dollar ADV -> capacity_results.json
python analysis/risk_tearsheet.py        # VaR/ES, stress table, rolling beta -> risk_tearsheet_results.json
python analysis/make_figure_tearsheet.py # one-page risk tearsheet: PNG figure + report/risk_tearsheet.pdf leave-behind
python analysis/make_figure_strategy.py  # hero, headline, research, and rolling figures
```

Data is fetched, not committed (SqueezeMetrics' terms bar redistribution and price history is large): SqueezeMetrics GEX/DIX, CBOE VIX, yfinance SPY/VIXY/VIX-family, FRED DGS3MO. The fetcher pins the window end to the vintage behind the committed results (pass `--end` to extend) and records every file in `data/raw/deep_manifest.json`. Window 2011-05 → 2026-05.

## References

- Alexander, C. and D. Korovilas (2012). *Diversification of Equity with VIX Futures: Personal Views and Skewness Preference.* Working paper, ICMA Centre.
- Cheng, I.-H. (2019). *The VIX Premium.* Review of Financial Studies 32(1), 180–227.
- Cooper, T. (2013). *Easy Volatility Investing.* SSRN working paper 2255327.
- Lo, A. W. (2002). *The Statistics of Sharpe Ratios.* Financial Analysts Journal 58(4), 36–52.
- Simon, D. and J. Campasano (2014). *The VIX Futures Basis: Evidence and Trading Strategies.* Journal of Derivatives 21(3), 54–69.
- Whaley, R. (2013). *Trading Volatility: At What Cost?* Journal of Portfolio Management 40(1), 95–108.
