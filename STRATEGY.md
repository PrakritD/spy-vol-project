# A Risk-Managed Short-Volatility Strategy — and an Honest Account of What It Is and Isn't

**One-line result:** Harvesting the variance risk premium by **shorting VIXY only when the VIX term structure is in contango** delivers, over **15 years (2011–2026, 3,730 trading days) including every short-vol blowup**, a **Sharpe of 0.74** with a **−15% maximum drawdown** — net of costs and borrow. It does **not** beat buy-and-hold SPY on Sharpe (SPY is 0.78 excess-of-rf / 0.88 total-return) or on Sortino. Its **genuine, convention-robust edge is drawdown control: Calmar 0.56 vs SPY 0.38, a maximum drawdown of −15% vs SPY's −34%** — a short-vol book that walked through Volmageddon and COVID with a third of the equity-market drawdown. The honest verdict is a **capital-efficient absolute-return premium, not a Sharpe-beater and not uncorrelated alpha.**

This document deliberately leads with what the strategy is *not*, because an earlier version of this project (`docs/v1-retrospective.md`) fooled itself with a manufactured headline, and because a first internal draft of *this* document over-claimed a "beats SPY Sharpe" result that a self-run adversarial audit then dismantled. The numbers below are what survived that audit.

The companion [`FINDINGS.md`](FINDINGS.md) is the **signal investigation** (does dealer gamma carry RV information beyond VIX?); runnable evidence: [`analysis/strategy_two_sleeve.py`](analysis/strategy_two_sleeve.py).

---

## 1. The edge and the vehicle

**The premium.** Implied volatility (what VIX prices) persistently exceeds subsequently-realized volatility — the **variance risk premium (VRP)**. A seller of volatility is paid this premium plus, on short-dated VIX futures, a **roll yield** when the futures curve is upward-sloping (contango). It is a real, well-documented premium — and a dangerous one: it pays steadily and then, in a vol spike, can give back years of gains in days.

**The vehicle — short VIXY.** VIXY (ProShares VIX Short-Term Futures ETF, inception Jan-2011) rolls front-two-month VIX futures and **decays ≈ −48%/year (CAGR)** from negative roll yield — the long-only series lost essentially all its value over the window; *shorting* it harvests the VRP + roll. Crucially, VIXY is a **single, un-spliced, free series that lived through every disaster** — Aug-2015, **Volmageddon Feb-2018**, **COVID Mar-2020**, 2022. There is no survivorship gloss: the blowups that ruined the short-vol industry (the XIV note was liquidated in Feb-2018) are **in the sample, in full**. VIXY is also **chronically hard-to-borrow**, which the cost analysis in §5 takes seriously.

## 2. The strategy (pre-registered, zero-parameter)

> **Short a fixed, modest notional of VIXY whenever `VIX < VIX3M` (1-month implied vol below 3-month — the term structure in contango). Flatten otherwise.**

`VIX/VIX3M` is the single most-documented term-structure signal; the rule has **no tuned thresholds** — the `<1` boundary is the structural point where the curve crosses from contango to backwardation (`VIX = VIX3M`). The signal is computed from the **prior close** (VIX and VIX3M are end-of-day indices) and the position is held over the next close-to-close day, so there is **no look-ahead** (verified by future-perturbation tests in the internal audit; see §9). P&L is close-to-close; costs are 10 bps per unit turnover plus a borrow fee on the short (stressed in §5).

## 3. Headline performance (full sample, net, blowups in-sample)

Both the strategy and "SPY (excess)" are quoted excess-of-rf (avg rf ≈ 1.6%/yr over the window); "SPY (total)" is the figure investors actually quote for buy-and-hold.

| | Sharpe | Sortino | Calmar | CAGR | Ann vol | maxDD |
|---|---|---|---|---|---|---|
| **Contango-filtered carry (this strategy)** | 0.74 | 0.81 | **0.56** | 8.5% | 11.9% | **−15.3%** |
| Buy-hold SPY (excess-of-rf) | **0.78** | **0.96** | 0.38 | 12.7% | 17.2% | −33.8% |
| Buy-hold SPY (total return) | **0.88** | **1.07** | 0.43 | 14.6% | 17.2% | −33.7% |
| 60/40 (SPY/cash, excess) | 0.78 | 0.96 | 0.37 | 7.8% | 10.3% | −21.3% |

**Read this honestly:** the carry **trails SPY on Sharpe and Sortino** (SPY is the better risk-adjusted *return* engine), and **beats SPY only on drawdown-based metrics** (Calmar, maxDD). The short-vol left tail (skew −1.31, kurtosis 6.1) is exactly what Sortino penalizes — so the strategy losing on Sortino is expected and disclosed. The value proposition is **a third of the drawdown**, not a higher Sharpe.

![Headline dashboard: equity, drawdown, blowup-dodging, borrow sensitivity](analysis/figures/strategy_headline.png)

**It dodges the blowups — verified in-sample.** The filter flattens the short *as the term structure inverts into a spike* and re-enters after:

| Event | strategy P&L over window | long-VIXY move | time in-market |
|---|---|---|---|
| **Volmageddon 2018** | **−3.9%** | +75% | 54% |
| **COVID crash 2020** | **−5.6%** | +273% | 12% |
| 2022 bear | −1.9% | −25% | 94% |

(The daily signal cannot dodge the *intraday* Feb-5-2018 spike — hence −3.9% — but it sidesteps the catastrophe a constant short suffers.)

## 4. Research depth — where the edge is, and where it is NOT

### 4a. Construction ladder: the contango filter is the lever

| Construction | Sharpe | Calmar | maxDD |
|---|---|---|---|
| 1. constant short, no controls | 0.57 | 0.23 | −31.9% |
| 2. + causal vol-targeting (no filter) | 0.44 | 0.22 | −23.1% |
| 3. **+ contango filter ← headline** | **0.74** | **0.56** | **−15.3%** |
| &nbsp;&nbsp;&nbsp;alt: continuous roll-yield sizing | 0.63 | 0.38 | −32.0% |
| 4. + extra signal gates (gamma/vvix/vix_z/liq) | 0.45 | 0.26 | −14.2% |

Causal vol-targeting is **roughly neutral** on VIXY (it does not approach the filter; a first draft's claim that it *hurts* was a look-ahead artifact in the normalization, caught and fixed). The term-structure filter is the real control. Piling additional gates on top (row 4) only sacrifices carry.

![Research: construction ladder + signal attribution](analysis/figures/strategy_research.png)

### 4b. Signal attribution (add-one): every extra signal hurts

Adding any single risk signal *on top of* the contango filter (ΔCalmar vs the filter-only headline):

| Add-on | ΔSharpe | ΔCalmar | verdict |
|---|---|---|---|
| + dealer gamma (neg-γ reduce) | −0.05 | −0.07 | **≈ null** — consistent with [`FINDINGS.md`](FINDINGS.md): gamma is ~95% a VIX echo |
| + vol-of-vol (VVIX z) | −0.10 | −0.14 | hurts |
| + VIX z-score | −0.08 | −0.03 | hurts (slightly improves maxDD only) |
| + liquidity (Amihud) | −0.07 | −0.10 | hurts |

That dealer gamma adds essentially nothing here **corroborates** the signal study: gamma's incremental information beyond VIX is real but economically tiny, and a trading overlay is exactly where "tiny" rounds to zero.

### 4c. The SPY-timing sleeve is an honest null

A walk-forward logistic (expanding window, monthly refit, 5-day embargo, train-only scaling) predicting next-day SPY direction from **DIX flow, dealer gamma, trend, momentum, VIX regime, relative volume** scores an **out-of-sample AUC of 0.51** — a coin flip. The fit collapses to closet-long (70% long / 3% short, corr +0.75 to SPY). **DIX and the other daily signals do not predict next-day SPY direction.** Reported as the null it is.

### 4d. Correlation honesty: a premium, not "alpha"

The carry is **+0.61 correlated to SPY** — short volatility ≈ short tail risk ≈ long equity. It is an **absolute-return risk premium, not an uncorrelated diversifier**; blending it into a SPY book does not materially lift the book's Sharpe. Stated explicitly rather than pitched.

## 5. Robustness (and its limits)

- **Costs / borrow — borrow is the binding axis.** Turnover is ~12 flips/yr (spread is near-irrelevant: 5→30 bps moves Sharpe ~0.05). But the strategy is **short — paying borrow — on ~92% of days**, and VIXY is chronically hard-to-borrow, so borrow is a near-constant cost, not a rare-stress one:

  | VIXY borrow (%/yr) | 0 | 3 (headline) | 5 | 8 | 12 | 18 | 25 |
  |---|---|---|---|---|---|---|---|
  | carry Sharpe | 0.79 | 0.74 | 0.71 | 0.67 | 0.60 | 0.51 | 0.40 |
  | carry Calmar | 0.61 | 0.56 | 0.52 | 0.47 | 0.40 | 0.31 | 0.21 |

  **Sharpe parity with SPY breaks at ~0% borrow** — i.e., net of any realistic borrow the carry does **not** beat SPY's Sharpe. But Calmar/maxDD continue to beat SPY out to ~20%/yr borrow. A VIX-conditioned borrow (base 5% + stress, avg ~6% on short days) gives Sharpe 0.69 / Calmar 0.51 / maxDD −16%. **The drawdown edge is borrow-robust; there is no Sharpe edge over SPY.**

- **Deflated Sharpe = 0.66–0.81** (N=22 variants). This is the honest range: the lower end uses the empirical Sharpe-dispersion across a *genuinely diverse* trial set, the upper end the theory-grounded Bailey–López-de-Prado H₀ per-trial std. An earlier draft reported **DSR 0.98 — that figure was invalid**, inflated by feeding the deflation a set of ~0.87-correlated short-VIXY clones (one a literal duplicate); it is not cited.

- **Block-bootstrap 95% CI on Sharpe: [+0.27, +1.20]; P(Sharpe ≤ 0) = 0.001** — but this is **significance vs zero, with no multiple-testing adjustment**; the selection-aware bar is the DSR above, not this p-value.

- **Out-of-sample persistence (sub-period splits).** The carry is parameter-light, but the edge should persist out of the period it was framed in:

  | Sub-period | carry Sharpe (HAC t) | carry Calmar | maxDD | SPY-excess Sharpe |
  |---|---|---|---|---|
  | 2011–2018 | +0.82 (t=2.4) | 0.75 | −12% | 0.78 |
  | 2019–2026 | +0.67 (t=2.1) | 0.60 | −13% | 0.80 |
  | 2018+ (post-XIV) | +0.50 (t=1.6) | 0.37 | −15% | 0.68 |

  **The edge roughly halves post-2018** (post-XIV, post-0DTE) but stays positive; the carry's Sharpe is at or below SPY's in every split.

- **Per-regime — only pre-2020 is individually significant.** With HAC t-stats and a multiplicity caveat (3 simultaneous blocks): pre-2020 Sharpe +0.81 (t=2.52, **significant**); 2020–21 +0.83 (t=1.42, *not* significant, n=505); 2022+ +0.57 (t=1.35, *not* significant). The positive 2020–21 and 2022+ point estimates should **not** be read as robust — they collapse toward the minus-top-3-days figures (+0.59, +0.42).

- **Threshold stability.** Across contango thresholds 0.97–1.05, Sharpe spans 0.56–0.77 (all positive); the structural `1.00` point gives 0.74 and is **not** the grid maximum (1.05 → 0.77), so it is not cherry-picked.

- **Fragility:** Sharpe minus the single best day = 0.73; minus top-5 = 0.67; minus top-10 = 0.61. Not a few-days artifact.

## 6. Honest limitations

- **Close-to-close fills.** No intraday execution; the daily signal cannot react within a crash day (the −3.9% Volmageddon figure reflects exactly this).
- **Borrow is the swing factor.** VIXY borrow can be expensive; net of realistic borrow there is **no Sharpe advantage over SPY**, only a drawdown advantage. This must be stated plainly.
- **It is short-vol.** The premium is real but the left tail is real too (skew −1.31, kurt 6.1): it earns small and steady and is designed to *flatten* before the gap, not be long it. A true overnight gap through the filter is the residual risk.
- **Edge decay.** Post-2018 the Sharpe roughly halves; a forward-looking reader should anchor on the recent-regime numbers (~0.50–0.57), not the pooled 0.74.
- **Capacity / vehicle.** Results ride on VIXY's tradability and on VIX/VIX3M as the curve proxy. A futures-level implementation (SPVXSTR roll) is the natural next step.

## 7. Where this goes next

A cited, OOS-screened extension roadmap is in **[`docs/strategy-extensions-research.md`](docs/strategy-extensions-research.md)**. The three highest-value free-data upgrades, all consistent with the honest framing above (they attack the *drawdown*, not the Sharpe):

1. **Continuous, magnitude-scaled roll/slope sizing** (walk-forward) to replace the binary contango switch — the value is in sizing on the *predicted magnitude* of the roll, not its sign.
2. **Explicit forward-VRP conditioning** — size on model-free implied variance minus a Yang–Zhang realized-variance forecast, cutting exposure as the *ex-ante* premium collapses (the regime that precedes blowups).
3. **A convex left-tail floor + downside-jump de-gross overlay** — to attack the one thing a daily term-structure gate provably cannot: the *intraday* Feb-2018-style spike. Honestly negative carry, bought to cap the tail.

The roadmap is equally explicit about **dead-ends to drop** (gamma/DIX timing, naive vol-targeting, fixed roll thresholds, rough-vol on daily data) and the **structural verdict**: daily short-vol is a beta-like premium, not alpha — its only honest differentiation is the drawdown profile.

## 8. Reproduce

```bash
# env with numpy/scipy/scikit-learn/pandas/pyarrow + matplotlib; fetchers download free data
/opt/anaconda3/envs/trading/bin/python analysis/strategy_two_sleeve.py   # full backtest + tables -> strategy_results.json
/opt/anaconda3/envs/trading/bin/python analysis/make_figure_strategy.py  # the two figures
```

Data is **fetched, not committed** (SqueezeMetrics' terms bar redistribution; price history is large): SqueezeMetrics GEX/DIX, CBOE VIX, yfinance SPY/VIXY/VIX-family, FRED DGS3MO. Window 2011-07 → 2026-05.

## 9. What this demonstrates

A complete, honest trading study — and, just as importantly, a demonstration of *catching one's own over-claim*. The real result is modest and specific: a short-vol VRP harvest that survived Volmageddon and COVID with a **−15% drawdown vs SPY's −34% (Calmar 0.56 vs 0.38)**, established with pre-registration, no look-ahead, realistic borrow, an honest Deflated Sharpe range, bootstrap, per-regime HAC significance, sub-period persistence, and fragility checks. Equally on the record: it **does not beat SPY on Sharpe or Sortino**, its DSR is 0.66–0.81 (not the clone-inflated 0.98 a first draft reported), its edge halves post-2018, the SPY-timing sleeve is a coin flip, and gamma adds nothing — the same calibrated truth-telling that, in [`FINDINGS.md`](FINDINGS.md), refused a fake edge and refused to dismiss a small real one. The headline is what remained after an internal adversarial audit tried to break it.
