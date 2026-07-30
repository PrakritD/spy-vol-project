# Robustness appendix

Extended studies behind [`STRATEGY.md`](../STRATEGY.md) §5. Each is standalone and writes its own
`analysis/*_results.json`. These are here rather than in the main document because they answer
"what if I built this differently" rather than "does the headline hold".

---

## Cross-vehicle generalization

The same untuned rule, unmodified, run on three other vol ETPs at the identical 0.20 notional
([`analysis/cross_vehicle.py`](../analysis/cross_vehicle.py)):

| Vehicle | Window | Sharpe | Calmar | maxDD |
|---|---|---|---|---|
| VIXY (headline) | 2011–2026 | 0.72 | 0.53 | −15.4% |
| VXX (short) | 2018–2026 | 0.52 | 0.36 | −15.6% |
| VIXY, same window as VXX | 2018–2026 | 0.50 | 0.36 | −15.4% |
| SVXY (long, no borrow), pooled | 2011–2026 | 0.73 | 0.55 | −11.6% |
| SVXY, pre deleverage (−1x) | 2011–2018-02 | 1.00 | 0.99 | −11.6% |
| SVXY, post deleverage (−0.5x) | 2018-02–2026 | 0.44 | 0.31 | −8.3% |
| UVXY (short, 2x) | 2011–2026 | 0.85 | 0.73 | −22.5% |

VXX, the relaunched note, tracks VIXY closely over identical dates. Both sit near Sharpe 0.5 in the
lower-Sharpe post-2018 sub-period and VXX's maxDD is marginally worse, consistent with two products
sharing the same underlying VIX-futures roll with small fee and methodology differences.

SVXY answers the obvious "why pay borrow shorting a hard-to-borrow ETF" question directly, since
going long in contango needs no borrow at all. Over 2011–2018, while still −1x, it posts the
strongest numbers in the table. ProShares' February 2018 deleverage to −0.5x roughly halves both
Sharpe and Calmar afterwards. The two periods are never pooled, because they are different products
in substance.

UVXY's 2x leverage cuts the other way: the best Sharpe and Calmar in the table, and by far the worst
drawdown at −22.5%, past both the −15.4% headline and the −20.1% next-open-fill figure. The
leverage-decay-and-tail tradeoff shows up where expected.

None of this is a borrow-free shortcut. VIXY and VXX pay the same borrow drag; SVXY's −0.5x NAV
decay is a real cost baked into its price rather than a borrow line item.

## ETF-free implementation: real VIX futures

Everything in the main document trades index-tracking products with their own fees and NAV decay.
CBOE's free per-contract settlement archive turns out to be badly incomplete once probed directly
([`ingest/vix_futures_pull.py`](../ingest/vix_futures_pull.py)): gap-free and correctly scaled
against spot VIX only for 2008-01 to 2013-12, patchy for 2014–2018 (front 8 months only), and the
whole archive stops dead after 2018-02-23. Extending past that needs a paid feed, out of scope at a
$0 budget.

Within the clean window, [`analysis/vix_futures_curve.py`](../analysis/vix_futures_curve.py) builds
a constant-30-day-maturity short directly on the front two VX contracts. No ETP wrapper and no
borrow, since this is a real futures short funded by margin rather than a stock loan.

| | Window | Sharpe | Calmar | maxDD |
|---|---|---|---|---|
| VX constant-maturity short (no borrow) | 2008–2013 | 1.10 | 1.05 | −11.9% |
| VIXY, 2011–2013 subset | 2011–2013 | 1.14 | 1.48 | −8.9% |
| VX constant-maturity short, matched dates | 2011–2013 | 0.86 | 0.98 | −11.0% |

Daily-return correlation over the 2011–2013 overlap is 0.96, so the futures build is a faithful
reconstruction of the same roll VIXY tracks rather than a different signal. On identical dates the
futures version does not beat the ETP despite paying zero borrow: removing the borrow drag was not
enough to offset the approximation in a constant-maturity weighting against VIXY's more granular
roll. This does not settle the capacity question, since a book too large for VIXY's ADV would still
need the futures market, but "no borrow" alone is not a free Sharpe upgrade here.

## Term-structure PCA sizing

[`analysis/vix_futures_term_pca.py`](../analysis/vix_futures_term_pca.py) replaces the binary
contango switch with a signal sized to the roll's predicted magnitude, on the real futures curve
rather than the two-index proxy. Six constant-maturity tenors from 30 to 180 days, a causal PCA
(expanding window, monthly refit, 5-day embargo) extracting a slope score from PC2, sign-fixed each
refit against measured contango depth, used to continuously scale the existing binary gate rather
than replace it. On the post-warmup test window (n = 931, 2010–2013):

| | Sharpe | Calmar | maxDD | Hit rate |
|---|---|---|---|---|
| Binary gate (same matched window) | 1.10 | 1.13 | −12.3% | 55.7% |
| PCA slope-scaled gate | 1.64 | 2.23 | −8.8% | 30.5% |

A real effect in this sample, not one lucky day: HAC t = 3.24, Sharpe only drops to 1.09 with the
ten best days removed, and the two halves of the test window post similar Sharpes (1.81, 1.46). The
hit rate falls because the multiplier sits at zero on about half the days the binary gate would be
short, so this is a fewer-but-better-timed-bets profile.

It is one untuned specification on 931 observations, with no multiplicity sweep of the kind the
headline's 22-variant deflated Sharpe provides. Proof of concept, not a validated result.

## Black-76 groundwork for a tail floor

[`analysis/black76.py`](../analysis/black76.py) is the pricing primitive, validated against put-call
parity and the standard zero-vol, deep-ITM and deep-OTM boundary limits.
[`analysis/black76_tail_floor_demo.py`](../analysis/black76_tail_floor_demo.py) demonstrates it on
the constant-maturity futures curve: a 30-day, 20%-OTM call averaged 1.9% of the forward level
across 2008–2013 (median 1.3%, range 0.2–6.8%).

Illustrative rather than a real quote. No VIX-options market data is ingested in this project, so
sigma is a trailing-60-day realized-vol proxy rather than an implied vol, and real VIX-option IV
trades persistently above realized vol, which is the same premium the whole strategy sells. Every
number here understates the true hedge cost.

One finding survives that caveat. The proxy price was cheaper than the sample average right before
two of the three crisis snapshots checked: 0.99% of forward the day before Lehman and 0.79% the day
before the 2010 flash crash, against a 1.9% mean. That is the same blind spot a realized-vol-only
hedge would have had walking into both.

## Deflated Sharpe under a larger trial count

The headline reports DSR 0.64–0.79 at N = 22 variants. Holding the same dispersion fixed and asking
what the bar would be under more trials than were actually run: 0.49–0.69 at N = 50, and 0.37–0.60
at N = 100. The theory-grounded Bailey–López-de-Prado estimate stays above 0.5 across every N
tabulated; the empirical-dispersion estimate drops below 0.5 by N = 100. Which conclusion holds
depends on which dispersion estimate is trusted, and the empirical one is more conservative.

The Bailey–López-de-Prado formula also assumes i.i.d. returns, and Lo (2002) shows serial
correlation biases the implied standard error of a Sharpe ratio. Carry positions are held for
multi-day stretches between the ~12 annual flips, so the daily return series is autocorrelated and
this DSR carries no Lo-style correction. The block bootstrap in the main document is the more
dependence-aware check of the two.

## Block-length sensitivity in the bootstraps

Both the paired drawdown bootstrap and the gap-risk bootstrap are sensitive to block length in the
same direction, because shorter blocks chop up multi-week crisis clustering and flatter the tail.

Paired drawdown bootstrap, P(maxDD worse than −25%): 0.27 at 15-day blocks, 0.084 at the 90-day
mean blocks used for reported drawdown claims, with 180-day blocks in agreement.

Gap-risk bootstrap, P(maxDD worse than −20% / −25% / −30%): 49%/19%/7% at 30-day mean blocks,
31%/8%/2% at the reported 90-day, 20%/4%/1% at 180-day.

Both resample the rule's own historical dynamics. Neither forecasts the next drawdown.
