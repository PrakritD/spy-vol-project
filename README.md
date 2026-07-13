# Short-Volatility Carry on SPY

[![ci](https://github.com/PrakritD/spy-vol-project/actions/workflows/ci.yml/badge.svg)](https://github.com/PrakritD/spy-vol-project/actions/workflows/ci.yml)

**The idea in plain terms.** Selling insurance makes money because people pay extra for certainty. Options on the S&P 500 are crash insurance, and buyers persistently overpay for them: the volatility that option prices imply is almost always higher than the volatility that actually arrives. That gap is the variance risk premium, and whoever sells the insurance collects it. The catch is the one every insurer faces: premiums trickle in daily, claims land all at once. A single volatility spike can hand back years of collected premium in days.

**So the design problem is not finding the premium; it is surviving it.** This strategy sells volatility only while a market-based warning light is off. When the VIX futures curve slopes upward (one-month VIX below three-month, the normal calm state), it shorts VIXY, an ETF built to bleed value in exactly that state. When the curve inverts, the classic pre-spike signature, it steps aside. The curve usually inverts before the spike, not during it, so the book is already flat when the damage lands.

**The result, 2011–2026, net of costs and borrow:** 8.5%/yr (excess of cash) with a maximum drawdown of −15%, under half of what SPY surrendered through Volmageddon and COVID, with every short-vol blowup of the modern era inside the sample. The durable edge is drawdown control: Calmar 0.56 vs 0.38 for SPY, −15% vs −34% peak-to-trough. Sharpe (0.74) runs just behind buy-and-hold SPY, and the write-up says so plainly.

**What makes this repo different:**

- **The trade is deliberately the best-documented one in the volatility literature** (Simon & Campasano 2014, and the family around it). The contribution is not a claimed discovery. It is the evidence standard: full cost and borrow realism, construction-ladder attribution, and selection-aware inference over every variant tried.
- **The no-lookahead guarantee is an executable test, not a promise.** A CI-enforced property test perturbs raw inputs strictly in the future and asserts that earlier positions and P&L are byte-identical. This is leakage detection as evaluation engineering, the same discipline that makes an ML validation trustworthy.
- **Failed models ship as results.** A walk-forward Ridge sizing layer, a logistic direction sleeve, and a dealer-gamma overlay all lose to the parameter-free rule. Each null is reported with its own deflated Sharpe instead of being deleted.

The rule itself is one line, and no parameter in it was fit on this sample: **short VIXY while `VIX < VIX3M` (the curve in contango), flat otherwise.**

**→ Full write-up, attribution, and robustness: [`STRATEGY.md`](STRATEGY.md).**

![Headline dashboard](analysis/figures/strategy_headline.png)

## The two deliverables

1. **[`STRATEGY.md`](STRATEGY.md), the strategy.** The contango-filtered VRP carry above. The construction ladder in §4 isolates where the risk-adjusted return comes from: the term-structure filter more than doubles Calmar (0.23 → 0.56) and halves drawdown (−32% → −15%) by being absent during the regime that produces the losses. Full attribution, cost and borrow stress, and per-regime robustness inside.

2. **[`FINDINGS.md`](FINDINGS.md), the signal investigation.** Does dealer gamma carry next-day realized-volatility information beyond VIX? Mostly not: gamma is almost entirely a VIX echo, a clean null on a calm 21-month options window. But on 15 years across real stress regimes there is a small, statistically robust, gamma-specific increment over a full VIX/HAR baseline (Diebold-Mariano on CRPS, **p = 0.001**). The increment is real and economically marginal, which is why §4b of the strategy finds gamma adds nothing once VIX is already in the model.

![Deep-history result](analysis/figures/deep_history_result.png)

## Method: evaluation engineering

Both deliverables are built so the result can be trusted whether it is large, small, or null:

- **Strict no-lookahead**: every predictor at `t` uses information available by the close of `t`; same-day sources are lagged; gamma is lagged for OCC's T-1 open interest. Enforced by the future-perturbation test in CI.
- **Out-of-sample walk-forward everywhere a model is fit**: expanding windows, refit embargoes, train-only scaling. The signal study scores on the CRPS differential with Diebold-Mariano inference rather than raw correlation.
- **Selection-aware significance**: deflated Sharpe computed over every variant tried (0.66–0.81), not a single p-value on the winner.
- **Per-regime reporting**, never pooled across the 0DTE structural break, with a confound decomposition separating gamma from the DIX flow signal.
- All on **free data**: SqueezeMetrics GEX/DIX (2011→), CBOE VIX (1990→), yfinance SPY/VIXY/VIX-family, FRED.

## Repository layout

| Path | What |
|---|---|
| `analysis/` | the deliverables: `strategy_two_sleeve.py` (strategy), `phase1_*` (deep-history signal study), `make_figure_*` |
| `STRATEGY.md` / `FINDINGS.md` | the two write-ups |
| `notebooks/strategy_walkthrough.ipynb` | a rendered, re-runnable narrative tying both together |
| `features/`, `ingest/`, `configs/` | feature engineering, the free-data fetchers, and the two-stage Databento OPRA pull |
| `tests/` | data-free test suite; the no-lookahead gate runs on synthetic panels, green in CI |
| `docs/ARCHITECTURE.md` | design notes: data flow, invariants, and the reasoning behind them |

## Reproduce

```bash
make install        # editable install + dev tools (pandas/numpy/scipy/scikit-learn/pyarrow/matplotlib)
make test           # data-free test suite (no-lookahead gate on synthetic panels); also runs in CI
make deep           # fetch the free inputs (yfinance, CBOE, FRED, SqueezeMetrics) + validate VIXY splits
make strategy       # VRP-carry backtest + robustness -> analysis/strategy_results.json
make findings       # deep-history gamma study + robustness decomposition
make figures        # regenerate the committed figures
make all            # everything above + execute the walkthrough notebook
```

The notebook **[`notebooks/strategy_walkthrough.ipynb`](notebooks/strategy_walkthrough.ipynb)** renders on GitHub and re-runs from committed, ToS-clean artifacts, so it needs no licensed data. The `make strategy`/`make findings` targets need the free data present first: `make deep` fetches all of it into the git-ignored `data/` tree, records every file's row count and sha256 in `data/raw/deep_manifest.json`, and cross-validates VIXY's reverse-split-adjusted series against VXX. The window end is pinned to the vintage behind the committed results, so a fresh clone reproduces the headline numbers. Raw data is fetched, not committed, because SqueezeMetrics' terms bar redistribution and price history is large.

## License

MIT.
