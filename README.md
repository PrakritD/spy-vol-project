# Short-Volatility Carry on SPY

[![ci](https://github.com/PrakritD/spy-vol-project/actions/workflows/ci.yml/badge.svg)](https://github.com/PrakritD/spy-vol-project/actions/workflows/ci.yml)

Options dealers hedge their inventory, and which way they hedge flips with their gamma position. Long gamma, they lean against moves and volatility gets pinned. Short gamma, they chase moves and volatility gets worse. I found that mechanism convincing enough to build a project around it, and the question I set out to answer was whether it could improve a volatility strategy.

It can't, and the reason is the interesting part.

I tested it against the strongest baseline I could build: a HAR model of realized volatility augmented with the whole VIX family, walk-forward, scored on CRPS, over fifteen years. Dealer gamma does track volatility (short-gamma days carry a Welch t of +28), but **97.8% of that is already inside VIX**. The remaining sliver survives on deep history at p = 0.001 and is far too small to trade. The full study is in **[`RESEARCH.md`](RESEARCH.md)**.

Two explanations, and I think both are true. The first is that VIX is close to a sufficient statistic for this: the same dealers who hedge the inventory are pricing the options, so positioning shows up in the price before it shows up in anyone's flow data. The second is horizon. Gamma hedging is an intraday phenomenon and I tested it on next-day realized volatility, because daily is the granularity I could get for free. I was probably looking at the wrong timescale from the start.

## What survived

The benchmark I was trying to beat. Short VIXY while `VIX < VIX3M`, flat otherwise. One line, no parameter fit on this sample, taken from Simon and Campasano (2014) rather than discovered here. Over 2011–2026, net of costs and borrow, it returns 8.1%/yr excess of cash against a **−15.4% maximum drawdown**, where SPY gave up −33.8%. It is shallower than SPY in 96% of paired bootstrap resamples, and still −20% under next-open fills instead of same-close.

**Sharpe is 0.72, which loses to buy-and-hold SPY at 0.76.** Drawdown depth is the claim the data supports; risk-adjusted return is not, and alpha against SPY is +3.0%/yr at t = 1.22. Every signal I added on top of the rule, gamma included, made it worse. Full write-up in **[`STRATEGY.md`](STRATEGY.md)**.

![Equity and drawdown, VRP carry vs buy-hold SPY](analysis/figures/strategy_hero.png)

## Layout

| Path | What |
|---|---|
| [`STRATEGY.md`](STRATEGY.md) | the strategy: construction, attribution, robustness, risk |
| [`RESEARCH.md`](RESEARCH.md) | the gamma question, and a forecasting benchmark where ML does win |
| [`analysis/`](analysis) | `strategy_two_sleeve.py` (the backtest), `phase1_*` (the gamma study), `forecast_bench.py`, standalone robustness studies |
| [`features/`](features), [`ingest/`](ingest) | raw OPRA decode, feature panels, the free-data fetchers |
| [`tests/`](tests) | data-free suite; the no-lookahead gates run on synthetic panels in CI |
| [`docs/`](docs) | architecture notes, the robustness appendix, and a pre-registered crowding study |
| [`notebooks/strategy_walkthrough.ipynb`](notebooks/strategy_walkthrough.ipynb) | renders on GitHub, re-runs from committed data |

A separate study asked whether crowding in the short-vol trade explains the drawdown edge. I pre-registered four predictions and pushed them before writing the test code; two were falsified and two failed to survive a placebo I ran on myself, where the volume denominator alone reproduced the effect with no crowding data in it. It is a null, and it is in [`docs/CROWDING.md`](docs/CROWDING.md) with its [pre-registration](docs/CROWDING-PREREG.md).

## Reproduce

```bash
make install        # editable install + dev tools
make test           # data-free suite; no-lookahead gates on synthetic panels
make deep           # fetch free inputs (yfinance, CBOE, FRED, SqueezeMetrics) + VIXY split check
make strategy       # backtest + robustness -> analysis/strategy_results.json
make research       # gamma study + ML forecasting benchmark
make figures        # regenerate committed figures
make all            # everything, plus the walkthrough notebook
```

Data is fetched rather than committed, because SqueezeMetrics' terms bar redistribution. `make deep` records every file's row count and sha256 in a manifest, cross-validates VIXY's split-adjusted series against VXX, and pins the window end to the vintage behind the committed results, so a fresh clone reproduces the headline numbers. The one paid input across the whole project is a $95 Databento OPRA pull for the 21-month options sub-study. [`requirements-lock.txt`](requirements-lock.txt) pins the environment. A live signal log runs since 2026-07-13 in [`analysis/paper_log.csv`](analysis/paper_log.csv).

## License

MIT.
