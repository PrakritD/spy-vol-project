# Short-Volatility Carry on SPY

[![ci](https://github.com/PrakritD/spy-vol-project/actions/workflows/ci.yml/badge.svg)](https://github.com/PrakritD/spy-vol-project/actions/workflows/ci.yml)

A systematic short-volatility strategy that harvests the variance risk premium and stays in the trade through the spikes that wiped out everyone who held the short unhedged. Over **2011–2026, net of costs and borrow, it compounds at 8.5%/yr against a −15% maximum drawdown**, under half of what SPY surrendered through Volmageddon and COVID, with every short-vol blowup of the modern era inside the sample.

The reasoning is first-principles. Index options are structurally expensive: investors pay up for crash protection they will not sell, so implied volatility prints above realized and the VIX futures curve sits in contango, each contract rolling down toward a lower spot as it nears expiry. Selling that curve collects the implied-vs-realized gap and the roll-down together, which is the variance risk premium. The catch is the left tail: one vol spike hands back years of carry in days. So the rule is one line, with no tuned parameter. Short VIXY while `VIX < VIX3M` (the curve in contango), flat otherwise, so the book is already closed by the time the curve inverts into a spike.

**→ Full write-up, attribution, and robustness: [`STRATEGY.md`](STRATEGY.md).**

![Headline dashboard](analysis/figures/strategy_headline.png)

## The two deliverables

1. **[`STRATEGY.md`](STRATEGY.md), the strategy.** The contango-filtered VRP carry above. The construction ladder in §4 isolates where the risk-adjusted return comes from: the term-structure filter more than doubles Calmar (0.23 → 0.56) and halves drawdown (−32% → −15%) by being absent during the regime that produces the losses. Sharpe 0.74, Calmar 0.56, maxDD −15%, full attribution and per-regime robustness inside.

2. **[`FINDINGS.md`](FINDINGS.md), the signal investigation.** Does dealer gamma carry next-day realized-volatility information *beyond* VIX? Mostly not: gamma is ~95% a VIX echo, a clean null on a calm 21-month options window. But on 15 years across real stress regimes there is a small, statistically robust, gamma-specific increment over a full VIX/HAR baseline (Diebold-Mariano on CRPS **p = 0.001**), which is genuinely gamma and not the flow signal that ships alongside it. The increment is real and economically marginal, which is why §4b of the strategy finds gamma adds nothing once VIX is already in the model.

![Deep-history result](analysis/figures/deep_history_result.png)

## Method

Both deliverables are built to make the result trustworthy whether it is large or small:

- **Strict no-lookahead** (every predictor at `t` uses information available by the close of `t`; gamma lagged for OCC's T-1 open interest), enforced by a future-perturbation test in CI.
- **Out-of-sample expanding walk-forward**; the signal study uses the nested **Diebold-Mariano test on the CRPS differential** rather than raw correlation, and block-bootstrap for the strategy.
- **Per-regime reporting**, never pooled across the 0DTE structural break; a confound decomposition that separates gamma from the DIX flow signal and from stale-VIX.
- **Selection-aware significance**: Deflated Sharpe over every variant tried (0.66–0.81), not a single p-value.
- All on **free data**: SqueezeMetrics GEX/DIX (2011→), CBOE VIX (1990→), yfinance SPY/VIXY/VIX-family, FRED.

## Repository layout

| Path | What |
|---|---|
| `analysis/` | the deliverables: `strategy_two_sleeve.py` (strategy), `phase1_*` (deep-history signal study), `make_figure_*` |
| `STRATEGY.md` / `FINDINGS.md` | the two write-ups |
| `notebooks/strategy_walkthrough.ipynb` | a rendered, re-runnable narrative tying both together |
| `features/`, `ingest/`, `configs/` | feature engineering and the two-stage Databento OPRA pull (the 21-month options sub-study in `FINDINGS.md`) |
| `tests/` | data-free test suite; the no-lookahead gate runs on synthetic panels, green in CI |
| `CLAUDE.md` | build commands, architecture, data-ingest, and design notes in one place |

## Reproduce

```bash
make install        # editable install + dev tools (pandas/numpy/scipy/scikit-learn/pyarrow/matplotlib)
make test           # data-free test suite (no-lookahead gate on synthetic panels); also runs in CI
make strategy       # VRP-carry backtest + robustness -> analysis/strategy_results.json
make findings       # deep-history gamma study + robustness decomposition
make figures        # regenerate the committed figures
make all            # everything above + execute the walkthrough notebook
```

The notebook **[`notebooks/strategy_walkthrough.ipynb`](notebooks/strategy_walkthrough.ipynb)** renders on GitHub and re-runs from committed, ToS-clean artifacts, so it needs no licensed data. The `make strategy`/`make findings` targets do need the free data present; their fetchers download it into the git-ignored `data/` tree. Raw data is fetched, not committed, because SqueezeMetrics' terms bar redistribution and price history is large.

## License

MIT.
