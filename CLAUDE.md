# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

A short-volatility **VRP-carry strategy** on SPY (`STRATEGY.md`, the flagship: short VIXY when
`VIX < VIX3M`, flat otherwise; Sharpe 0.74, Calmar 0.56, maxDD −15%; the durable edge is drawdown
control, not a Sharpe beat) plus the signal investigation behind one candidate input
(`FINDINGS.md`: dealer gamma is almost entirely a VIX echo, with a small, robust, economically
marginal increment on deep history). Design notes, data-flow detail, and the reasoning behind
both live in `docs/ARCHITECTURE.md`; read it before structural changes.

Analyses run in the `trading` conda env (`python`; pyarrow/scikit-learn/scipy; statsmodels is
absent, so OLS/Newey-West/CRPS are hand-rolled). Data is fetched, not committed (vendor ToS).

## Map (start here when navigating)

| Where | What lives there |
|---|---|
| `analysis/strategy_two_sleeve.py` | the flagship backtest; writes `strategy_results.json`, `strategy_equity.parquet`, `strategy_curves.csv` |
| `analysis/phase1_*.py` | the FINDINGS deep-history signal study and its confound decomposition |
| `analysis/strategy_results.json` | the single source of every number quoted in STRATEGY.md |
| `analysis/strategy_curves.csv` | committed, ToS-clean equity curves; the notebook's only data input |
| `analysis/execution_lag.py`, `factor_regression.py`, `drawdown_inference.py` | standalone robustness studies; each writes its own `*_results.json` quoted in STRATEGY.md §4e–5 |
| `ingest/deep_pull.py` | fetches every flagship data input; manifest in `data/raw/deep_manifest.json` |
| `docs/ARCHITECTURE.md` | data flow, no-lookahead invariants, GEX convention, Databento pull detail, design principles |
| `tests/test_strategy.py` | the no-lookahead perturbation gates, golden metric values, pinned synthetic headline |
| `ai/HANDOFF.md` (untracked, private) | REQUIRED READING for any multi-step session: protocol, hard rules, stage docs (`ai/stages/`), canonical numbers (`ai/FACTS.md`), roadmap (`ai/improvement-plan.md`) |

## Commands

```bash
make install                 # pip install -e ".[dev]"
make deep                    # fetch free deep-history inputs (yfinance/CBOE/FRED/SqueezeMetrics)
                             # -> data/raw/deep/, manifest + VIXY split check. No charge.
make test                    # pytest -q  (data-free; no-lookahead gate on synthetic panels)
make lint                    # ruff check analysis tests
make strategy                # STRATEGY.md backtest -> analysis/strategy_results.json
make findings                # FINDINGS.md deep-history + robustness
make figures                 # regenerate committed figures
make notebook                # execute notebooks/strategy_walkthrough.ipynb in place
make all                     # findings + strategy + figures + notebook + test
```

Databento ingest is gated to prevent accidental spend (`make quote` estimates cost with no
charge; `make sample` / `make data` charge real money; see `docs/ARCHITECTURE.md` for the
two-stage OPRA flow). Use the trading env python directly for ad-hoc runs:
`python analysis/strategy_two_sleeve.py`.

## Rules

- The no-lookahead gate (`tests/test_strategy.py::test_no_lookahead_end_to_end` and the ML
  variant) is the project's core guarantee. Do not relax it.
- Every number quoted in a doc must reconcile to `analysis/strategy_results.json` or the
  artifact that produced it.
- Configs are the source of truth for windows, thresholds, and feature toggles. Defaults live in
  dataclasses inside each module; production runs go through YAML.
- All parquet/manifest writes are relative to `REPO_ROOT = Path(__file__).resolve().parents[1]`.
  Do not hardcode absolute paths.
- New features land in `features/`, expose `run(df, cfg) -> daily_frame`, and are joined in
  `features/assemble.py` with explicit `shift(1)` if they use any same-day information.
- Python 3.11+, `from __future__ import annotations` at the top of every module.
- Prose voice: clear, declarative, concise. No em dashes, no "honest/honestly", no
  self-flagellation, no filler tells ("notably", "crucially", "it's worth noting").
