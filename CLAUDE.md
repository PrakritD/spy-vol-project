# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Two top-level documents. `RESEARCH.md` is the narrative spine: dealer gamma is almost entirely a
VIX echo (97.8%), with a small, robust, economically marginal increment on deep history, followed
by §6's ML forecasting benchmark where quantile gradient boosting beats a VIX-augmented HAR
baseline on next-day RV CRPS and a small MLP does not. `STRATEGY.md` is the short-volatility
**VRP-carry strategy** the gamma work failed to improve: short VIXY when `VIX < VIX3M`, flat
otherwise; Sharpe 0.72, Calmar 0.53, maxDD −15%; the durable edge is drawdown control, not a
Sharpe beat.

Secondary material lives in `docs/`: `ROBUSTNESS-APPENDIX.md` (cross-vehicle, real-futures
reconstruction, PCA sizing, Black-76), and `CROWDING.md` + `CROWDING-PREREG.md` (a pre-registered
null, demoted from root but kept intact for the timestamped prereg). Design notes and data-flow
detail are in `docs/ARCHITECTURE.md`; read it before structural changes.

Prose in the two root docs is deliberately first-person and low-hedge, aimed at a non-quant
reader first. Do not reintroduce inline caveats on every claim; caveats belong grouped, or in the
appendix.

Analyses run in the `trading` conda env (`python`; pyarrow/scikit-learn/scipy; statsmodels is
absent, so OLS/Newey-West/CRPS are hand-rolled). Data is fetched, not committed (vendor ToS).

## Map (start here when navigating)

| Where | What lives there |
|---|---|
| `analysis/strategy_two_sleeve.py` | the flagship backtest; writes `strategy_results.json`, `strategy_equity.parquet`, `strategy_curves.csv` |
| `analysis/phase1_*.py`, `phase0_gonogo.py`, `phase05*.py`, `phase_skew.py` | RESEARCH.md's deep-history + 21-month OPRA sub-studies (gamma level/path/profile, then put-call skew) |
| `features/opra_panel.py`, `assemble.py`, `gex.py`, `skew.py`, `fast_iv.py` | raw OPRA DBN -> `options_panel.parquet` -> `features_panel.parquet`; `fast_iv.py` is a vectorized IV solver validated against `gex.py`'s scalar one (see `tests/test_fast_iv.py`), used for bulk panel builds only |
| `analysis/strategy_results.json` | the single source of every number quoted in STRATEGY.md |
| `analysis/strategy_curves.csv` | committed, ToS-clean equity curves; the notebook's only data input |
| `analysis/execution_lag.py`, `factor_regression.py`, `drawdown_inference.py`, `gap_risk_mc.py`, `cross_vehicle.py`, `vix_futures_curve.py`, `vix_futures_term_pca.py`, `black76.py`, `black76_tail_floor_demo.py`, `risk_tearsheet.py` | standalone robustness studies; each writes its own `*_results.json` quoted in STRATEGY.md §4e–6a or docs/ROBUSTNESS-APPENDIX.md |
| `analysis/forecast_bench.py` | RESEARCH.md §6's walk-forward ML benchmark (HAR/HAR+VIX vs quantile GBM/MLP); writes `forecast_bench_results.json` |
| `analysis/paper_log.py` | live paper-trade log; appends one row/session to committed `paper_log.csv` |
| `ingest/deep_pull.py` | fetches every flagship data input; manifest in `data/raw/deep_manifest.json` |
| `ingest/vix_futures_pull.py` | free CBOE per-contract VIX futures archive; manifest in `data/raw/vix_futures_manifest.json` |
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
make research                # RESEARCH.md: gamma deep-history + ML benchmark (findings/forecast are aliases)
make figures                 # regenerate committed figures
make notebook                # execute notebooks/strategy_walkthrough.ipynb in place
make research-pdf            # render RESEARCH.md -> report/RESEARCH.pdf (pandoc + LaTeX)
make log                     # append today's close to the live paper-trade log (idempotent)
make all                     # research + strategy + crowding + figures + notebook + test
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
