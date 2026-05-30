# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SPY next-day realised-volatility regime classifier + VXX long-flat trading strategy. Binary target: `RV_{t+1} > trailing 21-day mean RV`. Signals are GEX (dealer gamma exposure from OPRA options) and lagged VIX-family term structure. The microstructure feature group (OBI, signed flow from intraday SPY tbbo) was designed but **not built** — ARCX SPY tbbo was never pulled because the project ran on the free $100 Databento credit which OPRA stage-2 consumed.

**Status (v2, 2026-05 — supersedes the v1 framing above).** The v1 "classifier → VXX strategy" was audited and its headline "+0.84 Sharpe" shown to be a single-day artifact (`docs/v1-retrospective.md`). The project was reframed to a **dealer-gamma vs realized-volatility signal investigation**. Current deliverable: **`FINDINGS.md`** — dealer gamma is ~95% a VIX echo (a clean null on the calm 21-month OPRA window across six pre-registered formulations) but carries a **small, robust, gamma-specific increment** on 15 years of free deep-history data (gamma-only Diebold-Mariano on CRPS p=0.001; not DIX; survives a richer VIX baseline). Evidence: `analysis/phase1_deep_history.py`, `analysis/phase1_robustness.py`, `analysis/phase0*.py`. Design + scope: `docs/specs/2026-05-29-gamma-regime-vol-design.md`. Analyses run in the `trading` conda env (`python`; pyarrow/scikit-learn/scipy; statsmodels absent — OLS/Newey-West/CRPS are hand-rolled). Free deep data is fetched, not committed (vendor ToS).

**Second deliverable — `STRATEGY.md` (2026-05, audit-hardened).** A risk-managed short-vol **VRP carry** (`analysis/strategy_two_sleeve.py`): short VIXY only when `VIX<VIX3M` (contango). 2011–2026, net of costs+borrow: Sharpe 0.74, Calmar 0.56, maxDD −15%. Honest verdict (after a self-run multi-agent adversarial audit dismantled a first draft's over-claims): it does **not** beat SPY on Sharpe/Sortino; the durable edge is **drawdown control** (Calmar 0.56 vs 0.38). DSR is a **range 0.66–0.81** (not the clone-inflated 0.98 a draft reported); gamma/DIX/timing all null. Figures: `analysis/make_figure_strategy.py`; cited extension roadmap: `docs/strategy-extensions-research.md`; walkthrough: `notebooks/strategy_walkthrough.ipynb`.

**Repo layout (v2).** The live tree is `analysis/` (v2 deliverables), `features/`+`ingest/`+`configs/` (feature-eng + Databento ingest, retained for the 21-month OPRA sub-study), `tests/` (data-free v2 tests), `notebooks/`. **The v1 strategy pipeline is quarantined under `legacy/`** (`models/ backtest/ report/ live/ paper/` + v1 tests + `STATISTICAL_RIGOR.md`); see `legacy/README.md`. The v2 core has zero `features/` dependency — the Yang-Zhang RV helper was vendored to `analysis/rvutil.py`.

## Commands

```bash
make install                 # pip install -e ".[dev]"
make test                    # pytest -q  (data-free; no-lookahead gate on synthetic panels)
make lint                    # ruff check analysis tests
make strategy                # STRATEGY.md backtest -> analysis/strategy_results.json
make findings                # FINDINGS.md deep-history + robustness
make figures                 # regenerate committed figures
make notebook                # execute notebooks/strategy_walkthrough.ipynb in place
make all                     # findings + strategy + figures + notebook + test
```

The v2 gate is `tests/test_strategy.py::test_no_lookahead_end_to_end` (perturbs raw inputs strictly
in the future, asserts earlier positions & cumulative P&L are byte-identical). Use the trading env
python directly for ad-hoc runs: `python analysis/strategy_two_sleeve.py`.

Databento ingest is gated to prevent accidental spend:

```bash
make quote                   # dry-run cost estimate via metadata API. No charge.
make sample                  # 5-day sample pull (small charge). Verify quote first.
make data                    # full pull (real charge). Verify quote first.
```

`make data` only runs stage 1 (OPRA definitions; the `arcx_spy_tbbo` entry in the YAML is `enabled: false` because intraday wasn't pulled). Stage 2 (filtered OPRA statistics) is a separate two-step flow — see Data ingest below.

## Data ingest: two-stage Databento pull

Naive `OPRA.PILLAR.statistics SPY.OPT parent` is far too expensive. The pipeline filters the contract universe first, then pulls statistics against only the kept instrument_ids:

1. **Stage 1** (`make data` / `databento_pull --confirm`): pull `OPRA.PILLAR/definition` for `SPY.OPT` (cheap; small per-contract rows) and `ARCX.PILLAR/tbbo` for SPY equity. tbbo = trade + BBO snapshot at trade time, chosen instead of mbp-1 (~10× cheaper for SPY 2023→2026).
2. **Build id list** (`python -m ingest.build_id_list configs/databento_pulls.yaml`): join definitions against daily SPY spot, filter to live contracts within ±20% moneyness band, DTE ∈ [7, 60], **monthly expiries only** (3rd Friday — dealer OI concentrates here, and the filter cuts ~80% of contract count to fit the free credit). Per-month id-list chunks land in `data/interim/id_list_chunks/*.json`.
3. **Stage 2** (`python -m ingest.databento_pull --quote-stage2` then `--confirm-stage2`): submit one filtered-statistics job per (month, chunk) using `stype_in=instrument_id`. Total ~$94.74 for 2024-08 → 2026-04.

`data/raw/manifest.json` records every batch job (job_id, sha256, dates, sample-vs-full). Resume an in-flight job without re-charging via `python -m ingest.databento_pull --resume <JOB_ID> --name <hint> configs/databento_pulls.yaml`.

Free data (`yfinance` VIX/VIX9D/VIX3M/VVIX/SPY/VXX and FRED `DGS3MO`) lands under `data/raw/yfinance/` and `data/raw/fred/` via separate pull scripts, driven by `configs/free_pulls.yaml`.

## Architecture (v1 — now under `legacy/`)

> The DAG below describes the **quarantined v1 pipeline** (`models/ backtest/ report/ live/` now live in `legacy/`). It is retained for the retrospective; its conclusions are superseded by `FINDINGS.md` / `STRATEGY.md`. The live v2 work is the self-contained scripts in `analysis/`. `ingest/` and `features/` remain at top level (feature-eng + data provenance for the 21-month OPRA sub-study).

The pipeline is a linear DAG of CLI entry points, each reading YAML config and writing parquet under `data/`. Every stage is invokable in isolation:

```
ingest/        Databento batch jobs + free-data pulls   →  data/raw/
ingest/build_id_list.py                                 →  data/interim/id_list_chunks/
features/      gex.py, rv_target.py, vix_termstructure.py — each emits a
               daily panel; features/opra_panel.py reads DBN and produces
               the contract-day panel; features/assemble.py joins them   →  data/processed/features_panel.parquet
models/        Six classes (logistic, logistic_interactions, har_x,
               xgb_calibrated, mlp_small, bayesian_head). All conform to
               the `Model` protocol in models/base.py (`fit`, `predict_proba`).
               models/factory.make_model wires config strings → instances.
               models/sequence_lstm.py exists but is NOT in the shipped pipeline.
backtest/      walk_forward.py (model-agnostic, calls a `model_factory`),
               execution.py (p_hat → VXX size → P&L, confidence sizing +
               regime-conditional turnover costs), sizing.py (linear + Kelly),
               metrics.py (Sharpe + block-bootstrap CI, PSR, Sortino, CAGR,
               VaR/CVaR, …), runner.py (full orchestrator).
report/        figures.py (matplotlib, 10 PNGs to report/_build/),
               render.py (optional Quarto → PDF).
```

### Critical: no-lookahead invariants

The whole project is fiction if the target leaks. `tests/test_no_lookahead.py::test_target_no_future_leak` is the gate — it perturbs RV strictly in the future and asserts earlier labels are unchanged. Do not relax this test.

Label alignment: `features/rv_target.py` produces a daily frame where `y_next` at row `t` reflects `RV_{t+1}` vs the rolling mean ending at `t`. A model sees features at `t` and predicts `y_next[t]`. The `features/assemble.py` joins apply `shift(1)` to any contemporaneous-day source (VIX close, GEX) so date-`t` rows contain only information available at the close of `t`.

### Walk-forward harness

`backtest/walk_forward.run` takes a `model_factory: Callable[[], Model]` and a feature column list. Per segment it trains on all data with `date < train_end` (or rolling window if `rolling_train_months` set) and predicts on `[seg_start, seg_end]`. Output is a long frame of `(date, y_true, p_hat, model_name)` — model-agnostic, so adding a new model means writing a class that conforms to `models/base.Model` and registering it under `configs/experiment.yaml` `models:`.

### Execution and sizing

`backtest/execution.backtest` is the only place that turns probabilities into P&L. Default sizing is linear-confidence (`size = clip(2*p_hat - 1, 0, 1)`); half-Kelly via `backtest.sizing.estimate_kelly_b(train_pnl)` is available per-fold. Costs are `|Δsize| × (base_bps + extra_bps × high_vol_indicator)` where the high-vol flag uses an out-of-sample VIX z-score, so cost scales with turnover rather than position presence. Approximation: P&L uses VXX close-to-close because intraday VXX is not yet pulled — switch to open-to-open when it lands.

### GEX dealer convention

`features/gex.run` follows the practitioner simplification: dealers long calls, short puts. `gex_net = gex_calls - gex_puts`. Document this in any report; it is not a universal convention.

## Conventions

- Configs are the source of truth for windows, thresholds, and feature toggles. Defaults live in dataclasses inside each module, but production runs go through YAML.
- All parquet/manifest writes are relative to `REPO_ROOT = Path(__file__).resolve().parents[1]`. Don't hardcode absolute paths.
- New features should land in `features/`, expose a `run(df, cfg) -> daily_frame` function, and be joined in `features/assemble.py` with explicit `shift(1)` if they use any same-day information.
- Python 3.11+, `from __future__ import annotations` at the top of every module.
