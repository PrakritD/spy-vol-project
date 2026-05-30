# `legacy/` — the superseded v1 pipeline (quarantined, kept on purpose)

This directory holds the **original v1** of the project: a "volatility-regime classifier → VXX long-flat-short strategy." Its conclusions are **superseded** and should not be cited. It is kept — not deleted — because *the reason it was abandoned is itself part of the story* (see [`../docs/v1-retrospective.md`](../docs/v1-retrospective.md)).

## Why it's here and not in the live tree

v1 reported a headline **"+0.84 Sharpe over 82 OOS days."** An audit showed that number was a **single-day artifact** (2026-01-16) that a constant-long position beats — a textbook small-sample / multiple-testing self-deception. The project was reframed around a **falsifiable signal question** ([`../FINDINGS.md`](../FINDINGS.md)) and an **honestly-benchmarked strategy** ([`../STRATEGY.md`](../STRATEGY.md)). The v1 code still runs, but believing its output is the mistake the rest of the repo exists to correct.

## What's in here

| Path | What it was | Status |
|---|---|---|
| `models/` | six v1 classifiers (logistic, HAR-X, calibrated XGBoost, MLP, Gaussian-process head, PPO sizing) | superseded |
| `backtest/` | v1 walk-forward, execution, sizing, metrics, runner | superseded (the v2 strategy has its own self-contained backtest in `../analysis/strategy_two_sleeve.py`) |
| `report/` | v1 figures + Quarto render | superseded |
| `live/` | v1 `predict_today.py` | superseded |
| `paper/` | v1 LaTeX write-up | superseded |
| `tests/` | v1 pipeline/metrics/sizing/execution/no-lookahead tests | superseded (v2 tests live in `../tests/`) |
| `notebooks/` | v1 thesis / signal-construction / models notebooks | superseded (v2 walkthrough is `../notebooks/strategy_walkthrough.ipynb`) |
| `STATISTICAL_RIGOR.md` | rigor note scoped to the 21-month window / old README headline | superseded framing |

## What was deliberately **kept** in the live tree

- `analysis/` — the v2 deliverables (deep-history gamma study + the VRP-carry strategy).
- `features/`, `ingest/`, `configs/` — feature engineering, Databento ingest, and pull configs. These are **shared infrastructure and data provenance** still used by the 21-month OPRA sub-study in `FINDINGS.md` (Part 1), so they are not v1-only and remain live. The v2 headline code does **not** import from them (the Yang-Zhang RV helper was vendored into `analysis/rvutil.py`).
