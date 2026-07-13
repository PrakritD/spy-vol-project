# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

A short-volatility **VRP-carry strategy** on SPY, plus the signal investigation behind one of its candidate inputs (dealer gamma versus realized volatility). Two deliverables:

- **`STRATEGY.md`** (the flagship): short VIXY only when `VIX < VIX3M` (term structure in contango), flat otherwise. 2011–2026, net of costs and borrow: Sharpe 0.74, Calmar 0.56, maxDD −15%, CAGR 8.5%. The durable edge is drawdown control (Calmar 0.56 vs SPY 0.38, −15% vs −34%); it does not beat SPY on Sharpe, and the construction ladder attributes the risk-adjusted gain to the contango filter. A walk-forward Ridge ML sizing layer (`ml_size_positions`) was tested and does not beat the parameter-free rule (Calmar 0.29 within-gate / 0.11 replacing it, vs 0.56; STRATEGY.md §4c), so the rule ships. Code: `analysis/strategy_two_sleeve.py`; figures: `analysis/make_figure_strategy.py`.
- **`FINDINGS.md`** (the signal study): dealer gamma is ~95% a VIX echo, a clean null on the calm 21-month OPRA window across six pre-registered formulations, but carries a small, robust, gamma-specific increment over a full VIX/HAR baseline on 15 years of free deep-history data (gamma-only Diebold-Mariano on CRPS p=0.001; not DIX; survives a richer VIX baseline). Economically marginal, which is why gamma adds nothing to the strategy once VIX is in the model. Code: `analysis/phase1_deep_history.py`, `analysis/phase1_robustness.py`, `analysis/phase0*.py`.

Analyses run in the `trading` conda env (`python`; pyarrow/scikit-learn/scipy; statsmodels is absent, so OLS/Newey-West/CRPS are hand-rolled). Free deep data is fetched, not committed (vendor ToS). An earlier version of the project (a volatility-regime classifier feeding a VXX long-flat strategy) was retired; see "v1 lessons" below.

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

The gate is `tests/test_strategy.py::test_no_lookahead_end_to_end` (perturbs raw inputs strictly
in the future, asserts earlier positions and cumulative P&L are byte-identical). Do not relax it.
Use the trading env python directly for ad-hoc runs: `python analysis/strategy_two_sleeve.py`.

Databento ingest is gated to prevent accidental spend:

```bash
make quote                   # dry-run cost estimate via metadata API. No charge.
make sample                  # 5-day sample pull (small charge). Verify quote first.
make data                    # full pull (real charge). Verify quote first.
```

`make data` only runs stage 1 (OPRA definitions; the `arcx_spy_tbbo` entry in the YAML is
`enabled: false` because intraday was never pulled). Stage 2 (filtered OPRA statistics) is a
separate two-step flow; see Data ingest below.

## Architecture (v2)

The live work is the self-contained scripts in `analysis/`, each runnable in isolation:

```
analysis/strategy_two_sleeve.py   the STRATEGY backtest: signals -> contango-filtered carry
                                  -> ladder/attribution/ablation -> robustness -> strategy_results.json
analysis/phase1_deep_history.py   the FINDINGS deep-history DM-on-CRPS test, per regime block
analysis/phase1_robustness.py     gamma-vs-DIX + richer-VIX confound decomposition
analysis/phase0_gonogo.py         21-month OPRA sub-study (level claim)
analysis/phase05_reframe.py       21-month sub-study (path/dynamics/tails/regime)
analysis/phase05b_profile.py      21-month sub-study (by-strike profile shape)
analysis/phase2_learned_flip.py   growth probe: daily gamma->RV is linear, not a threshold
analysis/rvutil.py                vendored Yang-Zhang RV helper (so analysis/ has no features/ dep)
analysis/make_figure*.py          figures; make_figure_deep.py has stats inline (no data needed),
                                  make_figure_strategy.py reads strategy_equity.parquet + _results.json
```

`features/`, `ingest/`, `configs/` are feature engineering and the Databento OPRA pull, retained
because the 21-month options sub-study in `FINDINGS.md` (the signed by-strike gamma profile) uses
them. The strategy and deep-history code do not import from them.

### No-lookahead invariants

The project is fiction if the target leaks. Every predictor at `t` uses only information available
by the close of `t`; same-day sources (VIX close, GEX) are `shift(1)`-ed, and gamma is lagged one
trading day for OCC's T-1 open interest. The synthetic-panel gate test above enforces this by
perturbing future inputs and asserting earlier positions and P&L are byte-identical.

### GEX dealer convention

`features/gex.run` follows the practitioner simplification: dealers long calls, short puts, so
`gex_net = gex_calls - gex_puts`. State this in any report; it is not a universal convention.

## Data ingest: two-stage Databento pull

Naive `OPRA.PILLAR.statistics SPY.OPT parent` is far too expensive. The pipeline filters the
contract universe first, then pulls statistics against only the kept instrument_ids:

1. **Stage 1** (`make data` / `databento_pull --confirm`): pull `OPRA.PILLAR/definition` for
   `SPY.OPT` (cheap) and `ARCX.PILLAR/tbbo` for SPY equity (tbbo = trade + BBO snapshot at trade
   time, ~10x cheaper than mbp-1 for SPY 2023→2026).
2. **Build id list** (`python -m ingest.build_id_list configs/databento_pulls.yaml`): join
   definitions against daily SPY spot, filter to live contracts within ±20% moneyness, DTE ∈ [7, 60],
   monthly expiries only (3rd Friday, where dealer OI concentrates; cuts ~80% of contract count to
   fit the free credit). Per-month id-list chunks land in `data/interim/id_list_chunks/*.json`.
3. **Stage 2** (`python -m ingest.databento_pull --quote-stage2` then `--confirm-stage2`): submit
   one filtered-statistics job per (month, chunk) using `stype_in=instrument_id`. Total ~$94.74 for
   2024-08 → 2026-04.

`data/raw/manifest.json` records every batch job (job_id, sha256, dates, sample-vs-full). Resume an
in-flight job without re-charging via `python -m ingest.databento_pull --resume <JOB_ID> --name <hint>`.

Free data comes in two trees. The **flagship inputs** (STRATEGY.md + FINDINGS.md deep history) are
fetched by `make deep` / `python -m ingest.deep_pull`: yfinance SPY/VIXY/VXX/SVXY/UVXY and
VIX3M/VIX9D/VVIX into `data/raw/deep/`, FRED `DGS3MO` into `data/raw/fred/dgs3mo_deep.parquet`,
CBOE `VIX_History.csv` into `data/raw/cboe_vix.csv`, and SqueezeMetrics DIX/GEX into
`data/raw/squeeze_dix.csv` (personal-use fetch; if it 403s, download manually from
squeezemetrics.com/monitor/dix). Every file's rows/dates/sha256 land in `data/raw/deep_manifest.json`;
the default `--end` is pinned to the committed-results vintage; `--check` validates VIXY's
split-adjusted series against VXX. The **21-month sub-study tree** (`yfinance` shallow pulls and FRED)
lands under `data/raw/yfinance/` via `ingest/yfinance_pull.py` + `ingest/fred_pull.py`, driven by
`configs/free_pulls.yaml`.

## Conventions

- Configs are the source of truth for windows, thresholds, and feature toggles. Defaults live in
  dataclasses inside each module; production runs go through YAML.
- All parquet/manifest writes are relative to `REPO_ROOT = Path(__file__).resolve().parents[1]`.
  Do not hardcode absolute paths.
- New features land in `features/`, expose `run(df, cfg) -> daily_frame`, and are joined in
  `features/assemble.py` with explicit `shift(1)` if they use any same-day information.
- Python 3.11+, `from __future__ import annotations` at the top of every module.

## v1 lessons (so they are not repeated)

The original version shipped a clean, leakage-disciplined classifier pipeline but pointed it at the
wrong question. The lessons that survive:

- **Test the cheapest kill-switch first.** v1 spent the entire ~$95 Databento credit pulling OPRA to
  build a scalar GEX feature before checking whether GEX beats free VIX. It does not: `corr(gex, vix_z)
  ≈ −0.69`, and cross-validated AUC *falls* when GEX is added. A two-line free-data baseline would
  have de-risked the most expensive decision in the project.
- **Signal before strategy.** v1 headlined a "+0.84 Sharpe" that was a single day (2026-01-16); an
  equal-notional constant-long beat it, and the ML signal subtracted value. Do not build or headline a
  strategy until the signal is established.
- **At small N, spend on inductive bias and inference, not model count.** Six models where the
  effective sample cannot distinguish any of them is wasted effort.
- **One pinned config, every number reconciled to the artifact that produced it.** v1's docs drifted
  out of sync with the shipped config, which destroys the credibility a quant project sells.

The v2 strategy's extension roadmap (continuous roll/slope sizing, forward-VRP conditioning, a convex
left-tail floor; dead-ends are gamma/DIX timing and naive vol-targeting) is folded into `STRATEGY.md` §7.
