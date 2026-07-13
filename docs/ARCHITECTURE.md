# Architecture and design notes

How the repo is put together, and why. The short version: two self-contained deliverables
(`STRATEGY.md`, `FINDINGS.md`), each backed by scripts in `analysis/` that run in isolation, with
the no-lookahead property enforced by an executable test rather than by convention.

## Layout

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

`features/`, `ingest/`, `configs/` hold feature engineering and the Databento OPRA pull. The
21-month options sub-study in `FINDINGS.md` (the signed by-strike gamma profile) uses them; the
strategy and deep-history code do not import from them.

## No-lookahead invariants

The project is fiction if the target leaks. Every predictor at `t` uses only information available
by the close of `t`; same-day sources (VIX close, GEX) are `shift(1)`-ed, and gamma is lagged one
trading day for OCC's T-1 open interest. The enforcement is a property test
(`tests/test_strategy.py::test_no_lookahead_end_to_end`): perturb raw inputs strictly in the
future, assert earlier positions and cumulative P&L are byte-identical. Walk-forward models get
the same treatment (`test_ml_sizing_is_causal_and_lagged`). Do not relax these tests.

## GEX dealer convention

`features/gex.run` follows the practitioner simplification: dealers long calls, short puts, so
`gex_net = gex_calls - gex_puts`. State this in any report; it is not a universal convention.

## Data ingest

Free data comes in two trees:

- **Flagship inputs** (STRATEGY.md + FINDINGS.md deep history): `make deep` /
  `python -m ingest.deep_pull` fetches yfinance SPY/VIXY/VXX/SVXY/UVXY and VIX3M/VIX9D/VVIX
  (CBOE CDN fallback) into `data/raw/deep/`, FRED `DGS3MO` into
  `data/raw/fred/dgs3mo_deep.parquet`, CBOE `VIX_History.csv` into `data/raw/cboe_vix.csv`, and
  SqueezeMetrics DIX/GEX into `data/raw/squeeze_dix.csv` (personal-use fetch; if it fails,
  download manually from squeezemetrics.com/monitor/dix). Every file's rows/dates/sha256 land in
  `data/raw/deep_manifest.json`. The default `--end` is pinned to the committed-results vintage;
  `--check` validates VIXY's split-adjusted series against VXX.
- **21-month sub-study tree**: `ingest/yfinance_pull.py` + `ingest/fred_pull.py` write to
  `data/raw/yfinance/` and `data/raw/fred/`, driven by `configs/free_pulls.yaml`.

### Two-stage Databento OPRA pull

Naive `OPRA.PILLAR.statistics SPY.OPT parent` is far too expensive. The pipeline filters the
contract universe first, then pulls statistics against only the kept instrument_ids:

1. **Stage 1** (`make data` / `databento_pull --confirm`): pull `OPRA.PILLAR/definition` for
   `SPY.OPT` (cheap) and `ARCX.PILLAR/tbbo` for SPY equity (tbbo = trade + BBO snapshot at trade
   time, ~10x cheaper than mbp-1 for SPY 2023→2026).
2. **Build id list** (`python -m ingest.build_id_list configs/databento_pulls.yaml`): join
   definitions against daily SPY spot, filter to live contracts within ±20% moneyness,
   DTE ∈ [7, 60], monthly expiries only (3rd Friday, where dealer OI concentrates; cuts ~80% of
   contract count to fit the free credit). Per-month id-list chunks land in
   `data/interim/id_list_chunks/*.json`.
3. **Stage 2** (`python -m ingest.databento_pull --quote-stage2` then `--confirm-stage2`): submit
   one filtered-statistics job per (month, chunk) using `stype_in=instrument_id`. Total ~$94.74
   for 2024-08 → 2026-04.

`data/raw/manifest.json` records every batch job (job_id, sha256, dates, sample-vs-full). Resume
an in-flight job without re-charging via
`python -m ingest.databento_pull --resume <JOB_ID> --name <hint>`.

## Design principles

- **Test the cheapest kill-switch first.** Before spending on data or models, check whether a
  free baseline already answers the question (here: GEX never beat free VIX, so nothing expensive
  should have been built on GEX alone).
- **Signal before strategy.** No strategy is built or headlined until the underlying signal is
  established with proper inference.
- **At small N, spend on inductive bias and inference, not model count.** Six models the sample
  cannot distinguish are wasted effort.
- **One pinned config; every number reconciled to the artifact that produced it.** Docs quote
  `strategy_results.json`, never a remembered run.

The extension roadmap (continuous roll/slope sizing, forward-VRP conditioning, a convex left-tail
floor) and the dead-ends (gamma/DIX timing, naive vol-targeting) live in `STRATEGY.md` §7.
