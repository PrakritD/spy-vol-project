PY ?= python

.PHONY: install test lint strategy findings forecast figures notebook all clean \
        deep quote sample data log findings-pdf

# ---------------------------------------------------------------- v2 deliverables ----
install:                        ## editable install + dev tools (pytest, ruff)
	pip install -e ".[dev]"

test:                           ## data-free test suite (the no-lookahead gate runs on synthetic panels)
	$(PY) -m pytest -q

lint:                           ## ruff over the live v2 code
	$(PY) -m ruff check analysis tests

strategy:                       ## STRATEGY.md: VRP-carry backtest + robustness -> analysis/strategy_results.json
	$(PY) analysis/strategy_two_sleeve.py

findings:                       ## FINDINGS.md: deep-history gamma study + robustness decomposition
	$(PY) analysis/phase1_deep_history.py
	$(PY) analysis/phase1_robustness.py

forecast:                       ## FORECASTING.md: walk-forward ML benchmark -> analysis/forecast_bench_results.json
	$(PY) analysis/forecast_bench.py

figures:                        ## regenerate every committed figure
	$(PY) analysis/make_figure_deep.py
	$(PY) analysis/make_figure_strategy.py
	$(PY) analysis/make_figure_forecast.py

notebook:                       ## execute the narrative walkthrough in-place (embeds outputs)
	$(PY) -m nbconvert --to notebook --execute --inplace notebooks/strategy_walkthrough.ipynb

findings-pdf:                   ## render FINDINGS.md -> report/FINDINGS.pdf (pandoc + LaTeX)
	mkdir -p report
	pandoc FINDINGS.md -o report/FINDINGS.pdf \
		--pdf-engine=xelatex --toc -V geometry:margin=1in -V fontsize=11pt -V colorlinks=true \
		-V mainfont="Arial Unicode MS" --include-in-header=report/findings_pdf_header.tex

all: findings strategy forecast figures notebook test   ## regenerate the whole v2 deliverable from scratch

log:                             ## append today's close to the live paper-trade log (idempotent)
	$(PY) analysis/paper_log.py

# ---------------------------------------------------------------- data ingest ----
# (free data is fetched, not committed; the 21-month OPRA sub-study used the Databento flow below)
deep:                           ## free deep-history inputs behind STRATEGY.md/FINDINGS.md (no charge)
	$(PY) -m ingest.deep_pull
	$(PY) -m ingest.deep_pull --check
quote:                          ## dry-run Databento cost estimate (no charge)
	$(PY) -m ingest.databento_pull --quote configs/databento_pulls.yaml

sample:                         ## small sample pull (small charge — verify quote first)
	$(PY) -m ingest.databento_pull --sample configs/databento_pulls.yaml
	$(PY) -m ingest.yfinance_pull configs/free_pulls.yaml
	$(PY) -m ingest.fred_pull configs/free_pulls.yaml

data:                           ## full pull (real charge — verify quote first)
	$(PY) -m ingest.databento_pull --confirm configs/databento_pulls.yaml
	$(PY) -m ingest.yfinance_pull configs/free_pulls.yaml
	$(PY) -m ingest.fred_pull configs/free_pulls.yaml

clean:
	rm -rf data/interim/* __pycache__ analysis/__pycache__ tests/__pycache__ .pytest_cache .ruff_cache
