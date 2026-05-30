PY ?= python

.PHONY: install test lint strategy findings figures notebook all clean \
        quote sample data

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

figures:                        ## regenerate every committed figure
	$(PY) analysis/make_figure_deep.py
	$(PY) analysis/make_figure_strategy.py

notebook:                       ## execute the narrative walkthrough in-place (embeds outputs)
	$(PY) -m nbconvert --to notebook --execute --inplace notebooks/strategy_walkthrough.ipynb

all: findings strategy figures notebook test   ## regenerate the whole v2 deliverable from scratch

# ---------------------------------------------------------------- data ingest ----
# (free data is fetched, not committed; the 21-month OPRA sub-study used the Databento flow below)
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
