PY := python -m

.PHONY: install sample quote data features backtest report explain sensitivity paper backtest-daily profitable test clean

install:
	pip install -e ".[dev]"

quote:
	$(PY) ingest.databento_pull --quote configs/databento_pulls.yaml

sample:
	$(PY) ingest.databento_pull --sample configs/databento_pulls.yaml
	$(PY) ingest.yfinance_pull configs/free_pulls.yaml
	$(PY) ingest.fred_pull configs/free_pulls.yaml

data:
	$(PY) ingest.databento_pull --confirm configs/databento_pulls.yaml
	$(PY) ingest.yfinance_pull configs/free_pulls.yaml
	$(PY) ingest.fred_pull configs/free_pulls.yaml

features:
	$(PY) features.assemble configs/features.yaml

backtest:
	$(PY) backtest.runner configs/experiment.yaml

report:
	$(PY) report.figures

explain:
	$(PY) report.explain

sensitivity:
	$(PY) backtest.sensitivity

paper:
	cd paper && pdflatex -interaction=nonstopmode spy_vol.tex

backtest-daily:
	$(PY) backtest.runner_v2 configs/experiment.yaml

profitable: backtest-daily

test:
	pytest -q

clean:
	rm -rf data/interim/* data/processed/* report/_build/*
