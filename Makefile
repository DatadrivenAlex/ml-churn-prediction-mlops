PYTHON ?= python3

.DEFAULT_GOAL := help

.PHONY: help install lint format test generate train drift serve ci docker-build

help:
	@echo "Targets:"
	@echo "  install       Install the package and dev tools"
	@echo "  lint          Ruff check and format check"
	@echo "  format        Apply Ruff formatting and autofixes"
	@echo "  test          Offline pytest suite with coverage"
	@echo "  generate      Write seeded baseline and shifted CSVs"
	@echo "  train         Train, track, and register a champion model"
	@echo "  drift         PSI of the shifted batch against the champion baseline"
	@echo "  serve         Run the FastAPI app on port 8000"
	@echo "  ci            Lint and test, same as GitHub Actions"
	@echo "  docker-build  Build the scoring image"

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

format:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

test:
	$(PYTHON) -m pytest

generate:
	mkdir -p data
	$(PYTHON) -m churn generate --rows 8000 --seed 42 --output data/churn.csv
	$(PYTHON) -m churn generate --rows 8000 --seed 42 --scenario shifted --output data/churn_shifted.csv

train:
	mkdir -p reports
	$(PYTHON) -m churn train --data data/churn.csv --seed 42 --report reports/training_report.json

drift:
	mkdir -p reports
	$(PYTHON) -m churn drift --current data/churn_shifted.csv --output reports/drift_report.json

serve:
	$(PYTHON) -m uvicorn churn.serving.app:app --host 0.0.0.0 --port 8000

ci: lint test

docker-build:
	docker build -t churn-mlops:local .
