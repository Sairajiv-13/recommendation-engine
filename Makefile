PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: setup data train test lint serve clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -e ".[dev]"

data:
	$(BIN)/python data/get_data.py --out data

train:
	PYTHONPATH=src $(BIN)/python -m recs.train --data-dir data/ml-latest-small --out artifacts

test:
	$(BIN)/pytest --cov=recs --cov-report=term-missing

lint:
	$(BIN)/ruff check src tests data
	$(BIN)/ruff format --check src tests data

serve:
	RECS_ARTIFACTS=artifacts PYTHONPATH=src $(BIN)/uvicorn recs.api:app --host 0.0.0.0 --port 8000

clean:
	rm -rf artifacts .pytest_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
