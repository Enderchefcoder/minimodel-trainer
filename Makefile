SHELL := /bin/bash
PYTHON ?= python3
VENV ?= venv
BIN := $(VENV)/bin

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip wheel setuptools

.PHONY: venv
venv: $(BIN)/python ## Create the virtualenv

.PHONY: install
install: venv ## Install the package in editable mode with dev extras
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e .

.PHONY: install-cpu
install-cpu: venv ## Install with the CPU-only PyTorch wheel (CI/laptops)
	$(BIN)/pip install --index-url https://download.pytorch.org/whl/cpu torch
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e .

.PHONY: test
test: ## Run the test suite
	$(BIN)/pytest

.PHONY: coverage
coverage: ## Run tests with a coverage report
	$(BIN)/pytest --cov=minimodel --cov-report=term-missing --cov-report=xml

.PHONY: lint
lint: ## Lint the source tree
	$(BIN)/ruff check src tests

.PHONY: format
format: ## Auto-format and auto-fix lint errors
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

.PHONY: smoke
smoke: ## Run the end-to-end smoke pipeline (tokenizer -> data -> train -> eval -> generate)
	$(BIN)/python scripts/smoke_e2e.py

.PHONY: demo
demo: ## Train the bundled demo model for a few hundred steps
	$(BIN)/minimodel train --config configs/pretrain/demo_tiny.yaml

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

.PHONY: clean-runs
clean-runs: ## Remove generated run/artifact directories
	rm -rf runs artifacts outputs
