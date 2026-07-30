# Targets are added as the corresponding code lands, so every target here works.
.DEFAULT_GOAL := help
.PHONY: help install lint format format-check typecheck test test-fast check clean

UV ?= uv
COV_MIN ?= 85

help:
	@echo Available targets: install lint format format-check typecheck test test-fast check clean

install:
	$(UV) sync --frozen

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

format-check:
	$(UV) run ruff format --check .

typecheck:
	$(UV) run mypy --strict src/

test:
	$(UV) run pytest --cov=src --cov-report=term-missing --cov-fail-under=$(COV_MIN)

test-fast:
	$(UV) run pytest

check: lint format-check typecheck test

clean:
	$(UV) run python -c "import shutil, pathlib; [shutil.rmtree(d, ignore_errors=True) for d in ('.pytest_cache', '.mypy_cache', '.ruff_cache', '.hypothesis', 'htmlcov')]; pathlib.Path('.coverage').unlink(missing_ok=True)"
