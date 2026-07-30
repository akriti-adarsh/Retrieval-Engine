# Targets are added as the corresponding code lands, so every target here works.
.DEFAULT_GOAL := help
.PHONY: help install lint format format-check typecheck clean

UV ?= uv

help:
	@echo Available targets: install lint format format-check typecheck clean

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

clean:
	$(UV) run python -c "import shutil, pathlib; [shutil.rmtree(d, ignore_errors=True) for d in ('.pytest_cache', '.mypy_cache', '.ruff_cache', '.hypothesis', 'htmlcov')]; pathlib.Path('.coverage').unlink(missing_ok=True)"
