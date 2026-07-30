# Targets are added as the corresponding code lands, so every target here works.
.DEFAULT_GOAL := help
.PHONY: help install lint format format-check typecheck test test-fast check clean \
        corpus golden-validate golden-stats eval eval-ablate eval-gate \
        ui up down migrate docker-test

UV ?= uv
COV_MIN ?= 85

help:
	@echo Quality: install lint format format-check typecheck test test-fast check clean
	@echo Data and eval: corpus golden-validate golden-stats eval eval-ablate eval-gate
	@echo Run: ui up down migrate docker-test

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

corpus:
	$(UV) run python scripts/download_corpus.py --limit 300 --days-back 60 --with-html

golden-validate:
	$(UV) run python scripts/build_golden_set.py --validate

golden-stats:
	$(UV) run python scripts/build_golden_set.py --stats

eval:
	$(UV) run python scripts/run_eval.py

eval-ablate:
	$(UV) run python scripts/run_eval.py --ablate

eval-gate:
	$(UV) run pytest tests/eval -v

ui:
	$(UV) run streamlit run ui/streamlit_app.py --server.port 8501

up:
	docker compose up -d --build

down:
	docker compose down

migrate:
	$(UV) run python scripts/migrate.py

docker-test:
	$(UV) run pytest -m docker -v

clean:
	$(UV) run python -c "import shutil, pathlib; [shutil.rmtree(d, ignore_errors=True) for d in ('.pytest_cache', '.mypy_cache', '.ruff_cache', '.hypothesis', 'htmlcov')]; pathlib.Path('.coverage').unlink(missing_ok=True)"
