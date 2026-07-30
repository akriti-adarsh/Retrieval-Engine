"""Shared fixtures.

Two invariants this file enforces for the whole suite:

1. No test sees the developer's environment. Every ``RE_`` variable is stripped and
   the settings cache is cleared before each test.
2. No test touches the network. ``HF_HUB_OFFLINE`` is forced on, so a test that
   accidentally reaches for a model download fails loudly instead of hanging in CI.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from retrieval_engine.config import DEFAULT_SEED, Settings, reset_settings_cache, set_seeds
from retrieval_engine.models import LLMKind, StoreKind


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip RE_* vars, force offline mode, and reseed before every test."""
    for key in list(os.environ):
        if key.startswith("RE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    reset_settings_cache()
    set_seeds(DEFAULT_SEED)
    yield
    reset_settings_cache()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a temp directory, using only backends that need no server."""
    return Settings(
        _env_file=None,
        env="test",
        store=StoreKind.MEMORY,
        llm=LLMKind.EXTRACTIVE,
        data_dir=tmp_path / "data",
        eval_results_dir=tmp_path / "eval_results",
    )
