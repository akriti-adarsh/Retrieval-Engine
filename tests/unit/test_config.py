"""Settings: defaults, env parsing, derived paths, and the guards that fail fast."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from retrieval_engine.config import (
    DEFAULT_SEED,
    Settings,
    get_settings,
    reset_settings_cache,
    set_seeds,
)
from retrieval_engine.models import ChunkStrategy, EmbedderKind, LLMKind, StoreKind


def test_defaults_are_fully_local() -> None:
    """A clone with no environment must not require any API key."""
    settings = Settings(_env_file=None)

    assert settings.seed == DEFAULT_SEED == 42
    assert settings.embedder is EmbedderKind.LOCAL
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_dimension == 384
    assert settings.llm is LLMKind.OLLAMA
    assert settings.store is StoreKind.PGVECTOR
    assert settings.openai_api_key is None
    assert settings.chunk_strategy is ChunkStrategy.RECURSIVE_STRUCTURAL


def test_bge_query_instruction_is_on_by_default() -> None:
    """The instruction prefix is documented for queries; it must default to on."""
    settings = Settings(_env_file=None)

    assert settings.use_query_instruction is True
    assert settings.query_instruction.startswith("Represent this sentence")
    assert settings.retrieval.use_query_instruction is True


def test_flat_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RE_SEED", "7")
    monkeypatch.setenv("RE_EMBEDDING_BATCH_SIZE", "64")
    monkeypatch.setenv("RE_LOG_JSON", "false")

    settings = Settings(_env_file=None)

    assert settings.seed == 7
    assert settings.embedding_batch_size == 64
    assert settings.log_json is False


def test_nested_retrieval_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested retrieval knobs use the double-underscore delimiter."""
    monkeypatch.setenv("RE_RETRIEVAL__TOP_K_DENSE", "123")
    monkeypatch.setenv("RE_RETRIEVAL__FUSION", "weighted")
    monkeypatch.setenv("RE_RETRIEVAL__USE_RERANK", "false")

    settings = Settings(_env_file=None)

    assert settings.retrieval.top_k_dense == 123
    assert settings.retrieval.fusion.value == "weighted"
    assert settings.retrieval.use_rerank is False


def test_derived_paths_follow_data_dir() -> None:
    settings = Settings(_env_file=None, data_dir=Path("/srv/re-data"))

    assert settings.corpus_dir == Path("/srv/re-data/corpus")
    assert settings.golden_set_path == Path("/srv/re-data/golden/golden_set.jsonl")
    assert settings.bm25_dir == Path("/srv/re-data/bm25")


def test_openai_embedder_without_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="requires RE_OPENAI_API_KEY"):
        Settings(_env_file=None, embedder=EmbedderKind.OPENAI)


def test_every_generation_backend_has_an_implementation() -> None:
    """No enum value may exist without code behind it, so there is no remote LLM option."""
    assert {kind.value for kind in LLMKind} == {"ollama", "extractive"}


def test_openai_embedder_with_key_is_accepted() -> None:
    settings = Settings(_env_file=None, embedder=EmbedderKind.OPENAI, openai_api_key="sk-test")

    assert settings.embedder is EmbedderKind.OPENAI
    # SecretStr must not leak the key through repr or str.
    assert "sk-test" not in repr(settings)
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-test"


def test_overlap_must_leave_forward_progress() -> None:
    """overlap >= size would make fixed-token chunking loop forever."""
    with pytest.raises(ValidationError, match="must be < chunk_size"):
        Settings(_env_file=None, chunk_size=256, chunk_overlap=256)


def test_min_tokens_cannot_exceed_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_min_tokens must not exceed"):
        Settings(_env_file=None, chunk_size=128, chunk_overlap=16, chunk_min_tokens=256)


def test_rerank_candidates_must_cover_final_top_k() -> None:
    """Reranking 3 candidates while returning 5 would silently truncate results."""
    with pytest.raises(ValidationError, match="rerank_candidates must be >="):
        Settings(_env_file=None, retrieval={"rerank_candidates": 3, "final_top_k": 5})


def test_get_settings_is_cached() -> None:
    first = get_settings()
    second = get_settings()

    assert first is second

    reset_settings_cache()
    assert get_settings() is not first


def test_set_seeds_makes_runs_identical() -> None:
    set_seeds(DEFAULT_SEED)
    first = (random.random(), float(np.random.rand()))

    set_seeds(DEFAULT_SEED)
    second = (random.random(), float(np.random.rand()))

    assert first == second


def test_set_seeds_with_torch_already_imported() -> None:
    """torch is seeded only when it is already in sys.modules, and must not error."""
    import torch

    set_seeds(DEFAULT_SEED)
    first = torch.randn(3).tolist()
    set_seeds(DEFAULT_SEED)
    second = torch.randn(3).tolist()

    assert first == second
