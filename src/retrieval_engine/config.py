"""Settings, loaded from the environment with ``.env`` support.

Every knob is an env var prefixed ``RE_``; nested retrieval knobs use a double
underscore, e.g. ``RE_RETRIEVAL__TOP_K_DENSE=100``. Defaults are chosen so that a
clone with no environment at all runs entirely locally with no API keys.
"""

from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from retrieval_engine.models import (
    ChunkStrategy,
    EmbedderKind,
    LLMKind,
    RetrievalConfig,
    StoreKind,
)

#: Default seed for every stochastic component. Two eval runs on the same corpus with
#: the same seed produce identical numbers.
DEFAULT_SEED = 42


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    """Seed the standard library and numpy, and torch when it is already imported.

    torch is imported lazily elsewhere because importing it costs seconds; the local
    embedder calls this again once torch is in ``sys.modules``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        manual_seed = getattr(torch_module, "manual_seed", None)
        if callable(manual_seed):
            manual_seed(seed)


class Settings(BaseSettings):
    """Runtime configuration for the service."""

    model_config = SettingsConfigDict(
        env_prefix="RE_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        validate_default=True,
    )

    # -- core -------------------------------------------------------------------------
    app_name: str = Field(default="retrieval-engine", description="Service name in logs.")
    env: Literal["dev", "test", "prod"] = Field(default="dev", description="Deployment env.")
    seed: int = Field(default=DEFAULT_SEED, description="Global random seed.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_json: bool = Field(default=True, description="JSON logs; set false for dev consoles.")

    # -- paths ------------------------------------------------------------------------
    data_dir: Path = Field(default=Path("data"), description="Root for corpus and indexes.")
    eval_results_dir: Path = Field(default=Path("eval_results"), description="Eval output root.")

    # -- embedding --------------------------------------------------------------------
    embedder: EmbedderKind = Field(default=EmbedderKind.LOCAL, description="Embedding backend.")
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", description="sentence-transformers model id."
    )
    embedding_dimension: int = Field(default=384, gt=0, description="Vector width.")
    embedding_batch_size: int = Field(default=32, gt=0, description="Chunks per encode call.")
    embedding_device: str | None = Field(default=None, description="torch device, None = auto.")
    normalize_embeddings: bool = Field(
        default=True, description="L2-normalise vectors so dot product is cosine."
    )
    use_query_instruction: bool = Field(
        default=True,
        description="Prefix queries (never passages) with the bge retrieval instruction.",
    )
    query_instruction: str = Field(
        default="Represent this sentence for searching relevant passages: ",
        description="The bge query-side instruction prefix.",
    )
    openai_api_key: SecretStr | None = Field(default=None, description="Only for embedder=openai.")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    # -- reranking --------------------------------------------------------------------
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base", description="Cross-encoder model id."
    )
    reranker_batch_size: int = Field(default=16, gt=0)
    reranker_cache_size: int = Field(default=10_000, gt=0, description="LRU entries.")

    # -- store ------------------------------------------------------------------------
    store: StoreKind = Field(default=StoreKind.PGVECTOR, description="Vector store backend.")
    collection: str = Field(default="default", description="Logical collection name.")
    postgres_dsn: str = Field(
        default="postgresql://retrieval:retrieval@localhost:5432/retrieval",
        description="Postgres connection string.",
    )
    pg_pool_min_size: int = Field(default=1, ge=0)
    pg_pool_max_size: int = Field(default=8, gt=0)
    hnsw_m: int = Field(default=16, gt=0, description="HNSW graph degree.")
    hnsw_ef_construction: int = Field(default=64, gt=0, description="HNSW build effort.")

    # -- chunking ---------------------------------------------------------------------
    chunk_strategy: ChunkStrategy = Field(default=ChunkStrategy.RECURSIVE_STRUCTURAL)
    chunk_size: int = Field(default=512, gt=0, description="Target tokens per chunk.")
    chunk_overlap: int = Field(default=64, ge=0, description="Token overlap, fixed strategy.")
    chunk_min_tokens: int = Field(default=96, gt=0, description="Merge fragments below this.")
    semantic_threshold_percentile: float = Field(
        default=95.0, gt=0.0, le=100.0, description="Semantic split threshold."
    )

    # -- ingestion --------------------------------------------------------------------
    ingest_concurrency: int = Field(default=4, gt=0, description="Concurrent document loads.")

    # -- generation -------------------------------------------------------------------
    llm: LLMKind = Field(default=LLMKind.OLLAMA, description="Generation backend.")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.1:8b")
    ollama_timeout_s: float = Field(default=60.0, gt=0.0)
    openai_chat_model: str = Field(default="gpt-4o-mini")
    generation_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    generation_max_tokens: int = Field(default=512, gt=0)
    extractive_sentences: int = Field(
        default=3, gt=0, description="Sentences in a fallback answer."
    )

    # -- guardrails -------------------------------------------------------------------
    grounding_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.3, description="Refuse below this rerank score.")
    min_sources: int = Field(default=1, ge=0, description="Refuse below this many usable sources.")

    # -- api --------------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", description="Bind address; containers need 0.0.0.0.")
    api_port: int = Field(default=8000, gt=0, le=65535)
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_per_minute: int = Field(default=60, gt=0)
    max_upload_mb: int = Field(default=25, gt=0)
    sse_heartbeat_seconds: float = Field(default=15.0, gt=0.0)

    # -- retrieval --------------------------------------------------------------------
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    @property
    def corpus_dir(self) -> Path:
        """Where the downloaded corpus lives."""
        return self.data_dir / "corpus"

    @property
    def golden_set_path(self) -> Path:
        """The committed golden set."""
        return self.data_dir / "golden" / "golden_set.jsonl"

    @property
    def bm25_dir(self) -> Path:
        """Where the serialised BM25 index and its fingerprint live."""
        return self.data_dir / "bm25"

    @model_validator(mode="after")
    def _check_credentials(self) -> Settings:
        """Fail fast when a remote backend is selected without a key."""
        if self.embedder is EmbedderKind.OPENAI and self.openai_api_key is None:
            msg = "embedder=openai requires RE_OPENAI_API_KEY"
            raise ValueError(msg)
        if self.llm is LLMKind.OPENAI and self.openai_api_key is None:
            msg = "llm=openai requires RE_OPENAI_API_KEY"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        """Overlap must leave forward progress, or fixed-token chunking never advances."""
        if self.chunk_overlap >= self.chunk_size:
            msg = f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            raise ValueError(msg)
        if self.chunk_min_tokens > self.chunk_size:
            msg = "chunk_min_tokens must not exceed chunk_size"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_rerank_budget(self) -> Settings:
        """Reranking fewer candidates than we return would silently truncate results."""
        if (
            self.retrieval.use_rerank
            and self.retrieval.rerank_candidates < self.retrieval.final_top_k
        ):
            msg = "retrieval.rerank_candidates must be >= retrieval.final_top_k"
            raise ValueError(msg)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton, cached so the env is read once."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings. Tests use this after changing the environment."""
    get_settings.cache_clear()


__all__ = [
    "DEFAULT_SEED",
    "Settings",
    "get_settings",
    "reset_settings_cache",
    "set_seeds",
]
