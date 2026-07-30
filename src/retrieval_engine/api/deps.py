"""Wiring: build the service graph once, hand it to routes through app state.

Everything expensive (the embedder, the reranker, the store connection) is constructed once
per process and shared. The heavy pieces load lazily on first use, so building this graph is
cheap and the app can start before any model is in memory.

Dependencies come off ``request.app.state`` rather than module-level globals, because tests
build several apps in one process and a global would leak one test's store into the next.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from fastapi import Depends, Request

from retrieval_engine.config import Settings, get_settings
from retrieval_engine.embed import build_embedder
from retrieval_engine.embed.base import Embedder
from retrieval_engine.generate import build_llm
from retrieval_engine.generate.base import LLM
from retrieval_engine.ingest.pipeline import IngestPipeline
from retrieval_engine.models import IngestJob, StoreKind
from retrieval_engine.retrieve.lexical import BM25Retriever
from retrieval_engine.retrieve.pipeline import RetrievalPipeline
from retrieval_engine.retrieve.rerank import CrossEncoderReranker
from retrieval_engine.service import AnswerService
from retrieval_engine.store.base import VectorStore
from retrieval_engine.store.memory import MemoryVectorStore


@dataclass
class Services:
    """Everything the routes need, built once per application."""

    settings: Settings
    store: VectorStore
    embedder: Embedder
    retrieval: RetrievalPipeline
    answers: AnswerService
    ingest: IngestPipeline
    lexical: BM25Retriever
    reranker: CrossEncoderReranker
    llm: LLM | None = None
    jobs: dict[str, IngestJob] = field(default_factory=dict)
    #: Strong references to running background jobs. A task with no reference can be
    #: garbage collected mid-run, which presents as an ingestion that silently stops.
    tasks: set[asyncio.Task[None]] = field(default_factory=set)

    async def aclose(self) -> None:
        for task in list(self.tasks):
            task.cancel()
        await self.store.close()


def build_store(settings: Settings) -> VectorStore:
    """Construct the configured vector store.

    The pgvector backend is imported lazily so that running with the in-memory store never
    requires psycopg to be importable, which is what keeps the test suite database-free.
    """
    if settings.store is StoreKind.MEMORY:
        return MemoryVectorStore(settings.collection)
    from retrieval_engine.store.pgvector import PgVectorStore

    return PgVectorStore(settings)


def build_services(
    settings: Settings | None = None,
    *,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> Services:
    """Assemble the service graph, allowing any piece to be injected for tests."""
    active = settings if settings is not None else get_settings()
    resolved_store = store if store is not None else build_store(active)
    resolved_embedder = embedder if embedder is not None else build_embedder(active)
    resolved_llm = llm if llm is not None else build_llm(active)
    resolved_reranker = reranker if reranker is not None else CrossEncoderReranker(active)
    lexical = BM25Retriever(resolved_store, Path(active.bm25_dir))

    retrieval = RetrievalPipeline(
        active,
        resolved_store,
        resolved_embedder,
        reranker=resolved_reranker,
        lexical=lexical,
        llm=resolved_llm,
    )
    return Services(
        settings=active,
        store=resolved_store,
        embedder=resolved_embedder,
        retrieval=retrieval,
        answers=AnswerService(active, retrieval, resolved_embedder, llm=resolved_llm),
        ingest=IngestPipeline(active, resolved_store, resolved_embedder),
        lexical=lexical,
        reranker=resolved_reranker,
        llm=resolved_llm,
    )


def get_services(request: Request) -> Services:
    """Pull the service graph off application state."""
    services = request.app.state.services
    if not isinstance(services, Services):  # pragma: no cover - defensive
        msg = "application state is missing its service graph"
        raise RuntimeError(msg)
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


def get_settings_dep(services: ServicesDep) -> Settings:
    return services.settings


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


def get_request_id(request: Request) -> str:
    """The current request id, injected by the middleware."""
    value = getattr(request.state, "request_id", "")
    return value if isinstance(value, str) else ""


RequestIdDep = Annotated[str, Depends(get_request_id)]


__all__ = [
    "RequestIdDep",
    "Services",
    "ServicesDep",
    "SettingsDep",
    "build_services",
    "build_store",
    "get_request_id",
    "get_services",
]
