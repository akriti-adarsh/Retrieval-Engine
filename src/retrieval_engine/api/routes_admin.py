"""Document administration and health.

``/health`` answers without touching anything, so a liveness probe cannot be made to fail by
a slow dependency. ``/health/ready`` deliberately does the opposite: it embeds a short string
to force the model to load and queries the store, because a readiness probe that only checks
its own process is worthless for deciding whether to send traffic.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from retrieval_engine import __version__
from retrieval_engine.api.deps import ServicesDep
from retrieval_engine.errors import DocumentNotFoundError
from retrieval_engine.logging_config import get_logger
from retrieval_engine.models import (
    AnswerType,
    DocumentPage,
    ErrorResponse,
    GroundingReport,
    HealthResponse,
    ReadyResponse,
    StageTimings,
)

logger = get_logger(__name__)

router = APIRouter(tags=["admin"])
documents_router = APIRouter(prefix="/v1/documents", tags=["documents"])

# A dedicated registry rather than the process-global default. Tests build several
# applications in one process, and global collectors would leak one test's counters into the
# next and raise duplicate-registration errors on the second app.
REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "re_requests_total",
    "HTTP requests by route, method, and status.",
    ["route", "method", "status"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "re_request_duration_seconds",
    "End to end request latency by route.",
    ["route"],
    registry=REGISTRY,
)
STAGE_LATENCY = Histogram(
    "re_stage_duration_seconds",
    "Per-stage latency inside a query, which is where the interesting cost lives.",
    ["stage"],
    # Buckets chosen for this pipeline: dense search is single-digit milliseconds while
    # cross-encoder reranking is hundreds, so default buckets would put both in one bin.
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)
ANSWERS = Counter(
    "re_answers_total",
    "Answers by type. Refusal rate is refused over the sum across types.",
    ["answer_type"],
    registry=REGISTRY,
)
GROUNDING = Counter(
    "re_grounding_total",
    "Grounding verdicts. Failure rate is failed over the total.",
    ["result"],
    registry=REGISTRY,
)
RERANK_CACHE_HIT_RATE = Gauge(
    "re_rerank_cache_hit_rate",
    "Fraction of reranker score lookups served from cache.",
    registry=REGISTRY,
)
RERANK_CACHE_ENTRIES = Gauge(
    "re_rerank_cache_entries",
    "Entries currently held in the reranker score cache.",
    registry=REGISTRY,
)


def record_answer(
    answer_type: AnswerType, timings: StageTimings, grounding: GroundingReport
) -> None:
    """Record one answer's metrics.

    Called from the query routes rather than from middleware, because middleware only sees a
    serialised body and would have to parse it back to learn any of this.
    """
    ANSWERS.labels(answer_type=answer_type.value).inc()
    GROUNDING.labels(result="grounded" if grounding.grounded else "failed").inc()
    for stage, milliseconds in (
        ("expansion", timings.expansion_ms),
        ("dense", timings.dense_ms),
        ("lexical", timings.lexical_ms),
        ("fusion", timings.fusion_ms),
        ("rerank", timings.rerank_ms),
        ("generation", timings.generation_ms),
        ("grounding", timings.grounding_ms),
    ):
        STAGE_LATENCY.labels(stage=stage).observe(milliseconds / 1000.0)


@router.get("/metrics", summary="Prometheus metrics")
async def metrics(services: ServicesDep) -> Response:
    """Expose metrics in the Prometheus text format.

    The cache gauges are sampled here rather than written on every lookup: they are derived
    state, and reading them at scrape time keeps the hot path free of metric writes.
    """
    RERANK_CACHE_HIT_RATE.set(services.reranker.cache.hit_rate)
    RERANK_CACHE_ENTRIES.set(len(services.reranker.cache))
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@router.get("/health", response_model=HealthResponse, summary="Liveness")
async def health() -> HealthResponse:
    """Whether the process is up. Touches no dependency, by design."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/health/ready", response_model=ReadyResponse, summary="Readiness")
async def ready(services: ServicesDep) -> ReadyResponse:
    """Whether the store, the embedder, and the generator are actually usable."""
    detail: dict[str, str] = {}

    try:
        store_ok = await services.store.health()
    except Exception as exc:  # a probe must report, never raise
        store_ok = False
        detail["store"] = f"{type(exc).__name__}: {exc}"

    embedder_ok = True
    try:
        # Embedding one short string forces the lazy model load, which is the thing a
        # readiness check is actually asking about.
        await services.embedder.embed_query("ready")
    except Exception as exc:
        embedder_ok = False
        detail["embedder"] = f"{type(exc).__name__}: {exc}"

    if services.llm is None:
        # The extractive path needs no server, so generation is always available.
        generator_ok = True
        detail["generator"] = "extractive (no model server required)"
    else:
        generator_ok = await services.llm.health()
        if not generator_ok:
            # Not fatal: the service degrades to extraction rather than failing.
            detail["generator"] = "model server unreachable, answers will be extractive"

    return ReadyResponse(
        # Generation is excluded from the verdict on purpose. A stopped model server is a
        # quality reduction, not an outage, so it must not take the service out of rotation.
        ready=store_ok and embedder_ok,
        store=store_ok,
        embedder=embedder_ok,
        generator=generator_ok,
        detail=detail,
    )


@documents_router.get("", response_model=DocumentPage, summary="List indexed documents")
async def list_documents(
    services: ServicesDep,
    limit: int = Query(default=50, gt=0, le=500),
    offset: int = Query(default=0, ge=0),
    title: str | None = Query(default=None, description="Exact-match metadata filter."),
) -> DocumentPage:
    """Paginated document listing."""
    filters = {"title": title} if title else None
    return await services.store.list_documents(limit=limit, offset=offset, filters=filters)


@documents_router.delete(
    "/{doc_id}",
    responses={404: {"model": ErrorResponse, "description": "Unknown document"}},
    summary="Delete a document and its chunks",
)
async def delete_document(doc_id: str, services: ServicesDep) -> dict[str, object]:
    """Remove a document. Returns the number of chunks deleted.

    The store returns 0 for an unknown id rather than raising, which leaves the decision
    here: a delete of something that never existed is a 404, not a silent success.
    """
    removed = await services.store.delete_document(doc_id)
    if removed == 0:
        msg = f"no document with id {doc_id!r}"
        raise DocumentNotFoundError(msg)
    logger.info("document_deleted", doc_id=doc_id, chunks_removed=removed)
    return {"doc_id": doc_id, "chunks_deleted": removed}


__all__ = ["documents_router", "router"]
