"""Document administration and health.

``/health`` answers without touching anything, so a liveness probe cannot be made to fail by
a slow dependency. ``/health/ready`` deliberately does the opposite: it embeds a short string
to force the model to load and queries the store, because a readiness probe that only checks
its own process is worthless for deciding whether to send traffic.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from retrieval_engine import __version__
from retrieval_engine.api.deps import ServicesDep
from retrieval_engine.errors import DocumentNotFoundError
from retrieval_engine.logging_config import get_logger
from retrieval_engine.models import (
    DocumentPage,
    ErrorResponse,
    HealthResponse,
    ReadyResponse,
)

logger = get_logger(__name__)

router = APIRouter(tags=["admin"])
documents_router = APIRouter(prefix="/v1/documents", tags=["documents"])


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
