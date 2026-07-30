"""The query route.

Thin on purpose. The orchestration lives in :class:`retrieval_engine.service.AnswerService`
so the eval harness measures the same path this serves, and this handler's only jobs are to
translate the request into service arguments and to hand back the validated response.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from retrieval_engine.api.deps import RequestIdDep, ServicesDep
from retrieval_engine.models import ErrorResponse, QueryRequest, QueryResponse, RetrievalConfig

router = APIRouter(prefix="/v1", tags=["query"])

RESPONSES: dict[int | str, dict[str, object]] = {
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    500: {"model": ErrorResponse, "description": "Unexpected error"},
}


def resolve_config(body: QueryRequest, base: RetrievalConfig) -> RetrievalConfig:
    """Apply per-request overrides to the configured defaults.

    The config is frozen, so this returns a copy rather than mutating shared state. That is
    what keeps two concurrent requests with different modes from interfering.
    """
    updates: dict[str, object] = {}
    if body.mode is not None:
        updates["expansion"] = body.mode
    if body.top_k is not None:
        updates["final_top_k"] = body.top_k
        # Reranking cannot return more than it scores, and the config validates that
        # relationship, so widen the shortlist to match an unusually large top_k.
        if body.top_k > base.rerank_candidates:
            updates["rerank_candidates"] = body.top_k
    return base.model_copy(update=updates) if updates else base


@router.post(
    "/query",
    response_model=QueryResponse,
    responses=RESPONSES,
    summary="Answer a question from the indexed corpus",
)
async def query(
    body: QueryRequest,
    services: ServicesDep,
    request_id: RequestIdDep,
    debug: bool = Query(default=False, description="Include the retrieval trail."),
) -> QueryResponse:
    """Retrieve, decide whether to answer, generate, and verify grounding.

    A low-confidence query comes back as a 200 with ``answer_type: "refused"``. That is not
    an error: refusing is a correct outcome, and a client should not have to parse a 4xx to
    discover the system was being careful.
    """
    config = resolve_config(body, services.settings.retrieval)
    return await services.answers.answer(
        body.query,
        config=config,
        filters=body.filters,
        top_k=body.top_k,
        request_id=request_id,
        debug=debug or body.debug,
    )


__all__ = ["resolve_config", "router"]
