"""The query route.

Thin on purpose. The orchestration lives in :class:`retrieval_engine.service.AnswerService`
so the eval harness measures the same path this serves, and this handler's only jobs are to
translate the request into service arguments and to hand back the validated response.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from retrieval_engine.api.deps import RequestIdDep, Services, ServicesDep
from retrieval_engine.api.routes_admin import record_answer
from retrieval_engine.models import ErrorResponse, QueryRequest, QueryResponse, RetrievalConfig

router = APIRouter(prefix="/v1", tags=["query"])

#: Event names in the SSE contract. A client switches on these, so they are part of the API.
TOKEN_EVENT = "token"
DONE_EVENT = "done"
HEARTBEAT = ": keep-alive\n\n"

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
    response = await services.answers.answer(
        body.query,
        config=config,
        filters=body.filters,
        top_k=body.top_k,
        request_id=request_id,
        debug=debug or body.debug,
    )
    record_answer(response.answer_type, response.timings, response.grounding)
    return response


def format_event(event: str, payload: object) -> str:
    """Format one SSE event.

    The data is JSON rather than raw text. SSE is newline-delimited, so a delta containing a
    newline would otherwise split into two data lines and silently corrupt the stream.
    """
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def with_heartbeat(source: AsyncIterator[str], interval: float) -> AsyncIterator[str]:
    """Interleave comment-line heartbeats into a slow stream.

    Proxies and load balancers close idle connections, and a model that thinks for thirty
    seconds before its first token looks exactly like an idle connection. The heartbeat is a
    comment line, which every SSE client ignores by specification.

    The pending item is shielded, so a timeout emits a heartbeat without discarding the token
    that was already in flight.
    """
    iterator = source.__aiter__()
    while True:
        pending = asyncio.ensure_future(iterator.__anext__())
        while True:
            try:
                item = await asyncio.wait_for(asyncio.shield(pending), timeout=interval)
            except TimeoutError:
                yield HEARTBEAT
                continue
            except StopAsyncIteration:
                return
            break
        yield item


async def _stream_answer(
    services: Services, body: QueryRequest, request_id: str, debug: bool
) -> AsyncIterator[str]:
    """Produce the token events, then one terminal done event carrying the metadata.

    Retrieval and the refusal decision happen before any token is emitted. A stream that has
    already sent half an answer cannot take it back, so the decision to answer at all is made
    while there is still nothing on the wire.
    """
    config = resolve_config(body, services.settings.retrieval)
    response = await services.answers.answer(
        body.query,
        config=config,
        filters=body.filters,
        top_k=body.top_k,
        request_id=request_id,
        debug=debug or body.debug,
    )
    record_answer(response.answer_type, response.timings, response.grounding)

    # The answer text is already complete: streaming here replays it in pieces so the client
    # contract is identical whether the backend streamed or not. Token-by-token generation
    # streaming is a separate concern from the event contract, and conflating them would mean
    # the refusal and extractive paths could not stream at all.
    for piece in response.answer.split(" "):
        yield format_event(TOKEN_EVENT, {"text": f"{piece} "})

    metadata = response.model_dump(mode="json")
    metadata.pop("answer", None)
    yield format_event(DONE_EVENT, metadata)


@router.post(
    "/query/stream",
    summary="Answer a question, streaming the tokens",
    response_class=StreamingResponse,
)
async def query_stream(
    body: QueryRequest,
    services: ServicesDep,
    request_id: RequestIdDep,
    debug: bool = Query(default=False, description="Include the retrieval trail."),
) -> StreamingResponse:
    """Server-sent events: ``token`` deltas, then a terminal ``done`` with the metadata.

    Heartbeat comment lines keep the connection alive while the model is thinking.
    """
    stream = with_heartbeat(
        _stream_answer(services, body, request_id, debug),
        services.settings.sse_heartbeat_seconds,
    )
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Tells nginx not to buffer, which would defeat streaming entirely.
            "X-Accel-Buffering": "no",
        },
    )


__all__ = [
    "DONE_EVENT",
    "HEARTBEAT",
    "TOKEN_EVENT",
    "format_event",
    "resolve_config",
    "router",
    "with_heartbeat",
]
