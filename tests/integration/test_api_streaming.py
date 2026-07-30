"""SSE streaming, Prometheus metrics, and the token-bucket rate limiter.

The rate limiter is tested with an injected clock rather than by sleeping, so the assertions
are about refill arithmetic instead of about how fast the machine happens to be.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from retrieval_engine.api.app import (
    UNLIMITED_PATHS,
    RateLimiter,
    TokenBucket,
    create_app,
)
from retrieval_engine.api.deps import Services, build_services
from retrieval_engine.api.routes_query import (
    DONE_EVENT,
    HEARTBEAT,
    TOKEN_EVENT,
    format_event,
    with_heartbeat,
)
from retrieval_engine.config import Settings
from retrieval_engine.ingest.pipeline import IngestPipeline
from retrieval_engine.models import AnswerType, LLMKind, StoreKind
from retrieval_engine.retrieve.rerank import CrossEncoderReranker
from retrieval_engine.store.memory import MemoryVectorStore
from tests.conftest import FAKE_DIMENSION, FakeEmbedder, FakeLLM


class StubCrossEncoder:
    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> Any:
        return [
            len(set(query.lower().split()) & set(passage.lower().split())) / 10.0
            for query, passage in sentences
        ]


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "_env_file": None,
        "env": "test",
        "store": StoreKind.MEMORY,
        "llm": LLMKind.EXTRACTIVE,
        "data_dir": tmp_path / "data",
        "eval_results_dir": tmp_path / "eval",
        "embedding_dimension": FAKE_DIMENSION,
        "chunk_size": 64,
        "chunk_overlap": 8,
        "chunk_min_tokens": 16,
        "log_json": False,
        "min_confidence": 0.01,
        "rate_limit_enabled": False,
    }
    base.update(overrides)
    return Settings(**base)


def _graph(settings: Settings, llm: FakeLLM | None = None) -> Services:
    return build_services(
        settings,
        store=MemoryVectorStore(settings.collection),
        embedder=FakeEmbedder(dimension=FAKE_DIMENSION),
        llm=llm,
        reranker=CrossEncoderReranker(settings, StubCrossEncoder),
    )


async def _client(services: Services, clock: Any = None) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(services.settings, services, clock=clock)
    async with (
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
        app.router.lifespan_context(app),
    ):
        yield client


@pytest.fixture
async def seeded(tmp_path: Path, corpus_dir: Path) -> AsyncIterator[httpx.AsyncClient]:
    settings = _settings(tmp_path)
    services = _graph(settings)
    await IngestPipeline(settings, services.store, services.embedder).ingest_directory(
        corpus_dir, progress=False
    )
    async for client in _client(services):
        yield client


def _parse_events(payload: str) -> list[tuple[str, str]]:
    """Parse an SSE payload into (event, data) pairs, ignoring comment lines."""
    events: list[tuple[str, str]] = []
    for block in payload.split("\n\n"):
        if not block.strip() or block.lstrip().startswith(":"):
            continue
        name = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if name:
            events.append((name, data))
    return events


# --- event formatting -------------------------------------------------------------------


def test_event_data_is_json_so_newlines_cannot_split_the_stream() -> None:
    """SSE is newline-delimited, so a raw delta with a newline would corrupt the stream."""
    formatted = format_event(TOKEN_EVENT, {"text": "line one\nline two"})

    assert formatted.startswith(f"event: {TOKEN_EVENT}\ndata: ")
    assert formatted.endswith("\n\n")
    assert formatted.count("\n\n") == 1
    body = formatted.split("data: ", 1)[1].strip()
    assert "\n" not in body


# --- heartbeat --------------------------------------------------------------------------


async def test_heartbeat_fires_while_the_source_is_slow() -> None:
    """A model thinking for thirty seconds looks exactly like an idle connection."""

    async def slow() -> AsyncIterator[str]:
        await asyncio.sleep(0.05)
        yield "first"
        yield "second"

    chunks = [chunk async for chunk in with_heartbeat(slow(), interval=0.01)]

    assert HEARTBEAT in chunks
    assert [chunk for chunk in chunks if chunk != HEARTBEAT] == ["first", "second"]


async def test_heartbeat_does_not_drop_the_pending_item() -> None:
    """The in-flight token is shielded, so a timeout must not discard it."""

    async def slow() -> AsyncIterator[str]:
        await asyncio.sleep(0.03)
        yield "kept"

    chunks = [chunk async for chunk in with_heartbeat(slow(), interval=0.005)]

    assert "kept" in chunks
    assert chunks.count("kept") == 1


async def test_a_fast_source_needs_no_heartbeat() -> None:
    async def fast() -> AsyncIterator[str]:
        yield "a"
        yield "b"

    chunks = [chunk async for chunk in with_heartbeat(fast(), interval=5.0)]

    assert chunks == ["a", "b"]


async def test_heartbeat_on_an_empty_source_terminates() -> None:
    async def empty() -> AsyncIterator[str]:
        return
        yield ""  # pragma: no cover - unreachable, defines an async generator

    assert [chunk async for chunk in with_heartbeat(empty(), interval=0.01)] == []


# --- the stream route -------------------------------------------------------------------


async def test_stream_sends_tokens_then_one_done(seeded: httpx.AsyncClient) -> None:
    response = await seeded.post("/v1/query/stream", json={"query": "layered proximity graph"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_events(response.text)
    names = [name for name, _ in events]
    assert names.count(DONE_EVENT) == 1
    assert names[-1] == DONE_EVENT, "done must be terminal"
    assert names.count(TOKEN_EVENT) >= 1


async def test_the_streamed_tokens_reassemble_into_the_answer(
    seeded: httpx.AsyncClient,
) -> None:
    """A client that concatenates deltas must get exactly what the JSON route returns."""
    import json

    plain = (await seeded.post("/v1/query", json={"query": "bm25 saturation"})).json()
    streamed = await seeded.post("/v1/query/stream", json={"query": "bm25 saturation"})

    deltas = [
        json.loads(data)["text"]
        for name, data in _parse_events(streamed.text)
        if name == TOKEN_EVENT
    ]
    assert "".join(deltas).strip() == plain["answer"]


async def test_the_done_event_carries_the_metadata(seeded: httpx.AsyncClient) -> None:
    import json

    response = await seeded.post("/v1/query/stream?debug=true", json={"query": "hnsw graph"})

    done = next(data for name, data in _parse_events(response.text) if name == DONE_EVENT)
    metadata = json.loads(done)
    assert metadata["citations"]
    assert metadata["sources"]
    assert metadata["grounding"]
    assert metadata["timings"]["total_ms"] > 0
    assert metadata["request_id"]
    assert metadata["debug"] is not None
    # The answer text arrived as tokens, so repeating it in the terminal event would double it.
    assert "answer" not in metadata


async def test_a_refusal_still_streams(tmp_path: Path) -> None:
    """The refusal path must reach a streaming client, not hang or error."""
    import json

    services = _graph(_settings(tmp_path))
    async for client in _client(services):
        response = await client.post("/v1/query/stream", json={"query": "nothing indexed"})

        events = _parse_events(response.text)
        done = json.loads(next(data for name, data in events if name == DONE_EVENT))
        assert done["answer_type"] == AnswerType.REFUSED.value
        assert any(name == TOKEN_EVENT for name, _ in events)


async def test_stream_validation_errors_use_the_error_envelope(
    seeded: httpx.AsyncClient,
) -> None:
    response = await seeded.post("/v1/query/stream", json={"query": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# --- metrics ----------------------------------------------------------------------------


async def test_metrics_exposes_the_prometheus_format(seeded: httpx.AsyncClient) -> None:
    await seeded.post("/v1/query", json={"query": "bm25 scoring"})

    response = await seeded.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    for metric in (
        "re_requests_total",
        "re_request_duration_seconds",
        "re_stage_duration_seconds",
        "re_answers_total",
        "re_grounding_total",
        "re_rerank_cache_hit_rate",
    ):
        assert metric in body, f"{metric} missing from /metrics"


async def test_answers_are_counted_by_type(seeded: httpx.AsyncClient) -> None:
    await seeded.post("/v1/query", json={"query": "layered proximity graph"})

    body = (await seeded.get("/metrics")).text

    assert 're_answers_total{answer_type="extractive"}' in body
    assert "re_grounding_total" in body


async def test_refusals_are_counted_so_the_rate_is_derivable(tmp_path: Path) -> None:
    services = _graph(_settings(tmp_path))
    async for client in _client(services):
        await client.post("/v1/query", json={"query": "nothing indexed at all"})

        body = (await client.get("/metrics")).text

        assert 're_answers_total{answer_type="refused"}' in body


async def test_stage_latencies_are_recorded(seeded: httpx.AsyncClient) -> None:
    await seeded.post("/v1/query", json={"query": "cross encoder"})

    body = (await seeded.get("/metrics")).text

    for stage in ("dense", "lexical", "fusion", "rerank", "generation", "grounding"):
        assert f'stage="{stage}"' in body


async def test_request_labels_use_the_route_template_not_the_raw_path(
    seeded: httpx.AsyncClient,
) -> None:
    """A path parameter as a label would grow cardinality without bound."""
    await seeded.delete("/v1/documents/doc-hnsw")

    body = (await seeded.get("/metrics")).text

    assert "/v1/documents/{doc_id}" in body
    assert 'route="/v1/documents/doc-hnsw"' not in body


# --- the token bucket -------------------------------------------------------------------


def test_a_fresh_bucket_starts_full() -> None:
    bucket = TokenBucket(capacity=3.0, refill_per_second=1.0)

    assert [bucket.take(0.0) for _ in range(4)] == [True, True, True, False]


def test_a_bucket_refills_continuously() -> None:
    """Continuous refill, not a window reset: a fixed window allows a double-rate burst."""
    bucket = TokenBucket(capacity=2.0, refill_per_second=1.0)
    assert bucket.take(0.0) is True
    assert bucket.take(0.0) is True
    assert bucket.take(0.0) is False

    assert bucket.take(1.0) is True, "one second refills exactly one token"
    assert bucket.take(1.0) is False


def test_a_bucket_never_exceeds_its_capacity() -> None:
    bucket = TokenBucket(capacity=2.0, refill_per_second=1.0)
    bucket.take(0.0)

    # A long idle period must not accumulate an unbounded allowance.
    assert [bucket.take(1000.0) for _ in range(3)] == [True, True, False]


def test_time_going_backwards_does_not_grant_tokens() -> None:
    bucket = TokenBucket(capacity=1.0, refill_per_second=1.0)
    assert bucket.take(10.0) is True

    assert bucket.take(5.0) is False


# --- the limiter ------------------------------------------------------------------------


def test_callers_are_limited_independently() -> None:
    now = [0.0]
    limiter = RateLimiter(2, clock=lambda: now[0])

    assert limiter.allow("1.1.1.1") is True
    assert limiter.allow("1.1.1.1") is True
    assert limiter.allow("1.1.1.1") is False
    # A different caller has its own allowance.
    assert limiter.allow("2.2.2.2") is True


def test_the_limiter_prunes_idle_buckets() -> None:
    """Unbounded per-caller state is a memory leak an attacker controls."""
    now = [0.0]
    limiter = RateLimiter(60, clock=lambda: now[0], max_buckets=4)

    for index in range(4):
        limiter.allow(f"10.0.0.{index}")
    now[0] = 600.0
    limiter.allow("10.0.0.99")

    assert len(limiter._buckets) <= 4


# --- limiting through the app -----------------------------------------------------------


async def test_requests_over_the_limit_get_429(tmp_path: Path, corpus_dir: Path) -> None:
    now = [0.0]
    settings = _settings(tmp_path, rate_limit_enabled=True, rate_limit_per_minute=2)
    services = _graph(settings)
    await IngestPipeline(settings, services.store, services.embedder).ingest_directory(
        corpus_dir, progress=False
    )

    async for client in _client(services, clock=lambda: now[0]):
        first = await client.post("/v1/query", json={"query": "bm25"})
        second = await client.post("/v1/query", json={"query": "bm25"})
        third = await client.post("/v1/query", json={"query": "bm25"})

        assert first.status_code == 200
        assert second.status_code == 200
        assert third.status_code == 429
        assert third.json()["error"]["code"] == "rate_limit_exceeded"
        assert "2 requests per minute" in third.json()["error"]["message"]


async def test_the_allowance_refills_over_time(tmp_path: Path, corpus_dir: Path) -> None:
    now = [0.0]
    settings = _settings(tmp_path, rate_limit_enabled=True, rate_limit_per_minute=60)
    services = _graph(settings)
    await IngestPipeline(settings, services.store, services.embedder).ingest_directory(
        corpus_dir, progress=False
    )

    async for client in _client(services, clock=lambda: now[0]):
        for _ in range(60):
            await client.post("/v1/query", json={"query": "bm25"})
        assert (await client.post("/v1/query", json={"query": "bm25"})).status_code == 429

        # 60 per minute is one per second, so one second buys exactly one request.
        now[0] = 1.0
        assert (await client.post("/v1/query", json={"query": "bm25"})).status_code == 200


async def test_health_is_never_rate_limited(tmp_path: Path) -> None:
    """A limiter that can starve the liveness probe takes a healthy service out of rotation."""
    now = [0.0]
    settings = _settings(tmp_path, rate_limit_enabled=True, rate_limit_per_minute=1)
    services = _graph(settings)

    async for client in _client(services, clock=lambda: now[0]):
        for _ in range(5):
            assert (await client.get("/health")).status_code == 200
        assert (await client.get("/metrics")).status_code == 200
        assert "/health" in UNLIMITED_PATHS


async def test_limiting_can_be_disabled(tmp_path: Path) -> None:
    settings = _settings(tmp_path, rate_limit_enabled=False, rate_limit_per_minute=1)
    services = _graph(settings)

    async for client in _client(services):
        for _ in range(5):
            assert (await client.get("/v1/documents")).status_code == 200
