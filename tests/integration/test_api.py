"""The API, driven through ASGI with no live server.

Every test here goes through the real application: real routing, real validation, the real
exception handler. The only substitutions are the store (in memory) and the models (fakes),
which is what lets this run in CI with no database, no downloads, and no network.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from retrieval_engine.api.app import REQUEST_ID_HEADER, create_app
from retrieval_engine.api.deps import Services, build_services
from retrieval_engine.api.routes_ingest import resolve_corpus_path
from retrieval_engine.api.routes_query import resolve_config
from retrieval_engine.config import Settings
from retrieval_engine.errors import DocumentLoadError
from retrieval_engine.ingest.pipeline import IngestPipeline
from retrieval_engine.models import (
    AnswerType,
    ExpansionMode,
    LLMKind,
    QueryRequest,
    RetrievalConfig,
    StoreKind,
)
from retrieval_engine.retrieve.rerank import CrossEncoderReranker
from retrieval_engine.store.memory import MemoryVectorStore
from tests.conftest import FAKE_DIMENSION, FakeEmbedder, FakeLLM


class StubCrossEncoder:
    """Shared-word scoring, so reranking is real but needs no 1.5 GB model."""

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
        # The fake reranker's shared-word scores are small, so keep the bar low enough that
        # a genuinely relevant answer is not refused for the wrong reason.
        "min_confidence": 0.01,
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


async def _client(services: Services) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(services.settings, services)
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        app.router.lifespan_context(app),
    ):
        yield client


@pytest.fixture
async def seeded(tmp_path: Path, corpus_dir: Path) -> AsyncIterator[httpx.AsyncClient]:
    """An API backed by a really ingested twelve document corpus."""
    settings = _settings(tmp_path)
    services = _graph(settings)
    await IngestPipeline(settings, services.store, services.embedder).ingest_directory(
        corpus_dir, progress=False
    )
    async for client in _client(services):
        yield client


@pytest.fixture
async def empty(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """An API with nothing indexed, for the degenerate paths."""
    settings = _settings(tmp_path)
    async for client in _client(_graph(settings)):
        yield client


# --- health -----------------------------------------------------------------------------


async def test_liveness_touches_nothing(empty: httpx.AsyncClient) -> None:
    response = await empty.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"]


async def test_readiness_reports_each_dependency(seeded: httpx.AsyncClient) -> None:
    response = await seeded.get("/health/ready")

    body = response.json()
    assert response.status_code == 200
    assert body["ready"] is True
    assert body["store"] is True
    assert body["embedder"] is True
    # No model server is configured, so generation is the extractive path.
    assert body["generator"] is True
    assert "extractive" in body["detail"]["generator"]


async def test_a_dead_model_server_does_not_make_the_service_unready(
    tmp_path: Path,
) -> None:
    """A stopped model server is a quality reduction, not an outage."""
    settings = _settings(tmp_path, llm=LLMKind.OLLAMA)
    services = _graph(settings, llm=FakeLLM(available=False))

    async for client in _client(services):
        body = (await client.get("/health/ready")).json()

        assert body["generator"] is False
        assert body["ready"] is True
        assert "extractive" in body["detail"]["generator"]


# --- query ------------------------------------------------------------------------------


async def test_a_query_returns_a_cited_answer(seeded: httpx.AsyncClient) -> None:
    response = await seeded.post("/v1/query", json={"query": "layered proximity graph"})

    body = response.json()
    assert response.status_code == 200
    assert body["answer"]
    assert body["answer_type"] == AnswerType.EXTRACTIVE.value
    assert body["sources"], "an answer must come with the sources it used"
    assert body["citations"], "the extractive path always cites"
    assert all(citation["resolved"] for citation in body["citations"])
    assert body["prompt_version"]
    assert body["request_id"]
    assert body["timings"]["total_ms"] > 0


async def test_the_answer_cites_a_source_that_was_returned(seeded: httpx.AsyncClient) -> None:
    """A citation pointing outside the returned sources would be unverifiable."""
    body = (await seeded.post("/v1/query", json={"query": "bm25 saturation"})).json()

    indexes = {source["index"] for source in body["sources"]}
    assert {citation["marker"] for citation in body["citations"]} <= indexes


async def test_grounding_is_reported_on_every_answer(seeded: httpx.AsyncClient) -> None:
    """A grounding report that only appears when it is clean is marketing."""
    body = (await seeded.post("/v1/query", json={"query": "cross encoder"})).json()

    assert "grounding" in body
    assert "grounded" in body["grounding"]
    assert body["grounding"]["threshold"] > 0


async def test_debug_returns_the_retrieval_trail(seeded: httpx.AsyncClient) -> None:
    body = (
        await seeded.post("/v1/query?debug=true", json={"query": "reciprocal rank fusion"})
    ).json()

    assert body["debug"] is not None
    assert body["debug"]["config"]["rrf_k"] == 60
    assert body["debug"]["candidate_counts"]["dense"] > 0
    assert len(body["debug"]["stages"]) == len(body["sources"])


async def test_debug_is_absent_by_default(seeded: httpx.AsyncClient) -> None:
    body = (await seeded.post("/v1/query", json={"query": "ndcg"})).json()

    assert body["debug"] is None


async def test_top_k_is_honoured(seeded: httpx.AsyncClient) -> None:
    body = (await seeded.post("/v1/query", json={"query": "retrieval", "top_k": 2})).json()

    assert len(body["sources"]) <= 2


async def test_filters_are_honoured(seeded: httpx.AsyncClient) -> None:
    body = (
        await seeded.post(
            "/v1/query",
            json={"query": "graph search", "filters": {"title": "HNSW Graph Indexes"}},
        )
    ).json()

    assert body["sources"]
    assert all(source["doc_id"] == "doc-hnsw" for source in body["sources"])


async def test_an_empty_index_refuses_rather_than_inventing(empty: httpx.AsyncClient) -> None:
    """The whole point of the refusal path."""
    response = await empty.post("/v1/query", json={"query": "anything at all"})

    body = response.json()
    assert response.status_code == 200, "refusing is a correct outcome, not an error"
    assert body["answer_type"] == AnswerType.REFUSED.value
    assert "don't have enough information" in body["answer"]
    assert body["citations"] == []


async def test_a_dead_model_server_falls_back_to_extraction(
    tmp_path: Path, corpus_dir: Path
) -> None:
    """The Session B check: with the model server down, a real cited answer, not a 500."""
    settings = _settings(tmp_path, llm=LLMKind.OLLAMA)
    services = _graph(settings, llm=FakeLLM(available=False))
    await IngestPipeline(settings, services.store, services.embedder).ingest_directory(
        corpus_dir, progress=False
    )

    async for client in _client(services):
        response = await client.post("/v1/query", json={"query": "layered proximity graph"})

        body = response.json()
        assert response.status_code == 200
        assert body["answer_type"] == AnswerType.EXTRACTIVE.value
        assert body["citations"]
        assert body["model"] == "extractive"


async def test_a_working_model_server_produces_a_generated_answer(
    tmp_path: Path, corpus_dir: Path
) -> None:
    settings = _settings(tmp_path, llm=LLMKind.OLLAMA)
    llm = FakeLLM(["A layered proximity graph indexes vectors for search [1]."])
    services = _graph(settings, llm=llm)
    await IngestPipeline(settings, services.store, services.embedder).ingest_directory(
        corpus_dir, progress=False
    )

    async for client in _client(services):
        body = (await client.post("/v1/query", json={"query": "layered proximity graph"})).json()

        assert body["answer_type"] == AnswerType.GENERATED.value
        assert body["model"] == "fake-llm"
        assert body["citations"][0]["marker"] == 1


async def test_queries_are_deterministic(seeded: httpx.AsyncClient) -> None:
    first = (await seeded.post("/v1/query", json={"query": "hnsw graph"})).json()
    second = (await seeded.post("/v1/query", json={"query": "hnsw graph"})).json()

    assert first["answer"] == second["answer"]
    assert [s["chunk_id"] for s in first["sources"]] == [s["chunk_id"] for s in second["sources"]]


# --- validation and errors --------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "x", "top_k": 0},
        {"query": "x", "top_k": 999},
        {"query": "x", "mode": "not-a-mode"},
        {"query": "x", "unknown_field": 1},
    ],
)
async def test_bad_requests_use_the_one_error_envelope(
    seeded: httpx.AsyncClient, payload: dict[str, Any]
) -> None:
    response = await seeded.post("/v1/query", json=payload)

    body = response.json()
    assert response.status_code == 422
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["message"]
    assert body["error"]["request_id"]


async def test_a_typed_error_maps_to_its_own_status_and_code(
    seeded: httpx.AsyncClient,
) -> None:
    response = await seeded.delete("/v1/documents/does-not-exist")

    body = response.json()
    assert response.status_code == 404
    assert body["error"]["code"] == "document_not_found"
    assert "does-not-exist" in body["error"]["message"]


async def test_an_upload_of_an_unsupported_type_is_415(seeded: httpx.AsyncClient) -> None:
    response = await seeded.post(
        "/v1/ingest/upload", files={"files": ("notes.xyz", b"content", "text/plain")}
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_format"


# --- request ids ------------------------------------------------------------------------


async def test_an_inbound_request_id_is_preserved(seeded: httpx.AsyncClient) -> None:
    """A caller's correlation id must survive, or tracing across services breaks."""
    response = await seeded.post(
        "/v1/query", json={"query": "bm25"}, headers={REQUEST_ID_HEADER: "caller-supplied-id"}
    )

    assert response.headers[REQUEST_ID_HEADER] == "caller-supplied-id"
    assert response.json()["request_id"] == "caller-supplied-id"


async def test_a_request_id_is_minted_when_absent(seeded: httpx.AsyncClient) -> None:
    response = await seeded.post("/v1/query", json={"query": "bm25"})

    assert response.headers[REQUEST_ID_HEADER]
    assert response.json()["request_id"] == response.headers[REQUEST_ID_HEADER]


async def test_request_ids_differ_between_requests(seeded: httpx.AsyncClient) -> None:
    first = await seeded.get("/health")
    second = await seeded.get("/health")

    assert first.headers[REQUEST_ID_HEADER] != second.headers[REQUEST_ID_HEADER]


async def test_an_error_body_carries_the_request_id(seeded: httpx.AsyncClient) -> None:
    response = await seeded.post("/v1/query", json={}, headers={REQUEST_ID_HEADER: "trace-me"})

    assert response.json()["error"]["request_id"] == "trace-me"


# --- documents --------------------------------------------------------------------------


async def test_documents_are_listed_with_pagination(seeded: httpx.AsyncClient) -> None:
    body = (await seeded.get("/v1/documents?limit=5&offset=0")).json()

    assert body["total"] == 12
    assert len(body["items"]) == 5
    assert body["items"][0]["chunk_count"] > 0


async def test_document_listing_filters(seeded: httpx.AsyncClient) -> None:
    body = (await seeded.get("/v1/documents?title=HNSW%20Graph%20Indexes")).json()

    assert body["total"] == 1
    assert body["items"][0]["doc_id"] == "doc-hnsw"


async def test_deleting_a_document_removes_its_chunks(seeded: httpx.AsyncClient) -> None:
    body = (await seeded.delete("/v1/documents/doc-hnsw")).json()

    assert body["chunks_deleted"] > 0
    listing = (await seeded.get("/v1/documents?limit=50")).json()
    assert listing["total"] == 11


# --- ingestion --------------------------------------------------------------------------


async def test_uploaded_files_are_ingested(empty: httpx.AsyncClient) -> None:
    markdown = b"---\ntitle: Uploaded\n---\n\n# Uploaded\n\nA paragraph about fusion methods.\n"

    response = await empty.post(
        "/v1/ingest/upload", files={"files": ("uploaded.md", markdown, "text/markdown")}
    )

    body = response.json()
    assert response.status_code == 200
    assert body["docs_changed"] == 1
    assert body["chunks_created"] > 0


async def test_ingesting_a_server_directory(tmp_path: Path, corpus_dir: Path) -> None:
    """The corpus lives under the data directory, which is the only place paths may point."""
    settings = _settings(tmp_path, data_dir=corpus_dir.parent)
    services = _graph(settings)

    async for client in _client(services):
        response = await client.post("/v1/ingest", json={"path": str(corpus_dir)})

        body = response.json()
        assert response.status_code == 200
        assert body["docs_changed"] == 12


async def test_a_path_outside_the_data_directory_is_refused(
    tmp_path: Path, corpus_dir: Path
) -> None:
    """Ingesting an arbitrary server path would be a file-read primitive."""
    settings = _settings(tmp_path, data_dir=corpus_dir.parent)
    services = _graph(settings)

    async for client in _client(services):
        response = await client.post("/v1/ingest", json={"path": "/etc"})

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "document_load_error"
        assert "outside the configured data directory" in response.json()["error"]["message"]


async def test_an_async_job_reports_its_status(tmp_path: Path, corpus_dir: Path) -> None:
    settings = _settings(tmp_path, data_dir=corpus_dir.parent)
    services = _graph(settings)

    async for client in _client(services):
        started = await client.post("/v1/ingest", json={"path": str(corpus_dir), "async_job": True})
        job_id = started.json()["job_id"]
        assert started.json()["status"] in {"queued", "running", "succeeded"}

        # The job runs on the event loop, so polling the route lets it progress.
        for _ in range(50):
            body = (await client.get(f"/v1/ingest/{job_id}")).json()
            if body["status"] in {"succeeded", "failed"}:
                break

        assert body["status"] == "succeeded"
        assert body["summary"]["docs_changed"] == 12


async def test_an_unknown_job_is_404(empty: httpx.AsyncClient) -> None:
    response = await empty.get("/v1/ingest/no-such-job")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "job_not_found"


# --- unit-level helpers -----------------------------------------------------------------


def test_config_overrides_widen_the_rerank_shortlist() -> None:
    """rerank_candidates must never be smaller than final_top_k, or results truncate."""
    base = RetrievalConfig(rerank_candidates=20, final_top_k=5)

    config = resolve_config(QueryRequest(query="q", top_k=40), base)

    assert config.final_top_k == 40
    assert config.rerank_candidates == 40


def test_config_overrides_leave_the_base_untouched() -> None:
    base = RetrievalConfig()

    resolve_config(QueryRequest(query="q", mode=ExpansionMode.HYDE), base)

    assert base.expansion is ExpansionMode.NONE


def test_no_overrides_returns_the_same_config() -> None:
    base = RetrievalConfig()

    assert resolve_config(QueryRequest(query="q"), base) is base


def test_a_relative_path_resolves_inside_the_data_directory(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    services = _graph(settings)

    resolved = resolve_corpus_path(services, "corpus")

    assert resolved == (Path(settings.data_dir) / "corpus").resolve()


def test_a_traversal_attempt_is_refused(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    services = _graph(settings)

    with pytest.raises(DocumentLoadError, match="outside the configured data directory"):
        resolve_corpus_path(services, "../../etc")


def test_no_path_defaults_to_the_configured_corpus(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    services = _graph(settings)

    assert resolve_corpus_path(services, None) == Path(settings.corpus_dir)
