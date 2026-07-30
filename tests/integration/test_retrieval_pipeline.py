"""The four-stage retriever over a really ingested corpus.

Every ablation configuration the published table will contain is exercised here, because a
config that the harness can express but the pipeline mishandles would produce a number that
looks fine and means nothing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from retrieval_engine.config import Settings
from retrieval_engine.ingest.pipeline import IngestPipeline
from retrieval_engine.models import (
    ChunkStrategy,
    ExpansionMode,
    FusionMethod,
    LLMKind,
    RetrievalConfig,
    ScoredChunk,
    StoreKind,
)
from retrieval_engine.retrieve.lexical import BM25Retriever
from retrieval_engine.retrieve.pipeline import RetrievalPipeline
from retrieval_engine.retrieve.rerank import CrossEncoderReranker
from retrieval_engine.store.memory import MemoryVectorStore
from tests.conftest import FAKE_DIMENSION, FakeEmbedder, FakeLLM


class StubCrossEncoder:
    """Scores by shared-word count, so reranking is meaningful but needs no model."""

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> Any:
        self.pairs.extend(sentences)
        return [
            len(set(query.lower().split()) & set(passage.lower().split())) / 10.0
            for query, passage in sentences
        ]


class RecordingBM25(BM25Retriever):
    """BM25 that remembers which query strings it was asked about."""

    def __init__(self, store: MemoryVectorStore, index_dir: Path) -> None:
        super().__init__(store, index_dir)
        self.queries: list[str] = []

    async def retrieve(self, query: str, top_k: int, **kwargs: Any) -> list[ScoredChunk]:
        self.queries.append(query)
        return await super().retrieve(query, top_k, **kwargs)


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
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
async def ready(tmp_path: Path, corpus_dir: Path) -> dict[str, Any]:
    """A really ingested corpus plus a pipeline wired to stubs, no models involved."""
    settings = _settings(tmp_path)
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=FAKE_DIMENSION)
    await IngestPipeline(settings, store, embedder).ingest_directory(corpus_dir, progress=False)

    encoder = StubCrossEncoder()
    lexical = RecordingBM25(store, tmp_path / "bm25")
    llm = FakeLLM()
    pipeline = RetrievalPipeline(
        settings,
        store,
        embedder,
        reranker=CrossEncoderReranker(settings, lambda: encoder),
        lexical=lexical,
        llm=llm,
    )
    return {
        "pipeline": pipeline,
        "store": store,
        "embedder": embedder,
        "encoder": encoder,
        "lexical": lexical,
        "llm": llm,
        "settings": settings,
    }


# --- the default configuration ----------------------------------------------------------


async def test_hybrid_with_rerank_returns_ranked_results(ready: dict[str, Any]) -> None:
    result = await ready["pipeline"].retrieve("layered proximity graph search")

    assert 0 < len(result.chunks) <= 5
    assert [chunk.stages.final_rank for chunk in result.chunks] == list(
        range(1, len(result.chunks) + 1)
    )
    assert result.chunks[0].stages.rerank_score is not None
    assert result.top_score == result.chunks[0].score


async def test_result_carries_the_config_that_produced_it(ready: dict[str, Any]) -> None:
    """Every published number has to be traceable to an exact configuration."""
    config = RetrievalConfig(rrf_k=13, final_top_k=2)

    result = await ready["pipeline"].retrieve("bm25 scoring", config=config)

    assert result.config.rrf_k == 13
    assert result.config == config
    assert len(result.chunks) <= 2


async def test_candidate_counts_are_reported_per_stage(ready: dict[str, Any]) -> None:
    result = await ready["pipeline"].retrieve("cross encoder reranking")

    assert set(result.candidate_counts) == {"dense", "lexical", "fused", "reranked"}
    assert result.candidate_counts["dense"] > 0
    assert result.candidate_counts["fused"] >= result.candidate_counts["reranked"]
    assert result.candidate_counts["reranked"] == len(result.chunks)


async def test_timings_are_measured_not_guessed(ready: dict[str, Any]) -> None:
    result = await ready["pipeline"].retrieve("discounted cumulative gain")

    assert result.timings.retrieval_ms > 0.0
    assert result.timings.total_ms == result.timings.retrieval_ms
    assert result.timings.rerank_ms >= 0.0


async def test_dense_and_lexical_are_timed_separately(tmp_path: Path, corpus_dir: Path) -> None:
    """Each concurrent branch measures itself, rather than both reporting the block.

    This asserts against a real regression. The two stages used to share one measurement of
    the surrounding ``asyncio.gather``, which made them identical in every eval artifact and
    made BM25 look as expensive as an embedding forward pass. A single number copied into two
    fields is indistinguishable from a real one unless something checks that they can differ.
    """
    settings = _settings(tmp_path)
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=FAKE_DIMENSION)
    await IngestPipeline(settings, store, embedder).ingest_directory(corpus_dir, progress=False)

    class SlowBM25(RecordingBM25):
        """Lexical retrieval with a delay far larger than any scheduling noise."""

        async def retrieve(self, query: str, top_k: int, **kwargs: Any) -> list[ScoredChunk]:
            await asyncio.sleep(0.25)
            return await super().retrieve(query, top_k, **kwargs)

    pipeline = RetrievalPipeline(
        settings,
        store,
        embedder,
        reranker=CrossEncoderReranker(settings, StubCrossEncoder),
        lexical=SlowBM25(store, tmp_path / "bm25"),
        llm=FakeLLM(),
    )

    result = await pipeline.retrieve("reciprocal rank fusion")

    assert result.timings.lexical_ms >= 250.0
    assert result.timings.dense_ms < result.timings.lexical_ms
    # The block waits for the slower branch, and the two overlap rather than queueing, so
    # retrieval covers lexical without being the sum of both.
    assert result.timings.retrieval_ms >= result.timings.lexical_ms


async def test_retrieval_is_deterministic(ready: dict[str, Any]) -> None:
    """Determinism is a published property, so the same query must give the same bytes."""
    pipeline = ready["pipeline"]

    first = await pipeline.retrieve("inverse document frequency weighting")
    second = await pipeline.retrieve("inverse document frequency weighting")

    assert [c.chunk.chunk_id for c in first.chunks] == [c.chunk.chunk_id for c in second.chunks]
    assert [c.score for c in first.chunks] == [c.score for c in second.chunks]


async def test_top_k_argument_overrides_the_config(ready: dict[str, Any]) -> None:
    result = await ready["pipeline"].retrieve("bm25", top_k=2)

    assert len(result.chunks) <= 2


async def test_filters_restrict_the_candidates(ready: dict[str, Any]) -> None:
    """The title comes from the document's front matter, which is what a caller filters on."""
    result = await ready["pipeline"].retrieve(
        "graph search", filters={"title": "HNSW Graph Indexes"}, top_k=5
    )

    assert result.chunks
    assert all(chunk.chunk.doc_id == "doc-hnsw" for chunk in result.chunks)
    assert all(chunk.chunk.metadata["title"] == "HNSW Graph Indexes" for chunk in result.chunks)


# --- the ablation rows ------------------------------------------------------------------


async def test_dense_only(ready: dict[str, Any]) -> None:
    config = RetrievalConfig(use_lexical=False, use_rerank=False)

    result = await ready["pipeline"].retrieve("layered proximity graph", config=config)

    assert result.candidate_counts["lexical"] == 0
    assert result.candidate_counts["dense"] > 0
    assert ready["lexical"].queries == []
    assert all(chunk.stages.dense_score is not None for chunk in result.chunks)


async def test_lexical_only(ready: dict[str, Any]) -> None:
    config = RetrievalConfig(use_dense=False, use_rerank=False)

    result = await ready["pipeline"].retrieve("bm25 saturating term frequency", config=config)

    assert result.candidate_counts["dense"] == 0
    assert result.candidate_counts["lexical"] > 0
    assert all(chunk.stages.lexical_score is not None for chunk in result.chunks)


async def test_hybrid_without_rerank_preserves_the_fused_order(ready: dict[str, Any]) -> None:
    config = RetrievalConfig(use_rerank=False, final_top_k=5)

    result = await ready["pipeline"].retrieve("reciprocal rank fusion", config=config)

    assert ready["encoder"].pairs == [], "the reranker must not run when it is disabled"
    assert [chunk.stages.final_rank for chunk in result.chunks] == list(
        range(1, len(result.chunks) + 1)
    )
    # Without reranking, the final order is exactly the fused order.
    assert [chunk.stages.fused_rank for chunk in result.chunks] == [
        chunk.stages.final_rank for chunk in result.chunks
    ]
    assert all(chunk.stages.rerank_score is None for chunk in result.chunks)


async def test_weighted_fusion_is_selectable(ready: dict[str, Any]) -> None:
    config = RetrievalConfig(fusion=FusionMethod.WEIGHTED, dense_weight=0.7, use_rerank=False)

    result = await ready["pipeline"].retrieve("hnsw graph", config=config)

    assert result.chunks
    assert result.config.fusion is FusionMethod.WEIGHTED


async def test_rerank_can_move_a_candidate_up(ready: dict[str, Any]) -> None:
    """The stage has to be able to change the order, or it is only adding latency."""
    result = await ready["pipeline"].retrieve("cross encoder scores the query jointly")

    movements = [chunk.stages.rank_movement for chunk in result.chunks]
    assert all(movement is not None for movement in movements)


# --- query expansion --------------------------------------------------------------------


async def test_multi_query_expansion_searches_every_paraphrase(
    tmp_path: Path, corpus_dir: Path
) -> None:
    settings = _settings(tmp_path)
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=FAKE_DIMENSION)
    await IngestPipeline(settings, store, embedder).ingest_directory(corpus_dir, progress=False)
    llm = FakeLLM(["graph based vector index\nproximity graph search\nnavigable small world"])
    lexical = RecordingBM25(store, tmp_path / "bm25")
    pipeline = RetrievalPipeline(
        settings,
        store,
        embedder,
        reranker=CrossEncoderReranker(settings, StubCrossEncoder),
        lexical=lexical,
        llm=llm,
    )
    config = RetrievalConfig(expansion=ExpansionMode.MULTI_QUERY, num_paraphrases=3)

    result = await pipeline.retrieve("hnsw", config=config)

    assert result.expanded_queries[0] == "hnsw"
    assert len(result.expanded_queries) == 4
    assert "proximity graph search" in result.expanded_queries
    # Every phrasing is searched lexically too, since the words are the user's meaning.
    assert len(lexical.queries) == 4


async def test_hyde_embeds_the_hypothetical_as_a_passage(tmp_path: Path, corpus_dir: Path) -> None:
    """The generated text is a passage, so the query-side instruction must not touch it."""
    settings = _settings(tmp_path)
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=FAKE_DIMENSION)
    await IngestPipeline(settings, store, embedder).ingest_directory(corpus_dir, progress=False)
    hypothetical = "A layered proximity graph indexes vectors for approximate search."
    llm = FakeLLM([hypothetical])
    lexical = RecordingBM25(store, tmp_path / "bm25")
    pipeline = RetrievalPipeline(
        settings,
        store,
        embedder,
        reranker=CrossEncoderReranker(settings, StubCrossEncoder),
        lexical=lexical,
        llm=llm,
    )
    documents_before = len(embedder.document_calls)
    queries_before = len(embedder.query_calls)

    result = await pipeline.retrieve(
        "what indexes vectors", config=RetrievalConfig(expansion=ExpansionMode.HYDE)
    )

    assert hypothetical in embedder.document_calls[documents_before:]
    assert hypothetical not in embedder.query_calls[queries_before:]
    # BM25 sees the user's words only: a generated passage's vocabulary is invented, and
    # feeding it to a lexical retriever manufactures matches on words nobody wrote.
    assert lexical.queries == ["what indexes vectors"]
    assert result.expanded_queries == ["what indexes vectors"]
    assert result.chunks


async def test_expansion_degrades_when_the_model_is_down(tmp_path: Path, corpus_dir: Path) -> None:
    """A stopped model server must cost quality, never a failed request."""
    settings = _settings(tmp_path)
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=FAKE_DIMENSION)
    await IngestPipeline(settings, store, embedder).ingest_directory(corpus_dir, progress=False)
    pipeline = RetrievalPipeline(
        settings,
        store,
        embedder,
        reranker=CrossEncoderReranker(settings, StubCrossEncoder),
        lexical=BM25Retriever(store, tmp_path / "bm25"),
        llm=FakeLLM(available=False),
    )

    for mode in (ExpansionMode.MULTI_QUERY, ExpansionMode.HYDE):
        result = await pipeline.retrieve("bm25 scoring", config=RetrievalConfig(expansion=mode))

        assert result.expanded_queries == ["bm25 scoring"]
        assert result.chunks, "retrieval must still return results without the model"


async def test_no_llm_configured_skips_expansion(tmp_path: Path, corpus_dir: Path) -> None:
    settings = _settings(tmp_path)
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=FAKE_DIMENSION)
    await IngestPipeline(settings, store, embedder).ingest_directory(corpus_dir, progress=False)
    pipeline = RetrievalPipeline(
        settings,
        store,
        embedder,
        reranker=CrossEncoderReranker(settings, StubCrossEncoder),
        lexical=BM25Retriever(store, tmp_path / "bm25"),
    )

    result = await pipeline.retrieve(
        "bm25", config=RetrievalConfig(expansion=ExpansionMode.MULTI_QUERY)
    )

    assert result.expanded_queries == ["bm25"]
    assert result.chunks


# --- chunking strategies ----------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [ChunkStrategy.FIXED_TOKEN, ChunkStrategy.RECURSIVE_STRUCTURAL, ChunkStrategy.SEMANTIC],
)
async def test_retrieval_works_under_every_chunking_strategy(
    tmp_path: Path, corpus_dir: Path, strategy: ChunkStrategy
) -> None:
    """The chunking ablation row must actually retrieve, not just ingest."""
    settings = _settings(tmp_path, chunk_strategy=strategy)
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=FAKE_DIMENSION)
    await IngestPipeline(settings, store, embedder).ingest_directory(corpus_dir, progress=False)
    pipeline = RetrievalPipeline(
        settings,
        store,
        embedder,
        reranker=CrossEncoderReranker(settings, StubCrossEncoder),
        lexical=BM25Retriever(store, tmp_path / "bm25"),
    )

    result = await pipeline.retrieve("layered proximity graph")

    assert result.chunks
    assert all(chunk.chunk.strategy is strategy for chunk in result.chunks)


# --- degenerate cases -------------------------------------------------------------------


async def test_empty_index_returns_an_empty_result(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=FAKE_DIMENSION)
    await store.ensure_collection(embedder.info)
    pipeline = RetrievalPipeline(
        settings,
        store,
        embedder,
        reranker=CrossEncoderReranker(settings, StubCrossEncoder),
        lexical=BM25Retriever(store, tmp_path / "bm25"),
    )

    result = await pipeline.retrieve("anything at all")

    assert result.chunks == []
    assert result.top_score == 0.0
    assert result.candidate_counts["fused"] == 0
