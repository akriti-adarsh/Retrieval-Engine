"""Ingestion end to end: real files, real chunking, real store writes, fake embedder only.

The no-op re-ingest is the headline assertion here. A pipeline that re-embeds an unchanged
corpus takes minutes per run on CPU and stops getting run, so "0 changed" is a feature with
a test, not an implementation detail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retrieval_engine.config import Settings
from retrieval_engine.embed.base import EmbedderInfo
from retrieval_engine.errors import EmbeddingSpaceMismatchError
from retrieval_engine.ingest.pipeline import IngestPipeline
from retrieval_engine.models import ChunkStrategy, LLMKind, StoreKind
from retrieval_engine.store.memory import MemoryVectorStore
from tests.conftest import FAKE_DIMENSION, FIXTURE_CORPUS, FakeEmbedder

CORPUS_SIZE = len(FIXTURE_CORPUS)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
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


def _pipeline(
    tmp_path: Path, store: MemoryVectorStore, embedder: FakeEmbedder, **overrides: object
) -> IngestPipeline:
    return IngestPipeline(_settings(tmp_path, **overrides), store, embedder)


@pytest.fixture
def store() -> MemoryVectorStore:
    return MemoryVectorStore()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dimension=FAKE_DIMENSION)


# --- first pass --------------------------------------------------------------------------


async def test_first_ingest_creates_chunks_for_every_document(
    tmp_path: Path, corpus_dir: Path, store: MemoryVectorStore, embedder: FakeEmbedder
) -> None:
    summary = await _pipeline(tmp_path, store, embedder).ingest_directory(
        corpus_dir, progress=False
    )

    assert summary.docs_seen == CORPUS_SIZE
    assert summary.docs_changed == CORPUS_SIZE
    assert summary.docs_unchanged == 0
    assert summary.docs_failed == 0
    assert summary.chunks_created > CORPUS_SIZE
    assert summary.tokens_embedded > 0
    assert summary.errors == []
    assert await store.count_chunks() == summary.chunks_created


async def test_collection_records_the_embedder(
    tmp_path: Path, corpus_dir: Path, store: MemoryVectorStore, embedder: FakeEmbedder
) -> None:
    await _pipeline(tmp_path, store, embedder).ingest_directory(corpus_dir, progress=False)

    info = await store.collection_info()

    assert info is not None
    assert info.embedder == "fake-embedder"
    assert info.dimension == FAKE_DIMENSION


async def test_ingested_corpus_is_searchable_end_to_end(
    tmp_path: Path, corpus_dir: Path, store: MemoryVectorStore, embedder: FakeEmbedder
) -> None:
    """The point of ingestion: a topical query finds the document that discusses it."""
    await _pipeline(tmp_path, store, embedder).ingest_directory(corpus_dir, progress=False)

    hits = await store.search(
        await embedder.embed_query("layered proximity graph navigable small world"),
        top_k=3,
    )

    assert hits
    assert hits[0].chunk.doc_id == "doc-hnsw"


# --- change detection -------------------------------------------------------------------


async def test_reingesting_an_unchanged_corpus_is_a_no_op(
    tmp_path: Path, corpus_dir: Path, store: MemoryVectorStore, embedder: FakeEmbedder
) -> None:
    pipeline = _pipeline(tmp_path, store, embedder)
    first = await pipeline.ingest_directory(corpus_dir, progress=False)

    second = await pipeline.ingest_directory(corpus_dir, progress=False)

    assert second.docs_changed == 0
    assert second.docs_unchanged == CORPUS_SIZE
    assert second.chunks_created == 0
    assert second.chunks_skipped == first.chunks_created
    assert second.tokens_embedded == 0
    assert "0 changed" in second.render()
    # Nothing was re-embedded on the second pass.
    assert await store.count_chunks() == first.chunks_created


async def test_no_op_reingest_embeds_nothing(
    tmp_path: Path, corpus_dir: Path, store: MemoryVectorStore, embedder: FakeEmbedder
) -> None:
    pipeline = _pipeline(tmp_path, store, embedder)
    await pipeline.ingest_directory(corpus_dir, progress=False)
    calls_after_first = len(embedder.document_calls)

    await pipeline.ingest_directory(corpus_dir, progress=False)

    assert len(embedder.document_calls) == calls_after_first


async def test_touching_a_file_does_not_count_as_a_change(
    tmp_path: Path, corpus_dir: Path, store: MemoryVectorStore, embedder: FakeEmbedder
) -> None:
    """Detection is by content hash, not mtime; a git checkout rewrites mtimes."""
    pipeline = _pipeline(tmp_path, store, embedder)
    await pipeline.ingest_directory(corpus_dir, progress=False)

    (corpus_dir / "doc-bm25.md").touch()
    summary = await pipeline.ingest_directory(corpus_dir, progress=False)

    assert summary.docs_changed == 0
    assert summary.docs_unchanged == CORPUS_SIZE


async def test_editing_one_document_reingests_only_that_one(
    tmp_path: Path, corpus_dir: Path, store: MemoryVectorStore, embedder: FakeEmbedder
) -> None:
    pipeline = _pipeline(tmp_path, store, embedder)
    await pipeline.ingest_directory(corpus_dir, progress=False)

    target = corpus_dir / "doc-bm25.md"
    target.write_text(
        target.read_text(encoding="utf-8") + "\n\nAn added paragraph about term saturation.\n",
        encoding="utf-8",
    )
    summary = await pipeline.ingest_directory(corpus_dir, progress=False)

    assert summary.docs_changed == 1
    assert summary.docs_unchanged == CORPUS_SIZE - 1
    assert summary.chunks_created > 0


async def test_reingest_replaces_rather_than_duplicating_chunks(
    tmp_path: Path, corpus_dir: Path, store: MemoryVectorStore, embedder: FakeEmbedder
) -> None:
    pipeline = _pipeline(tmp_path, store, embedder)
    await pipeline.ingest_directory(corpus_dir, progress=False)
    before = await store.count_chunks()

    target = corpus_dir / "doc-mrr.md"
    target.write_text("---\ntitle: Replaced\n---\n\n# Replaced\n\nShort.\n", encoding="utf-8")
    await pipeline.ingest_directory(corpus_dir, progress=False)

    after = await store.count_chunks()
    assert after < before
    listing = await store.list_documents(limit=100)
    assert {item.doc_id for item in listing.items} == set(FIXTURE_CORPUS)


async def test_chunk_ids_are_identical_across_a_reingest(
    tmp_path: Path, corpus_dir: Path, store: MemoryVectorStore, embedder: FakeEmbedder
) -> None:
    """Stable ids are what let change detection and upsert coexist safely."""
    pipeline = _pipeline(tmp_path, store, embedder)
    await pipeline.ingest_directory(corpus_dir, progress=False)
    before = sorted(chunk.chunk_id for chunk in await store.all_chunks())

    fresh_store = MemoryVectorStore()
    await _pipeline(tmp_path, fresh_store, FakeEmbedder(dimension=FAKE_DIMENSION)).ingest_directory(
        corpus_dir, progress=False
    )
    after = sorted(chunk.chunk_id for chunk in await fresh_store.all_chunks())

    assert before == after


# --- failure handling -------------------------------------------------------------------


async def test_one_unreadable_file_does_not_abort_the_run(
    tmp_path: Path, corpus_dir: Path, store: MemoryVectorStore, embedder: FakeEmbedder
) -> None:
    (corpus_dir / "broken.pdf").write_bytes(b"definitely not a pdf")

    summary = await _pipeline(tmp_path, store, embedder).ingest_directory(
        corpus_dir, progress=False
    )

    assert summary.docs_failed == 1
    assert summary.docs_changed == CORPUS_SIZE
    assert len(summary.errors) == 1
    assert "broken.pdf" in summary.errors[0]


async def test_a_different_embedder_cannot_write_into_the_collection(
    tmp_path: Path, corpus_dir: Path, store: MemoryVectorStore, embedder: FakeEmbedder
) -> None:
    """Mixing embedding spaces craters recall invisibly, so ingestion must refuse."""
    await _pipeline(tmp_path, store, embedder).ingest_directory(corpus_dir, progress=False)

    other = FakeEmbedder(dimension=FAKE_DIMENSION, name="a-different-model")
    assert other.info == EmbedderInfo(
        name="a-different-model",
        dimension=FAKE_DIMENSION,
        normalized=True,
        query_instruction="query: ",
    )

    with pytest.raises(EmbeddingSpaceMismatchError):
        await _pipeline(tmp_path, store, other).ingest_directory(corpus_dir, progress=False)


async def test_empty_directory_produces_an_empty_summary(
    tmp_path: Path, store: MemoryVectorStore, embedder: FakeEmbedder
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    summary = await _pipeline(tmp_path, store, embedder).ingest_directory(empty, progress=False)

    assert summary.docs_seen == 0
    assert summary.chunks_created == 0
    assert "0 changed" in summary.render()


# --- configuration ----------------------------------------------------------------------


async def test_summary_is_independent_of_concurrency(
    tmp_path: Path, corpus_dir: Path, embedder: FakeEmbedder
) -> None:
    """Each task returns its own result, so interleaving cannot change the totals."""
    serial_store = MemoryVectorStore()
    serial = await _pipeline(
        tmp_path, serial_store, embedder, ingest_concurrency=1
    ).ingest_directory(corpus_dir, progress=False)

    parallel_store = MemoryVectorStore()
    parallel = await _pipeline(
        tmp_path, parallel_store, FakeEmbedder(dimension=FAKE_DIMENSION), ingest_concurrency=8
    ).ingest_directory(corpus_dir, progress=False)

    assert serial.docs_changed == parallel.docs_changed
    assert serial.chunks_created == parallel.chunks_created
    assert serial.tokens_embedded == parallel.tokens_embedded


@pytest.mark.parametrize(
    "strategy",
    [ChunkStrategy.FIXED_TOKEN, ChunkStrategy.RECURSIVE_STRUCTURAL, ChunkStrategy.SEMANTIC],
)
async def test_every_chunking_strategy_ingests(
    tmp_path: Path, corpus_dir: Path, embedder: FakeEmbedder, strategy: ChunkStrategy
) -> None:
    store = MemoryVectorStore()

    summary = await _pipeline(tmp_path, store, embedder, chunk_strategy=strategy).ingest_directory(
        corpus_dir, progress=False
    )

    assert summary.docs_changed == CORPUS_SIZE
    assert summary.chunks_created > 0
    chunks = await store.all_chunks()
    assert {chunk.strategy for chunk in chunks} == {strategy}


async def test_progress_bar_can_be_enabled_without_breaking_anything(
    tmp_path: Path, corpus_dir: Path, store: MemoryVectorStore, embedder: FakeEmbedder
) -> None:
    summary = await _pipeline(tmp_path, store, embedder).ingest_directory(corpus_dir, progress=True)

    assert summary.docs_changed == CORPUS_SIZE
