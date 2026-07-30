"""The in-memory store: guards, replacement semantics, filters, and total ordering.

This store backs the entire test suite, so it is tested as the real thing rather than as a
convenience: if it quietly diverged from the pgvector contract, every downstream test would
be measuring the wrong system.
"""

from __future__ import annotations

import hashlib

import pytest

from retrieval_engine.embed.base import EmbedderInfo
from retrieval_engine.errors import EmbeddingSpaceMismatchError, StoreUnavailableError
from retrieval_engine.models import (
    Chunk,
    Document,
    EmbeddedChunk,
    Metadata,
    make_chunk_id,
)
from retrieval_engine.store.base import VectorStore
from retrieval_engine.store.memory import MemoryVectorStore
from tests.conftest import FakeEmbedder

DIMENSION = 32


def _document(doc_id: str, text: str, metadata: Metadata | None = None) -> Document:
    return Document(
        doc_id=doc_id,
        source_path=f"data/corpus/{doc_id}.md",
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        media_type="text/markdown",
        metadata=metadata or {"title": doc_id},
    )


async def _records(
    embedder: FakeEmbedder, doc_id: str, texts: list[str], metadata: Metadata | None = None
) -> list[EmbeddedChunk]:
    vectors = await embedder.embed_documents(texts)
    records = []
    for index, (text, vector) in enumerate(zip(texts, vectors, strict=True)):
        start = index * 1000
        records.append(
            EmbeddedChunk(
                chunk=Chunk(
                    chunk_id=make_chunk_id(doc_id, start),
                    doc_id=doc_id,
                    text=text,
                    start_char=start,
                    end_char=start + len(text),
                    token_count=len(text.split()),
                    metadata=metadata or {},
                ),
                embedding=vector,
            )
        )
    return records


async def _seeded() -> tuple[MemoryVectorStore, FakeEmbedder]:
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=DIMENSION)
    await store.ensure_collection(embedder.info)
    document = _document("doc-a", "body")
    records = await _records(
        embedder,
        "doc-a",
        [
            "BM25 scores lexical overlap using inverse document frequency",
            "HNSW builds a layered proximity graph for approximate search",
            "A cross encoder scores the query and passage jointly",
        ],
    )
    await store.upsert_document(document, records)
    return store, embedder


# --- protocol ---------------------------------------------------------------------------


def test_store_satisfies_the_protocol() -> None:
    assert isinstance(MemoryVectorStore(), VectorStore)


# --- collection lifecycle ---------------------------------------------------------------


async def test_ensure_collection_records_the_embedding_space() -> None:
    store = MemoryVectorStore("chunks")
    embedder = FakeEmbedder(dimension=DIMENSION)

    info = await store.ensure_collection(embedder.info)

    assert info.name == "chunks"
    assert info.embedder == "fake-embedder"
    assert info.dimension == DIMENSION
    assert await store.collection_info() is not None


async def test_collection_info_is_none_before_creation() -> None:
    assert await MemoryVectorStore().collection_info() is None


async def test_a_different_embedder_is_refused() -> None:
    """A config change to another model must not silently mix two embedding spaces."""
    store = MemoryVectorStore()
    await store.ensure_collection(FakeEmbedder(dimension=DIMENSION).info)

    other = EmbedderInfo(name="e5-base-v2", dimension=DIMENSION)
    with pytest.raises(EmbeddingSpaceMismatchError, match="refusing vectors from"):
        await store.ensure_collection(other)


async def test_a_different_dimension_is_refused() -> None:
    store = MemoryVectorStore()
    await store.ensure_collection(FakeEmbedder(dimension=DIMENSION).info)

    with pytest.raises(EmbeddingSpaceMismatchError, match="refusing 768-dim"):
        await store.ensure_collection(EmbedderInfo(name="fake-embedder", dimension=768))


async def test_reopening_with_the_same_embedder_is_fine() -> None:
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=DIMENSION)

    first = await store.ensure_collection(embedder.info)
    second = await store.ensure_collection(embedder.info)

    assert first.embedder == second.embedder


# --- write guards -----------------------------------------------------------------------


async def test_upsert_before_ensure_collection_is_refused() -> None:
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=DIMENSION)
    records = await _records(embedder, "doc-a", ["text"])

    with pytest.raises(StoreUnavailableError, match="ensure_collection"):
        await store.upsert_document(_document("doc-a", "text"), records)


async def test_wrong_width_records_are_refused() -> None:
    store = MemoryVectorStore()
    await store.ensure_collection(FakeEmbedder(dimension=DIMENSION).info)
    narrow = await _records(FakeEmbedder(dimension=8), "doc-a", ["text"])

    with pytest.raises(EmbeddingSpaceMismatchError, match="8-dim vector"):
        await store.upsert_document(_document("doc-a", "text"), narrow)


# --- replacement semantics --------------------------------------------------------------


async def test_reupsert_replaces_chunks_rather_than_duplicating() -> None:
    """A re-ingest must not strand chunks from a previous chunking strategy."""
    store, embedder = await _seeded()
    assert await store.count_chunks() == 3

    document = _document("doc-a", "new body")
    await store.upsert_document(document, await _records(embedder, "doc-a", ["only one chunk now"]))

    assert await store.count_chunks() == 1
    chunks = await store.all_chunks()
    assert [chunk.text for chunk in chunks] == ["only one chunk now"]


async def test_content_hash_is_recorded_for_change_detection() -> None:
    store, _embedder = await _seeded()

    hashes = await store.document_hashes()

    assert set(hashes) == {"doc-a"}
    assert hashes["doc-a"] == _document("doc-a", "body").content_hash


async def test_hash_changes_after_reingesting_changed_content() -> None:
    store, embedder = await _seeded()
    changed = _document("doc-a", "different body")

    await store.upsert_document(changed, await _records(embedder, "doc-a", ["chunk"]))

    assert (await store.document_hashes())["doc-a"] == changed.content_hash


async def test_delete_removes_chunks_and_returns_the_count() -> None:
    store, _ = await _seeded()

    removed = await store.delete_document("doc-a")

    assert removed == 3
    assert await store.count_chunks() == 0
    assert await store.document_hashes() == {}
    assert await store.all_chunks() == []


async def test_deleting_an_unknown_document_is_zero_not_an_error() -> None:
    """The caller decides whether a missing document is a 404 or nothing to do."""
    store, _ = await _seeded()

    assert await store.delete_document("never-existed") == 0


async def test_collection_chunk_count_tracks_writes() -> None:
    store, _embedder = await _seeded()

    info = await store.collection_info()
    assert info is not None
    assert info.chunk_count == 3

    await store.delete_document("doc-a")
    info = await store.collection_info()
    assert info is not None
    assert info.chunk_count == 0


# --- search -----------------------------------------------------------------------------


async def test_search_ranks_the_relevant_chunk_first() -> None:
    store, embedder = await _seeded()

    hits = await store.search(await embedder.embed_query("layered proximity graph"), top_k=3)

    assert len(hits) == 3
    assert "HNSW" in hits[0].chunk.text
    # Scores must be non-increasing.
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


async def test_search_respects_top_k() -> None:
    store, embedder = await _seeded()

    hits = await store.search(await embedder.embed_query("bm25"), top_k=2)

    assert len(hits) == 2


@pytest.mark.parametrize("top_k", [0, -1])
async def test_non_positive_top_k_returns_nothing(top_k: int) -> None:
    store, embedder = await _seeded()

    assert await store.search(await embedder.embed_query("bm25"), top_k=top_k) == []


async def test_search_on_an_empty_store_returns_nothing() -> None:
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=DIMENSION)
    await store.ensure_collection(embedder.info)

    assert await store.search(await embedder.embed_query("anything"), top_k=5) == []


async def test_search_before_ensure_collection_is_refused() -> None:
    with pytest.raises(StoreUnavailableError):
        await MemoryVectorStore().search([0.1] * DIMENSION, top_k=1)


async def test_zero_magnitude_query_returns_nothing_rather_than_an_arbitrary_order() -> None:
    store, _ = await _seeded()

    assert await store.search([0.0] * DIMENSION, top_k=3) == []


async def test_wrong_width_query_is_refused() -> None:
    store, _ = await _seeded()

    with pytest.raises(StoreUnavailableError, match="dimensions but the collection holds"):
        await store.search([0.5] * 8, top_k=1)


async def test_ef_search_is_accepted_and_ignored() -> None:
    """A brute-force store has no search effort to tune, but the signature is shared."""
    store, embedder = await _seeded()
    query = await embedder.embed_query("cross encoder")

    with_ef = await store.search(query, top_k=3, ef_search=200)
    without_ef = await store.search(query, top_k=3)

    assert [hit.chunk.chunk_id for hit in with_ef] == [hit.chunk.chunk_id for hit in without_ef]


async def test_identical_searches_are_byte_identical_including_ties() -> None:
    """Determinism is a published property of this system, so ties must break stably."""
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=DIMENSION)
    await store.ensure_collection(embedder.info)
    # Three chunks with identical text, so their vectors and therefore scores tie exactly.
    records = await _records(embedder, "doc-t", ["same text here"] * 3)
    await store.upsert_document(_document("doc-t", "body"), records)
    query = await embedder.embed_query("same text here")

    first = await store.search(query, top_k=3)
    second = await store.search(query, top_k=3)

    assert [hit.score for hit in first] == [hit.score for hit in second]
    assert [hit.chunk.chunk_id for hit in first] == [hit.chunk.chunk_id for hit in second]
    ids = [hit.chunk.chunk_id for hit in first]
    assert ids == sorted(ids), "tied scores must fall back to chunk_id order"


# --- filters ----------------------------------------------------------------------------


async def test_filters_restrict_results_by_chunk_metadata() -> None:
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=DIMENSION)
    await store.ensure_collection(embedder.info)
    await store.upsert_document(
        _document("doc-2024", "body"),
        await _records(embedder, "doc-2024", ["retrieval text"], metadata={"year": "2024"}),
    )
    await store.upsert_document(
        _document("doc-2025", "body"),
        await _records(embedder, "doc-2025", ["retrieval text"], metadata={"year": "2025"}),
    )
    query = await embedder.embed_query("retrieval text")

    hits = await store.search(query, top_k=5, filters={"year": "2024"})

    assert len(hits) == 1
    assert hits[0].chunk.doc_id == "doc-2024"


async def test_filters_fall_back_to_document_metadata() -> None:
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=DIMENSION)
    await store.ensure_collection(embedder.info)
    await store.upsert_document(
        _document("doc-a", "body", metadata={"title": "A", "venue": "acl"}),
        await _records(embedder, "doc-a", ["retrieval text"]),
    )
    query = await embedder.embed_query("retrieval text")

    assert len(await store.search(query, top_k=5, filters={"venue": "acl"})) == 1
    assert await store.search(query, top_k=5, filters={"venue": "emnlp"}) == []


async def test_filter_matches_inside_a_list_valued_field() -> None:
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=DIMENSION)
    await store.ensure_collection(embedder.info)
    await store.upsert_document(
        _document("doc-a", "body", metadata={"title": "A", "authors": ["X. Yang", "Y. Li"]}),
        await _records(embedder, "doc-a", ["retrieval text"]),
    )
    query = await embedder.embed_query("retrieval text")

    assert len(await store.search(query, top_k=5, filters={"authors": "Y. Li"})) == 1
    assert await store.search(query, top_k=5, filters={"authors": "Z. Nobody"}) == []


async def test_unknown_filter_key_matches_nothing() -> None:
    store, embedder = await _seeded()
    query = await embedder.embed_query("bm25")

    assert await store.search(query, top_k=5, filters={"nope": "value"}) == []


# --- listing ----------------------------------------------------------------------------


async def _multi_doc_store() -> tuple[MemoryVectorStore, FakeEmbedder]:
    store = MemoryVectorStore()
    embedder = FakeEmbedder(dimension=DIMENSION)
    await store.ensure_collection(embedder.info)
    for index in range(5):
        doc_id = f"doc-{index}"
        await store.upsert_document(
            _document(doc_id, f"body {index}", metadata={"title": f"Title {index}"}),
            await _records(embedder, doc_id, [f"chunk one {index}", f"chunk two {index}"]),
        )
    return store, embedder


async def test_listing_is_paginated_with_a_correct_total() -> None:
    store, _ = await _multi_doc_store()

    page = await store.list_documents(limit=2, offset=0)

    assert page.total == 5
    assert len(page.items) == 2
    assert page.items[0].doc_id == "doc-0"
    assert page.items[0].chunk_count == 2
    assert page.items[0].title == "Title 0"


async def test_listing_offset_walks_the_whole_set() -> None:
    store, _ = await _multi_doc_store()

    seen: list[str] = []
    for offset in (0, 2, 4):
        page = await store.list_documents(limit=2, offset=offset)
        seen.extend(item.doc_id for item in page.items)

    assert seen == ["doc-0", "doc-1", "doc-2", "doc-3", "doc-4"]


async def test_listing_offset_past_the_end_is_empty_not_an_error() -> None:
    store, _ = await _multi_doc_store()

    page = await store.list_documents(limit=2, offset=99)

    assert page.items == []
    assert page.total == 5


async def test_listing_limit_larger_than_the_total() -> None:
    store, _ = await _multi_doc_store()

    page = await store.list_documents(limit=100, offset=0)

    assert len(page.items) == 5


async def test_listing_filters_on_document_metadata() -> None:
    store, _ = await _multi_doc_store()

    page = await store.list_documents(limit=10, filters={"title": "Title 3"})

    assert page.total == 1
    assert page.items[0].doc_id == "doc-3"


# --- lifecycle --------------------------------------------------------------------------


async def test_health_and_close() -> None:
    store, embedder = await _seeded()

    assert await store.health() is True
    await store.close()
    await store.close()

    # Closing releases the cached matrix but must not lose the data.
    assert await store.count_chunks() == 3
    hits = await store.search(await embedder.embed_query("bm25 lexical"), top_k=1)
    assert len(hits) == 1
