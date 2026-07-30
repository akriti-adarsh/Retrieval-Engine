"""Dense and lexical candidate generation, including BM25 index staleness.

The staleness tests are the ones that matter operationally: an index that fails to rebuild
serves stale results forever, and an index that rebuilds when nothing changed makes every
query pay for a rebuild.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from retrieval_engine.models import Chunk, Document, EmbeddedChunk, Metadata, make_chunk_id
from retrieval_engine.retrieve.dense import DenseRetriever
from retrieval_engine.retrieve.lexical import BM25Retriever, tokenize
from retrieval_engine.store.memory import MemoryVectorStore
from tests.conftest import FAKE_DIMENSION, FakeEmbedder

TEXTS = {
    "bm25": "BM25 scores lexical overlap with inverse document frequency weighting",
    "hnsw": "HNSW builds a layered proximity graph for approximate nearest neighbour search",
    "rerank": "A cross encoder scores the query and the passage jointly in one pass",
    "ndcg": "Discounted cumulative gain divides each gain by the logarithm of its rank",
}


def _document(doc_id: str = "doc-a") -> Document:
    return Document(
        doc_id=doc_id,
        source_path=f"data/corpus/{doc_id}.md",
        text="body",
        content_hash=hashlib.sha256(doc_id.encode()).hexdigest(),
        media_type="text/markdown",
        metadata={"title": doc_id},
    )


async def _seeded(
    texts: dict[str, str] | None = None, metadata: Metadata | None = None
) -> tuple[MemoryVectorStore, FakeEmbedder]:
    store = MemoryVectorStore()
    embedder = FakeEmbedder()
    await store.ensure_collection(embedder.info)
    body = texts if texts is not None else TEXTS
    vectors = await embedder.embed_documents(list(body.values()))
    records = [
        EmbeddedChunk(
            chunk=Chunk(
                chunk_id=make_chunk_id("doc-a", index * 500),
                doc_id="doc-a",
                text=text,
                start_char=index * 500,
                end_char=index * 500 + len(text),
                token_count=len(text.split()),
                metadata=metadata or {"topic": key},
            ),
            embedding=vector,
        )
        for index, ((key, text), vector) in enumerate(zip(body.items(), vectors, strict=True))
    ]
    await store.upsert_document(_document(), records)
    return store, embedder


# --- dense ------------------------------------------------------------------------------


async def test_dense_records_scores_and_ranks() -> None:
    store, embedder = await _seeded()

    results = await DenseRetriever(embedder, store).retrieve("layered proximity graph", top_k=3)

    assert len(results) == 3
    assert [result.stages.dense_rank for result in results] == [1, 2, 3]
    for result in results:
        assert result.stages.dense_score == result.score
        assert result.stages.lexical_score is None
    assert "HNSW" in results[0].chunk.text


async def test_dense_scores_are_non_increasing() -> None:
    store, embedder = await _seeded()

    results = await DenseRetriever(embedder, store).retrieve("cross encoder", top_k=4)

    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


async def test_dense_applies_the_query_instruction() -> None:
    """The query goes through embed_query, which is where the bge prefix is applied."""
    store, embedder = await _seeded()
    before = len(embedder.query_calls)

    await DenseRetriever(embedder, store).retrieve("some question", top_k=2)

    assert embedder.query_calls[before:] == ["some question"]


async def test_dense_forwards_filters() -> None:
    store, embedder = await _seeded()

    results = await DenseRetriever(embedder, store).retrieve(
        "graph search", top_k=5, filters={"topic": "hnsw"}
    )

    assert len(results) == 1
    assert "HNSW" in results[0].chunk.text


async def test_dense_retrieve_vector_does_not_re_embed() -> None:
    """Multi-query expansion searches with vectors it already has."""
    store, embedder = await _seeded()
    vector = await embedder.embed_query("layered proximity graph")
    before = len(embedder.query_calls)

    results = await DenseRetriever(embedder, store).retrieve_vector(vector, top_k=2)

    assert len(embedder.query_calls) == before
    assert results


async def test_dense_on_an_empty_store_returns_nothing() -> None:
    store = MemoryVectorStore()
    embedder = FakeEmbedder()
    await store.ensure_collection(embedder.info)

    assert await DenseRetriever(embedder, store).retrieve("anything", top_k=5) == []


# --- tokenization -----------------------------------------------------------------------


def test_tokenize_lowercases_and_splits_on_non_alphanumerics() -> None:
    assert tokenize("BM25's IDF-weighting, v2!") == ["bm25", "s", "idf", "weighting", "v2"]


def test_tokenize_empty_text() -> None:
    assert tokenize("   ") == []


# --- lexical ----------------------------------------------------------------------------


async def test_bm25_finds_the_exact_rare_token(tmp_path: Path) -> None:
    """The reason lexical retrieval sits next to a dense retriever."""
    store, _ = await _seeded()

    results = await BM25Retriever(store, tmp_path).retrieve("bm25 idf weighting", top_k=3)

    assert results
    assert "BM25" in results[0].chunk.text
    assert results[0].stages.lexical_rank == 1
    assert results[0].stages.lexical_score == results[0].score
    assert results[0].stages.dense_score is None


async def test_bm25_excludes_zero_scoring_chunks(tmp_path: Path) -> None:
    """A chunk sharing no query term is not a candidate at all."""
    store, _ = await _seeded()

    results = await BM25Retriever(store, tmp_path).retrieve("hnsw", top_k=10)

    assert len(results) < len(TEXTS)
    assert all(result.score > 0.0 for result in results)


async def test_bm25_respects_top_k(tmp_path: Path) -> None:
    store, _ = await _seeded()

    results = await BM25Retriever(store, tmp_path).retrieve("scores the query gain rank", top_k=2)

    assert len(results) <= 2


@pytest.mark.parametrize("top_k", [0, -3])
async def test_bm25_non_positive_top_k(tmp_path: Path, top_k: int) -> None:
    store, _ = await _seeded()

    assert await BM25Retriever(store, tmp_path).retrieve("bm25", top_k=top_k) == []


async def test_bm25_query_with_no_usable_terms(tmp_path: Path) -> None:
    store, _ = await _seeded()

    assert await BM25Retriever(store, tmp_path).retrieve("!!! ??? ...", top_k=5) == []


async def test_bm25_on_an_empty_store(tmp_path: Path) -> None:
    store = MemoryVectorStore()
    await store.ensure_collection(FakeEmbedder().info)
    retriever = BM25Retriever(store, tmp_path)

    assert await retriever.retrieve("anything", top_k=5) == []
    assert retriever.rebuild_count == 0


async def test_bm25_tolerates_a_chunk_with_no_terms(tmp_path: Path) -> None:
    """rank_bm25 cannot score an empty document, so one must not break the index."""
    store, _ = await _seeded(texts={**TEXTS, "empty": "..."})

    results = await BM25Retriever(store, tmp_path).retrieve("bm25 idf", top_k=5)

    assert results
    assert "BM25" in results[0].chunk.text
    assert all(result.chunk.text != "..." for result in results)


async def test_bm25_contributes_nothing_on_a_tiny_corpus(tmp_path: Path) -> None:
    """Pins a real rank_bm25 behaviour so it cannot surprise anyone reading the ablation.

    IDF is log(N - df + 0.5) - log(df + 0.5) with no smoothing, so on a two-chunk corpus a
    term appearing in one chunk has an IDF of exactly zero and scores nothing. Lexical
    retrieval only starts earning its place at realistic corpus sizes.
    """
    store, _ = await _seeded(texts={"a": "bm25 lexical scoring", "b": "hnsw graph search"})

    assert await BM25Retriever(store, tmp_path).retrieve("bm25", top_k=5) == []


async def test_bm25_forwards_filters(tmp_path: Path) -> None:
    store, _ = await _seeded()

    results = await BM25Retriever(store, tmp_path).retrieve(
        "scores", top_k=5, filters={"topic": "rerank"}
    )

    assert all(result.chunk.metadata["topic"] == "rerank" for result in results)


async def test_bm25_filter_matches_inside_a_list_field(tmp_path: Path) -> None:
    store, _ = await _seeded(metadata={"authors": ["X. Yang", "Y. Li"]})

    assert await BM25Retriever(store, tmp_path).retrieve(
        "bm25", top_k=5, filters={"authors": "Y. Li"}
    )
    assert (
        await BM25Retriever(store, tmp_path).retrieve(
            "bm25", top_k=5, filters={"authors": "Z. Nobody"}
        )
        == []
    )


async def test_bm25_is_deterministic(tmp_path: Path) -> None:
    store, _ = await _seeded()
    retriever = BM25Retriever(store, tmp_path)

    first = await retriever.retrieve("scores the query", top_k=4)
    second = await retriever.retrieve("scores the query", top_k=4)

    assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]
    assert [r.score for r in first] == [r.score for r in second]


# --- index staleness --------------------------------------------------------------------


async def test_index_is_built_once_and_reused(tmp_path: Path) -> None:
    store, _ = await _seeded()
    retriever = BM25Retriever(store, tmp_path)

    await retriever.ensure_index()
    await retriever.ensure_index()
    await retriever.retrieve("bm25", top_k=1)

    assert retriever.rebuild_count == 1
    assert retriever.index_path.is_file()


async def test_adding_one_chunk_triggers_a_rebuild(tmp_path: Path) -> None:
    """Required by the spec: the fingerprint must notice a single added chunk."""
    store, embedder = await _seeded()
    retriever = BM25Retriever(store, tmp_path)
    await retriever.ensure_index()
    first_fingerprint = retriever.fingerprint

    extra_vector = (await embedder.embed_documents(["an entirely new chunk about fusion"]))[0]
    await store.upsert_document(
        _document("doc-b"),
        [
            EmbeddedChunk(
                chunk=Chunk(
                    chunk_id=make_chunk_id("doc-b", 0),
                    doc_id="doc-b",
                    text="an entirely new chunk about fusion",
                    start_char=0,
                    end_char=34,
                    token_count=6,
                ),
                embedding=extra_vector,
            )
        ],
    )
    await retriever.ensure_index()

    assert retriever.rebuild_count == 2
    assert retriever.fingerprint != first_fingerprint


async def test_a_no_op_reingest_does_not_rebuild(tmp_path: Path) -> None:
    """Required by the spec: re-upserting identical chunks must not force a rebuild."""
    store, _embedder = await _seeded()
    retriever = BM25Retriever(store, tmp_path)
    await retriever.ensure_index()

    # Re-upsert the same document with the same chunk ids and text.
    same_store, _ = await _seeded()
    fresh = BM25Retriever(same_store, tmp_path)
    await fresh.ensure_index()

    assert fresh.rebuild_count == 0, "an identical chunk set must load from disk"
    assert fresh.fingerprint == retriever.fingerprint


async def test_persisted_index_is_reused_by_a_new_process(tmp_path: Path) -> None:
    store, _ = await _seeded()
    await BM25Retriever(store, tmp_path).ensure_index()

    second = BM25Retriever(store, tmp_path)
    results = await second.retrieve("bm25 idf", top_k=2)

    assert second.rebuild_count == 0
    assert results


async def test_a_corrupt_index_file_is_a_cache_miss_not_a_crash(tmp_path: Path) -> None:
    store, _ = await _seeded()
    retriever = BM25Retriever(store, tmp_path)
    await retriever.ensure_index()
    retriever.index_path.write_text("{ not valid json", encoding="utf-8")

    fresh = BM25Retriever(store, tmp_path)
    results = await fresh.retrieve("bm25", top_k=2)

    assert fresh.rebuild_count == 1
    assert results


async def test_a_fingerprint_mismatch_on_disk_forces_a_rebuild(tmp_path: Path) -> None:
    store, _ = await _seeded()
    retriever = BM25Retriever(store, tmp_path)
    await retriever.ensure_index()
    retriever.index_path.write_text(
        '{"fingerprint": "stale", "chunk_ids": [], "tokens": []}', encoding="utf-8"
    )

    fresh = BM25Retriever(store, tmp_path)
    await fresh.ensure_index()

    assert fresh.rebuild_count == 1


async def test_fingerprint_ignores_chunk_order() -> None:
    """Two stores holding the same chunks in different orders share a fingerprint."""
    forward, _ = await _seeded()
    reversed_texts = dict(reversed(list(TEXTS.items())))
    backward, _ = await _seeded(texts=reversed_texts)

    first = BM25Retriever(forward, Path("unused"))
    second = BM25Retriever(backward, Path("unused"))

    assert await first.ensure_index() == await second.ensure_index()


async def test_dimension_constant_is_shared() -> None:
    """Guards against a test file drifting from the conftest fake's width."""
    assert FakeEmbedder().info.dimension == FAKE_DIMENSION
