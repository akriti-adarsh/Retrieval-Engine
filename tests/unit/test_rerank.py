"""Cross-encoder reranking: cache behaviour, batching, ordering, and rank movement.

The reranker is the most expensive stage, so the cache is not a nicety and gets tested like
load-bearing code: eviction order, hit accounting, key isolation between queries, and the
fact that a hit performs no forward pass at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from retrieval_engine.config import Settings
from retrieval_engine.models import Chunk, ScoredChunk, StageScores
from retrieval_engine.retrieve.rerank import (
    CrossEncoderReranker,
    ScoreCache,
    query_hash,
)


class StubCrossEncoder:
    """Scores a pair by counting shared words, and records every batch it was given."""

    def __init__(self) -> None:
        self.batches: list[list[tuple[str, str]]] = []

    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> Any:
        self.batches.append(list(sentences))
        scores = []
        for query, passage in sentences:
            shared = set(query.lower().split()) & set(passage.lower().split())
            scores.append(len(shared) / 10.0)
        return scores

    @property
    def pairs_seen(self) -> int:
        return sum(len(batch) for batch in self.batches)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"_env_file": None, "env": "test", "reranker_batch_size": 4}
    base.update(overrides)
    return Settings(**base)


def _candidate(name: str, text: str, fused_rank: int) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=name,
            doc_id="doc-a",
            text=text,
            start_char=0,
            end_char=len(text),
            token_count=len(text.split()),
        ),
        score=1.0 / fused_rank,
        stages=StageScores(fusion_score=1.0 / fused_rank, fused_rank=fused_rank),
    )


def _shortlist() -> list[ScoredChunk]:
    return [
        _candidate("c1", "unrelated text about cooking", 1),
        _candidate("c2", "layered proximity graph search", 2),
        _candidate("c3", "something else entirely", 3),
    ]


def _reranker(
    model: StubCrossEncoder | None = None, **overrides: Any
) -> tuple[CrossEncoderReranker, StubCrossEncoder]:
    stub = model if model is not None else StubCrossEncoder()
    return CrossEncoderReranker(_settings(**overrides), lambda: stub), stub


# --- the cache --------------------------------------------------------------------------


def test_cache_rejects_a_non_positive_size() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ScoreCache(0)


def test_cache_counts_hits_and_misses() -> None:
    cache = ScoreCache(4)

    assert cache.get(("q", "c1")) is None
    cache.put(("q", "c1"), 0.9)
    assert cache.get(("q", "c1")) == 0.9

    assert cache.misses == 1
    assert cache.hits == 1
    assert cache.hit_rate == pytest.approx(0.5)


def test_hit_rate_before_any_lookup_is_zero() -> None:
    assert ScoreCache(4).hit_rate == 0.0


def test_cache_evicts_the_least_recently_used() -> None:
    cache = ScoreCache(2)
    cache.put(("q", "a"), 1.0)
    cache.put(("q", "b"), 2.0)

    # Touch "a" so "b" becomes the least recently used.
    assert cache.get(("q", "a")) == 1.0
    cache.put(("q", "c"), 3.0)

    assert len(cache) == 2
    assert cache.get(("q", "a")) == 1.0
    assert cache.get(("q", "b")) is None


def test_cache_never_exceeds_its_bound() -> None:
    cache = ScoreCache(3)
    for index in range(20):
        cache.put(("q", f"c{index}"), float(index))

    assert len(cache) == 3


def test_query_hash_is_stable_and_distinguishing() -> None:
    """Stable across processes, because the builtin hash is salted per run."""
    assert query_hash("what is rrf") == query_hash("what is rrf")
    assert query_hash("what is rrf") != query_hash("what is hnsw")
    assert len(query_hash("x")) == 16


# --- reranking --------------------------------------------------------------------------


async def test_rerank_reorders_by_cross_encoder_score() -> None:
    """The point of the stage: a candidate the fusion ranked third can win."""
    reranker, _ = _reranker()

    results = await reranker.rerank("layered proximity graph", _shortlist(), top_k=3)

    assert [result.chunk.chunk_id for result in results] == ["c2", "c1", "c3"]
    assert results[0].score > results[1].score


async def test_rerank_stamps_final_rank_and_keeps_the_fused_rank() -> None:
    """Rank movement is only inspectable if the earlier rank survives this stage."""
    reranker, _ = _reranker()

    results = await reranker.rerank("layered proximity graph", _shortlist(), top_k=3)

    winner = results[0]
    assert winner.stages.final_rank == 1
    assert winner.stages.fused_rank == 2
    assert winner.stages.rerank_score == winner.score
    assert winner.stages.rank_movement == 1


async def test_rerank_truncates_to_top_k() -> None:
    reranker, _ = _reranker()

    results = await reranker.rerank("graph", _shortlist(), top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c2"


@pytest.mark.parametrize("top_k", [0, -2])
async def test_rerank_non_positive_top_k(top_k: int) -> None:
    reranker, stub = _reranker()

    assert await reranker.rerank("graph", _shortlist(), top_k=top_k) == []
    assert stub.pairs_seen == 0


async def test_rerank_with_no_candidates_does_no_work() -> None:
    reranker, stub = _reranker()

    assert await reranker.rerank("graph", [], top_k=5) == []
    assert stub.batches == []


async def test_rerank_is_deterministic() -> None:
    reranker, _ = _reranker()
    shortlist = _shortlist()

    first = await reranker.rerank("graph search", shortlist, top_k=3)
    second = await reranker.rerank("graph search", shortlist, top_k=3)

    assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]
    assert [r.score for r in first] == [r.score for r in second]


async def test_ties_break_on_chunk_id() -> None:
    """Identical passages score identically, so the order must still be total."""
    reranker, _ = _reranker()
    tied = [
        _candidate("zzz", "identical passage text", 1),
        _candidate("aaa", "identical passage text", 2),
    ]

    results = await reranker.rerank("identical passage", tied, top_k=2)

    assert [result.chunk.chunk_id for result in results] == ["aaa", "zzz"]


# --- model loading and batching ---------------------------------------------------------


def test_model_is_not_loaded_until_rerank_runs() -> None:
    """The reranker weights are around 1.5 GB, so construction must stay cheap."""
    calls = 0

    def factory() -> StubCrossEncoder:
        nonlocal calls
        calls += 1
        return StubCrossEncoder()

    CrossEncoderReranker(_settings(), factory)

    assert calls == 0


async def test_pairs_are_batched_by_the_configured_size() -> None:
    reranker, stub = _reranker(reranker_batch_size=2)
    many = [_candidate(f"c{index}", f"passage {index} graph", index + 1) for index in range(5)]

    await reranker.rerank("graph", many, top_k=5)

    assert [len(batch) for batch in stub.batches] == [2, 2, 1]


async def test_the_query_is_paired_with_every_passage() -> None:
    reranker, stub = _reranker()

    await reranker.rerank("my query", _shortlist(), top_k=3)

    pairs = [pair for batch in stub.batches for pair in batch]
    assert all(pair[0] == "my query" for pair in pairs)
    assert {pair[1] for pair in pairs} == {c.chunk.text for c in _shortlist()}


# --- caching in the retrieval path ------------------------------------------------------


async def test_a_repeated_query_performs_no_forward_pass() -> None:
    reranker, stub = _reranker()
    shortlist = _shortlist()

    await reranker.rerank("graph search", shortlist, top_k=3)
    after_first = stub.pairs_seen

    await reranker.rerank("graph search", shortlist, top_k=3)

    assert after_first == 3
    assert stub.pairs_seen == 3, "a fully cached query must not call the model at all"
    assert reranker.cache.hits == 3


async def test_a_different_query_is_a_separate_cache_key() -> None:
    """Caching on chunk id alone would serve one query's scores to another."""
    reranker, stub = _reranker()
    shortlist = _shortlist()

    await reranker.rerank("graph search", shortlist, top_k=3)
    await reranker.rerank("completely different question", shortlist, top_k=3)

    assert stub.pairs_seen == 6


async def test_only_the_uncached_candidates_are_scored() -> None:
    """A partly warm shortlist must cost only the new chunks."""
    reranker, stub = _reranker()

    await reranker.rerank("graph", _shortlist()[:2], top_k=2)
    assert stub.pairs_seen == 2

    await reranker.rerank("graph", _shortlist(), top_k=3)

    assert stub.pairs_seen == 3, "only the third chunk was new"
    assert reranker.cache.hits == 2


async def test_cached_scores_produce_the_same_ordering() -> None:
    reranker, _ = _reranker()
    shortlist = _shortlist()

    cold = await reranker.rerank("layered proximity graph", shortlist, top_k=3)
    warm = await reranker.rerank("layered proximity graph", shortlist, top_k=3)

    assert [r.chunk.chunk_id for r in cold] == [r.chunk.chunk_id for r in warm]
    assert [r.score for r in cold] == [r.score for r in warm]


async def test_cache_size_comes_from_settings() -> None:
    reranker, stub = _reranker(reranker_cache_size=1)
    shortlist = _shortlist()

    await reranker.rerank("graph", shortlist, top_k=3)
    # A one-entry cache cannot hold three chunks, so the next call re-scores.
    await reranker.rerank("graph", shortlist, top_k=3)

    assert len(reranker.cache) == 1
    assert stub.pairs_seen > 3
