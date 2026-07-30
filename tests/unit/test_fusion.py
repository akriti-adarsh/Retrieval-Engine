"""Fusion, verified against values computed by hand rather than by the implementation.

The spec requires the hand-computed RRF case, and for good reason: a fusion bug does not
crash, it just quietly ranks slightly worse, and every downstream metric absorbs it. The
expected numbers below were worked out from the formula on paper and are written as explicit
reciprocals so a reader can check them without running anything.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from retrieval_engine.models import (
    Chunk,
    ExpansionMode,
    FusionMethod,
    RetrievalConfig,
    ScoredChunk,
    StageScores,
)
from retrieval_engine.retrieve.fusion import (
    fuse,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)


def _chunk(name: str) -> Chunk:
    return Chunk(
        chunk_id=name,
        doc_id="doc-a",
        text=f"text of {name}",
        start_char=0,
        end_char=10,
        token_count=3,
    )


def _ranked(
    names: Sequence[str], *, kind: str = "dense", scores: Sequence[float] | None = None
) -> list[ScoredChunk]:
    """Build a ranked list, best first, tagged as coming from one retriever."""
    results: list[ScoredChunk] = []
    for rank, name in enumerate(names, start=1):
        score = scores[rank - 1] if scores is not None else 1.0 / rank
        stages = (
            StageScores(dense_score=score, dense_rank=rank)
            if kind == "dense"
            else StageScores(lexical_score=score, lexical_rank=rank)
        )
        results.append(ScoredChunk(chunk=_chunk(name), score=score, stages=stages))
    return results


def _order(results: Sequence[ScoredChunk]) -> list[str]:
    return [result.chunk.chunk_id for result in results]


# --- the required hand-computed case ----------------------------------------------------


def test_rrf_matches_values_computed_by_hand() -> None:
    """Dense [A, B, C] and lexical [B, D], k = 60.

    Worked out from score(d) = sum of 1 / (k + rank(d)):

        A = 1/61                 = 0.016393442622950820
        B = 1/62 + 1/61          = 0.032522474881015336
        C = 1/63                 = 0.015873015873015872
        D = 1/62                 = 0.016129032258064516

    So the order is B, A, D, C. Note that A beats D and D beats C by margins smaller
    than a thousandth, which is exactly the kind of ordering an off-by-one in the rank
    would silently invert.
    """
    dense = _ranked(["A", "B", "C"])
    lexical = _ranked(["B", "D"], kind="lexical")

    fused = reciprocal_rank_fusion([dense, lexical], k=60)

    assert _order(fused) == ["B", "A", "D", "C"]
    expected = {
        "A": 1 / 61,
        "B": 1 / 62 + 1 / 61,
        "C": 1 / 63,
        "D": 1 / 62,
    }
    for result in fused:
        assert result.score == pytest.approx(expected[result.chunk.chunk_id], rel=1e-12)


def test_rrf_stamps_the_fused_rank_and_score() -> None:
    fused = reciprocal_rank_fusion([_ranked(["A", "B", "C"])], k=60)

    assert [result.stages.fused_rank for result in fused] == [1, 2, 3]
    for result in fused:
        assert result.stages.fusion_score == result.score


def test_rrf_keeps_both_retrievers_evidence_on_a_shared_chunk() -> None:
    """A chunk found by both must carry both scores, or the debug view lies."""
    dense = _ranked(["A", "B"], scores=[0.9, 0.5])
    lexical = _ranked(["B", "C"], kind="lexical", scores=[7.0, 3.0])

    fused = reciprocal_rank_fusion([dense, lexical], k=60)
    shared = next(result for result in fused if result.chunk.chunk_id == "B")

    assert shared.stages.dense_score == 0.5
    assert shared.stages.dense_rank == 2
    assert shared.stages.lexical_score == 7.0
    assert shared.stages.lexical_rank == 1


# --- what k actually does ---------------------------------------------------------------


def test_small_k_lets_a_single_top_rank_win() -> None:
    """Dense [A, B, C, D] and lexical [E, F, G, D], k = 1.

        A = 1/2               = 0.5
        D = 1/5 + 1/5         = 0.4

    A appears once at rank one and still beats D, which appears in both lists at rank
    four. Small k makes rank one nearly decisive.
    """
    dense = _ranked(["A", "B", "C", "D"])
    lexical = _ranked(["E", "F", "G", "D"], kind="lexical")

    fused = reciprocal_rank_fusion([dense, lexical], k=1)

    assert _order(fused)[0] == "A"
    scores = {result.chunk.chunk_id: result.score for result in fused}
    assert scores["A"] == pytest.approx(0.5)
    assert scores["D"] == pytest.approx(0.4)


def test_large_k_flattens_into_counting_appearances() -> None:
    """Same lists, k = 1000.

        A = 1/1001            = 0.000999000999000999
        D = 1/1004 + 1/1004   = 0.001992031872509960

    Now D wins purely for appearing in two lists. Large k flattens the rank curve until
    fusion is effectively a vote count, which is the trade k controls.
    """
    dense = _ranked(["A", "B", "C", "D"])
    lexical = _ranked(["E", "F", "G", "D"], kind="lexical")

    fused = reciprocal_rank_fusion([dense, lexical], k=1000)

    assert _order(fused)[0] == "D"
    scores = {result.chunk.chunk_id: result.score for result in fused}
    assert scores["A"] == pytest.approx(1 / 1001, rel=1e-12)
    assert scores["D"] == pytest.approx(2 / 1004, rel=1e-12)


def test_rrf_rejects_a_non_positive_k() -> None:
    """k = 0 would divide by zero at rank zero, and negative k is meaningless."""
    with pytest.raises(ValueError, match="must be positive"):
        reciprocal_rank_fusion([_ranked(["A"])], k=0)

    with pytest.raises(ValueError, match="must be positive"):
        reciprocal_rank_fusion([_ranked(["A"])], k=-5)


# --- structural behaviour ---------------------------------------------------------------


def test_rrf_is_scale_free() -> None:
    """The premise of using RRF: only ranks matter, so raw score magnitude cannot."""
    small = _ranked(["A", "B"], scores=[0.9, 0.5])
    huge = _ranked(["A", "B"], scores=[9000.0, 5000.0])

    assert [r.score for r in reciprocal_rank_fusion([small], k=60)] == [
        r.score for r in reciprocal_rank_fusion([huge], k=60)
    ]


def test_rrf_of_one_list_preserves_its_order() -> None:
    fused = reciprocal_rank_fusion([_ranked(["A", "B", "C", "D"])], k=60)

    assert _order(fused) == ["A", "B", "C", "D"]


def test_rrf_of_nothing_is_empty() -> None:
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_rrf_breaks_ties_on_chunk_id() -> None:
    """Two chunks at the same rank in different lists tie exactly, so order must be total."""
    first = reciprocal_rank_fusion([_ranked(["B"]), _ranked(["A"], kind="lexical")], k=60)
    second = reciprocal_rank_fusion([_ranked(["B"]), _ranked(["A"], kind="lexical")], k=60)

    assert _order(first) == ["A", "B"]
    assert _order(first) == _order(second)


def test_rrf_handles_more_than_two_lists() -> None:
    """Multi-query expansion fuses one list per paraphrase."""
    lists = [_ranked(["A", "B"]), _ranked(["A", "C"]), _ranked(["A", "D"])]

    fused = reciprocal_rank_fusion(lists, k=60)

    assert _order(fused)[0] == "A"
    assert fused[0].score == pytest.approx(3 / 61, rel=1e-12)


# --- weighted score fusion --------------------------------------------------------------


def test_weighted_fusion_matches_values_computed_by_hand() -> None:
    """Dense [A 0.9, B 0.5, C 0.1] and lexical [B 10.0, D 2.0], equal weights.

    Min-max within each list:
        dense:   A = 1.0, B = 0.5, C = 0.0
        lexical: B = 1.0, D = 0.0
    Weighted sum at 0.5 each:
        A = 0.5, B = 0.25 + 0.5 = 0.75, C = 0.0, D = 0.0

    This also shows the weakness the ablation is meant to expose: min-max sends the worst
    entry of every list to exactly zero, so C's real cosine and D's real BM25 score are
    both discarded.
    """
    dense = _ranked(["A", "B", "C"], scores=[0.9, 0.5, 0.1])
    lexical = _ranked(["B", "D"], kind="lexical", scores=[10.0, 2.0])

    fused = weighted_score_fusion([dense, lexical], weights=[0.5, 0.5])

    scores = {result.chunk.chunk_id: result.score for result in fused}
    assert scores["B"] == pytest.approx(0.75)
    assert scores["A"] == pytest.approx(0.5)
    assert scores["C"] == pytest.approx(0.0)
    assert scores["D"] == pytest.approx(0.0)
    assert _order(fused)[:2] == ["B", "A"]


def test_weighted_fusion_respects_asymmetric_weights() -> None:
    dense = _ranked(["A", "B"], scores=[1.0, 0.0])
    lexical = _ranked(["B", "A"], kind="lexical", scores=[1.0, 0.0])

    dense_heavy = weighted_score_fusion([dense, lexical], weights=[0.9, 0.1])
    lexical_heavy = weighted_score_fusion([dense, lexical], weights=[0.1, 0.9])

    assert _order(dense_heavy)[0] == "A"
    assert _order(lexical_heavy)[0] == "B"


def test_weighted_fusion_treats_an_all_equal_list_as_jointly_best() -> None:
    """A zero range cannot be normalised, so every entry is treated as the maximum."""
    flat = _ranked(["A", "B"], scores=[0.4, 0.4])

    fused = weighted_score_fusion([flat], weights=[1.0])

    assert all(result.score == pytest.approx(1.0) for result in fused)


def test_weighted_fusion_defaults_to_equal_weights() -> None:
    fused = weighted_score_fusion([_ranked(["A", "B"], scores=[1.0, 0.0])])

    assert fused[0].score == pytest.approx(1.0)


def test_weighted_fusion_rejects_a_weight_count_mismatch() -> None:
    with pytest.raises(ValueError, match="2 weights for 1 lists"):
        weighted_score_fusion([_ranked(["A"])], weights=[0.5, 0.5])


def test_weighted_fusion_of_nothing_is_empty() -> None:
    assert weighted_score_fusion([]) == []


# --- dispatch ---------------------------------------------------------------------------


def test_fuse_dispatches_to_rrf() -> None:
    config = RetrievalConfig(fusion=FusionMethod.RRF, rrf_k=60)
    dense = _ranked(["A", "B", "C"])
    lexical = _ranked(["B", "D"], kind="lexical")

    assert _order(fuse([dense, lexical], config)) == ["B", "A", "D", "C"]


def test_fuse_dispatches_to_weighted_and_uses_dense_weight() -> None:
    config = RetrievalConfig(fusion=FusionMethod.WEIGHTED, dense_weight=0.9)
    dense = _ranked(["A", "B"], scores=[1.0, 0.0])
    lexical = _ranked(["B", "A"], kind="lexical", scores=[1.0, 0.0])

    fused = fuse([dense, lexical], config)

    assert _order(fused)[0] == "A"
    assert fused[0].score == pytest.approx(0.9)


def test_fuse_passes_k_from_the_config() -> None:
    dense = _ranked(["A", "B", "C", "D"])
    lexical = _ranked(["E", "F", "G", "D"], kind="lexical")

    small_k = fuse([dense, lexical], RetrievalConfig(rrf_k=1))
    large_k = fuse([dense, lexical], RetrievalConfig(rrf_k=1000))

    assert _order(small_k)[0] == "A"
    assert _order(large_k)[0] == "D"


def test_fuse_with_more_than_two_lists_uses_uniform_weights() -> None:
    """Multi-query expansion produces more lists than there are configured weights."""
    config = RetrievalConfig(fusion=FusionMethod.WEIGHTED, expansion=ExpansionMode.MULTI_QUERY)
    lists = [
        _ranked(["A", "B"], scores=[1.0, 0.0]),
        _ranked(["A", "C"], scores=[1.0, 0.0]),
        _ranked(["A", "D"], scores=[1.0, 0.0]),
    ]

    fused = fuse(lists, config)

    assert _order(fused)[0] == "A"
    assert fused[0].score == pytest.approx(1.0)
