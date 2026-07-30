"""Metrics, every expected value computed by hand.

The point of writing these metrics out rather than importing them is that a reader can check
them. That is only true if the tests state the arithmetic, so each case below shows the
formula it is checking.

The span-to-chunk predicate gets the most attention, including the split-span case the spec
calls for, because it defines what "relevant" means and every other number inherits it.
"""

from __future__ import annotations

import math

import pytest

from retrieval_engine.eval.metrics import (
    MIN_SPAN_OVERLAP,
    answer_similarity,
    chunk_matches_golden,
    citation_precision,
    cosine_similarity,
    count_relevant,
    dcg_at_k,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    normalise,
    overlaps_span,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    refusal_accuracy,
    relevance_flags,
)
from retrieval_engine.models import Chunk, Difficulty, GoldenCategory, GoldenEntry

SPAN = (
    "Reciprocal rank fusion assigns each document a score equal to the sum over result "
    "lists of one divided by the quantity k plus the rank of that document."
)


def _chunk(text: str, chunk_id: str = "c1", doc_id: str = "doc-a") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
    )


def _entry(*spans: str, category: GoldenCategory = GoldenCategory.FACTUAL) -> GoldenEntry:
    # A negative entry may carry no evidence at all, which the schema enforces, so the
    # helper must not hand it document ids either.
    negative = category is GoldenCategory.NEGATIVE
    return GoldenEntry(
        qid="q001",
        question="how does reciprocal rank fusion score documents?",
        relevant_doc_ids=[] if negative else ["doc-a"],
        relevant_chunk_texts=list(spans),
        answer="It sums one over k plus rank across lists.",
        category=category,
        difficulty=Difficulty.MEDIUM,
    )


# --- the relevance predicate ------------------------------------------------------------


def test_whitespace_is_normalised_before_matching() -> None:
    """Chunks keep the source's newlines; golden spans are written on one line."""
    assert normalise("a\n  b\tc") == "a b c"


def test_a_chunk_containing_the_span_in_full_is_relevant() -> None:
    chunk = _chunk(f"Some preamble. {SPAN} Some trailing prose.")

    assert chunk_matches_golden(chunk, _entry(SPAN)) is True


def test_a_span_survives_the_chunk_line_breaks() -> None:
    """The span was copied from this very chunk, so it must match despite rewrapping."""
    wrapped = SPAN.replace(" ", "\n", 8)

    assert overlaps_span(wrapped, SPAN) is True


def test_a_span_split_across_two_chunks_makes_both_relevant() -> None:
    """Required by the spec. Without this the chunking ablation compares different truths."""
    midpoint = len(SPAN) // 2
    first = _chunk(SPAN[:midpoint], chunk_id="c1")
    second = _chunk(SPAN[midpoint:], chunk_id="c2")

    # Each half is well over the overlap threshold, so each shares enough of the span.
    assert len(SPAN[:midpoint]) > MIN_SPAN_OVERLAP
    assert chunk_matches_golden(first, _entry(SPAN)) is True
    assert chunk_matches_golden(second, _entry(SPAN)) is True


def test_an_unrelated_chunk_is_not_relevant() -> None:
    chunk = _chunk("HNSW builds a layered proximity graph for approximate search.")

    assert chunk_matches_golden(chunk, _entry(SPAN)) is False


def test_a_short_incidental_phrase_does_not_qualify() -> None:
    """An overlap below the threshold is coincidence, not evidence."""
    chunk = _chunk("The rank of that document is what matters here.")

    assert chunk_matches_golden(chunk, _entry(SPAN)) is False


def test_overlap_exactly_at_the_threshold_qualifies() -> None:
    piece = normalise(SPAN)[:MIN_SPAN_OVERLAP]

    assert overlaps_span(f"prefix {piece} suffix", SPAN) is True


def test_overlap_one_character_short_does_not_qualify() -> None:
    piece = normalise(SPAN)[: MIN_SPAN_OVERLAP - 1]

    assert overlaps_span(f"prefix {piece} suffix", SPAN) is False


def test_a_span_shorter_than_the_threshold_needs_full_containment() -> None:
    short = "nDCG at five"

    assert overlaps_span("we report nDCG at five in the table", short) is True
    assert overlaps_span("we report other metrics", short) is False


def test_any_matching_span_makes_the_chunk_relevant() -> None:
    other = "A cross encoder scores the query and passage jointly in a single forward pass."
    chunk = _chunk(f"Preamble. {other}")

    assert chunk_matches_golden(chunk, _entry(SPAN, other)) is True


def test_an_entry_with_no_spans_matches_nothing() -> None:
    """Negative entries carry no spans, so nothing can be relevant to them."""
    entry = _entry(category=GoldenCategory.NEGATIVE)

    assert chunk_matches_golden(_chunk(SPAN), entry) is False


def test_relevance_does_not_require_the_document_id_to_match() -> None:
    """A correct passage found in an unexpected document is still a correct passage."""
    chunk = _chunk(SPAN, doc_id="some-other-doc")

    assert chunk_matches_golden(chunk, _entry(SPAN)) is True


def test_empty_text_is_never_relevant() -> None:
    assert overlaps_span("", SPAN) is False
    assert overlaps_span(SPAN, "") is False


def test_count_relevant_scans_a_chunk_set() -> None:
    chunks = [
        _chunk(SPAN, chunk_id="c1"),
        _chunk("unrelated text about indexes", chunk_id="c2"),
        _chunk(f"also here: {SPAN}", chunk_id="c3"),
    ]

    assert count_relevant(chunks, _entry(SPAN)) == 2


def test_relevance_flags_preserve_rank_order() -> None:
    chunks = [_chunk("unrelated", chunk_id="c1"), _chunk(SPAN, chunk_id="c2")]

    assert relevance_flags(chunks, _entry(SPAN)) == [False, True]


# --- recall, precision, hit rate --------------------------------------------------------


def test_recall_at_k_by_hand() -> None:
    """Flags [T, F, T, F, F], 4 relevant in the corpus, k=3: 2 found / 4 = 0.5."""
    flags = [True, False, True, False, False]

    assert recall_at_k(flags, total_relevant=4, k=3) == pytest.approx(0.5)


def test_recall_is_capped_at_one() -> None:
    """Retrieving the same relevant document twice cannot exceed perfect recall."""
    assert recall_at_k([True, True, True], total_relevant=2, k=3) == pytest.approx(1.0)


def test_recall_with_no_relevant_documents_is_zero_not_one() -> None:
    """Scoring an unanswerable question 1.0 would inflate the average for free."""
    assert recall_at_k([False, False], total_relevant=0, k=2) == 0.0


def test_recall_with_non_positive_k() -> None:
    assert recall_at_k([True], total_relevant=1, k=0) == 0.0


def test_precision_at_k_by_hand() -> None:
    """Flags [T, F, T, F, F], k=4: 2 relevant / 4 = 0.5."""
    assert precision_at_k([True, False, True, False, False], k=4) == pytest.approx(0.5)


def test_precision_divides_by_k_not_by_what_came_back() -> None:
    """Returning 2 results when 5 were asked for is a failure, not perfect precision."""
    assert precision_at_k([True, True], k=5) == pytest.approx(0.4)


def test_hit_rate_is_binary() -> None:
    assert hit_rate_at_k([False, False, True], k=3) == 1.0
    assert hit_rate_at_k([False, False, True], k=2) == 0.0
    assert hit_rate_at_k([], k=5) == 0.0


# --- reciprocal rank --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([True, False, False], 1.0),
        ([False, True, False], 0.5),
        ([False, False, True], 1 / 3),
        ([False, False, False], 0.0),
        ([], 0.0),
    ],
)
def test_reciprocal_rank_by_hand(flags: list[bool], expected: float) -> None:
    assert reciprocal_rank(flags) == pytest.approx(expected)


def test_mrr_by_hand() -> None:
    """Three queries with first relevant at ranks 1, 3, and never: (1 + 1/3 + 0) / 3."""
    per_query = [[True], [False, False, True], [False, False]]

    assert mrr(per_query) == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)


def test_mrr_of_nothing() -> None:
    assert mrr([]) == 0.0


# --- dcg and ndcg -----------------------------------------------------------------------


def test_dcg_at_k_by_hand() -> None:
    """Gains [1, 0, 1], k=3: 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.0 + 0 + 0.5 = 1.5."""
    assert dcg_at_k([1.0, 0.0, 1.0], k=3) == pytest.approx(1.5)


def test_dcg_respects_k() -> None:
    assert dcg_at_k([1.0, 1.0, 1.0], k=1) == pytest.approx(1.0)
    assert dcg_at_k([1.0], k=0) == 0.0


def test_ndcg_at_k_by_hand() -> None:
    """Gains [0, 1, 1] at k=3.

    DCG   = 0/log2(2) + 1/log2(3) + 1/log2(4) = 0.63093 + 0.5 = 1.13093
    ideal = [1, 1, 0] -> 1/log2(2) + 1/log2(3) + 0 = 1.0 + 0.63093 = 1.63093
    nDCG  = 1.13093 / 1.63093 = 0.69343
    """
    expected_dcg = 1 / math.log2(3) + 1 / math.log2(4)
    expected_ideal = 1.0 + 1 / math.log2(3)

    assert ndcg_at_k([0.0, 1.0, 1.0], k=3) == pytest.approx(expected_dcg / expected_ideal)
    assert ndcg_at_k([0.0, 1.0, 1.0], k=3) == pytest.approx(0.693426, abs=1e-6)


def test_a_perfect_ranking_scores_one() -> None:
    assert ndcg_at_k([1.0, 1.0, 0.0, 0.0], k=4) == pytest.approx(1.0)


def test_ndcg_with_no_relevant_documents_is_zero() -> None:
    """The zero-relevant edge case the spec asks for: the ideal is zero, so guard it."""
    assert ndcg_at_k([0.0, 0.0, 0.0], k=3) == 0.0
    assert ndcg_at_k([], k=3) == 0.0
    assert ndcg_at_k([1.0], k=0) == 0.0


def test_ties_do_not_change_the_score() -> None:
    """All-relevant is already ideal in any order, so tie-breaking cannot matter."""
    assert ndcg_at_k([1.0, 1.0, 1.0], k=3) == pytest.approx(1.0)


def test_unretrieved_relevant_documents_lower_ndcg() -> None:
    """Without total_relevant, finding 1 of 10 and ranking it first would score 1.0.

    gains [1], k=1, total_relevant=1  -> ideal is [1]        -> 1.0
    gains [1], k=3, total_relevant=3  -> ideal is [1, 1, 1]  -> 1 / (1 + 0.63093 + 0.5)
    """
    assert ndcg_at_k([1.0], k=1, total_relevant=1) == pytest.approx(1.0)

    ideal = 1.0 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k([1.0], k=3, total_relevant=3) == pytest.approx(1.0 / ideal)


def test_graded_relevance_is_live_code() -> None:
    """The golden set is binary, so this synthetic case keeps the graded path from rotting.

    gains [3, 1, 2] at k=3:
        DCG   = 3/log2(2) + 1/log2(3) + 2/log2(4) = 3 + 0.6309298 + 1.0 = 4.6309298
        ideal = [3, 2, 1] -> 3 + 2/log2(3) + 1/log2(4) = 3 + 1.2618595 + 0.5 = 4.7618595
    """
    expected_dcg = 3.0 + 1 / math.log2(3) + 2 / math.log2(4)
    expected_ideal = 3.0 + 2 / math.log2(3) + 1 / math.log2(4)

    assert dcg_at_k([3.0, 1.0, 2.0], k=3) == pytest.approx(expected_dcg)
    assert ndcg_at_k([3.0, 1.0, 2.0], k=3) == pytest.approx(expected_dcg / expected_ideal)
    assert ndcg_at_k([3.0, 1.0, 2.0], k=3) == pytest.approx(0.9725045, abs=1e-6)


# --- answer metrics ---------------------------------------------------------------------


def test_refusal_accuracy_by_hand() -> None:
    """Four negative questions, three refused: 3/4 = 0.75."""
    assert refusal_accuracy([True, True, False, True]) == pytest.approx(0.75)


def test_refusal_accuracy_of_nothing() -> None:
    assert refusal_accuracy([]) == 0.0


def test_citation_precision_by_hand() -> None:
    """Three citations, two pointing at relevant chunks: 2/3."""
    assert citation_precision([True, False, True]) == pytest.approx(2 / 3)


def test_an_uncited_answer_scores_zero_not_one() -> None:
    """Citing nothing is an unsupported answer, not perfect precision."""
    assert citation_precision([]) == 0.0


def test_cosine_similarity_by_hand() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    # [1,1] against [1,0]: dot 1, norms sqrt(2) and 1, so 1/sqrt(2).
    assert cosine_similarity([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1 / math.sqrt(2))


def test_cosine_handles_degenerate_inputs() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], [1.0, 0.0]) == 0.0


def test_answer_similarity_is_cosine() -> None:
    assert answer_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)
