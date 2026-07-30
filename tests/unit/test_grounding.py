"""Grounding verification and refusal.

Both of these are the difference between a demo and something trustworthy, so the tests
target the cases that matter: one invented sentence among supported ones, a citation pointing
at a source that never existed, and a fused score being mistaken for a confidence.
"""

from __future__ import annotations

from typing import Any

import pytest

from retrieval_engine.config import Settings
from retrieval_engine.guardrails.grounding import (
    GroundingVerifier,
    resolve_citations,
    strip_citations,
)
from retrieval_engine.guardrails.refusal import RefusalPolicy
from retrieval_engine.models import (
    Chunk,
    RetrievalConfig,
    RetrievalResult,
    ScoredChunk,
    SourceRef,
    StageScores,
)
from tests.conftest import FAKE_DIMENSION, FakeEmbedder

HNSW_TEXT = (
    "A hierarchical navigable small world index is a layered proximity graph. "
    "Search descends from the sparse top layer toward the dense bottom layer."
)
BM25_TEXT = (
    "BM25 sums inverse document frequency weights over the query terms a document contains. "
    "A saturation function stops repeated terms from dominating the score."
)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "_env_file": None,
        "env": "test",
        "embedding_dimension": FAKE_DIMENSION,
    }
    base.update(overrides)
    return Settings(**base)


def _source(index: int, text: str) -> SourceRef:
    return SourceRef(
        index=index,
        chunk_id=f"chunk-{index}",
        doc_id=f"doc-{index}",
        title=f"Doc {index}",
        source_path=f"data/corpus/doc-{index}.md",
        text=text,
        score=0.9,
    )


def _verifier(**overrides: Any) -> GroundingVerifier:
    return GroundingVerifier(_settings(**overrides), FakeEmbedder(dimension=FAKE_DIMENSION))


def _sources() -> list[SourceRef]:
    return [_source(1, HNSW_TEXT), _source(2, BM25_TEXT)]


# --- citation handling ------------------------------------------------------------------


def test_strip_citations_removes_markers() -> None:
    """Markers are punctuation to a reader but tokens to an embedder."""
    assert strip_citations("A claim [1] and another [12].") == "A claim and another ."


def test_citations_resolve_to_their_sources() -> None:
    citations = resolve_citations("First [1]. Second [2].", _sources())

    assert [citation.marker for citation in citations] == [1, 2]
    assert all(citation.resolved for citation in citations)
    assert citations[0].chunk_id == "chunk-1"
    assert citations[1].doc_id == "doc-2"


def test_an_unknown_marker_is_reported_not_dropped() -> None:
    """A model citing a source it was never given is exactly what this must surface."""
    citations = resolve_citations("A claim [7].", _sources())

    assert len(citations) == 1
    assert citations[0].marker == 7
    assert citations[0].resolved is False
    assert citations[0].chunk_id is None


def test_repeated_markers_are_reported_once() -> None:
    citations = resolve_citations("One [1]. Two [1]. Three [1].", _sources())

    assert len(citations) == 1


def test_an_answer_with_no_markers_has_no_citations() -> None:
    assert resolve_citations("An uncited claim.", _sources()) == []


# --- grounding --------------------------------------------------------------------------


async def test_a_quoted_answer_is_grounded() -> None:
    verifier = _verifier()
    answer = "Search descends from the sparse top layer toward the dense bottom layer [1]."

    report = await verifier.verify(answer, _sources())

    assert report.grounded is True
    assert report.flagged_sentences == []
    assert report.sentences[0].grounded is True
    assert report.sentences[0].supporting_chunk_id == "chunk-1"
    assert report.sentences[0].max_similarity >= report.threshold


async def test_one_invented_sentence_among_supported_ones_is_flagged() -> None:
    """The failure mode this exists for: a whole-answer score would average it away."""
    verifier = _verifier()
    answer = (
        "Search descends from the sparse top layer toward the dense bottom layer [1]. "
        "The authors report a fourteen percent revenue increase for the Barcelona office."
    )

    report = await verifier.verify(answer, _sources())

    assert report.grounded is False
    assert len(report.sentences) == 2
    assert report.sentences[0].grounded is True
    assert report.sentences[1].grounded is False
    assert len(report.flagged_sentences) == 1
    assert "Barcelona" in report.flagged_sentences[0]


async def test_an_unresolved_citation_makes_the_answer_ungrounded() -> None:
    verifier = _verifier()
    answer = "Search descends from the sparse top layer toward the dense bottom layer [9]."

    report = await verifier.verify(answer, _sources())

    assert report.unresolved_citations == [9]
    assert report.grounded is False


async def test_the_threshold_comes_from_settings() -> None:
    lenient = _verifier(grounding_threshold=0.0)
    strict = _verifier(grounding_threshold=0.99)
    answer = "BM25 sums inverse document frequency weights over the query terms [2]."

    assert (await lenient.verify(answer, _sources())).grounded is True
    strict_report = await strict.verify(answer, _sources())
    assert strict_report.grounded is False
    assert strict_report.threshold == 0.99


async def test_an_insufficiency_statement_is_not_a_hallucination() -> None:
    """Scoring an honest refusal as ungrounded would be exactly backwards."""
    verifier = _verifier()

    report = await verifier.verify(
        "I don't have enough information in the provided sources.", _sources()
    )

    assert report.grounded is True
    assert report.sentences == []


async def test_an_answer_with_no_sources_cannot_be_grounded() -> None:
    verifier = _verifier()

    report = await verifier.verify("Some confident claim about nothing.", [])

    assert report.grounded is False
    assert report.flagged_sentences


async def test_an_empty_answer_is_not_grounded() -> None:
    assert (await _verifier().verify("", _sources())).grounded is False


async def test_mean_similarity_summarises_the_report() -> None:
    report = await _verifier().verify(
        "A saturation function stops repeated terms from dominating the score [2].",
        _sources(),
    )

    assert 0.0 < report.mean_similarity <= 1.0


async def test_grounding_is_deterministic() -> None:
    verifier = _verifier()
    answer = "Search descends from the sparse top layer toward the dense bottom layer [1]."

    first = await verifier.verify(answer, _sources())
    second = await verifier.verify(answer, _sources())

    assert [s.max_similarity for s in first.sentences] == [
        s.max_similarity for s in second.sentences
    ]


# --- refusal ----------------------------------------------------------------------------


def _result(*scores: float | None, reranked: bool = True) -> RetrievalResult:
    chunks = []
    for index, score in enumerate(scores, start=1):
        stages = (
            StageScores(rerank_score=score, final_rank=index, fused_rank=index)
            if reranked
            else StageScores(fusion_score=score, fused_rank=index, final_rank=index)
        )
        chunks.append(
            ScoredChunk(
                chunk=Chunk(
                    chunk_id=f"c{index}",
                    doc_id="doc-a",
                    text="some retrieved text",
                    start_char=0,
                    end_char=19,
                    token_count=3,
                ),
                score=score if score is not None else 0.0,
                stages=stages,
            )
        )
    return RetrievalResult(query="q", chunks=chunks, config=RetrievalConfig())


def test_strong_evidence_is_answered() -> None:
    policy = RefusalPolicy(_settings())

    decision = policy.decide(_result(0.95, 0.80))

    assert decision.refused is False
    assert decision.confidence == pytest.approx(0.95)
    assert decision.usable_sources == 2
    assert decision.reason == ""


def test_a_low_top_score_is_refused() -> None:
    policy = RefusalPolicy(_settings(min_confidence=0.3))

    decision = policy.decide(_result(0.05, 0.01))

    assert decision.refused is True
    assert "below the 0.30 confidence threshold" in decision.reason


def test_too_few_sources_clearing_the_bar_is_refused() -> None:
    policy = RefusalPolicy(_settings(min_confidence=0.3, min_sources=2))

    decision = policy.decide(_result(0.9, 0.1, 0.05))

    assert decision.refused is True
    assert decision.usable_sources == 1
    assert "2 required" in decision.reason


def test_nothing_retrieved_is_refused() -> None:
    decision = RefusalPolicy(_settings()).decide(_result())

    assert decision.refused is True
    assert decision.usable_sources == 0
    assert "nothing was retrieved" in decision.reason


def test_a_fused_score_is_never_treated_as_a_confidence() -> None:
    """An RRF score is around 0.03 by construction. Thresholding it at 0.3 would refuse
    every query in the no-rerank ablation rows and silently turn them into zeros.
    """
    policy = RefusalPolicy(_settings(min_confidence=0.3))

    decision = policy.decide(_result(0.032, 0.016, reranked=False))

    assert decision.refused is False
    assert decision.threshold_applied is False


def test_without_reranking_the_count_rule_still_applies() -> None:
    policy = RefusalPolicy(_settings(min_sources=3))

    decision = policy.decide(_result(0.032, 0.016, reranked=False))

    assert decision.refused is True
    assert decision.threshold_applied is False
    assert "2 source(s) retrieved" in decision.reason


def test_min_sources_zero_never_refuses_on_count() -> None:
    policy = RefusalPolicy(_settings(min_confidence=0.0, min_sources=0))

    assert policy.decide(_result(0.0)).refused is False


def test_the_explanation_reads_as_an_answer() -> None:
    policy = RefusalPolicy(_settings())
    decision = policy.decide(_result(0.01))

    explanation = policy.explanation(decision)

    assert explanation.startswith("I don't have enough information")
    assert decision.reason in explanation
    assert explanation.endswith(".")


def test_the_explanation_without_a_reason_is_still_a_sentence() -> None:
    policy = RefusalPolicy(_settings())

    explanation = policy.explanation(policy.decide(_result(0.99)))

    assert explanation.endswith(".")
    assert "()" not in explanation
