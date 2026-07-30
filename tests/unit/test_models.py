"""Schemas: deterministic ids, range guards, and the golden-set contract."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from retrieval_engine.models import (
    MAX_GOLDEN_SPAN_CHARS,
    Chunk,
    ChunkStrategy,
    Difficulty,
    Document,
    ExpansionMode,
    FusionMethod,
    GoldenCategory,
    GoldenEntry,
    GroundingReport,
    IngestSummary,
    PageSpan,
    QueryRequest,
    RetrievalConfig,
    RetrievalResult,
    ScoredChunk,
    SentenceGrounding,
    StageScores,
    make_chunk_id,
)

SHA = hashlib.sha256(b"body").hexdigest()


def _chunk(text: str = "some text", start: int = 0) -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id("doc-1", start),
        doc_id="doc-1",
        text=text,
        start_char=start,
        end_char=start + len(text),
        token_count=len(text.split()),
    )


# --- deterministic ids -----------------------------------------------------------------


def test_chunk_id_is_stable_across_calls() -> None:
    """Re-ingesting an unchanged document must reproduce byte-identical ids."""
    assert make_chunk_id("2401.12345", 512) == make_chunk_id("2401.12345", 512)


def test_chunk_id_varies_with_offset_and_doc() -> None:
    assert make_chunk_id("doc-a", 0) != make_chunk_id("doc-a", 1)
    assert make_chunk_id("doc-a", 0) != make_chunk_id("doc-b", 0)


def test_chunk_id_is_a_uuid5_hex_string() -> None:
    chunk_id = make_chunk_id("doc-a", 0)

    assert len(chunk_id) == 36
    assert chunk_id.count("-") == 4


# --- range and shape guards ------------------------------------------------------------


def test_page_span_rejects_inverted_range() -> None:
    with pytest.raises(ValidationError, match="must not precede"):
        PageSpan(page_number=1, start_char=100, end_char=10)


def test_chunk_rejects_inverted_range() -> None:
    with pytest.raises(ValidationError, match="must not precede"):
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            text="t",
            start_char=50,
            end_char=10,
            token_count=1,
        )


def test_models_forbid_unknown_fields() -> None:
    """extra=forbid catches typo'd field names at the boundary instead of silently."""
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            text="t",
            start_char=0,
            end_char=1,
            token_count=1,
            tokens=1,
        )


def test_document_title_prefers_front_matter() -> None:
    doc = Document(
        doc_id="2401.12345",
        source_path="data/corpus/2401.12345.md",
        text="body",
        content_hash=SHA,
        media_type="text/markdown",
        metadata={"title": "  Dense Passage Retrieval  ", "authors": ["A. Author"]},
    )

    assert doc.title == "Dense Passage Retrieval"


def test_document_title_falls_back_to_filename() -> None:
    doc = Document(
        doc_id="d1",
        source_path="data/corpus/2401.12345.md",
        text="body",
        content_hash=SHA,
        media_type="text/markdown",
    )

    assert doc.title == "2401.12345.md"


def test_document_rejects_short_content_hash() -> None:
    with pytest.raises(ValidationError):
        Document(
            doc_id="d1",
            source_path="p",
            text="body",
            content_hash="deadbeef",
            media_type="text/plain",
        )


# --- retrieval config ------------------------------------------------------------------


def test_retrieval_config_requires_a_retriever() -> None:
    with pytest.raises(ValidationError, match="at least one of use_dense or use_lexical"):
        RetrievalConfig(use_dense=False, use_lexical=False)


def test_retrieval_config_is_frozen() -> None:
    """Configs are recorded verbatim in eval reports, so they must not drift mid-run."""
    config = RetrievalConfig()

    with pytest.raises(ValidationError):
        config.top_k_dense = 10


def test_retrieval_config_defaults_match_the_spec() -> None:
    config = RetrievalConfig()

    assert config.top_k_dense == config.top_k_lexical == 50
    assert config.fusion is FusionMethod.RRF
    assert config.rrf_k == 60
    assert config.rerank_candidates == 20
    assert config.final_top_k == 5
    assert config.expansion is ExpansionMode.NONE


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (RetrievalConfig(), "hybrid (RRF) + rerank"),
        (RetrievalConfig(use_rerank=False), "hybrid (RRF)"),
        (RetrievalConfig(use_lexical=False, use_rerank=False), "dense only"),
        (RetrievalConfig(use_dense=False, use_rerank=False), "lexical only"),
        (
            RetrievalConfig(fusion=FusionMethod.WEIGHTED, use_rerank=False),
            "hybrid (WEIGHTED)",
        ),
        (
            RetrievalConfig(expansion=ExpansionMode.MULTI_QUERY),
            "hybrid (RRF) + rerank + multi_query",
        ),
        (
            RetrievalConfig(chunk_strategy=ChunkStrategy.SEMANTIC),
            "hybrid (RRF) + rerank + semantic chunking",
        ),
    ],
)
def test_retrieval_config_label(config: RetrievalConfig, expected: str) -> None:
    """Labels become ablation table row names, so they must be stable and readable."""
    assert config.label == expected


# --- stage bookkeeping -----------------------------------------------------------------


def test_rank_movement_is_signed_places_gained() -> None:
    assert StageScores(fused_rank=5, final_rank=2).rank_movement == 3
    assert StageScores(fused_rank=1, final_rank=4).rank_movement == -3


def test_rank_movement_is_none_without_reranking() -> None:
    assert StageScores(fused_rank=3).rank_movement is None
    assert StageScores().rank_movement is None


def test_retrieval_result_top_score_handles_empty() -> None:
    assert RetrievalResult(query="q").top_score == 0.0
    assert (
        RetrievalResult(query="q", chunks=[ScoredChunk(chunk=_chunk(), score=0.87)]).top_score
        == 0.87
    )


def test_ingest_summary_renders_the_no_op_case() -> None:
    """The 'nothing changed' wording is asserted on, because the spec requires it."""
    summary = IngestSummary(docs_seen=12, docs_unchanged=12, elapsed_seconds=0.4)

    rendered = summary.render()

    assert "0 changed" in rendered
    assert "12 unchanged" in rendered


def test_grounding_report_mean_similarity() -> None:
    assert GroundingReport().mean_similarity == 0.0

    report = GroundingReport(
        sentences=[
            SentenceGrounding(sentence="a", max_similarity=0.8, grounded=True),
            SentenceGrounding(sentence="b", max_similarity=0.4, grounded=False),
        ],
        grounded=False,
    )

    assert report.mean_similarity == pytest.approx(0.6)


# --- api request shape -----------------------------------------------------------------


def test_query_request_strips_whitespace() -> None:
    assert QueryRequest(query="  what is rrf?  ").query == "what is rrf?"


def test_query_request_rejects_blank_query() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        QueryRequest(query="   ")


# --- golden set contract ---------------------------------------------------------------


def test_negative_entry_must_carry_no_evidence() -> None:
    """A negative question the corpus cannot answer has nothing to point at."""
    entry = GoldenEntry(
        qid="q001",
        question="What is the authors' home address?",
        category=GoldenCategory.NEGATIVE,
        difficulty=Difficulty.EASY,
    )

    assert entry.relevant_chunk_texts == []

    with pytest.raises(ValidationError, match="must have no relevant docs or spans"):
        GoldenEntry(
            qid="q002",
            question="q",
            category=GoldenCategory.NEGATIVE,
            difficulty=Difficulty.EASY,
            relevant_doc_ids=["2401.1"],
        )


def test_answerable_entry_needs_at_least_one_span() -> None:
    with pytest.raises(ValidationError, match="needs at least one span"):
        GoldenEntry(
            qid="q003",
            question="q",
            category=GoldenCategory.FACTUAL,
            difficulty=Difficulty.EASY,
        )


def test_span_length_is_capped() -> None:
    """Spans are matching keys, not documents; the cap keeps the predicate strict."""
    with pytest.raises(ValidationError, match="over the 300-char cap"):
        GoldenEntry(
            qid="q004",
            question="q",
            category=GoldenCategory.FACTUAL,
            difficulty=Difficulty.HARD,
            relevant_doc_ids=["2401.1"],
            relevant_chunk_texts=["x" * (MAX_GOLDEN_SPAN_CHARS + 1)],
        )


def test_blank_span_is_rejected() -> None:
    with pytest.raises(ValidationError, match="blank span"):
        GoldenEntry(
            qid="q005",
            question="q",
            category=GoldenCategory.AMBIGUOUS,
            difficulty=Difficulty.MEDIUM,
            relevant_chunk_texts=["   "],
        )
