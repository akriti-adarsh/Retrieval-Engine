"""Chunking: exact boundaries by hand, plus the three properties the spec requires.

The property tests are sync functions that drive the async chunker with asyncio.run.
Hypothesis and pytest-asyncio do not compose (Hypothesis would run the coroutine factory
many times inside one event loop), so mixing them is a known way to get tests that pass for
the wrong reason.
"""

from __future__ import annotations

import asyncio
import hashlib
import re

import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from retrieval_engine.config import Settings
from retrieval_engine.errors import ConfigurationError
from retrieval_engine.ingest.chunker import (
    Chunker,
    FixedTokenChunker,
    RecursiveStructuralChunker,
    SemanticChunker,
    build_chunker,
)
from retrieval_engine.models import ChunkStrategy, Document, PageSpan
from tests.conftest import FakeEmbedder, FakeTokenizer

WHITESPACE = re.compile(r"\s")

STRUCTURED = """# Retrieval

Dense retrieval embeds queries and passages independently.

## Fusion

Reciprocal rank fusion needs no score calibration.

### Weights

Weighted fusion needs min-max normalisation first.

## Reranking

A cross-encoder scores each pair jointly.
"""


def _doc(text: str, spans: list[PageSpan] | None = None) -> Document:
    return Document(
        doc_id="doc-1",
        source_path="data/corpus/doc-1.md",
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        media_type="text/markdown",
        metadata={"title": "Fixture"},
        page_spans=spans or [],
    )


def _run(chunker: Chunker, document: Document) -> list:
    return asyncio.run(chunker.chunk(document))


def _tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


# --- protocol and construction ----------------------------------------------------------


def test_all_three_strategies_satisfy_the_protocol() -> None:
    tokenizer = _tokenizer()

    assert isinstance(FixedTokenChunker(tokenizer), Chunker)
    assert isinstance(RecursiveStructuralChunker(tokenizer), Chunker)
    assert isinstance(SemanticChunker(FakeEmbedder(), tokenizer), Chunker)


def test_overlap_at_or_above_size_is_rejected() -> None:
    """A stride of zero would loop forever, so this cannot be allowed to construct."""
    with pytest.raises(ConfigurationError, match="must be smaller than size"):
        FixedTokenChunker(_tokenizer(), size=64, overlap=64)


def test_non_positive_size_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="must be positive"):
        FixedTokenChunker(_tokenizer(), size=0)

    with pytest.raises(ConfigurationError, match="must be positive"):
        RecursiveStructuralChunker(_tokenizer(), target_size=0)


def test_out_of_range_percentile_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="threshold percentile"):
        SemanticChunker(FakeEmbedder(), _tokenizer(), threshold_percentile=120.0)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (ChunkStrategy.FIXED_TOKEN, FixedTokenChunker),
        (ChunkStrategy.RECURSIVE_STRUCTURAL, RecursiveStructuralChunker),
        (ChunkStrategy.SEMANTIC, SemanticChunker),
    ],
)
def test_build_chunker_dispatches(strategy: ChunkStrategy, expected: type) -> None:
    chunker = build_chunker(
        strategy, Settings(_env_file=None), _tokenizer(), embedder=FakeEmbedder()
    )

    assert isinstance(chunker, expected)


def test_semantic_without_an_embedder_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="needs an embedder"):
        build_chunker(ChunkStrategy.SEMANTIC, Settings(_env_file=None), _tokenizer())


def test_build_chunker_reads_sizes_from_settings() -> None:
    settings = Settings(_env_file=None, chunk_size=128, chunk_overlap=16)

    chunker = build_chunker(ChunkStrategy.FIXED_TOKEN, settings, _tokenizer())
    document = _doc(" ".join(f"w{index}" for index in range(300)))

    assert all(chunk.token_count <= 128 for chunk in _run(chunker, document))


# --- fixed token: exact boundaries ------------------------------------------------------


def test_fixed_windows_have_exact_boundaries_and_overlap() -> None:
    """Ten words, size four, overlap one, so stride three: windows 0-4, 3-7, 6-10."""
    text = " ".join(f"w{index}" for index in range(10))
    chunks = _run(FixedTokenChunker(_tokenizer(), size=4, overlap=1), _doc(text))

    assert [chunk.text for chunk in chunks] == [
        "w0 w1 w2 w3",
        "w3 w4 w5 w6",
        "w6 w7 w8 w9",
    ]
    assert [chunk.token_count for chunk in chunks] == [4, 4, 4]
    for chunk in chunks:
        assert text[chunk.start_char : chunk.end_char] == chunk.text
    # The overlap has to actually overlap, or "overlap" is a lie.
    assert chunks[1].start_char < chunks[0].end_char


def test_fixed_tail_is_not_duplicated() -> None:
    """A window that already reaches the end must not be followed by a redundant tail."""
    text = " ".join(f"w{index}" for index in range(7))
    chunks = _run(FixedTokenChunker(_tokenizer(), size=4, overlap=1), _doc(text))

    assert [chunk.text for chunk in chunks] == ["w0 w1 w2 w3", "w3 w4 w5 w6"]


def test_document_shorter_than_one_chunk_yields_one_chunk() -> None:
    chunks = _run(FixedTokenChunker(_tokenizer(), size=512), _doc("three short words"))

    assert len(chunks) == 1
    assert chunks[0].text == "three short words"
    assert chunks[0].start_char == 0


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t\n"])
def test_empty_documents_yield_no_chunks_rather_than_raising(text: str) -> None:
    for chunker in (
        FixedTokenChunker(_tokenizer()),
        RecursiveStructuralChunker(_tokenizer()),
        SemanticChunker(FakeEmbedder(), _tokenizer()),
    ):
        assert _run(chunker, _doc(text)) == []


# --- structural -------------------------------------------------------------------------


def test_structural_builds_a_nested_section_path() -> None:
    chunks = _run(RecursiveStructuralChunker(_tokenizer(), target_size=12), _doc(STRUCTURED))
    paths = [tuple(chunk.section_path) for chunk in chunks]

    assert ("Retrieval",) in paths
    assert ("Retrieval", "Fusion") in paths
    assert ("Retrieval", "Fusion", "Weights") in paths
    assert ("Retrieval", "Reranking") in paths


def test_structural_keeps_a_heading_with_its_text() -> None:
    chunks = _run(RecursiveStructuralChunker(_tokenizer(), target_size=64), _doc(STRUCTURED))
    fusion = next(chunk for chunk in chunks if tuple(chunk.section_path) == ("Retrieval", "Fusion"))

    assert fusion.text.startswith("## Fusion")
    assert "score calibration" in fusion.text


def test_structural_never_merges_across_a_heading() -> None:
    """A chunk spanning two sections could not carry an honest section_path."""
    chunks = _run(RecursiveStructuralChunker(_tokenizer(), target_size=1000), _doc(STRUCTURED))

    assert len(chunks) > 1
    for chunk in chunks:
        headings = re.findall(r"^#{1,6} ", chunk.text, re.MULTILINE)
        assert len(headings) == 1, f"chunk spans multiple sections: {chunk.text!r}"


def test_structural_merges_small_fragments_within_a_section() -> None:
    text = "## Section\n\nOne two three.\n\nFour five six.\n\nSeven eight nine.\n"
    chunks = _run(RecursiveStructuralChunker(_tokenizer(), target_size=64), _doc(text))

    assert len(chunks) == 1
    assert "Seven eight nine" in chunks[0].text


def test_structural_splits_an_oversized_block_on_sentences() -> None:
    text = "# H\n\n" + " ".join(f"Sentence {index} has words." for index in range(20))
    chunks = _run(RecursiveStructuralChunker(_tokenizer(), target_size=12), _doc(text))

    assert len(chunks) > 1
    assert all(chunk.token_count <= 12 for chunk in chunks)


def test_a_single_sentence_over_budget_is_still_split() -> None:
    """The hard cap has to hold even when there is no sentence boundary to use."""
    text = "# H\n\n" + " ".join(f"w{index}" for index in range(100))
    chunks = _run(RecursiveStructuralChunker(_tokenizer(), target_size=10), _doc(text))

    assert len(chunks) > 1
    assert all(chunk.token_count <= 10 for chunk in chunks)


# --- provenance -------------------------------------------------------------------------


def test_page_number_is_resolved_from_page_spans() -> None:
    text = "Alpha words here. Beta words there."
    spans = [
        PageSpan(page_number=1, start_char=0, end_char=18, label="Alpha"),
        PageSpan(page_number=2, start_char=18, end_char=len(text), label="Beta"),
    ]
    chunks = _run(FixedTokenChunker(_tokenizer(), size=3, overlap=0), _doc(text, spans))

    assert chunks[0].page_number == 1
    assert chunks[-1].page_number == 2
    assert chunks[0].section_path == ["Alpha"]


def test_document_metadata_is_inherited() -> None:
    chunks = _run(FixedTokenChunker(_tokenizer()), _doc("some text here"))

    assert chunks[0].metadata["title"] == "Fixture"
    assert chunks[0].doc_id == "doc-1"


def test_strategy_is_recorded_on_every_chunk() -> None:
    document = _doc(STRUCTURED)

    fixed = _run(FixedTokenChunker(_tokenizer(), size=8, overlap=2), document)
    structural = _run(RecursiveStructuralChunker(_tokenizer(), target_size=8), document)

    assert {chunk.strategy for chunk in fixed} == {ChunkStrategy.FIXED_TOKEN}
    assert {chunk.strategy for chunk in structural} == {ChunkStrategy.RECURSIVE_STRUCTURAL}


def test_chunk_ids_and_starts_are_unique_within_a_document() -> None:
    """Two chunks sharing a start_char would collide on their UUID5 id."""
    document = _doc(STRUCTURED)
    for chunker in (
        FixedTokenChunker(_tokenizer(), size=6, overlap=2),
        RecursiveStructuralChunker(_tokenizer(), target_size=10),
        SemanticChunker(FakeEmbedder(), _tokenizer(), threshold_percentile=50.0, max_tokens=10),
    ):
        chunks = _run(chunker, document)
        starts = [chunk.start_char for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        assert len(set(starts)) == len(starts)
        assert len(set(ids)) == len(ids)


# --- semantic ---------------------------------------------------------------------------


def test_semantic_splits_at_a_topic_change() -> None:
    """Two sentences on one topic, then one on another, with a median threshold.

    A high percentile on a very short document yields no boundary by construction, since
    the 95th percentile of two distances is essentially the larger of them. The percentile
    is a knob, so the test turns it down rather than pretending otherwise.
    """
    text = (
        "BM25 scores lexical overlap between query terms and documents. "
        "Lexical overlap scoring uses inverse document frequency weights. "
        "HNSW builds a layered proximity graph for vector search."
    )
    chunker = SemanticChunker(FakeEmbedder(), _tokenizer(), threshold_percentile=50.0)

    chunks = _run(chunker, _doc(text))

    assert len(chunks) == 2
    assert "BM25" in chunks[0].text
    assert "HNSW" in chunks[1].text


def test_semantic_does_not_split_uniform_text() -> None:
    text = "Retrieval quality matters. Retrieval quality matters. Retrieval quality matters."
    chunker = SemanticChunker(FakeEmbedder(), _tokenizer(), threshold_percentile=50.0)

    chunks = _run(chunker, _doc(text))

    assert len(chunks) == 1


def test_semantic_handles_a_single_sentence() -> None:
    chunks = _run(SemanticChunker(FakeEmbedder(), _tokenizer()), _doc("Only one sentence here."))

    assert len(chunks) == 1
    assert chunks[0].text == "Only one sentence here."


def test_semantic_embeds_sentences_in_one_batched_call() -> None:
    """Per-sentence calls dominate ingest wall-clock on a real corpus."""
    embedder = FakeEmbedder()
    text = "One sentence. Two sentence. Three sentence. Four sentence."

    _run(SemanticChunker(embedder, _tokenizer(), threshold_percentile=50.0), _doc(text))

    assert len(embedder.document_calls) == 4
    assert embedder.query_calls == []


def test_semantic_respects_the_token_cap() -> None:
    text = " ".join(f"Sentence {index} contains several words here." for index in range(15))
    chunker = SemanticChunker(FakeEmbedder(), _tokenizer(), threshold_percentile=99.0, max_tokens=8)

    chunks = _run(chunker, _doc(text))

    assert all(chunk.token_count <= 8 for chunk in chunks)


# --- properties required by the spec ----------------------------------------------------

PROSE = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=0,
    max_size=400,
)
HYPOTHESIS = hypothesis_settings(
    deadline=None,
    max_examples=60,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _covered(chunks: list) -> set[int]:
    covered: set[int] = set()
    for chunk in chunks:
        covered.update(range(chunk.start_char, chunk.end_char))
    return covered


def _required(text: str) -> set[int]:
    # Whitespace-only gaps are acceptable: chunk ranges are trimmed to token boundaries, so
    # the separators between chunks belong to no chunk. Every non-whitespace character must
    # still be inside some chunk, or content was silently dropped.
    return {index for index, char in enumerate(text) if not WHITESPACE.match(char)}


@given(text=PROSE, size=st.integers(min_value=1, max_value=32))
@HYPOTHESIS
def test_property_fixed_chunks_cover_the_document(text: str, size: int) -> None:
    document = _doc(text)
    overlap = size // 2
    chunks = _run(FixedTokenChunker(_tokenizer(), size=size, overlap=overlap), document)

    assert _required(text) <= _covered(chunks)


@given(text=PROSE, size=st.integers(min_value=2, max_value=32))
@HYPOTHESIS
def test_property_structural_chunks_cover_the_document(text: str, size: int) -> None:
    document = _doc(text)
    chunks = _run(RecursiveStructuralChunker(_tokenizer(), target_size=size), document)

    assert _required(text) <= _covered(chunks)


@given(text=PROSE, size=st.integers(min_value=1, max_value=32))
@HYPOTHESIS
def test_property_no_chunk_exceeds_the_token_budget(text: str, size: int) -> None:
    document = _doc(text)
    tokenizer = _tokenizer()
    for chunker in (
        FixedTokenChunker(tokenizer, size=size, overlap=size // 2),
        RecursiveStructuralChunker(tokenizer, target_size=max(size, 2)),
    ):
        for chunk in _run(chunker, document):
            assert chunk.token_count <= max(size, 2)


@given(text=PROSE, size=st.integers(min_value=1, max_value=32))
@HYPOTHESIS
def test_property_chunk_ids_are_stable_across_runs(text: str, size: int) -> None:
    """Re-ingesting an unchanged document must reproduce identical ids, or change
    detection and upsert semantics both break."""
    document = _doc(text)
    chunker = FixedTokenChunker(_tokenizer(), size=size, overlap=size // 2)

    first = _run(chunker, document)
    second = _run(chunker, document)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.start_char for chunk in first] == [chunk.start_char for chunk in second]


@given(text=PROSE, size=st.integers(min_value=1, max_value=32))
@HYPOTHESIS
def test_property_chunk_text_matches_its_offsets(text: str, size: int) -> None:
    """Citations resolve by offset, so text and offsets must never disagree."""
    document = _doc(text)
    chunks = _run(FixedTokenChunker(_tokenizer(), size=size, overlap=0), document)

    for chunk in chunks:
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.text.strip() == chunk.text
