"""Shared store guards: the embedding-space check and the BM25 staleness fingerprint.

These helpers live in store/base.py so both backends enforce them identically. They are
tested here rather than once per backend, because a guard that only one store applies is
worse than no guard: it makes the bug backend-dependent.
"""

from __future__ import annotations

import pytest

from retrieval_engine.embed.base import EmbedderInfo
from retrieval_engine.errors import EmbeddingSpaceMismatchError
from retrieval_engine.models import Chunk, CollectionInfo, EmbeddedChunk, make_chunk_id
from retrieval_engine.store.base import (
    check_embedding_space,
    check_record_dimensions,
    chunk_fingerprint,
)


def _collection(embedder: str = "bge-small", dimension: int = 384) -> CollectionInfo:
    return CollectionInfo(name="default", embedder=embedder, dimension=dimension)


def _embedder(name: str = "bge-small", dimension: int = 384) -> EmbedderInfo:
    return EmbedderInfo(name=name, dimension=dimension)


def _record(width: int, start: int = 0) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=Chunk(
            chunk_id=make_chunk_id("doc-1", start),
            doc_id="doc-1",
            text="text",
            start_char=start,
            end_char=start + 4,
            token_count=1,
        ),
        embedding=[0.1] * width,
    )


# --- fingerprint -----------------------------------------------------------------------


def test_fingerprint_ignores_insertion_order() -> None:
    """A re-ingest producing the same chunks in another order must not force a rebuild."""
    assert chunk_fingerprint(["c", "a", "b"]) == chunk_fingerprint(["a", "b", "c"])


def test_fingerprint_changes_when_one_chunk_is_added() -> None:
    """Adding a single chunk has to invalidate a persisted BM25 index."""
    before = chunk_fingerprint(["a", "b"])

    assert chunk_fingerprint(["a", "b", "c"]) != before


def test_fingerprint_is_delimited_so_concatenations_do_not_collide() -> None:
    """Without a delimiter, ids 'ab' + 'c' and 'a' + 'bc' would hash identically."""
    assert chunk_fingerprint(["ab", "c"]) != chunk_fingerprint(["a", "bc"])


def test_fingerprint_of_nothing_is_stable() -> None:
    assert chunk_fingerprint([]) == chunk_fingerprint([])


# --- embedding space guard -------------------------------------------------------------


def test_matching_embedding_space_is_accepted() -> None:
    check_embedding_space(_collection(), _embedder())


def test_different_embedder_name_is_refused() -> None:
    """A config change to another model must not silently mix two embedding spaces."""
    with pytest.raises(EmbeddingSpaceMismatchError, match="refusing vectors from"):
        check_embedding_space(_collection(embedder="bge-small"), _embedder(name="e5-base"))


def test_different_dimension_is_refused() -> None:
    with pytest.raises(EmbeddingSpaceMismatchError, match="refusing 768-dim vectors"):
        check_embedding_space(_collection(dimension=384), _embedder(dimension=768))


def test_mismatch_message_names_both_sides() -> None:
    """The error has to say what the collection holds and what was offered."""
    with pytest.raises(EmbeddingSpaceMismatchError) as caught:
        check_embedding_space(_collection(embedder="bge-small"), _embedder(name="e5-base"))

    assert "bge-small" in caught.value.message
    assert "e5-base" in caught.value.message


# --- record width guard ----------------------------------------------------------------


def test_correct_record_widths_are_accepted() -> None:
    check_record_dimensions([_record(384), _record(384, start=10)], 384)


def test_wrong_record_width_is_refused_and_names_the_chunk() -> None:
    record = _record(128)

    with pytest.raises(EmbeddingSpaceMismatchError) as caught:
        check_record_dimensions([_record(384), record], 384)

    assert record.chunk.chunk_id in caught.value.message
    assert "128-dim" in caught.value.message


def test_empty_record_list_is_accepted() -> None:
    """An ingest that produced no chunks is not a dimension error."""
    check_record_dimensions([], 384)
