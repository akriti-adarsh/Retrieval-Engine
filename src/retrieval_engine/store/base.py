"""Vector store protocol, shared across the pgvector and in-memory backends.

The unit of write is a whole document (``upsert_document``) rather than a chunk. That is
what makes re-ingestion safe: replacing a document's chunk set in one call cannot leave
orphaned chunks from a previous run, and it gives the store a natural place to record the
document's content hash for change detection.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from retrieval_engine.embed.base import EmbedderInfo
from retrieval_engine.errors import EmbeddingSpaceMismatchError
from retrieval_engine.models import (
    Base,
    Chunk,
    CollectionInfo,
    Document,
    DocumentPage,
    EmbeddedChunk,
)


class SearchHit(Base):
    """One chunk returned by a similarity search, with its raw store-level score."""

    chunk: Chunk
    score: float


def chunk_fingerprint(chunk_ids: Iterable[str]) -> str:
    """SHA256 over the sorted chunk-id list.

    Used to detect that a persisted BM25 index no longer matches the chunk set. Sorting
    makes the fingerprint independent of insertion order, so a re-ingest that produces
    the same chunks does not trigger a rebuild.
    """
    digest = hashlib.sha256()
    for chunk_id in sorted(chunk_ids):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def check_embedding_space(existing: CollectionInfo, incoming: EmbedderInfo) -> None:
    """Raise unless ``incoming`` vectors belong in the ``existing`` collection.

    A mixed embedding space is invisible until recall craters, so this is a hard error
    rather than a warning. Implemented once here and called by every backend.
    """
    if existing.embedder != incoming.name:
        msg = (
            f"collection {existing.name!r} was built with embedder {existing.embedder!r}, "
            f"refusing vectors from {incoming.name!r}"
        )
        raise EmbeddingSpaceMismatchError(msg)
    if existing.dimension != incoming.dimension:
        msg = (
            f"collection {existing.name!r} holds {existing.dimension}-dim vectors, "
            f"refusing {incoming.dimension}-dim vectors"
        )
        raise EmbeddingSpaceMismatchError(msg)


def check_record_dimensions(records: Sequence[EmbeddedChunk], dimension: int) -> None:
    """Raise if any record's vector width disagrees with the collection."""
    for record in records:
        if len(record.embedding) != dimension:
            msg = (
                f"chunk {record.chunk.chunk_id} has a {len(record.embedding)}-dim vector, "
                f"collection expects {dimension}"
            )
            raise EmbeddingSpaceMismatchError(msg)


@runtime_checkable
class VectorStore(Protocol):
    """Persistence and similarity search over embedded chunks."""

    async def ensure_collection(self, embedder: EmbedderInfo) -> CollectionInfo:
        """Create the collection if absent, else verify the embedding space matches.

        Raises:
            EmbeddingSpaceMismatchError: the collection exists with a different
                embedder name or dimension.
        """
        ...

    async def collection_info(self) -> CollectionInfo | None:
        """Current collection metadata, or ``None`` when it does not exist yet."""
        ...

    async def document_hashes(self) -> Mapping[str, str]:
        """``doc_id -> content_hash`` for every stored document.

        The ingestion pipeline diffs this against the corpus to decide what changed.
        """
        ...

    async def upsert_document(self, document: Document, records: Sequence[EmbeddedChunk]) -> int:
        """Replace ``document``'s chunks with ``records`` and record its content hash.

        Returns the number of chunks written.
        """
        ...

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        *,
        filters: Mapping[str, str] | None = None,
        ef_search: int | None = None,
    ) -> list[SearchHit]:
        """Cosine-similarity search, best first.

        Args:
            embedding: The query vector.
            top_k: Maximum hits to return.
            filters: Exact-match metadata filters, ANDed together.
            ef_search: Per-query HNSW search effort. Backends that have no such knob
                ignore it.
        """
        ...

    async def all_chunks(self) -> list[Chunk]:
        """Every stored chunk. The lexical retriever builds its BM25 index from this."""
        ...

    async def list_documents(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        filters: Mapping[str, str] | None = None,
    ) -> DocumentPage:
        """Paginated document listing for the admin endpoint."""
        ...

    async def delete_document(self, doc_id: str) -> int:
        """Delete a document and its chunks, returning the number of chunks removed."""
        ...

    async def count_chunks(self) -> int:
        """Total stored chunks."""
        ...

    async def health(self) -> bool:
        """Whether the backend is reachable and usable right now."""
        ...

    async def close(self) -> None:
        """Release connections. Safe to call more than once."""
        ...


__all__ = [
    "SearchHit",
    "VectorStore",
    "check_embedding_space",
    "check_record_dimensions",
    "chunk_fingerprint",
]
