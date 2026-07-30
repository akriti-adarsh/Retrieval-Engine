"""numpy-backed vector store, so the whole test suite needs no database.

This is a faithful stand-in for the pgvector backend rather than a toy: it enforces the
same embedding-space guard, the same document-replacement semantics, and the same total
ordering. If a test passes here and fails against Postgres, that is a bug in the pgvector
backend, which is only a useful signal because this store does not cut corners.

Search is brute force over a cached matrix. That is exact rather than approximate, which
means the memory store and pgvector can disagree slightly on recall at the same top_k, and
the eval harness records which store produced a run for that reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from retrieval_engine.embed.base import EmbedderInfo
from retrieval_engine.errors import StoreUnavailableError
from retrieval_engine.models import (
    Chunk,
    CollectionInfo,
    Document,
    DocumentInfo,
    DocumentPage,
    EmbeddedChunk,
    Metadata,
)
from retrieval_engine.store.base import (
    SearchHit,
    check_embedding_space,
    check_record_dimensions,
)


@dataclass
class _DocRecord:
    """Document-level facts the store needs without keeping the full text in memory."""

    doc_id: str
    source_path: str
    title: str
    content_hash: str
    metadata: Metadata = field(default_factory=dict)


class MemoryVectorStore:
    """In-process vector store implementing the :class:`VectorStore` protocol."""

    def __init__(self, collection: str = "default") -> None:
        self._collection = collection
        self._info: CollectionInfo | None = None
        self._chunk_ids: list[str] = []
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._doc_chunks: dict[str, list[str]] = {}
        self._documents: dict[str, _DocRecord] = {}
        self._matrix: np.ndarray | None = None
        self._matrix_ids: list[str] = []

    # -- collection -------------------------------------------------------------------

    async def ensure_collection(self, embedder: EmbedderInfo) -> CollectionInfo:
        if self._info is None:
            self._info = CollectionInfo(
                name=self._collection,
                embedder=embedder.name,
                dimension=embedder.dimension,
                chunk_count=len(self._chunks),
            )
            return self._info
        check_embedding_space(self._info, embedder)
        self._info.chunk_count = len(self._chunks)
        return self._info

    async def collection_info(self) -> CollectionInfo | None:
        if self._info is not None:
            self._info.chunk_count = len(self._chunks)
        return self._info

    def _require_collection(self) -> CollectionInfo:
        if self._info is None:
            msg = "collection has not been created; call ensure_collection first"
            raise StoreUnavailableError(msg)
        return self._info

    # -- writes -----------------------------------------------------------------------

    async def document_hashes(self) -> Mapping[str, str]:
        return {doc_id: record.content_hash for doc_id, record in self._documents.items()}

    def _drop_document(self, doc_id: str) -> int:
        removed = self._doc_chunks.pop(doc_id, [])
        for chunk_id in removed:
            self._chunks.pop(chunk_id, None)
            self._vectors.pop(chunk_id, None)
        if removed:
            dropped = set(removed)
            self._chunk_ids = [chunk_id for chunk_id in self._chunk_ids if chunk_id not in dropped]
            self._matrix = None
        self._documents.pop(doc_id, None)
        return len(removed)

    async def upsert_document(self, document: Document, records: Sequence[EmbeddedChunk]) -> int:
        info = self._require_collection()
        check_record_dimensions(records, info.dimension)

        # Replace rather than merge, so a re-ingest cannot leave chunks from a previous
        # run of a different chunking strategy stranded in the index.
        self._drop_document(document.doc_id)

        chunk_ids: list[str] = []
        for record in records:
            chunk_id = record.chunk.chunk_id
            self._chunks[chunk_id] = record.chunk
            self._vectors[chunk_id] = np.asarray(record.embedding, dtype=np.float32)
            self._chunk_ids.append(chunk_id)
            chunk_ids.append(chunk_id)

        self._doc_chunks[document.doc_id] = chunk_ids
        self._documents[document.doc_id] = _DocRecord(
            doc_id=document.doc_id,
            source_path=document.source_path,
            title=document.title,
            content_hash=document.content_hash,
            metadata=dict(document.metadata),
        )
        self._matrix = None
        info.chunk_count = len(self._chunks)
        return len(chunk_ids)

    async def delete_document(self, doc_id: str) -> int:
        """Remove a document and its chunks.

        Returns 0 for an unknown id rather than raising, so the caller decides whether a
        missing document is a 404 or simply nothing to do.
        """
        removed = self._drop_document(doc_id)
        if self._info is not None:
            self._info.chunk_count = len(self._chunks)
        return removed

    # -- reads ------------------------------------------------------------------------

    def _build_matrix(self) -> np.ndarray:
        """Stack and L2-normalise the stored vectors, cached until the next write."""
        if self._matrix is None or self._matrix_ids != self._chunk_ids:
            if not self._chunk_ids:
                self._matrix = np.zeros((0, 0), dtype=np.float32)
            else:
                stacked = np.vstack([self._vectors[cid] for cid in self._chunk_ids])
                norms = np.linalg.norm(stacked, axis=1)
                # A zero vector would divide by zero; leaving it unnormalised keeps its
                # similarity at zero, which is the honest answer for an empty embedding.
                safe = np.where(norms == 0.0, 1.0, norms)
                self._matrix = (stacked / safe[:, None]).astype(np.float32)
            self._matrix_ids = list(self._chunk_ids)
        return self._matrix

    def _metadata_for(self, chunk: Chunk) -> Metadata:
        """Chunk metadata, falling back to the document's for keys it did not inherit."""
        record = self._documents.get(chunk.doc_id)
        if record is None:
            return chunk.metadata
        merged: Metadata = dict(record.metadata)
        merged.update(chunk.metadata)
        return merged

    def _matches(self, chunk: Chunk, filters: Mapping[str, str] | None) -> bool:
        if not filters:
            return True
        metadata = self._metadata_for(chunk)
        for key, wanted in filters.items():
            value = metadata.get(key)
            if isinstance(value, list):
                if wanted not in value:
                    return False
            elif value is None or str(value) != wanted:
                return False
        return True

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        *,
        filters: Mapping[str, str] | None = None,
        ef_search: int | None = None,
    ) -> list[SearchHit]:
        """Exact cosine search, best first.

        ``ef_search`` is accepted and ignored: a brute-force store has no approximate
        search effort to tune. The parameter stays in the signature so callers do not have
        to know which backend they hold.
        """
        del ef_search
        self._require_collection()
        if top_k <= 0 or not self._chunk_ids:
            return []

        query = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            # Every cosine would be zero, so ranking is meaningless. Say so with an empty
            # result rather than returning an arbitrary order.
            return []
        matrix = self._build_matrix()
        if matrix.shape[1] != query.shape[0]:
            msg = (
                f"query has {query.shape[0]} dimensions but the collection holds {matrix.shape[1]}"
            )
            raise StoreUnavailableError(msg)

        scores = matrix @ (query / norm)
        candidates = [
            (float(scores[index]), chunk_id)
            for index, chunk_id in enumerate(self._chunk_ids)
            if self._matches(self._chunks[chunk_id], filters)
        ]
        # Total ordering: chunk_id breaks score ties, so two identical queries return
        # byte-identical results.
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [
            SearchHit(chunk=self._chunks[chunk_id], score=score)
            for score, chunk_id in candidates[:top_k]
        ]

    async def all_chunks(self) -> list[Chunk]:
        return [self._chunks[chunk_id] for chunk_id in self._chunk_ids]

    async def list_documents(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        filters: Mapping[str, str] | None = None,
    ) -> DocumentPage:
        selected = [
            record
            for doc_id, record in sorted(self._documents.items())
            if self._document_matches(record, filters)
        ]
        window = selected[offset : offset + limit] if limit > 0 else []
        items = [
            DocumentInfo(
                doc_id=record.doc_id,
                source_path=record.source_path,
                title=record.title,
                chunk_count=len(self._doc_chunks.get(record.doc_id, [])),
                metadata=record.metadata,
            )
            for record in window
        ]
        return DocumentPage(items=items, total=len(selected), limit=limit, offset=offset)

    @staticmethod
    def _document_matches(record: _DocRecord, filters: Mapping[str, str] | None) -> bool:
        if not filters:
            return True
        for key, wanted in filters.items():
            value = record.metadata.get(key)
            if isinstance(value, list):
                if wanted not in value:
                    return False
            elif value is None or str(value) != wanted:
                return False
        return True

    async def count_chunks(self) -> int:
        return len(self._chunks)

    async def health(self) -> bool:
        """Always reachable. There is no connection to lose."""
        return True

    async def close(self) -> None:
        """Drop the cached matrix. Idempotent, and the stored data survives."""
        self._matrix = None
        self._matrix_ids = []


__all__ = ["MemoryVectorStore"]
