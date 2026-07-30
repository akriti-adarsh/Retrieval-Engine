"""Dense candidate generation: embed the query, search the vector store.

Thin by design. The interesting decisions live in the embedder (the query instruction
prefix) and the store (the HNSW index and its per-query search effort), so this module's
whole job is to run them in the right order and record what each candidate scored, which is
what the fusion stage and the debug view both need.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from retrieval_engine.embed.base import Embedder
from retrieval_engine.models import ScoredChunk, StageScores
from retrieval_engine.store.base import VectorStore


class DenseRetriever:
    """Cosine similarity over stored chunk embeddings."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    async def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        filters: Mapping[str, str] | None = None,
        ef_search: int | None = None,
    ) -> list[ScoredChunk]:
        """Embed ``query`` (with the query-side instruction) and search."""
        embedding = await self._embedder.embed_query(query)
        return await self.retrieve_vector(embedding, top_k, filters=filters, ef_search=ef_search)

    async def retrieve_vector(
        self,
        embedding: Sequence[float],
        top_k: int,
        *,
        filters: Mapping[str, str] | None = None,
        ef_search: int | None = None,
    ) -> list[ScoredChunk]:
        """Search with an embedding that is already computed.

        Multi-query and HyDE expansion embed several strings and search with each, so this
        entry point exists to avoid re-embedding text the caller already has vectors for.
        """
        hits = await self._store.search(embedding, top_k, filters=filters, ef_search=ef_search)
        return [
            ScoredChunk(
                chunk=hit.chunk,
                score=hit.score,
                stages=StageScores(dense_score=hit.score, dense_rank=rank),
            )
            for rank, hit in enumerate(hits, start=1)
        ]


__all__ = ["DenseRetriever"]
