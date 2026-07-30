"""Postgres and pgvector backend, the primary store.

Three specifics that are the difference between this working and appearing to work:

* The HNSW index is built ``USING hnsw (embedding vector_cosine_ops)`` and the queries use
  the matching cosine distance operator ``<=>``. An index built for a different operator
  class is silently ignored by the planner, which presents as "pgvector is slow" rather than
  as a mistake.
* Per-query search effort is delivered with ``SET LOCAL hnsw.ef_search`` inside the query's
  own transaction. A config value that never becomes a ``SET LOCAL`` does nothing at all, so
  the parameter is applied in the same transaction as the search or not honoured.
* Distance is converted to similarity as ``1 - distance``, so scores are comparable with the
  in-memory store's cosine similarity and the two backends can be compared directly.

Writing a document is one transaction that deletes its old chunks and inserts the new ones,
so a failed re-ingest cannot leave a document half-replaced.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from retrieval_engine.config import Settings
from retrieval_engine.embed.base import EmbedderInfo
from retrieval_engine.errors import StoreUnavailableError
from retrieval_engine.logging_config import get_logger
from retrieval_engine.models import (
    Chunk,
    ChunkStrategy,
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

logger = get_logger(__name__)

CHUNK_COLUMNS = (
    "chunk_id, doc_id, text, start_char, end_char, token_count, "
    "section_path, page_number, strategy, metadata"
)


def _vector_literal(values: Sequence[float]) -> str:
    """pgvector's text input form, which avoids needing its Python adapter registered."""
    return "[" + ",".join(repr(float(value)) for value in values) + "]"


def _filter_clause(
    filters: Mapping[str, str] | None, column: str = "metadata"
) -> tuple[str, list[Any]]:
    """Build an ANDed JSONB containment clause and its parameters.

    Parameterised rather than interpolated: filter keys and values arrive over HTTP, and a
    string-formatted metadata filter is an injection point. Only the column name is
    interpolated, and it comes from this module rather than from a caller.

    Filters are sorted so the generated SQL is stable, which keeps the statement cache and
    any query log readable.
    """
    if not filters:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    for key, value in sorted(filters.items()):
        clauses.append(f"{column} @> %s::jsonb")
        params.append(json.dumps({key: value}))
    return " AND " + " AND ".join(clauses), params


def _to_chunk(row: Mapping[str, Any]) -> Chunk:
    raw_strategy = row.get("strategy") or ChunkStrategy.RECURSIVE_STRUCTURAL.value
    try:
        strategy = ChunkStrategy(raw_strategy)
    except ValueError:
        strategy = ChunkStrategy.RECURSIVE_STRUCTURAL
    metadata = row.get("metadata") or {}
    return Chunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        text=row["text"],
        start_char=row["start_char"],
        end_char=row["end_char"],
        token_count=row["token_count"],
        section_path=list(row.get("section_path") or []),
        page_number=row.get("page_number"),
        strategy=strategy,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


class PgVectorStore:
    """Vector store backed by Postgres with the pgvector extension."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._collection = settings.collection
        self._pool: Any | None = None
        self._info: CollectionInfo | None = None

    async def _get_pool(self) -> Any:
        """Open the connection pool on first use."""
        if self._pool is None:
            try:
                from psycopg_pool import AsyncConnectionPool
            except ImportError as exc:  # pragma: no cover - psycopg is a hard dependency
                msg = f"psycopg is required for the pgvector store: {exc}"
                raise StoreUnavailableError(msg) from exc
            try:
                pool = AsyncConnectionPool(
                    self._settings.postgres_dsn,
                    min_size=self._settings.pg_pool_min_size,
                    max_size=self._settings.pg_pool_max_size,
                    open=False,
                )
                await pool.open(wait=True, timeout=10.0)
            except Exception as exc:
                msg = f"cannot connect to postgres: {type(exc).__name__}: {exc}"
                raise StoreUnavailableError(msg) from exc
            self._pool = pool
        return self._pool

    async def ensure_collection(self, embedder: EmbedderInfo) -> CollectionInfo:
        from psycopg.rows import dict_row

        pool = await self._get_pool()
        async with pool.connection() as connection:
            connection.row_factory = dict_row
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT name, embedder, dimension, created_at FROM collections WHERE name = %s",
                    (self._collection,),
                )
                row = await cursor.fetchone()
                if row is not None:
                    existing = CollectionInfo(
                        name=row["name"],
                        embedder=row["embedder"],
                        dimension=row["dimension"],
                        created_at=row["created_at"],
                    )
                    check_embedding_space(existing, embedder)
                else:
                    await cursor.execute(
                        "INSERT INTO collections (name, embedder, dimension) VALUES (%s, %s, %s)",
                        (self._collection, embedder.name, embedder.dimension),
                    )
                    existing = CollectionInfo(
                        name=self._collection,
                        embedder=embedder.name,
                        dimension=embedder.dimension,
                    )
                await cursor.execute(
                    "SELECT count(*) AS total FROM chunks WHERE collection = %s",
                    (self._collection,),
                )
                counted = await cursor.fetchone()
        existing.chunk_count = int(counted["total"]) if counted else 0
        self._info = existing
        return existing

    async def collection_info(self) -> CollectionInfo | None:
        from psycopg.rows import dict_row

        pool = await self._get_pool()
        async with pool.connection() as connection:
            connection.row_factory = dict_row
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT c.name, c.embedder, c.dimension, c.created_at, "
                    "(SELECT count(*) FROM chunks WHERE collection = c.name) AS total "
                    "FROM collections c WHERE c.name = %s",
                    (self._collection,),
                )
                row = await cursor.fetchone()
        if row is None:
            return None
        return CollectionInfo(
            name=row["name"],
            embedder=row["embedder"],
            dimension=row["dimension"],
            chunk_count=int(row["total"]),
            created_at=row["created_at"],
        )

    def _require_info(self) -> CollectionInfo:
        if self._info is None:
            msg = "collection has not been created; call ensure_collection first"
            raise StoreUnavailableError(msg)
        return self._info

    async def document_hashes(self) -> Mapping[str, str]:
        pool = await self._get_pool()
        async with pool.connection() as connection, connection.cursor() as cursor:
            await cursor.execute(
                "SELECT doc_id, content_hash FROM documents WHERE collection = %s",
                (self._collection,),
            )
            rows = await cursor.fetchall()
        return {row[0]: row[1] for row in rows}

    async def upsert_document(self, document: Document, records: Sequence[EmbeddedChunk]) -> int:
        info = self._require_info()
        check_record_dimensions(records, info.dimension)

        pool = await self._get_pool()
        # One transaction: a failed re-ingest must not leave a document half-replaced.
        async with (
            pool.connection() as connection,
            connection.transaction(),
            connection.cursor() as cursor,
        ):
            await cursor.execute(
                "INSERT INTO documents "
                "(doc_id, collection, source_path, title, content_hash, media_type, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb) "
                "ON CONFLICT (doc_id) DO UPDATE SET "
                "source_path = EXCLUDED.source_path, title = EXCLUDED.title, "
                "content_hash = EXCLUDED.content_hash, media_type = EXCLUDED.media_type, "
                "metadata = EXCLUDED.metadata, ingested_at = now()",
                (
                    document.doc_id,
                    self._collection,
                    document.source_path,
                    document.title,
                    document.content_hash,
                    document.media_type,
                    json.dumps(document.metadata),
                ),
            )
            await cursor.execute("DELETE FROM chunks WHERE doc_id = %s", (document.doc_id,))
            for record in records:
                chunk = record.chunk
                await cursor.execute(
                    f"INSERT INTO chunks ({CHUNK_COLUMNS}, collection, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::vector)",
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.text,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.token_count,
                        list(chunk.section_path),
                        chunk.page_number,
                        chunk.strategy.value,
                        json.dumps(chunk.metadata),
                        self._collection,
                        _vector_literal(record.embedding),
                    ),
                )
        return len(records)

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        *,
        filters: Mapping[str, str] | None = None,
        ef_search: int | None = None,
    ) -> list[SearchHit]:
        from psycopg.rows import dict_row

        if top_k <= 0:
            return []
        pool = await self._get_pool()
        clause, filter_params = _filter_clause(filters)
        effort = ef_search if ef_search is not None else self._settings.retrieval.hnsw_ef_search
        vector = _vector_literal(embedding)
        query = (
            f"SELECT {CHUNK_COLUMNS}, 1 - (embedding <=> %s::vector) AS similarity "
            f"FROM chunks WHERE collection = %s{clause} "
            # chunk_id is the tie-break, so equal distances cannot reorder between runs.
            "ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s"
        )

        async with pool.connection() as connection:
            connection.row_factory = dict_row
            # SET LOCAL lasts only for the surrounding transaction, which is exactly the
            # scope wanted: per-query effort without touching the server's global setting.
            async with connection.transaction(), connection.cursor() as cursor:
                await cursor.execute(f"SET LOCAL hnsw.ef_search = {int(effort)}")
                await cursor.execute(
                    query,
                    (vector, self._collection, *filter_params, vector, top_k),
                )
                rows = await cursor.fetchall()

        return [SearchHit(chunk=_to_chunk(row), score=float(row["similarity"])) for row in rows]

    async def all_chunks(self) -> list[Chunk]:
        from psycopg.rows import dict_row

        pool = await self._get_pool()
        async with pool.connection() as connection:
            connection.row_factory = dict_row
            async with connection.cursor() as cursor:
                await cursor.execute(
                    f"SELECT {CHUNK_COLUMNS} FROM chunks WHERE collection = %s "
                    "ORDER BY doc_id, start_char",
                    (self._collection,),
                )
                rows = await cursor.fetchall()
        return [_to_chunk(row) for row in rows]

    async def list_documents(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        filters: Mapping[str, str] | None = None,
    ) -> DocumentPage:
        from psycopg.rows import dict_row

        pool = await self._get_pool()
        count_clause, filter_params = _filter_clause(filters)
        page_clause, _ = _filter_clause(filters, column="d.metadata")
        async with pool.connection() as connection:
            connection.row_factory = dict_row
            async with connection.cursor() as cursor:
                await cursor.execute(
                    f"SELECT count(*) AS total FROM documents WHERE collection = %s{count_clause}",
                    (self._collection, *filter_params),
                )
                counted = await cursor.fetchone()
                await cursor.execute(
                    "SELECT d.doc_id, d.source_path, d.title, d.metadata, "
                    "(SELECT count(*) FROM chunks WHERE doc_id = d.doc_id) AS chunk_count "
                    f"FROM documents d WHERE d.collection = %s{page_clause} "
                    "ORDER BY d.doc_id LIMIT %s OFFSET %s",
                    (self._collection, *filter_params, limit, offset),
                )
                rows = await cursor.fetchall()

        items = [
            DocumentInfo(
                doc_id=row["doc_id"],
                source_path=row["source_path"],
                title=row["title"],
                chunk_count=int(row["chunk_count"]),
                metadata=dict(row["metadata"]) if isinstance(row["metadata"], dict) else {},
            )
            for row in rows
        ]
        total = int(counted["total"]) if counted else 0
        return DocumentPage(items=items, total=total, limit=limit, offset=offset)

    async def delete_document(self, doc_id: str) -> int:
        pool = await self._get_pool()
        async with (
            pool.connection() as connection,
            connection.transaction(),
            connection.cursor() as cursor,
        ):
            await cursor.execute(
                "SELECT count(*) FROM chunks WHERE doc_id = %s AND collection = %s",
                (doc_id, self._collection),
            )
            row = await cursor.fetchone()
            removed = int(row[0]) if row else 0
            # Chunks go with the document by ON DELETE CASCADE.
            await cursor.execute(
                "DELETE FROM documents WHERE doc_id = %s AND collection = %s",
                (doc_id, self._collection),
            )
        return removed

    async def count_chunks(self) -> int:
        pool = await self._get_pool()
        async with pool.connection() as connection, connection.cursor() as cursor:
            await cursor.execute(
                "SELECT count(*) FROM chunks WHERE collection = %s", (self._collection,)
            )
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def health(self) -> bool:
        """Whether the database answers. Never raises, by contract."""
        try:
            pool = await self._get_pool()
            async with pool.connection() as connection, connection.cursor() as cursor:
                await cursor.execute("SELECT 1")
                await cursor.fetchone()
        except Exception:
            return False
        return True

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


def metadata_as_json(metadata: Metadata) -> str:
    """Serialise metadata for a JSONB column."""
    return json.dumps(metadata)


__all__ = ["PgVectorStore", "metadata_as_json"]
