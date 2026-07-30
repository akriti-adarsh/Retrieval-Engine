"""The pgvector backend's pure logic, plus a docker-marked live round trip.

The SQL-building and row-mapping functions are tested here without a database, because they
are where the real mistakes live: a filter clause built by string formatting is an injection
hole, and an operator-class mismatch in the index is invisible until someone profiles a
query. The live round trip is marked ``docker`` and excluded from the default run, so the
suite stays database-free as the spec requires.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from retrieval_engine.config import Settings
from retrieval_engine.errors import StoreUnavailableError
from retrieval_engine.models import ChunkStrategy, StoreKind
from retrieval_engine.store.base import VectorStore
from retrieval_engine.store.pgvector import (
    PgVectorStore,
    _filter_clause,
    _to_chunk,
    _vector_literal,
)

#: Deliberately NOT prefixed RE_: conftest strips every RE_ variable to isolate tests from the
#: developer environment, so an RE_POSTGRES_DSN override would be swallowed before it arrived.
#: Set this when the compose Postgres is remapped, which docker-compose.override.yml does when
#: 5432 is already taken on the host.
DSN_ENV = "TEST_POSTGRES_DSN"
DEFAULT_DSN = "postgresql://retrieval:retrieval@127.0.0.1:5432/retrieval"


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "_env_file": None,
        "env": "test",
        "store": StoreKind.PGVECTOR,
        "postgres_dsn": os.environ.get(DSN_ENV, DEFAULT_DSN),
    }
    base.update(overrides)
    return Settings(**base)


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "chunk_id": "c1",
        "doc_id": "doc-a",
        "text": "some chunk text",
        "start_char": 0,
        "end_char": 15,
        "token_count": 3,
        "section_path": ["Methods", "Retrieval"],
        "page_number": 2,
        "strategy": "fixed_token",
        "metadata": {"title": "A Paper"},
    }
    row.update(overrides)
    return row


# --- protocol ---------------------------------------------------------------------------


def test_store_satisfies_the_protocol() -> None:
    assert isinstance(PgVectorStore(_settings()), VectorStore)


def test_upsert_before_ensure_collection_is_refused() -> None:
    """Without a collection there is no dimension to validate against."""
    with pytest.raises(StoreUnavailableError, match="ensure_collection"):
        PgVectorStore(_settings())._require_info()


# --- vector literals --------------------------------------------------------------------


def test_vector_literal_is_pgvector_text_form() -> None:
    assert _vector_literal([1.0, -0.5, 0.25]) == "[1.0,-0.5,0.25]"


def test_vector_literal_coerces_integers() -> None:
    assert _vector_literal([1, 0]) == "[1.0,0.0]"


def test_vector_literal_of_nothing() -> None:
    assert _vector_literal([]) == "[]"


# --- filter clauses ---------------------------------------------------------------------


def test_no_filters_produces_no_clause() -> None:
    assert _filter_clause(None) == ("", [])
    assert _filter_clause({}) == ("", [])


def test_filters_are_parameterised_not_interpolated() -> None:
    """Filter values arrive over HTTP, so they must never reach the SQL string."""
    clause, params = _filter_clause({"title": "'; DROP TABLE chunks; --"})

    assert clause == " AND metadata @> %s::jsonb"
    assert "DROP TABLE" not in clause
    assert json.loads(params[0]) == {"title": "'; DROP TABLE chunks; --"}


def test_multiple_filters_are_anded_in_sorted_order() -> None:
    clause, params = _filter_clause({"year": "2024", "title": "A Paper"})

    assert clause == " AND metadata @> %s::jsonb AND metadata @> %s::jsonb"
    assert [json.loads(param) for param in params] == [{"title": "A Paper"}, {"year": "2024"}]


def test_filter_column_can_be_qualified() -> None:
    """The paginated listing joins documents as d, so its clause needs the alias."""
    clause, _ = _filter_clause({"title": "x"}, column="d.metadata")

    assert clause == " AND d.metadata @> %s::jsonb"


def test_placeholder_count_matches_parameter_count() -> None:
    clause, params = _filter_clause({"a": "1", "b": "2", "c": "3"})

    assert clause.count("%s") == len(params) == 3


# --- row mapping ------------------------------------------------------------------------


def test_row_maps_to_a_chunk() -> None:
    chunk = _to_chunk(_row())

    assert chunk.chunk_id == "c1"
    assert chunk.section_path == ["Methods", "Retrieval"]
    assert chunk.page_number == 2
    assert chunk.strategy is ChunkStrategy.FIXED_TOKEN
    assert chunk.metadata["title"] == "A Paper"


def test_an_unknown_strategy_falls_back_rather_than_failing() -> None:
    """A row written by a future version must not make an old reader crash."""
    chunk = _to_chunk(_row(strategy="some_future_strategy"))

    assert chunk.strategy is ChunkStrategy.RECURSIVE_STRUCTURAL


def test_null_columns_become_sensible_defaults() -> None:
    chunk = _to_chunk(_row(section_path=None, page_number=None, metadata=None, strategy=None))

    assert chunk.section_path == []
    assert chunk.page_number is None
    assert chunk.metadata == {}


def test_non_dict_metadata_is_discarded() -> None:
    assert _to_chunk(_row(metadata="not a mapping")).metadata == {}


# --- health contract --------------------------------------------------------------------


async def test_health_is_false_when_the_pool_cannot_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A readiness check must report a problem, not become one."""
    store = PgVectorStore(_settings())

    async def boom() -> Any:
        msg = "cannot connect to postgres"
        raise StoreUnavailableError(msg)

    monkeypatch.setattr(store, "_get_pool", boom)

    assert await store.health() is False


async def test_close_without_a_pool_is_a_no_op() -> None:
    store = PgVectorStore(_settings())

    await store.close()
    await store.close()


# --- live round trip --------------------------------------------------------------------


@pytest.mark.docker
async def test_live_round_trip() -> None:
    """Full round trip against a real pgvector database.

    Excluded from the default run. Bring the stack up and run it explicitly:

        docker compose up -d postgres
        uv run python scripts/migrate.py
        uv run pytest -m docker
    """
    import hashlib

    from retrieval_engine.models import Chunk, Document, EmbeddedChunk, make_chunk_id
    from tests.conftest import FakeEmbedder

    embedder = FakeEmbedder(dimension=384)
    store = PgVectorStore(_settings())
    try:
        info = await store.ensure_collection(embedder.info)
        assert info.dimension == 384

        text = "A layered proximity graph indexes vectors for approximate search."
        document = Document(
            doc_id="live-test-doc",
            source_path="data/corpus/live-test-doc.md",
            text=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            media_type="text/markdown",
            metadata={"title": "Live Test"},
        )
        vector = (await embedder.embed_documents([text]))[0]
        written = await store.upsert_document(
            document,
            [
                EmbeddedChunk(
                    chunk=Chunk(
                        chunk_id=make_chunk_id("live-test-doc", 0),
                        doc_id="live-test-doc",
                        text=text,
                        start_char=0,
                        end_char=len(text),
                        token_count=len(text.split()),
                        metadata={"title": "Live Test"},
                    ),
                    embedding=vector,
                )
            ],
        )
        assert written == 1

        hits = await store.search(await embedder.embed_query("proximity graph"), top_k=3)
        assert hits
        assert hits[0].chunk.doc_id == "live-test-doc"
        # Similarity, not distance: comparable with the in-memory store's scores.
        assert 0.0 <= hits[0].score <= 1.0

        # ef_search must reach the server as a SET LOCAL, not be silently dropped.
        assert await store.search(
            await embedder.embed_query("proximity graph"), top_k=3, ef_search=200
        )

        assert (await store.document_hashes())["live-test-doc"] == document.content_hash
        page = await store.list_documents(limit=10, filters={"title": "Live Test"})
        assert page.total >= 1
        assert await store.health() is True
        assert await store.delete_document("live-test-doc") == 1
    finally:
        await store.close()
