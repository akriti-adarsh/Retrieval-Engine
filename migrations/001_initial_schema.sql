-- Initial schema for the retrieval engine.
--
-- Three tables rather than one: collections records the embedding space so a config change
-- cannot silently mix two of them, documents holds the content hash that change detection
-- reads, and chunks holds the vectors. Deleting a document cascades to its chunks, which is
-- what makes re-ingestion unable to leave orphans behind.
--
-- The vector column has a fixed width, because an HNSW index cannot be built on an
-- unconstrained vector type. 384 matches the default embedder (BAAI/bge-small-en-v1.5).
-- Switching to a model with a different width needs a new migration, and the store refuses
-- mismatched vectors at runtime rather than letting them corrupt the index.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS collections (
    name            TEXT PRIMARY KEY,
    embedder        TEXT        NOT NULL,
    dimension       INTEGER     NOT NULL CHECK (dimension > 0),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    collection      TEXT        NOT NULL REFERENCES collections (name) ON DELETE CASCADE,
    source_path     TEXT        NOT NULL,
    title           TEXT        NOT NULL DEFAULT '',
    content_hash    CHAR(64)    NOT NULL,
    media_type      TEXT        NOT NULL DEFAULT '',
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_collection_idx ON documents (collection);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    doc_id          TEXT        NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    collection      TEXT        NOT NULL REFERENCES collections (name) ON DELETE CASCADE,
    text            TEXT        NOT NULL,
    start_char      INTEGER     NOT NULL CHECK (start_char >= 0),
    end_char        INTEGER     NOT NULL CHECK (end_char >= start_char),
    token_count     INTEGER     NOT NULL CHECK (token_count >= 0),
    section_path    TEXT[]      NOT NULL DEFAULT '{}',
    page_number     INTEGER,
    strategy        TEXT        NOT NULL DEFAULT '',
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    embedding       vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_collection_idx ON chunks (collection);
CREATE INDEX IF NOT EXISTS chunks_metadata_idx ON chunks USING gin (metadata);

-- Cosine operator class, matching the cosine distance operator the queries use. An index
-- built for a different operator class is silently ignored by the planner, which looks like
-- "pgvector is slow" rather than like a mistake.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
