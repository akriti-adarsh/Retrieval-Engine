# ADR 002: Postgres with pgvector over a dedicated vector database

Status: accepted.

## Context

The service needs to store, for every chunk: the text, its provenance (document, character
offsets, section path, page), arbitrary metadata used for filtering, and a 384-dimensional
embedding. It needs similarity search over the embeddings, exact-match filtering over the
metadata, and document-level operations (list, delete a document and all its chunks).

The obvious options are a purpose-built vector database (Qdrant, Weaviate, Milvus, Pinecone)
or a relational database with a vector extension.

## Decision

Use Postgres 16 with the `pgvector` extension as the primary store, behind a `VectorStore`
protocol that also has an in-memory numpy implementation for tests.

### Why one datastore rather than two

A dedicated vector database stores vectors well and document rows badly, so a system using
one almost always ends up with two datastores: the vector index, and something relational for
the documents and metadata. That means two systems to run, two backup and restore procedures,
two failure modes, and no transaction spanning them. The last point is the sharp one: with
two stores there is no way to delete a document and its vectors atomically, so every crash
between the two writes leaves orphans, and something has to reconcile them later.

With pgvector, `DELETE FROM documents` cascades to chunks inside one transaction. Orphaned
chunks are structurally impossible rather than merely unlikely. Re-ingesting a document is
one transaction that deletes its old chunks and inserts the new ones, so a failure mid-write
leaves the previous version intact rather than a half-replaced document.

At this corpus size the performance argument for a dedicated index does not bite, and the
operational argument against running two systems does.

### Two specifics that decide whether this works or only appears to

**Operator class.** The index is created `USING hnsw (embedding vector_cosine_ops)` and the
queries use the matching cosine distance operator `<=>`. An index built for a different
operator class is silently ignored by the planner, which presents as "pgvector is slow"
rather than as a mistake, and is the single most common way a pgvector deployment ends up
doing sequential scans.

**Per-query search effort.** `hnsw.ef_search` trades recall against latency at query time. It
is delivered with `SET LOCAL hnsw.ef_search = <n>` inside the same transaction as the search,
because `SET LOCAL` is scoped to the transaction and a configuration value that never becomes
a `SET LOCAL` does nothing at all. This is verified against a live database rather than
assumed.

Distance is converted to similarity as `1 - distance` so that scores are directly comparable
with the in-memory store's cosine similarity, which is what lets the two backends be compared.

### Schema and migrations

Three tables: `collections` records the embedding space (embedder name and dimension) so a
config change cannot silently mix two of them, `documents` holds the content hash that change
detection reads, and `chunks` holds the vectors with `ON DELETE CASCADE` from documents.

Migrations are ordered SQL files applied two ways: `docker-entrypoint-initdb.d` for a fresh
container, and an idempotent `scripts/migrate.py` that records applied filenames in a
`_migrations` table for everything else. No Alembic: there is one schema, and Alembic would be
ceremony without payoff, where a numbered-files scheme is something a reader can verify at a
glance.

## Consequences

**A fixed vector width in the schema.** An HNSW index cannot be built on an unconstrained
`vector` column, so the column is `vector(384)`, matching the default embedder. Switching to a
model with a different width needs a new migration, and the store refuses mismatched vectors
at runtime rather than letting them corrupt the index.

**A single-node ceiling.** HNSW build time and memory grow with the corpus. Somewhere around
the low tens of millions of chunks a single Postgres becomes the constraint, and the answer is
sharding by collection or moving to a dedicated index. That trades away the operational
simplicity this decision was made for, so it is worth doing only when the corpus demands it.

**No incremental BM25 rebuild.** The lexical index is separate from Postgres and rebuilds when
its fingerprint changes. At this size that is a sub-second cost. At ten million chunks it is
not, and the index would need to live outside the process with incremental posting-list
updates.

**Connection handling has sharp edges.** Setting `row_factory` on a pooled connection leaks
that setting to whoever borrows the connection next, so it is scoped per cursor. This was
found by running the live test, not by reading the code.

## Alternatives considered

**Qdrant or Weaviate.** Better raw vector performance, richer native filtering, built for
this. Rejected because it adds a second datastore for a corpus this size, and the cross-store
consistency problem is a real cost paid on every write.

**Pinecone or another hosted index.** Removes the operations burden entirely. Rejected because
the project's first constraint is that it runs with zero paid API keys and no external
dependency.

**FAISS in-process.** Fast, no server. Rejected as the primary store because it has no
persistence story, no metadata filtering, and no concurrent writer. It is essentially what the
in-memory store already is, and that is kept deliberately as a test backend rather than
promoted to production.

**IVFFlat instead of HNSW.** Cheaper to build and smaller. Rejected because it needs a
representative training set before it is useful and its recall is more sensitive to
`nprobe` tuning, whereas HNSW is good out of the box and exposes one clear query-time knob.
