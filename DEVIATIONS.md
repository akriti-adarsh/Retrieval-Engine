# DEVIATIONS

Every place the build differs from `docs/BUILD_SPEC.md`, one line each:
**spec said** / **reality is** / **what was done**.

1. **Commit 2 and commit 3 swapped.**
   *Spec said:* commit 2 is settings and logging, commit 3 is the pydantic schemas.
   *Reality is:* `config.py` imports `EmbedderKind`, `StoreKind`, `ChunkStrategy`, `LLMKind`
   and `RetrievalConfig` from `models.py`, so a config-first commit could not import, let
   alone run its own tests.
   *What was done:* schemas land as commit 2, settings and logging as commit 3. Both
   commits are green independently (commit 2 verified at 62 passed, 99% coverage with the
   later files stashed out of the tree).

2. **Added `src/retrieval_engine/errors.py`, which the section 2 tree does not list.**
   *Spec said:* the file tree in section 2, with `UnsupportedFormatError` raised from
   `ingest/loaders.py` and a typed embedding-space error raised from the store.
   *Reality is:* section 7 requires a single API exception handler producing one error
   shape, which needs every deliberate error to share a base class carrying a stable
   `code` and an HTTP status; scattering those across five modules would force the
   handler into an isinstance chain.
   *What was done:* one `errors.py` holds the hierarchy (`RetrievalEngineError` plus ten
   subclasses). `UnsupportedFormatError` and `EmbeddingSpaceMismatchError` are raised from
   the modules the spec names, they are just defined in one place.
