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

2. **arXiv Search API is rate-limiting this network, so the corpus download is unverified.**
   *Spec said:* pull the 300 most recent `cs.CL` papers from the arXiv API respecting the
   three second delay, with `export.arxiv.org` as the designated mirror if the standard
   endpoint throttles hard (section 15).
   *Reality is:* `export.arxiv.org/api/query` returns HTTP 429 "Rate exceeded" on the very
   first request from a cold process, and plain `http://` requests to it read-timeout. The
   OAI-PMH endpoint at `https://export.arxiv.org/oai2` answers normally (`verb=Identify`
   returned 200 with a valid response).
   *What was done:* the delay was NOT reduced, since arXiv blocks aggressive clients by IP.
   The Search-API downloader is written but parked unverified in `.drafts/` instead of
   committed, because it has never completed a real run. Its listing step moves to OAI-PMH,
   which is arXiv's designated interface for bulk metadata harvesting, and gets verified
   end to end before it lands.

3. **Embedder and store protocols landed ahead of their implementations.**
   *Spec said:* commit 6 is `embed/` with tests, commit 7 is `store/base.py` plus
   `memory.py` with tests.
   *Reality is:* four modules were being implemented in parallel and the run was cut short
   by a session usage limit, which left two implementation files written but with no tests,
   no mypy run, and no ruff run against them.
   *What was done:* only the protocol layer landed (`embed/base.py`, `store/base.py`, the
   shared embedding-space and fingerprint guards, and the test fakes), fully tested. The
   unverified implementation drafts sit in `.drafts/`, excluded from git, with a README
   stating what each still needs. They are tested and moved back in, or rewritten, before
   commits 6 and 7 can be claimed.

4. **Added `src/retrieval_engine/errors.py`, which the section 2 tree does not list.**
   *Spec said:* the file tree in section 2, with `UnsupportedFormatError` raised from
   `ingest/loaders.py` and a typed embedding-space error raised from the store.
   *Reality is:* section 7 requires a single API exception handler producing one error
   shape, which needs every deliberate error to share a base class carrying a stable
   `code` and an HTTP status; scattering those across five modules would force the
   handler into an isinstance chain.
   *What was done:* one `errors.py` holds the hierarchy (`RetrievalEngineError` plus ten
   subclasses). `UnsupportedFormatError` and `EmbeddingSpaceMismatchError` are raised from
   the modules the spec names, they are just defined in one place.
