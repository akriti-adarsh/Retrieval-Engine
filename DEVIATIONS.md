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

4. **Token counts are approximate under the optional remote embedder.**
   *Spec said:* fixed-token chunking is "token-based via the embedder's own tokenizer, never
   a naive character split" (section 4).
   *Reality is:* that holds exactly for the default local embedder, whose HuggingFace fast
   tokenizer gives real token offsets. The OpenAI embeddings API does not expose its
   tokenizer, and matching it locally would mean adding `tiktoken` for a non-default,
   key-requiring code path.
   *What was done:* `embed/openai.py` ships `ApproximateTokenizer`, a word-and-punctuation
   tokenizer, and its docstring states plainly that chunk sizes are approximate in that
   configuration and that the local embedder is the default because its counts are exact.
   No dependency was added and no approximation is hidden.

5. **The optional OpenAI backend uses httpx directly rather than the openai SDK.**
   *Spec said:* `embed/openai.py  # optional`, without naming a client library.
   *Reality is:* the entire surface used is one POST to `/embeddings`.
   *What was done:* called it with `httpx`, which is already a dependency. This avoids an
   SDK dependency bought for one request, and it lets every failure branch (non-2xx,
   transport error, malformed body, wrong vector count, wrong width) be tested with
   `httpx.MockTransport` instead of a patched client.

6. **`ingest_concurrency` parallelises loading and chunking, but not embedding.**
   *Spec said:* ingestion uses "concurrency via `asyncio.Semaphore`, default 4" (section 4).
   *Reality is:* the HuggingFace fast tokenizer is a Rust object that raises
   `RuntimeError: Already borrowed` when two threads touch it at once. Encoding runs in a
   worker thread (via `asyncio.to_thread`) while the chunker tokenizes on the event loop
   thread, so concurrent ingestion crashed outright on the first real run against
   `bge-small`. The fake embedder could never have caught this.
   *What was done:* the tokenizer wrapper and the encode path share one `threading.Lock`,
   so document loading, parsing, and chunk assembly still overlap, while tokenizing and
   encoding are serialised per embedder. This costs nothing real: a CPU forward pass
   already saturates the cores, so overlapping two would not have been faster. There is a
   regression test that reproduces the concurrent borrow and asserts it cannot happen.

7. **`count_tokens` and `token_spans` deliberately disagree.**
   *Spec said:* chunking is token-based via the embedder's own tokenizer (section 4).
   *Reality is:* markdown horizontal rules and table separators tokenize into pieces whose
   character offsets are empty. Counting only the spans that can be sliced undercounted, and
   on the first real run a 733-token chunk reached a model with a 512-token limit and was
   silently truncated, losing its tail from the embedding.
   *What was done:* `count_tokens` reports the tokenizer's true count (used for budgeting,
   which is the conservative direction) and `token_spans` returns only spans that consume
   characters (used for slicing). A regression test pins the divergence.

8. **sentence-transformers renamed the dimension accessor.**
   *Spec said:* nothing; it predates the rename.
   *Reality is:* sentence-transformers 5.x renamed `get_sentence_embedding_dimension` to
   `get_embedding_dimension` and emits a `FutureWarning` on the old name.
   *What was done:* `reported_dimension` prefers the new name, falls back to the old one so
   an older pin still works, and treats a model reporting neither as "unknown" so the
   configured dimension stays authoritative. Both paths are tested.

9. **Commit 8 (pgvector) deferred until after commit 9 (ingest pipeline).**
   *Spec said:* commit 8 is the pgvector backend, commit 9 is the orchestrated pipeline.
   *Reality is:* the Docker daemon was not running on this machine, so no pgvector claim
   could be verified, and rule 3 forbids calling unverified work done.
   *What was done:* the ingest pipeline landed first, fully verified end to end against the
   real `bge-small` model. pgvector follows once Docker is up. The test suite needs no
   database by design, so nothing else was blocked.

10. **BM25 contributes nothing on a very small index, which shapes how to read the ablation.**
    *Spec said:* lexical retrieval via `rank-bm25`, fused with dense (sections 1 and 5).
    *Reality is:* `rank_bm25` computes IDF as `log(N - df + 0.5) - log(df + 0.5)` with no
    smoothing, so a term appearing in half or more of a small corpus has an IDF of zero or
    below. On a two-chunk index a term in one chunk scores exactly zero.
    *What was done:* nothing was changed, because that is correct BM25 rather than a bug.
    It is documented in the module docstring and pinned by a test, so the "lexical only"
    ablation row is read as a statement about the real corpus size and not mistaken for a
    broken retriever.

11. **Added `src/retrieval_engine/errors.py`, which the section 2 tree does not list.**
   *Spec said:* the file tree in section 2, with `UnsupportedFormatError` raised from
   `ingest/loaders.py` and a typed embedding-space error raised from the store.
   *Reality is:* section 7 requires a single API exception handler producing one error
   shape, which needs every deliberate error to share a base class carrying a stable
   `code` and an HTTP status; scattering those across five modules would force the
   handler into an isinstance chain.
   *What was done:* one `errors.py` holds the hierarchy (`RetrievalEngineError` plus ten
   subclasses). `UnsupportedFormatError` and `EmbeddingSpaceMismatchError` are raised from
   the modules the spec names, they are just defined in one place.
