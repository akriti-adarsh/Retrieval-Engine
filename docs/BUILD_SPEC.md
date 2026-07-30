# Project 1 — `retrieval-engine`

**Lane:** AI / LLM Engineering (flagship)
**Build time with Claude Code:** ~6–10 hours across 2–3 sessions
**Pin on profile:** yes, position 1

---

## PASTE EVERYTHING BELOW INTO CLAUDE CODE

---

You are building a production-grade retrieval-augmented generation service called `retrieval-engine`. Build it completely. This repository is a portfolio centrepiece and will be read by senior engineers, so correctness, tests, and documentation matter as much as features.

**Precedence rule:** the "Review round" sections at the end of this file amend the sections above them; where they conflict, the review sections govern.

### 0. Absolute constraints — read before writing any code

1. **The entire system must run with zero paid API keys.** Default embedding model is local (`sentence-transformers`), default generation model is local via Ollama, and if Ollama is absent the system must fall back to an extractive answer built from retrieved spans. OpenAI/Anthropic support exists behind an env flag only.
2. **No placeholder code.** No `TODO`, no `pass  # implement later`, no function that raises `NotImplementedError`, no mock data standing in for a real code path. If you specify a module, finish it.
3. **Pin every dependency.** Resolve exact versions at build time and commit the lockfile. Do not invent version numbers — run the resolver and commit whatever it produces.
4. **Everything is typed.** Full type hints, `mypy --strict` passes on `src/`.
5. **Determinism.** Every random seed is configurable and defaults to 42. Two runs of the eval harness on the same corpus produce identical numbers.
6. **Commit as you go**, following the commit plan in section 12. Do not make one giant commit.

### 1. Stack

- Python 3.11+, dependency management with `uv` (commit `uv.lock`)
- FastAPI + Uvicorn for the API
- Pydantic v2 for all schemas and settings
- `sentence-transformers` with `BAAI/bge-small-en-v1.5` as the default embedder (384-dim, CPU-friendly, ~130MB)
- Postgres 16 + `pgvector` as the primary vector store, with a `numpy`-backed in-memory store used for tests so the test suite needs no database
- `rank-bm25` for lexical retrieval
- `BAAI/bge-reranker-base` cross-encoder for reranking
- `httpx` for the Ollama client
- pytest + pytest-asyncio + pytest-cov for tests
- ruff (lint + format), mypy (strict) for quality
- Docker + docker-compose for the full stack
- Streamlit for a small demo UI (single file, no React needed here)

### 2. Repository layout

Create exactly this structure. Every file listed must exist with real content.

```
retrieval-engine/
├── README.md
├── LICENSE                      # MIT
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── .dockerignore
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── .github/workflows/ci.yml
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   └── decisions/
│       ├── 001-hybrid-over-dense-only.md
│       ├── 002-pgvector-over-dedicated-db.md
│       └── 003-chunking-strategy.md
├── src/retrieval_engine/
│   ├── __init__.py
│   ├── config.py                # pydantic-settings
│   ├── models.py                # all Pydantic schemas
│   ├── logging_config.py        # structlog JSON logging w/ request ids
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── loaders.py           # pdf, md, txt, html, docx
│   │   ├── chunker.py
│   │   └── pipeline.py
│   ├── embed/
│   │   ├── __init__.py
│   │   ├── base.py              # Embedder protocol
│   │   ├── local.py             # sentence-transformers
│   │   └── openai.py            # optional
│   ├── store/
│   │   ├── __init__.py
│   │   ├── base.py              # VectorStore protocol
│   │   ├── pgvector.py
│   │   └── memory.py
│   ├── retrieve/
│   │   ├── __init__.py
│   │   ├── dense.py
│   │   ├── lexical.py
│   │   ├── fusion.py            # reciprocal rank fusion
│   │   ├── rerank.py
│   │   └── pipeline.py          # the orchestrated retriever
│   ├── generate/
│   │   ├── __init__.py
│   │   ├── base.py              # LLM protocol
│   │   ├── ollama.py
│   │   ├── extractive.py        # zero-dependency fallback
│   │   └── prompts.py
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── grounding.py         # citation verification
│   │   └── refusal.py           # low-confidence abstention
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── routes_query.py
│   │   ├── routes_ingest.py
│   │   ├── routes_admin.py
│   │   └── deps.py
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py           # ndcg@k, recall@k, mrr, hit_rate
│       ├── golden.py            # golden set loader/validator
│       ├── runner.py
│       └── report.py            # markdown + json report writer
├── scripts/
│   ├── download_corpus.py
│   ├── build_golden_set.py
│   ├── run_eval.py
│   └── seed_demo.py
├── ui/streamlit_app.py
├── data/
│   ├── corpus/.gitkeep
│   └── golden/golden_set.jsonl  # committed, ~60 questions
└── tests/
    ├── conftest.py
    ├── unit/
    │   ├── test_chunker.py
    │   ├── test_fusion.py
    │   ├── test_metrics.py
    │   ├── test_grounding.py
    │   ├── test_memory_store.py
    │   ├── test_loaders.py
    │   └── test_config.py
    ├── integration/
    │   ├── test_ingest_pipeline.py
    │   ├── test_retrieval_pipeline.py
    │   └── test_api.py
    └── eval/test_eval_regression.py
```

### 3. Corpus and golden set — this is the part most projects fake, do not fake it

**Corpus:** `scripts/download_corpus.py` downloads a real, license-clear corpus. Use the **CUAD-adjacent public alternative**: pull the 300 most-recent arXiv abstracts + full-text HTML from `cs.CL` via the arXiv API (respect the 3-second rate limit, cache to disk, resume on failure). This gives dense technical prose with real terminology. Save as individual `.md` files under `data/corpus/` with YAML front matter containing `arxiv_id`, `title`, `authors`, `published`, `url`.

The script must:
- Be idempotent (skip already-downloaded documents)
- Handle HTTP errors with exponential backoff, max 5 retries
- Print a progress line per document
- Write `data/corpus/manifest.json` with a SHA256 per file

**Golden set:** `data/golden/golden_set.jsonl` with ~60 entries, committed to the repo. Each line:

```json
{
  "qid": "q001",
  "question": "What metric do the authors use to evaluate retrieval quality?",
  "relevant_doc_ids": ["2401.12345", "2402.09876"],
  "relevant_chunk_texts": ["exact span 1", "exact span 2"],
  "answer": "reference answer used for answer-quality scoring",
  "category": "factual | multi_hop | negative | ambiguous",
  "difficulty": "easy | medium | hard"
}
```

`scripts/build_golden_set.py` generates candidate questions from the corpus using the local LLM, then writes them to a review file. **The committed golden set must be human-verified** — build the tooling, generate the candidates, and write the file, then have `scripts/build_golden_set.py --validate` check that every `relevant_chunk_texts` entry is an exact substring of some document in the corpus. Include at least 8 `negative` entries: questions the corpus genuinely cannot answer, whose correct behaviour is refusal.

### 4. Ingestion pipeline

`ingest/loaders.py` — a `Loader` protocol with `load(path: Path) -> Document`, and implementations for `.pdf` (pypdf), `.md`, `.txt`, `.html` (selectolax), `.docx` (python-docx). Each preserves: source path, page/section number where available, and any front matter as metadata. Unsupported extension raises a typed `UnsupportedFormatError`.

`ingest/chunker.py` — implement **three** strategies behind one interface, selectable by config, because the ADR compares them:

1. `FixedTokenChunker(size=512, overlap=64)` — token-based via the embedder's own tokenizer, never a naive character split
2. `RecursiveStructuralChunker` — splits on markdown headings → paragraphs → sentences, merging fragments below `min_size` up toward `target_size`
3. `SemanticChunker(threshold_percentile=95)` — embeds sentences, splits where cosine distance between consecutive sentences exceeds the percentile threshold

Every chunk carries: `chunk_id` (deterministic UUID5 of `doc_id + start_char`), `doc_id`, `text`, `start_char`, `end_char`, `token_count`, `section_path` (e.g. `["Methods", "Retrieval"]`), and inherited doc metadata. Chunk IDs must be stable across re-ingestion of unchanged documents.

`ingest/pipeline.py` — orchestrates load → chunk → embed → upsert with:
- Content-hash-based change detection so re-ingesting an unchanged corpus is a no-op that logs "0 changed, N unchanged"
- Batched embedding (configurable batch size, default 32) with a progress bar
- Concurrency via `asyncio.Semaphore`, default 4
- A structured summary object returned: docs processed, chunks created, chunks skipped, elapsed seconds, tokens embedded

### 5. Retrieval pipeline — the technical core

`retrieve/pipeline.py` implements a four-stage retriever. Every stage is individually toggleable via `RetrievalConfig` so the eval harness can ablate them.

**Stage 1 — Query processing.** Optional query expansion: generate 3 paraphrases with the LLM and retrieve for each (multi-query retrieval). Also implement HyDE as an alternative expansion mode. Both off by default, both eval-tested.

**Stage 2 — Parallel candidate generation.**
- Dense: cosine similarity over pgvector, `top_k=50` default, with an HNSW index (`m=16`, `ef_construction=64`) created in the migration
- Lexical: BM25 over the same chunk set, `top_k=50`, with the index persisted to disk and rebuilt only when the chunk set changes
- Both run concurrently with `asyncio.gather`

**Stage 3 — Fusion.** `retrieve/fusion.py` implements Reciprocal Rank Fusion: `score(d) = Σ_r 1/(k + rank_r(d))` with `k=60` configurable. Also implement weighted-score fusion (min-max normalise each list, then weighted sum) as an alternative, and make the choice a config enum. Unit-test RRF against a hand-computed example with known expected output — this is a required test.

**Stage 4 — Reranking.** Cross-encoder over the top 20 fused candidates, returning top 5. Batch the cross-encoder calls. Cache reranker scores in an LRU keyed by `(query_hash, chunk_id)`, size 10,000.

The retriever returns `RetrievalResult` with, per chunk: text, score, the score from each individual stage (so you can debug what contributed), and rank movement between stages. Expose this in the API response under `debug` when `?debug=true`.

### 6. Generation and guardrails

`generate/prompts.py` — a single versioned prompt template with the context injected as numbered sources, and an explicit instruction to cite with `[1]`, `[2]` markers and to answer "I don't have enough information in the provided sources" when the context is insufficient. Version the template string (`PROMPT_VERSION = "v1"`) and record the version in every response, so eval results are attributable to a prompt version.

`generate/extractive.py` — the no-LLM fallback. Given the query and top chunks, select the 2–3 highest-scoring sentences (score by embedding similarity to the query) and return them as a cited extract. This path must work with nothing installed beyond the base dependencies, and the API must never 500 because a model server is down.

`guardrails/grounding.py` — after generation, verify grounding:
- Split the answer into sentences
- For each sentence, compute max cosine similarity against the retrieved chunks
- Any sentence below `grounding_threshold` (default 0.55) is flagged
- Return a `GroundingReport` with per-sentence scores and an overall `grounded: bool`
- Validate that every `[n]` citation in the answer refers to a source actually present in the context; unresolvable citations are an error surfaced in the response

`guardrails/refusal.py` — abstain when the top reranker score is below `min_confidence` (default 0.3) or when fewer than `min_sources` (default 1) chunks clear the threshold. Refusal returns a 200 with `answer_type: "refused"` and an explanation, never an exception.

### 7. API

All routes return Pydantic-validated responses. All errors go through a single exception handler producing `{"error": {"code": str, "message": str, "request_id": str}}`.

- `POST /v1/query` — body `{query, top_k?, filters?, mode?, debug?}` → `{answer, answer_type, citations[], sources[], grounding, timings, prompt_version, request_id}`. `timings` breaks down retrieval/rerank/generation milliseconds.
- `POST /v1/query/stream` — same, but SSE-streams the answer tokens then a final metadata event.
- `POST /v1/ingest` — accepts uploaded files or a directory path, runs ingestion, returns the summary object. Async with a job id for large batches; `GET /v1/ingest/{job_id}` returns status.
- `GET /v1/documents` — paginated list with metadata filters.
- `DELETE /v1/documents/{doc_id}` — removes doc and its chunks, returns count deleted.
- `GET /health` — liveness. `GET /health/ready` — checks database and embedder are actually loaded.
- `GET /metrics` — Prometheus format: request count/latency histograms by route, retrieval stage latencies, refusal rate, grounding failure rate, cache hit rate.

Middleware: request-id injection (accept inbound `X-Request-ID` or generate), structured access logging, and a simple in-memory token-bucket rate limiter (configurable, default 60 req/min per IP).

### 8. Evaluation harness — this is what makes the repo credible

`eval/metrics.py` — implement from scratch, do not import a metrics library:
- `recall_at_k`, `precision_at_k`, `hit_rate_at_k`
- `mrr` (mean reciprocal rank)
- `ndcg_at_k` with graded relevance support
- `refusal_accuracy` — on the `negative` category, fraction correctly refused
- `citation_precision` — fraction of citations that point to a genuinely relevant chunk
- `answer_similarity` — embedding cosine between generated and reference answer

Each function has a docstring with the formula, and a unit test with a hand-computed expected value. The nDCG test must include the tie-breaking and the zero-relevant-documents edge case.

`eval/runner.py` — runs the golden set against a given `RetrievalConfig`, with:
- Concurrency control and a progress bar
- Per-query result rows written to `eval_results/{run_id}/rows.jsonl`
- Aggregate metrics, plus breakdowns by `category` and `difficulty`
- Wall-clock and p50/p95 latency per stage

`scripts/run_eval.py --ablate` runs a predefined **ablation matrix** and writes a comparison table:

| config | recall@5 | nDCG@5 | MRR | refusal acc | p95 latency |
|---|---|---|---|---|---|
| dense only | | | | | |
| lexical only | | | | | |
| hybrid (RRF) | | | | | |
| hybrid + rerank | | | | | |
| hybrid + rerank + multi-query | | | | | |
| hybrid + rerank, semantic chunking | | | | | |

Run this for real and paste the actual numbers into `README.md` and `docs/evaluation.md`. **Report the real numbers even if a fancy component loses** — an honest ablation showing multi-query expansion didn't help is far more impressive than an implausible clean sweep, and it gives her something real to discuss.

`tests/eval/test_eval_regression.py` — a fast subset (12 questions, in-memory store, extractive generation) asserting recall@5 stays above a floor committed in `eval/baseline.json`. This runs in CI and fails the build if retrieval quality regresses.

### 9. Testing requirements

- Target ≥85% line coverage on `src/`, enforced with `--cov-fail-under=85`
- `tests/conftest.py` provides: a tiny 12-document fixture corpus generated in-code, an in-memory store fixture, a deterministic fake embedder (hash-based, dimension-correct) for tests that don't need real semantics, and a fake LLM returning canned responses
- **No test may require Postgres, Ollama, or network access.** Integration tests use the memory store and fakes. Mark any test that needs Docker with `@pytest.mark.docker` and exclude it from the default run.
- Property-based tests with Hypothesis for the chunker: for any input text and any config, chunks must (a) cover the whole document when overlap is accounted for, (b) never exceed `max_tokens`, (c) produce stable ids across runs
- API tests use `httpx.ASGITransport`, no live server
- One test asserting the full pipeline is deterministic: run the same query twice, assert byte-identical results

### 10. CI, Docker, and tooling

`.github/workflows/ci.yml` — on push and PR:
- Matrix over Python 3.11 and 3.12
- `uv sync --frozen`
- `ruff check` + `ruff format --check`
- `mypy --strict src/`
- `pytest --cov=src --cov-fail-under=85`
- Build the Docker image (don't push)
- Upload the coverage report as an artifact
- A separate job that runs the eval regression test

Cache the `uv` and HuggingFace model directories so CI is fast. The workflow must pass on the first run — verify locally with the same commands before committing.

`docker-compose.yml` — services: `api`, `postgres` (pgvector image, with an init script creating the extension and schema), `ollama` (optional, in an `llm` profile so `docker compose up` works without it), `streamlit`. Healthchecks on all services, `depends_on: condition: service_healthy`.

`Dockerfile` — multi-stage, non-root user, model weights pre-downloaded in the build layer so container start is fast, `HEALTHCHECK` instruction.

`Makefile` targets: `install`, `dev`, `test`, `lint`, `typecheck`, `ingest`, `eval`, `eval-ablate`, `up`, `down`, `clean`. Each one works.

### 11. README specification

Structure, in this order:

1. One-sentence description, then a row of badges (CI, coverage, Python version, license)
2. **A results table above the fold** — the real ablation numbers. This is the first thing a reader should see.
3. A Mermaid architecture diagram showing ingestion and query paths as separate flows
4. Quickstart: exactly three commands from clone to a working query, then a `curl` example with real output pasted in
5. "How it works" — 5 short subsections, one per pipeline stage, each explaining the *why* not just the what
6. Configuration table: every env var, type, default, effect
7. Evaluation section: how the golden set was built, what each metric means, how to reproduce
8. "What I'd do differently at scale" — 4 bullets on the honest limitations (single-node pgvector ceiling, reranker latency cost, no incremental index rebuild, golden set size)
9. Project structure tree
10. License

Write the README last, after the numbers exist. Never write aspirational documentation.

### 12. Build order and commit plan

Work through these in order. Commit at each numbered step with a conventional-commit message. Run the tests before each commit.

1. `chore: scaffold project, tooling, and CI skeleton` — pyproject, ruff/mypy config, empty CI that passes
2. `feat(config): settings and logging` — config.py, logging_config.py, tests
3. `feat(models): pydantic schemas` — models.py with every schema, tests
4. `feat(ingest): document loaders` — loaders.py + tests for each format
5. `feat(ingest): three chunking strategies` — chunker.py + unit and property tests
6. `feat(embed): local and optional remote embedders` — embed/ + tests with the fake embedder
7. `feat(store): vector store protocol and in-memory implementation` — store/base.py, memory.py + tests
8. `feat(store): pgvector backend with hnsw index` — pgvector.py, migrations, docker-compose postgres
9. `feat(ingest): orchestrated pipeline with change detection` — pipeline.py + integration test
10. `feat(retrieve): dense and lexical retrievers` — dense.py, lexical.py + tests
11. `feat(retrieve): reciprocal rank fusion` — fusion.py + the hand-computed unit test
12. `feat(retrieve): cross-encoder reranking with cache` — rerank.py + tests
13. `feat(retrieve): orchestrated retrieval pipeline` — pipeline.py + integration test
14. `feat(generate): prompts, ollama client, extractive fallback` — generate/ + tests
15. `feat(guardrails): grounding verification and refusal` — guardrails/ + tests
16. `feat(api): query, ingest, admin routes` — api/ + API tests
17. `feat(api): streaming, metrics, rate limiting` — middleware and SSE + tests
18. `feat(eval): metrics implemented from scratch` — eval/metrics.py + hand-computed tests
19. `feat(eval): golden set tooling and validator` — golden.py, build_golden_set.py
20. `data: add human-verified golden set` — the jsonl file
21. `feat(eval): runner, reporting, ablation script` — runner.py, report.py, run_eval.py
22. `test: eval regression gate with committed baseline` — the CI gate
23. `feat(ui): streamlit demo` — ui/
24. `ci: full pipeline with coverage gate and docker build`
25. `docs: architecture, evaluation, and three ADRs`
26. `docs: readme with measured ablation results`

### 13. Definition of done

Do not stop until every line is true:

- [ ] `git clone` → `make install` → `make up` → a real query returns a cited answer, on a machine with no API keys
- [ ] `make test` passes, coverage ≥85%
- [ ] `mypy --strict src/` clean
- [ ] `ruff check` and `ruff format --check` clean
- [ ] `make eval-ablate` produces the full comparison table with real numbers
- [ ] Zero `TODO`, `FIXME`, `NotImplementedError`, or `pass  #` in `src/`
- [ ] Re-running ingestion on an unchanged corpus reports 0 changed
- [ ] The API returns a graceful refusal (not a 500) when Ollama is stopped mid-request
- [ ] Every negative-category golden question is either refused or answered with an explicit insufficiency statement
- [ ] CI is green
- [ ] README numbers match the committed `eval_results/` output

### 14. Session plan

**This file is fully self-contained — no companion document is required.** Hand this single file to Claude Code and run the build as the sessions below, under this protocol:

**Session 1, before any code:** save this prompt as `docs/BUILD_SPEC.md`, create `DEVIATIONS.md` (header only), and create `CLAUDE.md` exactly as follows — commit all three as part of commit 1, and keep CLAUDE.md's State section current at every commit thereafter.

```markdown
# CLAUDE.md — standing rules and state
The spec is docs/BUILD_SPEC.md. This file is rules and state; the spec defines the work.

## Rules — non-negotiable
1. No TODO, FIXME, NotImplementedError, or stub bodies anywhere in src/. Ever.
2. Dependency versions come from the resolver (`uv add` / `npm install`); commit the lockfile.
   Never hand-type a version number the resolver has not produced.
3. Nothing is "done" until its command has run in THIS session with the real output shown —
   the actual pytest summary line, the actual exit status. "Should pass" is not a status.
4. Every number in a README or doc must exist in a committed artifact (eval_results/,
   benchmarks/results/, a CI log). An estimated or remembered number is a defect.
5. When a library, API, or dataset differs from the spec — renamed function, changed endpoint,
   auth now required — adapt to reality and add one line to DEVIATIONS.md
   (spec said / reality is / what was done). Never mock a real path to fake compliance.
6. Never weaken, skip, or delete a test to make it pass. Fix the code or flag the conflict.
7. One commit per plan milestone; the full test suite runs green before every commit.
8. If the next milestone will not fit in the session's remaining capacity, stop at the last
   green commit and update State. Do not start work you cannot finish.

## State (update at every commit)
- Plan position: <n> of <total>. Last completed: "<commit message>"
- Suite at last commit: <pytest summary line> · Coverage: <n>%
- Open deviations: <count> · Next up: commits <n+1>–<m>
- Notes for next session: <blockers, decisions pending>
```

**Every session after the first opens with this message** (the human pastes it, filling the brackets):

> Read CLAUDE.md, DEVIATIONS.md, docs/BUILD_SPEC.md (skim), and `git log --oneline -15`. We are at commit [n] of the plan. First action: run `make test` and paste the summary line. If it is not green, fixing that is the entire session — no new work on a red suite. If green, proceed with commits [n+1]–[m] only, under the CLAUDE.md rules. Stop at the last green commit before context runs low and update State.

**Between sessions (human, ~15 minutes):** run `make test`; run `grep -rnE "TODO|FIXME|NotImplementedError" src/`; compare `git log --oneline` against the commit plan; open the newest test file and check it asserts something real rather than that a mock returned what it was told; pick one number from the README and trace it to its committed artifact. Any failure means the next session opens with "fix these findings" instead of new work.

**If the build starts thrashing** — rewriting working code, a test flip-flopping between attempts, quiet "simplifications" of the spec — stop, `git reset --hard <last-green-commit>`, and open a fresh session scoped to one milestone with the exact error text pasted in.

**The session slices for this project:**

Four sessions, each ending at a green commit; the protocol above governs every one of them.

| Session | Commits | Boundary check before closing |
|---|---|---|
| A | 1–9 (scaffold → ingestion pipeline) | `make test` green; ingest a 3-document sample end-to-end and show the "N chunks created" summary; re-run and show "0 changed". |
| B | 10–17 (retrieval → generation → API) | With Ollama **stopped**, `curl` a real query against the running API and get a cited extractive answer, not a 500 — this proves the fallback path before anything depends on it. |
| C | 18–23 (eval metrics → golden set → ablation → UI) | `make eval-ablate` completes on the real corpus; the comparison table exists in `eval_results/` with plausible, non-identical numbers per row. |
| D | 24–26 (CI → docs → README) + acceptance | Fresh-clone acceptance: clone into a brand-new directory, follow only the README quickstart character for character, and it must work; then every README number traced to its `eval_results/` artifact. |

The ablation in session C is the long-running step (~30–60 min of compute) — kick it off, let it run, and have the session write docs while it does rather than idling.

### 15. Failure recovery — project-specific

- **arXiv API slow or returning 503s:** the retry/backoff is specified; if the standard endpoint is rate-limiting hard, `export.arxiv.org` is the designated mirror for programmatic access — switch the base URL, keep the 3-second delay, note it in DEVIATIONS.md. Never drop the delay to "speed things up"; arXiv blocks aggressive clients by IP.
- **HuggingFace model downloads stalling CI:** cache `~/.cache/huggingface` with `actions/cache`, and set `HF_HUB_OFFLINE=1` in the unit-test job — those tests use the deterministic fake embedder by design and must never touch the network. If they fail offline, that's a test-isolation bug to fix, not a caching problem.
- **pgvector container on Apple Silicon:** use the multi-arch `pgvector/pgvector:pg16` image (pin the digest). If port 5432 is already taken locally, remap in a `docker-compose.override.yml` rather than editing the committed compose file.
- **Reranker too heavy for the dev machine:** `bge-reranker-base` needs ~1.5 GB. The config already allows disabling the rerank stage for local iteration — but the numbers published in the README must come from a run with it enabled. If dev and published configs differ, the eval report records which config produced it (the runner already logs `RetrievalConfig` per run — keep it that way).
- **Golden-set validation failures:** if `--validate` reports a `relevant_chunk_texts` entry that isn't an exact substring, the fix is always to correct the golden entry against the corpus — never to loosen the validator to fuzzy matching. The exact-substring rule is what makes retrieval metrics trustworthy.
- **Ollama model choice:** any 7–8B instruct model works; if the default isn't pulled, `make pull-model` should do it explicitly rather than the API call failing mid-demo. The extractive fallback means nothing hard-fails either way — that's the design, lean on it.

### 16. Review round 2 — added depth and corrections

- **Schema migrations, pinned down:** ordered SQL files in `migrations/NNN_*.sql`, applied by `docker-entrypoint-initdb.d` in compose and by an idempotent `scripts/migrate.py` (tracks applied filenames in a `_migrations` table) for non-Docker runs. Do not add Alembic for one schema — it's ceremony without payoff here.
- **Embedding-space guard:** the collection records its embedder name and dimension at creation; ingest refuses vectors from any other embedder with a clear typed error. This prevents a silently mixed vector space after a config change — the corruption is invisible until recall craters. Add a test.
- **BM25 staleness detection:** persist a fingerprint (SHA256 over the sorted chunk-id list) beside the serialised index; rebuild iff it differs. Test that adding one chunk triggers a rebuild and a no-op re-ingest does not.
- **SSE contract:** `event: token` carrying text deltas, a terminal `event: done` carrying the metadata JSON (citations, timings, grounding, request_id), and a comment-line heartbeat every 15 s so proxies don't kill idle streams. Put a working client snippet in the README.
- **Golden-set honesty:** the build generates and substring-validates the 60 questions, but the *human* review is Akriti's acceptance-pass task (1–2 hours). Until she has actually done it, the README describes the set as "model-generated, substring-validated against the corpus; human review pending" — the wording flips only after the review happens, never before.

### 17. Review round 3 — final audit findings

- **Span→chunk relevance — the eval's missing predicate:** the golden set stores document-level spans, but retrieval returns chunks whose IDs differ across chunking strategies. Define relevance once in `eval/metrics.py`: a retrieved chunk is relevant iff it contains a golden span in full, or overlaps one by ≥50 consecutive characters. Unit-test the predicate with a span deliberately split across two chunks (both halves count as relevant). Without this rule the chunking ablation compares incomparable numbers — it is the reason relevance was stored as text rather than as chunk ids.
- **bge query instruction:** `bge-small-en-v1.5` documents a query-side instruction prefix for retrieval ("Represent this sentence for searching relevant passages: "). Apply it to queries only — never to passages — inside the embedder wrapper, and record its presence in `RetrievalConfig` so every run is attributable.
- **pgvector specifics:** create the index `USING hnsw (embedding vector_cosine_ops)`, and deliver per-query search effort with `SET LOCAL hnsw.ef_search = <ef>` inside the query's transaction — that is how a per-request `ef` actually reaches pgvector; a config value that never becomes a `SET LOCAL` does nothing.
- **Relevance grading:** the golden set is binary; state that nDCG therefore uses binary gains, and cover the graded-relevance code path with a synthetic-grades unit test so it isn't dead code.

### 18. Review round 4 — closing ambiguities

- **arXiv HTML availability:** native HTML full text exists for most recent papers but not all. When a paper's HTML is unavailable, land the abstract-only document, set `"full_text": false` in its manifest entry, and report the split in the download summary — never silently skip the paper, and don't fall back to fetching the PDF (size and rate-limit cost for marginal gain).
- **Corpus stays out of git, explicitly:** `data/corpus/` is download-only (`.gitignore`d, `.gitkeep` placeholder). The manifest with SHA256s is committed; the texts are not — arXiv papers carry per-paper licenses, and wholesale redistribution isn't yours to grant. The fresh-clone acceptance test therefore includes running the download script.
- **Golden spans capped at 300 characters:** `relevant_chunk_texts` entries are matching keys, not documents — short spans make the span→chunk relevance predicate stricter and keep quoted material minimal. The `--validate` step enforces the cap alongside the exact-substring rule.

---

## After the build

**Repo description:** `Production RAG service: hybrid dense+BM25 retrieval, cross-encoder reranking, grounding verification, and a from-scratch evaluation harness with published ablations.`

**Topics:** `rag` `llm` `information-retrieval` `fastapi` `pgvector` `sentence-transformers` `evaluation` `python`

**LinkedIn Projects entry (fill in the real numbers):**

> Built a production retrieval-augmented generation service with hybrid dense + BM25 retrieval, reciprocal rank fusion, and cross-encoder reranking. Implemented nDCG/MRR/recall metrics from scratch and a 60-question human-verified golden set, then published the full ablation study: hybrid + reranking reached nDCG@5 of X.XXX versus X.XXX for dense-only retrieval, at p95 latency of XXXms. Includes grounding verification that flags unsupported sentences, confidence-based refusal, and a CI gate that fails the build on retrieval-quality regression. Python, FastAPI, pgvector, sentence-transformers, Docker.

**Interview questions to be ready for:**
- Why RRF instead of just normalising and summing the scores? (Answer: rank-based fusion is scale-free, so you don't need the two retrievers' scores to be comparable — and it's robust when one retriever's score distribution shifts.)
- What does `k=60` do in RRF? What happens at `k=1` and `k=1000`?
- Your reranker adds latency. When is it not worth it?
- Where does semantic chunking beat structural chunking, and where does it lose? (Your ablation has the answer — know it.)
- How would you handle a 10M-document corpus? (Honest answer: pgvector HNSW starts hurting; you'd shard or move to a dedicated index, and you'd need incremental BM25 index updates.)
