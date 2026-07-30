# Retrieval Engine

Production retrieval-augmented generation service: hybrid dense and BM25 retrieval,
reciprocal rank fusion, cross-encoder reranking, grounding verification, and an evaluation
harness whose metrics are implemented from scratch and whose ablation numbers are published
alongside the artifacts that produced them.

[![ci](https://github.com/akriti-adarsh/Retrieval-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/akriti-adarsh/Retrieval-Engine/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-555%20passing-brightgreen)
![coverage](https://img.shields.io/badge/coverage-89.11%25-brightgreen)

**It runs with zero API keys.** The default embedder is local, the default generator is a
local Ollama model, and if Ollama is not running the service falls back to extractive
answering rather than failing. That fallback is tested against a genuinely stopped server, not
mocked.

---

## Results

Seven configurations over **303 arXiv `cs.CL` papers** and **60 golden questions**, 2.15 hours
of CPU. Every cell is read from [`eval_results/ablation.json`](eval_results/ablation.json).

| config | recall@5 | nDCG@5 | MRR | refusal acc | false refusal | p95 latency |
|---|---|---|---|---|---|---|
| dense only | 0.400 | 0.318 | 0.302 | 0.000 | 0.000 | 3.5 s |
| lexical only | 0.683 | 0.568 | 0.553 | 0.000 | 0.000 | 7.0 s |
| hybrid (RRF) | 0.577 | 0.460 | 0.428 | 0.000 | 0.000 | 5.2 s |
| **hybrid + rerank** (default) | **0.703** | **0.603** | 0.589 | 1.000 | 0.140 | 21.6 s |
| hybrid (weighted) + rerank | 0.703 | 0.585 | 0.565 | 1.000 | 0.140 | 11.8 s |
| hybrid + rerank, fixed-token chunking | 0.670 | 0.584 | 0.594 | 0.700 | 0.180 | 18.7 s |
| hybrid + rerank, semantic chunking | 0.672 | 0.562 | 0.552 | 0.900 | 0.120 | 14.9 s |

**Read the awkward rows first.** [docs/evaluation.md](docs/evaluation.md) is the full analysis,
including what should not be trusted.

- **Reranking is the only large win.** It moved recall@5 from 0.577 to 0.703 and nDCG from
  0.460 to 0.603, and it is the only configuration that can refuse at all. It also costs about
  70 percent of end-to-end latency on CPU.
- **The full pipeline beats plain BM25 by 2.0 points of recall@5**, 0.703 against 0.683. That
  is a modest return on a lot of machinery, and it is the honest headline.
- **Fusion made things worse before reranking fixed them.** Hybrid RRF (0.577) scored below
  lexical alone (0.683). RRF gives every input list an equal vote, so fusing a weak list with
  a strong one drags the strong one down.
- **Semantic chunking, the most expensive strategy, came last.** It costs 2.02x the default to
  build and lost on recall, nDCG, MRR, and precision.
- **The 28 point dense-against-lexical gap is not a finding.** The golden questions were
  written with their evidence passages in view, which reuses source vocabulary and hands BM25
  exact-token overlap. That row is reported as unknown, not as "BM25 beats embeddings".
- **Refusal accuracy is 0.000 without a reranker by design**, not because the guardrail broke.
  RRF scores cluster near 0.03 and carry no absolute meaning, so the pipeline declines to
  threshold them rather than inventing a comparison.

`refusal accuracy` never appears without `false refusal` beside it, because refusing every
question also scores 1.000 on the first.

Two rows the spec asks for are missing: multi-query and HyDE. Both need a model to write the
expansions and no Ollama is installed here, so the harness omits them rather than running them
with expansion silently off, which would duplicate the row above under a label claiming
otherwise. `DEVIATIONS.md` entry 14 records it.

```bash
make eval-ablate    # reproduces the table, 2.15 hours measured on CPU
```

### What has actually been measured

These numbers come from real runs on this machine and each traces to a committed artifact or a
recorded command.

| Measurement | Value | Where it came from |
|---|---|---|
| Corpus harvested | 303 documents, 278 with full text, 18,042,170 characters | `data/corpus/manifest.json` |
| Ingest throughput | 2,654 tokens/second on CPU, structural chunking | `eval_results/index_build.md` |
| Full corpus index build | 26.5 min structural, 20.3 min fixed-token, 53.5 min semantic | `eval_results/index_build.md` |
| Full ablation matrix | 2.15 hours, seven configurations | `eval_results/ablation.json` |
| Test suite | 555 passing, 1 deselected (the docker-marked test) | `make test` |
| Coverage | 89.11 percent, gate at 85 | `make test` |
| `mypy --strict` | Clean across 43 source files | `make typecheck` |
| Regression gate, fixture corpus | recall@5 0.750, nDCG@5 0.570, MRR 0.510, refusal accuracy 0.250 | `src/retrieval_engine/eval/baseline.json` |
| Reranker score separation | Answered 0.4138 to 0.9996, refused 0.0001 to 0.2878, nothing in between | `eval_results/02042ae6-recursive_structural/rows.jsonl`, all 60 questions |
| Extractive fallback with Ollama stopped | HTTP 200, cited answer, `grounded: true` | Live `curl`, recorded in `CLAUDE.md` |

The fixture-corpus gate numbers are **not** retrieval quality claims. They are a pipeline
regression floor measured with a deterministic fake embedder, explained in
[the CI regression gate](#the-ci-regression-gate).

### Build status

| Artifact | State |
|---|---|
| Ingestion, retrieval, generation, guardrails, API | Complete, tested, CI green |
| Evaluation harness, metrics, validator, regression gate | Complete, tested |
| arXiv corpus (303 papers) | Downloaded, manifest committed |
| Golden set (`data/golden/golden_set.jsonl`) | 60 questions, validator reports 0 failures |
| pgvector, migrations, live round trip | Verified against a real database |
| Docker image, compose stack, full CI | Complete. Stack verified running: postgres, api, and streamlit all healthy, ingest and query end to end through the containers |
| Streamlit UI (`make ui`) | Complete |
| `docs/architecture.md`, three ADRs | Complete |
| Ablation results (`eval_results/`) | Complete. Seven configurations, 2.15 hours, artifacts committed |
| `docs/evaluation.md` | Complete, including the threats to validity |
| Multi-query and HyDE ablation rows | Not measured. No model server on this machine (`DEVIATIONS.md` 14) |

`DEVIATIONS.md` records every deviation from the specification as spec said, reality is, what
was done. `CLAUDE.md` carries the build state and the measurements above.

---

## Table of contents

- [What this is, in plain terms](#what-this-is-in-plain-terms)
- [Why the design looks like this](#why-the-design-looks-like-this)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [How it works, stage by stage](#how-it-works-stage-by-stage)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Project structure](#project-structure)
- [What I would do differently at scale](#what-i-would-do-differently-at-scale)
- [Known limitations and unverified claims](#known-limitations-and-unverified-claims)
- [License](#license)

---

## What this is, in plain terms

A language model asked a factual question will answer confidently whether or not it knows.
Retrieval-augmented generation (RAG) reduces that problem by finding real passages first and
requiring the answer to come from them.

This repository is a complete, running implementation of that idea, built to be inspected
rather than demoed. Concretely, you give it a folder of documents and it:

1. **Reads them.** PDF, markdown, HTML, plain text, and docx, preserving page and section
   information so a citation can point somewhere specific.
2. **Splits them into passages** and turns each into a vector.
3. **Answers a question** by finding the most relevant passages two different ways, combining
   the two rankings, re-scoring the best candidates with a slower and more accurate model, and
   writing an answer that cites the passages it used.
4. **Checks its own answer.** Every sentence is scored against the retrieved passages, and any
   sentence that is not supported is flagged in the response.
5. **Refuses when the evidence is weak,** returning an explanation instead of a guess.

The part that is unusual, and the part worth reviewing, is item 5 plus the evaluation harness.
Most RAG projects can produce an answer. Far fewer can tell you how often that answer is
wrong, and fewer still publish the number when it is unflattering.

### Who this is for

| If you are | Start here |
|---|---|
| Evaluating this as engineering work | The results table above, then [Evaluation](#evaluation), then [Known limitations](#known-limitations-and-unverified-claims) |
| Trying to run it | [Quickstart](#quickstart) |
| Trying to understand the retrieval design | [How it works](#how-it-works-stage-by-stage) and [docs/decisions/](docs/decisions/) |
| Looking for the honest caveats | [Known limitations](#known-limitations-and-unverified-claims) and `DEVIATIONS.md` |

---

## Why the design looks like this

Five decisions shape everything else. Each is written up as an ADR in [docs/decisions/](docs/decisions/).

**Two retrievers, not one.** A dense bi-encoder is good at paraphrase and blurs exact rare
tokens. BM25 is precise on a specific identifier, model name, or number, and blind to
rephrasing. They fail in different ways, which is the only reason combining them helps. See [ADR 001](docs/decisions/001-hybrid-over-dense-only.md).

**Reciprocal rank fusion, not score averaging.** Cosine similarity and BM25 scores live on
unrelated scales. RRF consumes ranks, so it needs no calibration between the two retrievers
and survives one of them shifting its score distribution. Weighted score fusion is implemented
too, so the ablation can measure the difference rather than assert it.

**Postgres with pgvector, not a dedicated vector database.** One datastore holds the documents
and the vectors, which is one system to operate, back up, and reason about transactionally.
The honest ceiling is in [ADR 002](docs/decisions/002-pgvector-over-dedicated-db.md).

**Relevance is defined on text spans, not chunk ids.** The golden set records verbatim spans.
If it recorded chunk ids, the chunking comparison would be measuring three different ground
truths and the numbers would look fine while meaning nothing.

**Refusing is a feature with a test.** A low-confidence question returns HTTP 200 with
`answer_type: "refused"` and an explanation. The evaluation reports refusal accuracy next to
false refusal rate, because refusing everything scores a perfect 1.000 on the first one.

---

## Architecture

Ingestion and querying are separate paths that meet only at the vector store.

```mermaid
flowchart LR
    subgraph Ingestion
        A[Source files<br/>pdf md html txt docx] --> B[Loader<br/>text + page spans + front matter]
        B --> C{Content hash<br/>changed?}
        C -- no --> C1[Skip<br/>0 re-embedded]
        C -- yes --> D[Chunker<br/>fixed / structural / semantic]
        D --> E[Embedder<br/>bge-small-en-v1.5]
        E --> F[(Vector store<br/>pgvector or memory)]
        D --> G[BM25 index<br/>fingerprinted]
    end
```

```mermaid
flowchart TD
    Q[Question] --> EXP[Stage 1<br/>optional expansion<br/>multi-query or HyDE]
    EXP --> DEN[Stage 2a<br/>dense search<br/>top_k 50]
    EXP --> LEX[Stage 2b<br/>BM25 search<br/>top_k 50]
    DEN --> FUSE[Stage 3<br/>reciprocal rank fusion<br/>k 60]
    LEX --> FUSE
    FUSE --> RER[Stage 4<br/>cross-encoder rerank<br/>top 20 to top 5]
    RER --> POL{Refusal policy<br/>score >= min_confidence?}
    POL -- no --> REF[200 refused<br/>+ explanation]
    POL -- yes --> GEN[Generate<br/>Ollama, else extract]
    GEN --> GRD[Grounding check<br/>per sentence + citations]
    GRD --> OUT[Answer + citations<br/>+ sources + grounding + timings]
```

Stages 2a and 2b run concurrently. The refusal decision happens **before** generation, so a
weak question costs no model call and the refusal cannot be talked out of by a fluent model.

Full module map, data model, and request lifecycle: [docs/architecture.md](docs/architecture.md).

---

## Quickstart

Three commands from clone to a working query, on a machine with no API keys.

```bash
git clone https://github.com/akriti-adarsh/Retrieval-Engine.git && cd Retrieval-Engine
make install                                    # uv sync --frozen
make corpus                                     # fetch 300 arXiv cs.CL papers
```

Then start the API. With Docker, in this order, because the service applies no schema of its
own and will not start against an empty database:

```bash
docker compose up -d postgres                          # database first
docker compose run --rm api python scripts/migrate.py  # apply the schema, idempotent
docker compose up -d                                   # api on :8000, streamlit on :8501
```

Ollama sits in the `llm` profile and is **not** started by default, so the stack above comes up
with no model server and answers extractively. Add it with `docker compose --profile llm up -d`.

Without Docker, using the in-memory store:

```bash
RE_STORE=memory uv run uvicorn retrieval_engine.api.app:create_app --factory --port 8000
```

Ingest and ask. Everything below is **real output from the containerised stack above**, with no
model server running, which is why `answer_type` is `extractive`. The corpus here is three of
this repository's own documents, so it takes a second rather than half an hour:

```bash
$ curl -s -X POST localhost:8000/v1/ingest \
    -H 'Content-Type: application/json' -d '{"path": "demo"}'
{"docs_seen":3,"docs_changed":3,"docs_unchanged":0,"docs_failed":0,
 "chunks_created":23,"chunks_skipped":0,"tokens_embedded":4870,
 "elapsed_seconds":1.103,"errors":[]}

$ curl -s localhost:8000/health/ready
{"ready":true,"store":true,"embedder":true,"generator":false,
 "detail":{"generator":"model server unreachable, answers will be extractive"}}

$ curl -s -X POST localhost:8000/v1/query \
    -H 'Content-Type: application/json' \
    -d '{"query": "how does reciprocal rank fusion combine two ranked lists?", "top_k": 2}'
{"answer":"## Decision Run both retrievers on every query and fuse their ranked lists with
            reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank(d)) with
            k = 60 ... [1]",
 "answer_type":"extractive",
 "citations":[{"marker":1,"doc_id":"001-hybrid-over-dense-only","resolved":true},
              {"marker":2,"doc_id":"001-hybrid-over-dense-only","resolved":true}],
 "grounding":{"grounded":true,"flagged_sentences":[],"threshold":0.55},
 "timings":{"expansion_ms":0.0,"dense_ms":78.1,"lexical_ms":62.5,"fusion_ms":0.3,
            "rerank_ms":5281.1,"retrieval_ms":5359.8,"generation_ms":4210.5,
            "grounding_ms":190.6,"total_ms":9761.2},
 "prompt_version":"v1","model":"extractive","request_id":"..."}
```

Note `generator: false` with `ready: true`. A dead model server is a quality reduction, not an
outage, so it must not pull the service out of rotation.

Note also that `dense_ms` and `lexical_ms` differ. They run concurrently and each times itself,
so they overlap and are not meant to sum to `retrieval_ms`. They were identical in earlier
artifacts, which was a bug, not a coincidence.

Re-run the ingest and nothing is re-embedded:

```bash
$ curl -s -X POST localhost:8000/v1/ingest -H 'Content-Type: application/json' -d '{"path": "demo"}'
{"docs_seen":3,"docs_changed":0,"docs_unchanged":3,"docs_failed":0,
 "chunks_created":0,"chunks_skipped":23,"tokens_embedded":0,
 "elapsed_seconds":0.011,"errors":[]}
```

A path outside the data directory is refused rather than read, and every failure uses the same
error envelope:

```bash
$ curl -s -X POST localhost:8000/v1/ingest -H 'Content-Type: application/json' -d '{"path": "../../etc"}'
{"error":{"code":"document_load_error",
          "message":"path '../../etc' is outside the configured data directory",
          "request_id":"41da823508374e12b0898f103db41b0f"}}
```

Change detection compares the document's content hash, not its modification time, because a
git checkout rewrites mtimes without changing content and would otherwise re-embed everything.

### Streaming

```bash
curl -N -X POST localhost:8000/v1/query/stream \
  -H 'Content-Type: application/json' -d '{"query": "what does k do in RRF?"}'
```

```
event: token
data: {"text": "Reciprocal "}

event: token
data: {"text": "rank "}

: keep-alive

event: done
data: {"citations":[...],"sources":[...],"grounding":{...},"timings":{...},"request_id":"..."}
```

The contract is `token` events carrying deltas, one terminal `done` event carrying the
metadata, and comment-line heartbeats every 15 seconds so a proxy does not close the
connection while the model is thinking. Event data is JSON rather than raw text, because SSE
is newline-delimited and a delta containing a newline would otherwise corrupt the stream.

A minimal client:

```python
import httpx, json

with httpx.stream(
    "POST", "http://localhost:8000/v1/query/stream", json={"query": "what does k do in RRF?"}
) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            payload = json.loads(line[6:])
            print(payload.get("text", ""), end="", flush=True)
```

### Demo UI

```bash
make ui    # streamlit on :8501
```

It talks to the API over HTTP like any other client rather than importing the pipeline, so it
shows what a consumer actually sees: the answer, the numbered sources to check each citation
against, the grounding verdict, and the per-stage timings.

---

## How it works, stage by stage

### 1. Ingestion

Loaders produce a `Document` with the extracted text, a SHA256 content hash, front matter as
metadata, and `PageSpan` entries recording where each page or heading sits. Every character
offset in the system is an offset into that text, so front matter is stripped and line endings
normalised **before** any offset is computed. A half-stripped offset would corrupt every
citation from that document silently, so the tests assert each recorded span really starts on
the heading it claims.

Front-matter metadata is coerced at the boundary. PyYAML turns an unquoted date into a
`datetime.date` and a nested block into a dict, neither of which the schema accepts, so dates
become ISO strings and nested mappings are dropped rather than stringified.

### 2. Chunking

Three strategies behind one interface, because the ADR compares them:

| Strategy | How it splits | Good at | Bad at |
|---|---|---|---|
| `fixed_token` | Token windows with overlap | Predictable size and cost | Cuts sentences in half, strands a claim from its subject |
| `recursive_structural` (default) | Headings, then paragraphs, then sentences | Keeps a heading with its text, so `section_path` is honest | Sections of wildly uneven size |
| `semantic` | Embeds sentences, cuts where consecutive sentences diverge | Finds topic boundaries no heading marks | Embeds every sentence at ingest time |

All three emit candidate character ranges and share one finaliser, which is where the hard
token cap is enforced and where ids, page numbers, and metadata are attached. That is what
makes the chunking comparison meaningful: the strategies differ only in where they cut.

Chunk ids are `UUID5(doc_id, start_char)`, so re-ingesting an unchanged document reproduces
byte-identical ids. That is what lets change detection and upsert coexist safely.

Five Hypothesis properties hold over generated text and configs: fixed and structural chunks
cover every non-whitespace character, no chunk exceeds its token budget, ids are identical
across runs, and a chunk's text always matches its own offsets.

### 3. Retrieval

Dense and lexical run concurrently, then fuse, then rerank.

**Dense** embeds the query with the bge query-side instruction prefix and searches the vector
store. The prefix is applied to queries and never to passages, which is the documented bge
behaviour; `embed_query` and `embed_documents` are separate methods precisely so a call site
cannot get that backwards.

**Lexical** is BM25 over the same chunk set. The index persists to disk and rebuilds only when
a SHA256 fingerprint over the sorted chunk-id list changes. No stemming: BM25's IDF already
discounts common words, and stemming would collapse the rare exact tokens that justify running
a lexical retriever next to a dense one.

**Fusion** is `score(d) = Σ_r 1 / (k + rank_r(d))` with `k = 60`. What `k` does is worth
knowing, and both ends are pinned by tests: at `k = 1` a document at rank one in a single list
beats one at rank four in both lists (0.5 against 0.4); at `k = 1000` the same lists invert
(0.000999 against 0.001992), because a flat rank curve turns fusion into a vote count.

**Reranking** runs a cross-encoder over the top 20 fused candidates and returns 5. A
cross-encoder concatenates query and passage into one sequence so attention runs across both,
which is why it beats a bi-encoder on ranking and why nothing can be precomputed. It is the
largest latency contributor in the pipeline. Scores are cached in an LRU keyed by
`(query_hash, chunk_id)`; keying on chunk id alone would serve one query's scores to another.

Every result carries the score and rank it had at each stage, so rank movement through fusion
and reranking is inspectable with `?debug=true` rather than requiring a second run.

### 4. Generation

The prompt injects sources as a numbered list and asks for `[n]` markers. That is not
cosmetic: a marker can be mechanically resolved back to a source and checked, whereas
"according to the first paper" cannot. `PROMPT_VERSION` travels on every response and into
every eval run, because a prompt edit moves metrics as much as a retrieval change does.

If Ollama is unreachable, or returns an empty completion, the service extracts instead. The
extractive path selects the sentences closest to the query in embedding space and attaches
their citations. Its central claim is that an extract cannot hallucinate, and there is a test
that walks the produced answer and asserts every sentence is quoted verbatim from a source.

### 5. Guardrails

**Grounding** is scored per sentence, not per answer. The failure worth catching is a fluent
answer that mixes supported claims with one invented detail, and a single similarity over the
whole answer averages that detail away. Citations are resolved too: a `[9]` pointing at a
source that was never in the context comes back `resolved: false` and makes the answer
ungrounded.

**Refusal** thresholds the reranker's score specifically, and that distinction matters. An RRF
score is about 0.03 by construction, so comparing it against a 0.3 confidence threshold would
refuse every query in the no-rerank rows and publish them as zeros. When no calibrated score
exists, the confidence check is skipped, only the source-count rule applies, and the decision
records `threshold_applied: false` so nothing downstream can present a fused score as a
probability.

The default threshold of 0.3 is validated by measurement rather than inherited. Across six
live queries against the real `bge-reranker-base`, relevant chunks scored 0.81 to 0.97 and an
out-of-corpus question ("who won the world cup in 1998?") scored 0.0000. The threshold sits in
the empty gap between those two modes.

---

## API reference

Every response is Pydantic-validated. Every error uses one envelope, so a client can switch on
`error.code` instead of discovering that some routes fail differently:

```json
{"error": {"code": "document_not_found", "message": "...", "request_id": "..."}}
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/query` | Answer a question. Returns answer, citations, sources, grounding, timings, prompt version |
| `POST` | `/v1/query/stream` | Same, as SSE: `token` events then a terminal `done` |
| `POST` | `/v1/ingest` | Ingest a server-side directory. `async_job: true` returns a job id |
| `POST` | `/v1/ingest/upload` | Ingest uploaded files |
| `GET` | `/v1/ingest/{job_id}` | Ingestion job status |
| `GET` | `/v1/documents` | Paginated document listing with metadata filters |
| `DELETE` | `/v1/documents/{doc_id}` | Remove a document and its chunks, returns the count |
| `GET` | `/health` | Liveness. Touches no dependency by design |
| `GET` | `/health/ready` | Readiness. Loads the embedder and queries the store |
| `GET` | `/metrics` | Prometheus format |

Query parameters on the query routes: `?debug=true` adds the retrieval trail (expanded
queries, candidate counts, per-stage scores, rank movement).

### Error codes

| Code | Status | Meaning |
|---|---|---|
| `validation_error` | 422 | Request body violated the schema |
| `unsupported_format` | 415 | No loader claims that file extension |
| `document_load_error` | 422 | Supported extension, unparseable content, or a path outside the data directory |
| `request_too_large` | 413 | Upload over `RE_MAX_UPLOAD_MB` |
| `embedding_space_mismatch` | 409 | Vectors from a different embedder or dimension |
| `store_unavailable` | 503 | Vector store unreachable or unmigrated |
| `document_not_found` / `job_not_found` | 404 | Unknown id |
| `rate_limit_exceeded` | 429 | Token bucket exhausted |
| `internal_error` | 500 | Unexpected. Detail is in the log under the request id |

A refused answer is **not** in this table. It is a 200 with `answer_type: "refused"`, because
refusing is a correct outcome and a client should not parse a 4xx to discover the system was
being careful.

### Observability

Every request carries a request id, taken from an inbound `X-Request-ID` or minted, returned
in both the header and the body, and bound to a contextvar so a log line from deep inside the
retrieval pipeline carries it without threading it through every signature.

`/metrics` exposes request counts and latency by route (labelled with the route template, not
the raw path, so a path parameter cannot grow label cardinality without bound), per-stage
latency histograms, answers by type, grounding verdicts, and the reranker cache hit rate.

---

## Configuration

Every setting is an environment variable prefixed `RE_`. Nested retrieval knobs use a double
underscore, for example `RE_RETRIEVAL__TOP_K_DENSE=100`. Defaults are chosen so a clone with
no environment at all runs entirely locally. See `.env.example`.

### Core

| Variable | Type | Default | Effect |
|---|---|---|---|
| `RE_ENV` | `dev\|test\|prod` | `dev` | Deployment environment label |
| `RE_SEED` | int | `42` | Global random seed. Two eval runs on one corpus give identical numbers |
| `RE_LOG_LEVEL` | enum | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `RE_LOG_JSON` | bool | `true` | JSON lines, or a readable console format |
| `RE_DATA_DIR` | path | `data` | Root for the corpus, golden set, and BM25 index |
| `RE_EVAL_RESULTS_DIR` | path | `eval_results` | Where eval artifacts land |

### Embedding

| Variable | Type | Default | Effect |
|---|---|---|---|
| `RE_EMBEDDER` | `local\|openai` | `local` | `local` needs no API key |
| `RE_EMBEDDING_MODEL` | str | `BAAI/bge-small-en-v1.5` | sentence-transformers model id |
| `RE_EMBEDDING_DIMENSION` | int | `384` | Vector width. Checked against the loaded model; a mismatch is fatal |
| `RE_EMBEDDING_BATCH_SIZE` | int | `32` | Chunks per encode call |
| `RE_EMBEDDING_DEVICE` | str | unset | torch device. Unset means auto |
| `RE_NORMALIZE_EMBEDDINGS` | bool | `true` | L2-normalise so dot product is cosine |
| `RE_USE_QUERY_INSTRUCTION` | bool | `true` | Apply the bge prefix to queries, never to passages |
| `RE_QUERY_INSTRUCTION` | str | `Represent this sentence for searching relevant passages: ` | The prefix itself |
| `RE_OPENAI_API_KEY` | secret | unset | Required only when `RE_EMBEDDER=openai` |
| `RE_OPENAI_BASE_URL` | str | `https://api.openai.com/v1` | For an OpenAI-compatible endpoint |
| `RE_OPENAI_EMBEDDING_MODEL` | str | `text-embedding-3-small` | Remote model id |

### Chunking

| Variable | Type | Default | Effect |
|---|---|---|---|
| `RE_CHUNK_STRATEGY` | enum | `recursive_structural` | `fixed_token`, `recursive_structural`, `semantic` |
| `RE_CHUNK_SIZE` | int | `512` | Target tokens per chunk |
| `RE_CHUNK_OVERLAP` | int | `64` | Token overlap. Must be less than `RE_CHUNK_SIZE` |
| `RE_CHUNK_MIN_TOKENS` | int | `96` | Fragments below this merge upward |
| `RE_SEMANTIC_THRESHOLD_PERCENTILE` | float | `95.0` | Split where sentence distance exceeds this percentile |
| `RE_INGEST_CONCURRENCY` | int | `4` | Concurrent document loads |

### Retrieval

| Variable | Type | Default | Effect |
|---|---|---|---|
| `RE_RETRIEVAL__USE_DENSE` | bool | `true` | Enable dense candidate generation |
| `RE_RETRIEVAL__USE_LEXICAL` | bool | `true` | Enable BM25 candidate generation |
| `RE_RETRIEVAL__TOP_K_DENSE` | int | `50` | Dense candidates |
| `RE_RETRIEVAL__TOP_K_LEXICAL` | int | `50` | Lexical candidates |
| `RE_RETRIEVAL__FUSION` | `rrf\|weighted` | `rrf` | Fusion method |
| `RE_RETRIEVAL__RRF_K` | int | `60` | RRF damping. Small favours top ranks, large counts appearances |
| `RE_RETRIEVAL__DENSE_WEIGHT` | float | `0.5` | Dense share under weighted fusion |
| `RE_RETRIEVAL__USE_RERANK` | bool | `true` | Enable the cross-encoder stage |
| `RE_RETRIEVAL__RERANK_CANDIDATES` | int | `20` | Shortlist size. Must be at least `FINAL_TOP_K` |
| `RE_RETRIEVAL__FINAL_TOP_K` | int | `5` | Results returned |
| `RE_RETRIEVAL__EXPANSION` | `none\|multi_query\|hyde` | `none` | Query expansion. Needs a reachable model |
| `RE_RETRIEVAL__NUM_PARAPHRASES` | int | `3` | Paraphrases under multi-query |
| `RE_RETRIEVAL__HNSW_EF_SEARCH` | int | `64` | Per-query HNSW effort, applied as `SET LOCAL` |
| `RE_RERANKER_MODEL` | str | `BAAI/bge-reranker-base` | Cross-encoder model id |
| `RE_RERANKER_BATCH_SIZE` | int | `16` | Pairs per forward pass |
| `RE_RERANKER_CACHE_SIZE` | int | `10000` | LRU entries |

### Store

| Variable | Type | Default | Effect |
|---|---|---|---|
| `RE_STORE` | `pgvector\|memory` | `pgvector` | `memory` needs no database |
| `RE_COLLECTION` | str | `default` | Logical collection. Records its embedder and dimension |
| `RE_POSTGRES_DSN` | str | `postgresql://retrieval:retrieval@localhost:5432/retrieval` | Connection string |
| `RE_PG_POOL_MIN_SIZE` / `RE_PG_POOL_MAX_SIZE` | int | `1` / `8` | Pool bounds |
| `RE_HNSW_M` | int | `16` | HNSW graph degree, dominates memory |
| `RE_HNSW_EF_CONSTRUCTION` | int | `64` | HNSW build effort |

### Generation and guardrails

| Variable | Type | Default | Effect |
|---|---|---|---|
| `RE_LLM` | `ollama\|extractive` | `ollama` | Generation backend. There is no remote option; the embedder is where that lives |
| `RE_OLLAMA_BASE_URL` | str | `http://localhost:11434` | Ollama server |
| `RE_OLLAMA_MODEL` | str | `llama3.1:8b` | Any 7B to 8B instruct model works |
| `RE_OLLAMA_TIMEOUT_S` | float | `60.0` | Request timeout |
| `RE_GENERATION_TEMPERATURE` | float | `0.0` | Zero, because an eval that changes on re-run measures nothing |
| `RE_GENERATION_MAX_TOKENS` | int | `512` | Completion cap |
| `RE_EXTRACTIVE_SENTENCES` | int | `3` | Sentences in a fallback answer |
| `RE_GROUNDING_THRESHOLD` | float | `0.55` | Per-sentence similarity below which a sentence is flagged |
| `RE_MIN_CONFIDENCE` | float | `0.3` | Refuse below this reranker score |
| `RE_MIN_SOURCES` | int | `1` | Refuse below this many usable sources |

### API

| Variable | Type | Default | Effect |
|---|---|---|---|
| `RE_API_HOST` / `RE_API_PORT` | str / int | `0.0.0.0` / `8000` | Bind address |
| `RE_RATE_LIMIT_ENABLED` | bool | `true` | Token-bucket limiting |
| `RE_RATE_LIMIT_PER_MINUTE` | int | `60` | Requests per caller per minute |
| `RE_MAX_UPLOAD_MB` | int | `25` | Upload size cap |
| `RE_SSE_HEARTBEAT_SECONDS` | float | `15.0` | Keep-alive interval on streams |

---

## Evaluation

This is the part that makes the rest of the repository checkable.

### The corpus

| Property | Value |
|---|---|
| Source | arXiv `cs.CL`, harvested over OAI-PMH |
| Documents | 303 |
| With full text | 278 |
| Abstract only | 25 |
| Total characters | 18,042,170 |
| Mean per document | 59,545 |
| Committed to git | No. The manifest with a SHA256 per file is; the texts are not |

The texts are not redistributed because arXiv papers carry per-paper licenses. `make corpus`
reproduces the download, and `data/corpus/manifest.json` records a digest per file so you can
verify what you fetched. A paper whose HTML full text is unavailable lands abstract-only and is
flagged `full_text: false` rather than being silently dropped.

### The golden set

Committed at `data/golden/golden_set.jsonl`. Each entry records the question, the document ids
that answer it, verbatim spans from those documents, a reference answer, a category, and a
difficulty.

Relevance is stored as **text spans, not chunk ids**, and this is the single most important
design choice in the evaluation. Chunk ids change with the chunking strategy, so a golden set
keyed by id would make the three chunking rows measure three different ground truths while
producing a table that looks perfectly reasonable.

`make golden-validate` enforces the contract:

- every span is an **exact** substring of a corpus document, not normalised and not fuzzy
- every span is at most 300 characters, so spans stay matching keys rather than quoted documents
- question ids are unique
- negative entries carry no evidence at all
- there are at least 8 negative entries

When a span fails, the fix is always to correct the span against the corpus. Loosening the
validator to fuzzy matching would make every number downstream unfalsifiable.

**Provenance, stated plainly:** the committed set is model-generated and substring-validated
against the corpus. Human review is pending. `scripts/build_golden_set.py --generate` is the
reproducible path for drafting candidates with a local model, and it writes to a review file,
never to the committed set, so the system cannot grade itself against its own guesses.

### What each metric means

| Metric | Definition | Why it is here |
|---|---|---|
| `recall@5` | Relevant chunks in the top 5, over all relevant chunks in the corpus | The headline retrieval number. The denominator is the true corpus-wide count, not the span count |
| `precision@5` | Relevant chunks in the top 5, over 5 | Divides by k, not by results returned, so returning 3 of 5 stays a failure |
| `hit_rate@5` | 1 if any of the top 5 is relevant | The "did it find anything at all" number |
| `mrr` | Mean of 1 / rank of the first relevant result | Right when a user reads one answer, wrong when breadth matters |
| `ndcg@5` | DCG over the ideal ordering's DCG | Rewards putting the relevant chunk higher. Binary gains, since the golden set is binary |
| `refusal_accuracy` | Fraction of negative questions correctly refused | Stops a system scoring well by answering everything |
| `false_refusal_rate` | Fraction of answerable questions wrongly refused | The counterpart. Refusing everything scores 1.000 on refusal accuracy |
| `citation_precision` | Fraction of citations pointing at a genuinely relevant chunk | An uncited answer scores 0.0, not 1.0 |
| `answer_similarity` | Embedding cosine to the reference answer | A blunt instrument, reported next to grounding rather than alone |
| `grounded_rate` | Fraction of answers with every sentence supported | The guardrail's own score |

Every metric is implemented from scratch in `src/retrieval_engine/eval/metrics.py`, with the
formula in its docstring and a unit test whose expected value was computed by hand. That
includes nDCG's tie-breaking and its zero-relevant-documents edge case.

**Relevance predicate.** A retrieved chunk counts as relevant if it contains a golden span in
full, or shares at least 50 consecutive characters with one. The 50-character rule is what
makes a span split across two chunks count for both halves, which is exactly the case the
chunking comparison depends on. There is a test that splits a span deliberately.

### Reproducing the numbers

```bash
make corpus            # 303 arXiv cs.CL papers, rate-limited to arXiv's 3-second ask
make golden-validate   # exact-substring check against the corpus
make eval-ablate       # the full matrix, 2.15 hours measured on CPU
```

Artifacts land in `eval_results/`: `ablation.md` and `ablation.json` for the matrix, plus a
per-run directory with `run.json` (aggregate metrics, the exact `RetrievalConfig`, embedder,
prompt version, and seed) and `rows.jsonl` (one line per question). Every number in this
README traces to those files.

### The CI regression gate

`make eval-gate` runs a separate, faster check that fails the build if retrieval quality drops
below a committed floor in `src/retrieval_engine/eval/baseline.json`.

It is deliberately **not** the published ablation, and the distinction matters. The gate
measures the 12-document fixture corpus with the deterministic fake embedder, because the test
suite may not touch the network. It catches pipeline regressions such as broken fusion or a
chunker that stops covering its input. It cannot catch a change in real-model retrieval
quality, which is what the ablation is for. One of its own tests asserts the run actually
retrieved something, because a gate that passes on an empty result is worse than no gate.

---

## Testing

```bash
make test        # full suite with the coverage gate
make test-fast   # no coverage, for iteration
make check       # lint, format check, mypy strict, then tests
```

| Property | Value |
|---|---|
| Tests | 555 passing, 1 deselected (requires Docker) |
| Coverage | 89.11 percent, gate at 85 |
| `mypy --strict` | Clean on `src/`, checked under both `--platform linux` and `--platform win32` |
| `ruff` | Clean, lint and format |
| `TODO` / `FIXME` / `NotImplementedError` in `src/` | Zero |

Four rules the suite holds to:

**No test needs the network, Postgres, Ollama, or a model download.** `HF_HUB_OFFLINE` is
forced on in `conftest.py`, so a test that reaches for a download fails loudly instead of
hanging in CI. Anything needing Docker is marked and excluded by default.

**The fakes are faithful, not inert.** The fake embedder hashes tokens into dimensions, so
vocabulary overlap really drives similarity, and it uses blake2b rather than the builtin
`hash`, which is salted per process and would make "deterministic" mean "deterministic until
you restart". Its width was chosen by measurement: at 32 dimensions collisions made an
unrelated document rank first, so it runs at 128.

**Properties, not just examples.** Hypothesis covers the chunker's coverage, token budget, and
id stability over generated inputs.

**Determinism is asserted, not hoped for.** Identical queries return byte-identical results,
ties break on `chunk_id` everywhere, and the eval gate asserts two runs produce identical
metrics.

That last rule earned its place. It caught a bug where the persisted BM25 index stored tokens
positionally while its staleness fingerprint was deliberately order-independent, so a reload
after a concurrent re-ingest could pair each chunk with **another chunk's tokens**. The index
is now keyed by chunk id and re-aligned on load.

---

## Project structure

```
retrieval-engine/
├── src/retrieval_engine/
│   ├── config.py              settings, every RE_ variable, seeding
│   ├── models.py              every Pydantic schema and enum
│   ├── errors.py              typed errors, each with a code and HTTP status
│   ├── logging_config.py      structlog JSON with a request-id contextvar
│   ├── service.py             the answer path, shared by the API and the eval harness
│   ├── ingest/
│   │   ├── loaders.py         pdf, md, html, txt, docx
│   │   ├── chunker.py         three strategies, one finaliser
│   │   └── pipeline.py        load, chunk, embed, upsert, with change detection
│   ├── embed/
│   │   ├── base.py            Embedder and Tokenizer protocols
│   │   ├── local.py           sentence-transformers, lazy load, thread-safe
│   │   └── openai.py          optional remote, httpx, no SDK
│   ├── store/
│   │   ├── base.py            VectorStore protocol, embedding-space guard, fingerprint
│   │   ├── memory.py          numpy, exact search, backs the test suite
│   │   └── pgvector.py        Postgres, HNSW, per-query SET LOCAL ef_search
│   ├── retrieve/
│   │   ├── dense.py           cosine over the store
│   │   ├── lexical.py         BM25 with a fingerprinted persisted index
│   │   ├── fusion.py          reciprocal rank fusion and weighted fusion
│   │   ├── rerank.py          cross-encoder with an LRU score cache
│   │   └── pipeline.py        the four stages, every one toggleable
│   ├── generate/
│   │   ├── base.py            LLM protocol
│   │   ├── prompts.py         the versioned answer template
│   │   ├── ollama.py          local generation over httpx
│   │   └── extractive.py      the no-LLM fallback
│   ├── guardrails/
│   │   ├── grounding.py       per-sentence verification, citation resolution
│   │   └── refusal.py         confidence-based abstention
│   ├── api/
│   │   ├── app.py             factory, one error handler, request ids, rate limiting
│   │   ├── deps.py            the service graph
│   │   ├── routes_query.py    query and SSE
│   │   ├── routes_ingest.py   directory and upload ingestion, jobs
│   │   └── routes_admin.py    documents, health, metrics
│   └── eval/
│       ├── metrics.py         every metric from scratch, plus the relevance predicate
│       ├── golden.py          loading and the exact-substring validator
│       ├── runner.py          the harness, index cached per chunking strategy
│       ├── report.py          markdown and JSON output
│       └── baseline.json      the committed CI floor
├── tests/                     unit, integration, and the eval gate
├── scripts/                   download_corpus, build_golden_set, run_eval, migrate
├── migrations/                ordered SQL, applied by compose or scripts/migrate.py
├── docs/
│   ├── architecture.md        module map, data model, request lifecycle
│   ├── evaluation.md          the measured results, discussed
│   ├── BUILD_SPEC.md          the specification this was built against
│   └── decisions/             three ADRs
├── ui/streamlit_app.py        demo client over HTTP
├── data/golden/               the committed golden set
├── eval_results/              ablation artifacts, committed
├── CLAUDE.md                  standing rules and build state
└── DEVIATIONS.md              every place this differs from the spec, and why
```

---

## What I would do differently at scale

Honest limits of these choices, not a wish list.

**Single-node pgvector has a ceiling.** HNSW index build time and memory grow with the corpus,
and at roughly 10 million chunks a single Postgres becomes the constraint. The fix is sharding
by collection or moving to a dedicated index, which trades the operational simplicity of one
datastore for horizontal room. That trade is only worth making once the corpus demands it.

**The BM25 index has no incremental update.** It rebuilds when the chunk-id fingerprint
changes, so adding one document re-tokenises the whole corpus. Fine at this size, wrong at ten
million chunks, where the index needs to live outside the process and support incremental
posting-list updates.

**Reranking dominates latency.** It is worth it when precision at the top matters and the
result feeds a human or a generator. It is not worth it for a bulk pipeline scoring millions of
pairs offline, or when the fused ranking is already good enough. The ablation is what tells you
which situation you are in.

**The golden set is 60 questions.** Enough to catch a regression and to make an ablation
directional. Not enough for a confident claim about a small difference between two rows.
Meaningful separation of near-neighbours needs several hundred questions and ideally more than
one annotator.

---

## Known limitations and unverified claims

Everything here is also in `DEVIATIONS.md`, which records each deviation as spec said, reality
is, what was done.

| Claim | Status |
|---|---|
| pgvector migration applies cleanly | **Verified** against pgvector/pgvector:pg16, and re-running is a no-op |
| pgvector live round trip | **Verified**, as a `@pytest.mark.docker` test excluded from the default run. The test uses its own collection and drops it, then asserts it is gone |
| `SET LOCAL hnsw.ef_search` reaches the planner | **Verified** |
| Docker image builds | **Verified** locally, and CI builds it on every push |
| Compose image pinned by digest | **Not done.** Pinned by tag |
| Multi-query and HyDE ablation rows | **Not measured.** No local model was installed, and the harness omits those rows rather than running them with expansion silently disabled, which would duplicate the row above while claiming to measure something |
| Generated answers (as opposed to extractive) | Code path tested against a fake and against a genuinely stopped server. Not measured end to end with a real Ollama model |
| Golden set human review | **Pending.** The set is model-authored and substring-validated |
| Dense against lexical retrieval quality | **Not measured.** The questions were written with their evidence in view, which favours BM25. Separating that bias from model quality needs a golden set authored without sight of the passages |
| Semantic chunking mechanism | The 2.02x build cost and the scores are measured. The explanation, that a document-relative percentile yields few boundaries on long papers, is a hypothesis: the chunk-size distribution was not recorded |

Two library behaviours worth knowing before reading the numbers:

**`rank_bm25` uses unsmoothed IDF**, `log(N - df + 0.5) - log(df + 0.5)`, so a term in half or
more of a small corpus scores zero or below. On a two-chunk index a term in one chunk scores
exactly zero. That is correct BM25, and it means the lexical-only row is a statement about real
corpus size rather than a broken retriever.

**Token counts are approximate under the optional remote embedder.** The OpenAI API does not
expose its tokenizer, so `ApproximateTokenizer` is used and chunk sizes are approximate in that
configuration. The local embedder is the default partly because its counts are exact.

---

## License

MIT. See [LICENSE](LICENSE).
