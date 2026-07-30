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
- Plan position: spec commits 1-19, 21, 22 landed (22 commits). Remaining: 20 (golden set data),
  23 (Streamlit UI), 24 (Dockerfile and full CI), 25 (architecture doc and three ADRs), 26
  (README numbers). Last completed: "docs: comprehensive README, plus eval Makefile targets"
- NEXT SESSION, in this order, because each unblocks the next:
  1. Author `data/golden/golden_set.jsonl`, 60 entries, and get
     `make golden-validate` to print PASS. Spans MUST be copied verbatim out of
     `data/corpus/*.md` body text (front matter is stripped by the loader) and stay inside one
     line, since a span crossing a line break must match the break exactly. Keep them 80 to 250
     chars. 10 negatives with empty relevant_doc_ids and relevant_chunk_texts.
  2. `make eval-ablate`. Measured throughput is 2,435 tok/s, so budget about 30 min per
     chunking index and about 90 min for the seven rows. It rewrites the report after every
     row, so an interrupted run still leaves real numbers on disk.
  3. Fill the README's Results section from `eval_results/ablation.md`, and write
     `docs/evaluation.md`. Delete the "ablation has not been run" note only once the numbers
     are in a committed artifact.
  4. Commits 23, 24, 25 (UI, containers, architecture doc and ADRs). The README's Build status
     table lists each as pending; update it as they land.
- Blocked, not forgotten: three pgvector claims and the Docker image build need a working Docker
  daemon (DEVIATIONS 9). Multi-query and HyDE ablation rows need a local Ollama; the harness
  omits them rather than running expansion silently disabled.
- Do NOT put a number in the README that is not in a committed artifact. The Results section is
  deliberately empty rather than estimated, and that is the point of the whole harness.
- Suite at last commit: 500 passed, 1 deselected in 20.43s · Coverage: 91%
- Open deviations: 14 · Next up: commits 18-22, the eval harness. That is the part that makes
  this repo credible, and it is also where the two open blockers bite: the ablation needs the
  real corpus, which needs the arXiv OAI-PMH rewrite (DEVIATIONS 2).
- Note: the SSE route currently streams a fully-computed answer in pieces, so the client
  contract is identical whether or not the backend streamed. Wiring LLM token-by-token
  streaming through it is a separate change and must keep the refusal and extractive paths
  streaming, since those have no token stream at all.

## Session B boundary check: PASSED
Run against a live uvicorn on 127.0.0.1:8077 with `RE_STORE=memory`, `RE_LLM=ollama`, and
Ollama genuinely not running (`/api/tags` timed out). Corpus was this repo's own four markdown
docs, ingested through the API.

- `POST /v1/ingest {"path":"corpus"}` gave `4 docs, 44 chunks, 13578 tokens in 12.1s`
- `GET /health/ready` gave `ready=true, store=true, embedder=true, generator=false` with
  detail "model server unreachable, answers will be extractive". Generation is excluded from
  the readiness verdict on purpose, so a dead model server cannot pull the service out of
  rotation.
- `POST /v1/query` with the model server down returned **HTTP 200, answer_type=extractive,
  model=extractive**, citations `[1]` and `[2]` both `resolved=true`, `grounded=true` with 0
  flagged sentences. No 500 anywhere. This is the gate the spec asks for.
- `POST /v1/query/stream` emitted `event: token` frames with JSON payloads (verified the em
  dash arrived as `—`, so the JSON encoding is doing its job).
- The reranker LRU works: an identical repeat query came back with `rerank_ms=0.0`.

### Measured reranker score distribution, and why min_confidence=0.3 stands
Six real queries against the real `bge-reranker-base`:

| top score | outcome | query |
|---|---|---|
| 0.9711 | extractive | how is reciprocal rank fusion scored |
| 0.8819 | extractive | what does the k parameter do in RRF |
| 0.8285 | extractive | which chunking strategies are compared |
| 0.8135 | extractive | what is the grounding threshold default |
| 0.2655 | refused | what line coverage does the build require |
| 0.0000 | refused | who won the world cup in 1998 |

The distribution is sharply bimodal: genuinely relevant chunks land at 0.81 to 0.97, and an
out-of-corpus question scores 0.0000. The spec's 0.3 default sits in the empty gap between
those modes, so it is well chosen and should NOT be tuned to make a demo look better.

The 0.2655 row is a real false refusal and worth understanding rather than hiding. Retrieval
was correct: the top chunk is the one containing "Target >=85% line coverage". The chunk is a
long mixed bullet list, and a cross-encoder dilutes across a passage that is mostly about
other things. That is a chunking problem, not a threshold problem, which is exactly what the
chunking ablation is built to measure. Do not "fix" it by lowering min_confidence; let the
golden set's refusal_accuracy metric settle it with data.
- Coverage note: it fell from 95% to 91% because pgvector.py's database code cannot run
  without Docker. That is the honest number, not a regression in test quality, and it should
  climb back once the docker-marked tests can run. Do not "fix" it by deleting assertions.
- Reminder for whoever gets Docker working: three pgvector claims are still unverified, listed
  in DEVIATIONS 9. Verify them before the README describes pgvector as working.
- Note for the API work: the refusal policy only thresholds a reranker score. If the API is
  ever wired to run without reranking, RefusalDecision.threshold_applied is False and the
  confidence field is a fused score, not a probability. Do not surface it as a confidence.
- Session A boundary check PASSED with the real model, not the fake: ingesting
  docs/BUILD_SPEC.md + README.md + CLAUDE.md gave "3 changed, 36 chunks created, 10316 tokens
  embedded in 18.95s", and the immediate re-run gave "0 changed, 3 unchanged, 0 chunks
  created, 0 tokens embedded". A real query for "what line coverage percentage does the build
  require" retrieved the section that states the 85% target, at 0.683 cosine.
- Notes for next session:
  - Do not trust anything in `.drafts/`: read `.drafts/README.md`, then either write real tests
    for a draft and verify it, or delete it and write the module fresh. The two module drafts
    there (`embed/local.py`, `store/memory.py`) have never been imported even once. Loaders were
    written fresh rather than recovered from a draft, which is the pattern to follow.
  - Build order matters here: the chunker needs the `Tokenizer` protocol from `embed/base.py`,
    so embedders come before the chunker regardless of the plan's numbering.
  - Both models are already in the local HuggingFace cache, so no download is needed:
    `bge-small-en-v1.5` (384 dim, tokenizer confirmed fast, so `return_offsets_mapping` works and
    the chunker can slice on exact token offsets) and `bge-reranker-base` (sanity check gave
    0.9997 for a relevant pair against 0.00004 for an irrelevant one).
  - arXiv: use OAI-PMH, not the Search API. See DEVIATIONS entry 2.
  - Docker daemon is not running on this machine, so the pgvector backend and compose stack are
    still entirely unvalidated. The default suite needs no database by design, so this does not
    block commits 4 to 7, but it does block commit 8's acceptance.
  - Local python is 3.11.14 via uv, matching the CI matrix floor. CI is green on all three
    pushed commits.
