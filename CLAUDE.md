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
- Plan position: spec commits 1-7 and 9-14 done; commit 8 (pgvector) deferred, see
  DEVIATIONS 9. Last completed: "feat(generate): prompts, ollama client, extractive fallback"
- Suite at last commit: 395 passed in 15.50s · Coverage: 95%
- Open deviations: 12 · Next up: commit 15 (grounding verification and refusal), then commits
  16-17 (the API). Commit 8 (pgvector) still waiting on Docker.
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
