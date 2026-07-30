# ADR 001: Hybrid retrieval over dense-only, fused by reciprocal rank

Status: accepted. The measured comparison is in [../evaluation.md](../evaluation.md) and
`eval_results/`; this document records the reasoning, not the numbers.

## Context

A dense bi-encoder embeds the query and each passage independently and compares them by
cosine similarity. That is cheap at query time, because passage vectors are computed once at
index time, and it handles paraphrase well: a question worded differently from the passage
still lands nearby in embedding space.

It has a specific weakness. A bi-encoder compresses a passage into a few hundred floats, and
rare exact tokens survive that compression poorly. A query naming a specific identifier,
model name, version number, or dataset is exactly where embeddings blur, and exactly where a
user is most confident they typed the right thing.

BM25 has the mirror-image profile. It is precise on rare exact tokens, because its inverse
document frequency term makes a rare term dominate the score, and it is blind to rephrasing,
because it has no notion that two different words mean the same thing.

Two retrievers only help if they fail differently. These do.

## Decision

Run both retrievers on every query and fuse their ranked lists with reciprocal rank fusion:

```
score(d) = sum over lists of 1 / (k + rank(d))     with k = 60
```

Weighted score fusion (min-max normalise each list, then take a weighted sum) is also
implemented and selectable, so the ablation can measure the difference rather than assert it.

### Why rank-based fusion rather than score normalisation

Cosine similarity lives roughly in [0, 1]. BM25 scores are unbounded and depend on corpus
statistics. To combine them by score, the two have to be made comparable, and every way of
doing that introduces a new problem:

- **Min-max normalisation** maps the worst entry of each list to exactly zero. A passage with
  a real cosine of 0.1 and a passage with a real BM25 score of 2.0 both become 0.0, so genuine
  signal is discarded at the bottom of each list. It is also sensitive to a single outlier at
  the top, which compresses everything below it.
- **Fixed scaling factors** need re-tuning whenever the corpus, the embedder, or the analyzer
  changes, and nothing warns you when they go stale.

RRF sidesteps this entirely by consuming only the ordering. It is scale free, needs no
calibration between the retrievers, and degrades gracefully when one retriever's score
distribution shifts, because only the order it produces is used.

### What k controls

`k` sets how sharply top ranks dominate. Small `k` makes rank one nearly decisive; large `k`
flattens the curve until fusion is effectively counting how many lists a document appears in.
Both ends are pinned by unit tests in `tests/unit/test_fusion.py`, which show the same two
lists producing opposite winners at `k = 1` and `k = 1000`.

`k = 60` is the value from the original formulation and is the default here. It is a config
value, not a constant, so the ablation can move it.

## Consequences

**Cost.** Every query runs two retrievers instead of one. Dense search is a vector operation;
BM25 is a scan over the term statistics. They run concurrently, so wall-clock cost is roughly
the slower of the two rather than the sum, but the CPU cost is genuinely doubled.

**A second index to keep current.** The BM25 index has to track the chunk set, which is why
it carries a SHA256 fingerprint over the sorted chunk-id list and rebuilds when that changes.
That is extra machinery a dense-only system would not need.

**Memory.** The lexical retriever holds the chunk set in process. That is honest for a
single-node corpus of this size and is the first thing that has to change for a very large
one.

**A library behaviour that shapes how the results read.** `rank_bm25` computes IDF as
`log(N - df + 0.5) - log(df + 0.5)` with no smoothing term, so a term occurring in half or
more of a small corpus gets an IDF of zero or below and contributes nothing. On a two-chunk
index a term in one chunk already scores exactly zero. That is correct BM25 rather than a
bug, and it means the lexical-only row of the ablation is a statement about corpus size as
much as about the retriever. This is recorded in `DEVIATIONS.md` and pinned by a test so it
cannot surprise a later reader.

## Alternatives considered

**Dense only.** Simplest, one index, no fusion stage. Rejected because the failure mode is
silent: a query for an exact rare token returns plausible neighbours rather than the passage
that contains it, and nothing in the response indicates that happened.

**Lexical only.** Cheap, interpretable, no model. Rejected because it cannot handle a
question worded differently from the source, which is the normal case for a natural-language
question over technical prose.

**Learned sparse retrieval (SPLADE and similar).** Attractive, because it gets term-level
interpretability with learned expansion, arguably subsuming the reason to run two retrievers.
Rejected for this project because it needs a second trained model and its own index format,
which is a large increase in operational surface for a benefit this corpus size cannot
demonstrate.

**Fusing more than two lists.** The implementation already supports it, and multi-query
expansion uses it (one list per paraphrase). Not enabled by default, because each extra list
costs a retrieval round and the ablation is the right place to decide whether it pays.
