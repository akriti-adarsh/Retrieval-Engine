# Evaluation

Every number on this page came out of a command that ran, and each one can be traced to a
committed file under [`eval_results/`](../eval_results). Nothing here is estimated, rounded up
from a similar project, or copied from a paper.

The generated report is [`eval_results/ablation.md`](../eval_results/ablation.md) and the
machine-readable form is [`eval_results/ablation.json`](../eval_results/ablation.json). This
page is the part a generator cannot write: what the numbers mean, and where they should not
be trusted.

## The short version

Reranking is the only change in this project that produced a large, unambiguous improvement.
Everything else was small, mixed, or explainable by how the test set was built. The full
pipeline beats BM25 alone by 2.0 points of recall@5, which is a much less impressive headline
than "hybrid retrieval with reranking", and it is the honest one.

Three results here work against the design rather than for it, and none of them was removed:

- **Semantic chunking, the most expensive strategy, finished last.** It costs 2.02x the
  default to build and scored below it on every retrieval metric.
- **Fusing dense with lexical made retrieval worse** than lexical alone, 0.577 against 0.683,
  before reranking rescued it.
- **The dense-against-lexical gap is not a measurement.** The golden questions were written
  with their evidence in view, which favours BM25, so that row is reported as unknown.

---

## What was measured, and on what

| property | value | where it comes from |
|---|---|---|
| Corpus documents | 303 | `ablation.json` → `corpus_docs` |
| Corpus chunks | 16,318 structural, 9,727 fixed-token, 16,454 semantic | [`eval_results/index_build.md`](../eval_results/index_build.md) |
| Golden questions | 60 | `ablation.json` → `golden_questions` |
| Embedder | `BAAI/bge-small-en-v1.5` | `ablation.json` → `embedder` |
| Generator | `extractive` | `ablation.json` → `generator` |
| Seed | 42 | `ablation.json` → `seed` |

Chunk counts are given per strategy because they are a property of the strategy, not of the
corpus. `ablation.json` carries a single top-level `corpus_chunks` of 16,454, and
`scripts/run_eval.py` computes it as the **maximum** across every index the run built, which
is the semantic one. It should not be read as "the corpus has 16,454 chunks" for any
particular configuration. That is a reporting wart in the artifact, recorded here rather than
papered over.

The corpus is 303 recent arXiv `cs.CL` papers, harvested through OAI-PMH by
`scripts/download_corpus.py`. The generator is the extractive fallback rather than Ollama,
deliberately: the ablation is about retrieval, and a language model in the loop would add
sampling noise to rows that are supposed to differ by one retrieval setting.

### The golden set

60 questions in `data/golden/golden_set.jsonl`, validated by
`python scripts/build_golden_set.py --validate` (PASS, 0 failures, 0 warnings).

| category | n | what it tests |
|---|---|---|
| factual | 32 | a single passage answers it |
| multi_hop | 10 | the answer needs two passages |
| ambiguous | 8 | the question underspecifies, so a good system asks or hedges |
| negative | 10 | the corpus does not contain the answer, so the correct output is a refusal |

| difficulty | n |
|---|---|
| easy | 19 |
| medium | 28 |
| hard | 13 |

Evidence spans: 60 spans, mean 189 characters, longest 247. Negative questions carry no
evidence at all, which is enforced by a validator rather than by convention.

Retrieval metrics are averaged over the 50 answerable questions only. Including the 10
negatives would drag every retrieval score down by a fifth for questions that have no correct
chunk to retrieve, which would make the table look worse without measuring anything.

---

## How relevance is defined

This is the most consequential choice in the whole evaluation, so it is stated before any
results.

The golden set records **verbatim text spans**, not chunk ids. A retrieved chunk counts as
relevant if it **contains a golden span in full**, or **shares at least 50 consecutive
characters** with one.

The reason is that chunk ids are derived from `(doc_id, start_char)`, so they change whenever
the chunking strategy changes. Had relevance been recorded as ids, every chunking row would
have been scored against different ground truth while producing a table that looked perfectly
reasonable. The span rule keeps one fixed ground truth across all seven configurations.

The 50-character window is what lets a span split across a chunk boundary count for both
halves, which is exactly the case a chunking comparison turns on. `tests/unit/test_metrics.py`
splits a span deliberately and asserts both halves qualify.

The matcher is exact substring, applied with a sliding window. It is never fuzzy. Fuzzy
matching would make every metric on this page a matter of threshold choice.

---

## The metrics, defined exactly

All of these are implemented from scratch in `src/retrieval_engine/eval/metrics.py`, with no
retrieval-metrics library, so the definitions are visible and testable rather than inherited.

| metric | definition | note |
|---|---|---|
| `recall@k` | relevant chunks retrieved in the top k, divided by all relevant chunks for that question | 0.0 when a question has no relevant chunks, rather than undefined |
| `precision@k` | relevant chunks in the top k, divided by **k** | divided by k, not by the number returned, so returning 3 results is not rewarded for the 2 it did not return |
| `hit_rate@k` | fraction of questions with at least one relevant chunk in the top k | the loosest measure here |
| `MRR` | mean of 1/rank of the first relevant chunk | 0 for a question with no relevant chunk in the list |
| `nDCG@k` | discounted cumulative gain over binary relevance, divided by the ideal | the ideal is extended to the true number of relevant chunks, so a question with 3 relevant chunks cannot score 1.0 from finding 1 |
| `refusal_accuracy` | fraction of questions where refusing or answering was the correct call | reported next to false refusal, always |
| `false_refusal_rate` | fraction of **answerable** questions that were refused | the number that stops refusal accuracy from being gamed |
| `citation_precision` | fraction of citations pointing at a chunk that is relevant | an answer with no citations scores 0.0, not 1.0 |
| `answer_similarity` | cosine between the generated answer and the reference answer | the weakest metric here, see the caveat below |
| `grounded_rate` | fraction of answers where every sentence cleared the grounding threshold | near-tautological for extractive answers, see below |

Two of these are deliberately paired. **Refusal accuracy alone is meaningless**: a system that
refuses every single question scores 1.000 on it. That is why `false_refusal_rate` sits next
to it in every table, including the generated one.

---

## Results

Seven configurations, one corpus, one golden set, one seed. Full per-row detail including
per-category and per-difficulty breakdowns is in
[`eval_results/ablation.md`](../eval_results/ablation.md).

| config | recall@5 | precision@5 | hit@5 | MRR | nDCG@5 | refusal acc | false refusal | p95 |
|---|---|---|---|---|---|---|---|---|
| dense only | 0.400 | 0.088 | 0.420 | 0.302 | 0.318 | 0.000 | 0.000 | 3.5 s |
| lexical only | 0.683 | 0.168 | 0.740 | 0.553 | 0.568 | 0.000 | 0.000 | 7.0 s |
| hybrid (RRF) | 0.577 | 0.136 | 0.620 | 0.428 | 0.460 | 0.000 | 0.000 | 5.2 s |
| **hybrid + rerank** (default) | **0.703** | 0.176 | 0.760 | 0.589 | **0.603** | 1.000 | 0.140 | 21.6 s |
| hybrid (weighted) + rerank | 0.703 | 0.176 | 0.760 | 0.565 | 0.585 | 1.000 | 0.140 | 11.8 s |
| hybrid + rerank, fixed-token chunking | 0.670 | 0.212 | 0.760 | 0.594 | 0.584 | 0.700 | 0.180 | 18.7 s |
| hybrid + rerank, semantic chunking | 0.672 | 0.164 | 0.740 | 0.552 | 0.562 | 0.900 | 0.120 | 14.9 s |

Total wall clock for the matrix: **2.15 hours** on CPU. Rows sharing a chunking strategy share
one cached index, so the first row using each strategy pays to build it.

Two rows the spec's matrix asks for are absent: `hybrid + rerank + multi-query`, and HyDE.
Both need a language model to write the expansions, and no Ollama is installed on this
machine. The harness omits them rather than running them with expansion silently disabled,
which would publish a copy of the `hybrid + rerank` row under a label claiming a component was
active. `DEVIATIONS.md` entry 14 records this.

---

## Reading the results

### Reranking is the only large win

Adding the cross-encoder to hybrid retrieval moved recall@5 from 0.577 to 0.703 and nDCG@5
from 0.460 to 0.603. That is the largest single improvement in the table, by a wide margin,
and it is the one result here that is not sensitive to how the golden set was written: the
reranker sees the same candidate list either way and simply orders it better.

It is also the only configuration that can refuse at all, which is discussed below.

The cost is severe. Reranking is about 70 percent of end-to-end p95 latency on CPU.

### Lexical beating dense is probably an artifact of how the test set was built

Lexical-only scored 0.683 recall@5 against dense-only at 0.400. A 28 point gap in favour of
BM25 over a modern embedding model should not be reported as a finding without saying why it
is suspicious.

Every golden question was written with its evidence passage in view. The set is
model-generated and substring-validated against the corpus, with human review still pending,
which the README states in the same words. Even when the instruction is to paraphrase, a
question written that way reuses the distinctive vocabulary of its source passage, and
distinctive shared vocabulary is exactly what BM25 scores. The test set is therefore biased
toward exact-token overlap, which hands BM25 an advantage no real user query would.

A second explanation cannot be ruled out with this data: `bge-small` is a 384-dimensional
model, and dense academic text across 16,318 chunks is a hard setting for it. Both
explanations predict the same table.

Separating them needs a second golden set authored without sight of the passages, which has
not been built. Until then, **the correct reading of this row is "unknown", not
"BM25 wins"**, and any conclusion that depends on the dense/lexical ordering should be
treated as unsupported.

### Fusing a weak list with a strong one made things worse

Hybrid RRF scored 0.577, below lexical-only at 0.683. Fusion lost 10 points against simply not
fusing.

This follows directly from what RRF does. It scores by rank position only, discarding the
scores underneath, and it gives every input list an equal vote. When one list is much better
than the other, an equal vote drags good results down with mediocre ones. RRF's robustness to
incomparable score scales is the same property that makes it unable to notice that one of its
inputs is weak.

Reranking then recovers the loss and more (0.703), because the cross-encoder reads the actual
query and passage text and does not care what rank fusion assigned.

### The full pipeline beats BM25 alone by 2.0 points, and that is the honest headline

Hybrid with reranking reached recall@5 0.703 against lexical-only 0.683. On nDCG@5 the gap is
larger (0.603 against 0.568) and on MRR similar (0.589 against 0.553), so the ordering
improves more than the retrieval does.

For a project that exists to demonstrate a hybrid pipeline, "2.0 points of recall over BM25"
is a modest result, and it is stated here rather than buried because the alternative is
publishing an architecture diagram that implies more than the measurement supports.

The pipeline does buy one thing BM25 alone cannot provide at any threshold: a calibrated score
that supports refusal. That is a capability difference, not a metric difference.

### RRF against weighted fusion is close to a tie

With reranking on both, recall@5 is identical at 0.703. RRF is slightly better on ordering
(nDCG 0.603 against 0.585, MRR 0.589 against 0.565).

RRF stays the default, but on this evidence the honest statement is that the fusion method
barely matters once a reranker is present, because the reranker reorders the candidates
anyway. Fusion's job in that configuration is recall of the candidate pool, not ordering.

### The most expensive chunking strategy lost

Three chunking strategies, all with hybrid retrieval and reranking, so chunking is the only
variable. Build cost is from [`eval_results/index_build.md`](../eval_results/index_build.md).

| strategy | recall@5 | nDCG@5 | MRR | precision@5 | chunks | build |
|---|---|---|---|---|---|---|
| `recursive_structural` (default) | **0.703** | **0.603** | 0.589 | 0.176 | 16,318 | 26.5 min |
| `fixed_token` | 0.670 | 0.584 | **0.594** | **0.212** | 9,727 | 20.3 min |
| `semantic` | 0.672 | 0.562 | 0.552 | 0.164 | 16,454 | 53.5 min |

**Semantic chunking is the most sophisticated strategy here and it came last.** It embeds
every sentence to find topic boundaries by cosine distance, costs 2.02x structural chunking to
build, and scored below it on recall, nDCG, MRR, and precision alike. If any result in this
project deserved to be quietly dropped, it is this one, which is exactly why it is here.

The 2.02x figure is worth pausing on, because ADR 003 predicted before any of this ran that
semantic chunking would "roughly double ingestion cost". Both builds embedded essentially the
same text (4,224,502 tokens against 4,224,426), so the extra hour bought nothing but the
boundary search.

There is a plausible mechanism, offered as a hypothesis rather than a measurement: the semantic
threshold is a percentile of the distances within a document, so it is document-relative, and
these are long papers. It produced 16,454 chunks against structural's 16,318, which is a
1 percent difference in granularity for twice the price. ADR 003 called this consequence out
in advance too. Confirming it would need the chunk-size distribution per strategy, which was
not recorded, so the mechanism stays a hypothesis and only the cost and the scores are claims.

**Structural against fixed-token is genuinely mixed**, and reporting it as a clean win would
be wrong. Structural takes recall@5 (0.703 against 0.670) and nDCG@5 (0.603 against 0.584).
Fixed-token takes precision@5 (0.212 against 0.176) and MRR (0.594 against 0.589). That shape
is consistent: uniform 512-token windows are more likely to contain a whole evidence span, so
each returned chunk is more often relevant, while structural sections of uneven size cover more
of the span set overall.

Structural stays the default on recall, nDCG, and the fact that its `section_path` is true of
the whole chunk, which is what a citation shows a reader. Fixed-token also had the worst
refusal behaviour of the three (0.700 accuracy at 0.180 false refusal).

### Refusal only works when a reranker is present, by design

Refusal accuracy is 0.000 on every row without reranking. That is not a broken guardrail, it
is the guardrail declining to run.

RRF scores cluster around 0.03 and carry no meaning on an absolute scale, so thresholding them
at 0.3 would refuse every question ever asked. The pipeline therefore sets
`threshold_applied=False` when no calibrated score exists, and the harness records refusal
accuracy as 0.000 rather than inventing a comparison.

Only the cross-encoder produces a score with a stable interpretation, and the evidence for
that is now recorded per question rather than asserted. Every eval row carries `top_score`,
the exact value the threshold is compared against. Across all 60 golden questions
([`eval_results/02042ae6-recursive_structural/rows.jsonl`](../eval_results/02042ae6-recursive_structural/rows.jsonl)):

| outcome | n | top_score range |
|---|---|---|
| answered | 43 | 0.4138 to 0.9996 |
| refused | 17 | 0.0001 to 0.2878 |

**Nothing at all falls between 0.2878 and 0.4138.** The configured threshold of 0.3 sits
inside that empty gap, so it is not a tuned number: anywhere in that interval produces exactly
the same decisions on this set. The threshold was originally justified by a distribution
observed by hand over six queries, which was a claim rather than a measurement, and recording
`top_score` is what turned it into one.

With reranking on, refusal accuracy is 1.000 **and** false refusal is 0.140. Both numbers are
needed: the first alone would also be produced by a system that refuses everything, and the
second shows it does not. All 10 negative questions were correctly refused, and 7 of the 50
answerable ones were refused when they should not have been.

One of those false refusals was traced by hand. The relevant chunk scored 0.2655, just under
the threshold, because it was a long bullet list mixing several topics and the reranker scored
the chunk as a whole. That is a chunking problem, and it is recorded as one. Lowering the
threshold to make it disappear would have traded a real defect for a worse one.

### Citation precision is low, and the reason is mechanical

Citation precision ranges from 0.210 to 0.337 across the table. Against recall@5 of about 0.70
that looks inconsistent, and the explanation is in the definition.

The extractive generator cites the passages it quotes, which are the top-ranked ones. A
citation counts as precise only when the cited chunk is relevant under the span rule. So
citation precision is roughly measuring precision at the very top of the list, not at 5, and
`precision@5` in the same table is 0.176 for the same row. Read that way the numbers agree.

Answers with no citations score 0.0 rather than being excluded, which pulls the average down
further and is the conservative choice.

### Grounded rate is close to a tautology here

Grounded rate sits at 0.960 to 0.980 everywhere. It should not be read as evidence that the
system does not hallucinate.

The ablation runs the extractive generator, whose output is quoted verbatim from retrieved
passages. Verifying that quoted text is similar to the passage it was quoted from is close to
a self-fulfilling check. The number is reported because it confirms the grounding path runs
and does not crash, and that is all it confirms.

Grounding is a meaningful measurement only against generated output, which means running the
ablation with Ollama. That is a documented gap rather than a claim.

### Answer similarity is the weakest number on the page

Values sit between 0.672 and 0.716 with very little spread. Cosine similarity between two
embeddings of short text is high for almost any pair of on-topic sentences, so this metric
barely separates the configurations. It is reported for completeness and should not be used
to rank anything.

---

## By category and by difficulty

Taken from the `hybrid + rerank` row, which is the default configuration.

| category | recall@5 | nDCG@5 | MRR | refused |
|---|---|---|---|---|
| factual (32) | 0.818 | 0.710 | 0.688 | 0.031 |
| multi_hop (10) | 0.600 | 0.442 | 0.445 | 0.300 |
| ambiguous (8) | 0.375 | 0.375 | 0.375 | 0.375 |
| negative (10) | n/a | n/a | n/a | 1.000 |

| difficulty | recall@5 | nDCG@5 | MRR |
|---|---|---|---|
| easy (9) | 0.963 | 0.838 | 0.806 |
| medium (28) | 0.625 | 0.543 | 0.527 |
| hard (13) | 0.692 | 0.571 | 0.573 |

The difficulty counts here (9 easy, 28 medium, 13 hard) sum to 50 rather than 60, because the
10 negatives are excluded from retrieval metrics and 10 of the easy questions are negatives.

Three things are worth naming:

**Ambiguous questions are the weakest category by a distance**, at 0.375 across every metric.
That is the expected shape of the problem. An underspecified question has no single passage
that answers it, so a retriever that returns one confident chunk is wrong regardless of which
chunk it picks. 3 of the 8 were refused, which is arguably the better outcome for that
category, but the harness scores those as false refusals because the questions are marked
answerable.

**Multi-hop sits between the two**, at 0.600. Single-vector retrieval finds one hop well and
the second hop only when it happens to share vocabulary with the first.

**Hard scores above medium** (0.692 against 0.625), which inverts the labels. The difficulty
labels are a judgement recorded while writing the questions, and this says they do not track
what the retriever finds difficult. The labels are kept as recorded rather than quietly
adjusted to match the results, since relabelling after seeing the scores is how a test set
stops measuring anything.

---

## Latency

Measured on CPU, on a laptop, with no GPU anywhere in the stack. p95 over 60 questions, from
the `hybrid + rerank` row of `ablation.json`.

| stage | p50 | p95 |
|---|---|---|
| expansion | 0.006 ms | 0.010 ms |
| fusion | 1.2 ms | 2.8 ms |
| rerank | 10,031 ms | 15,167 ms |
| retrieval (whole block) | 10,602 ms | 15,696 ms |
| generation (extractive) | 1,385 ms | 6,244 ms |
| grounding | 1,170 ms | 3,481 ms |
| **total** | **13,317 ms** | **21,644 ms** |

These are slow, and the shape is the interesting part: **the cross-encoder is about 97 percent
of retrieval time and about 70 percent of the total**. Reranking 60 candidate passages means
60 full transformer forward passes with no batching win to be had on CPU.

That is the trade this project makes explicit. Reranking bought the largest quality
improvement in the table and it costs an order of magnitude in latency. On this hardware the
service is a demonstration, not something to put in front of users at 20 seconds per query.
The first fix would be a GPU for the cross-encoder, and the second would be reranking 20
candidates instead of 60, which is a `top_k_rerank` change and would need re-measuring rather
than assuming.

### The dense against lexical split, from a separate clean run

`ablation.json` cannot answer this. Until commit `3a3267a` the pipeline timed the block
containing both concurrent branches and wrote that one figure into both fields, which made
them identical to thirteen decimal places in every row. The whole matrix was measured before
the fix, so the split had to come from re-running the default configuration afterwards:
[`eval_results/eval_default.json`](../eval_results/eval_default.json).

| stage | p50 | p95 |
|---|---|---|
| dense | 344.4 ms | 704.5 ms |
| lexical | 146.2 ms | 329.2 ms |

**Dense retrieval costs about 2.4x lexical**, which is the shape you would expect and is
nothing like the equality the broken instrumentation implied. The two run concurrently, so
they overlap and do not sum to the retrieval block.

### The same configuration, measured twice, differs by 27 percent

That re-run is worth more than the split it was for, because it accidentally measured
something else. It ran the identical configuration, corpus, golden set, and seed:

| | ablation run | clean re-run |
|---|---|---|
| recall@5, nDCG@5, MRR, citation precision | 0.703 / 0.603 / 0.589 / 0.320 | identical to four decimals |
| total p95 | 21,644 ms | 15,744 ms |
| rerank p95 | 15,167 ms | 11,355 ms |

**Every quality metric reproduced exactly. Latency did not, by 27 percent.** The difference is
machine load: the ablation was a two-hour job building indexes and running other rows, and it
shared the machine with the development work that was happening alongside it.

So the quality numbers on this page are reproducible in the strong sense, and the latency
numbers are accurate for the run that produced them and should be read as an order of
magnitude rather than to the millisecond. The p95 figures in the results table all come from
the same ablation, so they remain comparable against each other, which is what the table is
for.

Expansion reads as 0.006 ms because the ablation runs with `expansion=none`. It is the cost of
returning the query unchanged, not the cost of multi-query or HyDE, both of which require a
language model round trip.

Row wall-clock times in `ablation.json` (`elapsed_seconds`) include index construction for the
first row using each chunking strategy, so they are not per-query costs and should not be read
as such.

---

## Threats to validity

Stated plainly, because a results page without this section is advertising.

| # | Threat | Effect |
|---|---|---|
| 1 | The golden questions were written while reading their evidence passages | Biases the whole evaluation toward lexical overlap. The dense/lexical ordering should be treated as unmeasured. |
| 2 | 60 questions, of which 50 are answerable | Differences of 1 to 2 points are inside the noise. The 2.0 point gap between the full pipeline and BM25 alone is not a reliable ranking. |
| 3 | One author wrote the questions, the answers, and the system | No independent judgement anywhere in the loop. |
| 4 | Grounding is measured against extractive output | The 0.98 grounded rate is close to self-fulfilling and is not evidence about hallucination. |
| 5 | One corpus, one domain | 303 arXiv `cs.CL` papers. Nothing here transfers to a corpus of support tickets or contracts without re-measuring. |
| 6 | Difficulty labels do not predict measured difficulty | Hard scored above medium, so the per-difficulty breakdown describes the labelling, not the retriever. |
| 7 | Single seed | Everything is seeded at 42. Quality metrics reproduce exactly across processes, confirmed by re-running the default config, but variance across seeds is unmeasured. |
| 8 | Latency was measured under varying machine load | The same configuration re-measured on a quieter machine was 27 percent faster. p95 figures are comparable within the table and should not be read to the millisecond. |

The first two matter most. Together they mean this evaluation is good enough to justify a
design decision the size of "use a reranker" and not good enough to justify one the size of
"BM25 beats dense embeddings".

---

## The regression gate

`tests/eval/` runs on every push and fails the build if retrieval quality drops below a
committed floor. It uses the deterministic fake embedder, so it runs in CI in seconds with no
model download and no network.

The floor is recorded in a committed baseline file, and the gate compares against it rather
than against a number typed into a test. Regenerate it deliberately with
`RECORD_EVAL_BASELINE=1 pytest tests/eval`.

The threshold in the gate was recalibrated once during the build, and the reason is worth
recording. The gate originally ran at `min_confidence=0.01`, at which nothing is ever refused.
Its negative questions were decorative and its refusal accuracy was a constant 0.000: the gate
would have passed a build in which refusal was deleted outright. It now runs at 0.25, matching
the stub scorer's actual output scale, so the refusal path is genuinely exercised.

---

## Reproducing this

```bash
make corpus            # 303 arXiv cs.CL papers via OAI-PMH, 3s delay between requests
make golden-validate   # 60 entries, exact-substring span check
make eval              # the default configuration
make eval-ablate       # every row in the table above, 2.15 hours measured on CPU
```

The ablation took 2.15 hours on CPU, and most of that is index building plus the
cross-encoder. Row order
matters for wall clock but not for results: the first row using each chunking strategy pays
for building that index and the rest reuse it.

Everything is seeded at 42. Re-running against the same corpus reproduces the same numbers,
which is the property that makes a regression gate possible at all.
