# Index build cost per chunking strategy

Primary output, not a summary. These are the structlog lines emitted by `make eval-ablate`
on 2026-07-30, copied verbatim. The ablation caches one index per chunking strategy and
reuses it across the rows that share it, so each strategy appears once.

They are recorded here because `ablation.json` has a single top-level `corpus_chunks` field,
which holds whatever index was built last (the semantic one, 16454) rather than a property of
the corpus. Chunk counts differ per strategy and belong per strategy.

| strategy | chunks | tokens embedded | wall clock | throughput |
|---|---|---|---|---|
| `recursive_structural` (default) | 16,318 | 4,224,426 | 26.5 min | 2,654 tok/s |
| `fixed_token` | 9,727 | 4,815,270 | 20.3 min | 3,953 tok/s |
| `semantic` | 16,454 | 4,224,502 | 53.5 min | 1,316 tok/s |

Three things this measures, all on CPU with no GPU:

1. **Semantic chunking costs 2.02x structural chunking** on essentially identical token counts
   (4,224,502 against 4,224,426, the same text either way). ADR 003 predicted it would
   "roughly double ingestion cost" because every sentence is embedded to find boundaries
   before the resulting chunks are embedded. That prediction is now measured rather than
   asserted.
2. **Fixed-token chunking embeds 14 percent MORE tokens yet finishes fastest.** The overlap
   duplicates text, so there is more to embed, but it produces 40 percent fewer chunks and
   they are uniformly sized, which batches far better than many short uneven ones.
3. **Semantic chunking produced 16,454 chunks against structural's 16,318**, so on documents
   this long it did not yield meaningfully different granularity for twice the price.

## Verbatim log lines

```
11:25:56 [info     ] ingest_finished                chunks_created=16318 docs_changed=303 docs_seen=303 docs_unchanged=0 summary='303 changed, 0 unchanged, 16318 chunks created, 0 skipped, 4224426 tokens embedded in 1591.96s'
11:25:58 [info     ] bm25_index_rebuilt             chunks=16318 fingerprint=d9572aacbf1d
12:02:11 [info     ] ingest_finished                chunks_created=9727 docs_changed=303 docs_seen=303 docs_unchanged=0 summary='303 changed, 0 unchanged, 9727 chunks created, 0 skipped, 4815270 tokens embedded in 1218.08s'
12:02:13 [info     ] bm25_index_rebuilt             chunks=9727 fingerprint=57426fd75d3d
13:02:45 [info     ] ingest_finished                chunks_created=16454 docs_changed=303 docs_seen=303 docs_unchanged=0 summary='303 changed, 0 unchanged, 16454 chunks created, 0 skipped, 4224502 tokens embedded in 3209.73s'
13:02:46 [info     ] bm25_index_rebuilt             chunks=16454 fingerprint=91a2fca0b290
```
