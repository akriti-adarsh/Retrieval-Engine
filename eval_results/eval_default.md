# Ablation results

Generated 2026-07-30T20:16:40.998867+00:00

| property | value |
|---|---|
| corpus documents | 303 |
| corpus chunks | 16318 |
| golden questions | 60 |
| embedder | `BAAI/bge-small-en-v1.5` |
| generator | `extractive` |
| prompt version | `v1` |
| seed | 42 |

## Headline comparison

| config | recall@5 | nDCG@5 | MRR | refusal acc | false refusal | p95 latency |
|---|---|---|---|---|---|---|
| hybrid + rerank | 0.703 | 0.603 | 0.589 | 1.000 | 0.140 | 15744 ms |

`false refusal` is the fraction of answerable questions that were refused. It is
reported next to refusal accuracy because a system can score perfectly on one by
failing the other: refusing everything gives refusal accuracy 1.000.

## hybrid + rerank

Run `02042ae6-recursive_structural`, 60 questions, 1639.2s wall clock.

### Metrics

| metric | value |
|---|---|
| answer_similarity | 0.709 |
| citation_precision | 0.320 |
| false_refusal_rate | 0.140 |
| grounded_rate | 0.980 |
| hit_rate@5 | 0.760 |
| mrr | 0.589 |
| ndcg@5 | 0.603 |
| precision@5 | 0.176 |
| recall@5 | 0.703 |
| refusal_accuracy | 1.000 |

### By category

| category | n | recall@5 | nDCG@5 | MRR |
|---|---|---|---|---|
| ambiguous | 8 | 0.375 | 0.375 | 0.375 |
| factual | 32 | 0.818 | 0.710 | 0.688 |
| multi_hop | 10 | 0.600 | 0.442 | 0.445 |
| negative | 10 | 0.000 | 0.000 | 0.000 |

### By difficulty

| difficulty | n | recall@5 | nDCG@5 | MRR |
|---|---|---|---|---|
| easy | 9 | 0.963 | 0.838 | 0.806 |
| hard | 13 | 0.692 | 0.571 | 0.573 |
| medium | 28 | 0.625 | 0.543 | 0.527 |

### Stage latency

| stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| expansion | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| dense | 344.4 ms | 704.5 ms | 392.6 ms | 984.0 ms |
| lexical | 146.2 ms | 329.2 ms | 164.0 ms | 579.3 ms |
| fusion | 1.1 ms | 2.3 ms | 1.4 ms | 5.1 ms |
| rerank | 7703.6 ms | 11355.1 ms | 8056.7 ms | 22995.9 ms |
| retrieval | 8234.5 ms | 11950.3 ms | 8451.1 ms | 23349.1 ms |
| generation | 1539.3 ms | 4217.5 ms | 1829.3 ms | 5934.4 ms |
| grounding | 1334.0 ms | 2440.0 ms | 1204.5 ms | 3187.9 ms |
| total | 10907.6 ms | 15743.9 ms | 11485.5 ms | 25335.6 ms |

### Exact configuration

```json
{
  "use_dense": true,
  "use_lexical": true,
  "top_k_dense": 50,
  "top_k_lexical": 50,
  "fusion": "rrf",
  "rrf_k": 60,
  "dense_weight": 0.5,
  "use_rerank": true,
  "rerank_candidates": 20,
  "final_top_k": 5,
  "expansion": "none",
  "num_paraphrases": 3,
  "chunk_strategy": "recursive_structural",
  "use_query_instruction": true,
  "hnsw_ef_search": 64
}
```

