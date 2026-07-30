# Ablation results

Generated 2026-07-30T14:59:24.551180+00:00

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
| dense only | 0.400 | 0.318 | 0.302 | 0.000 | 0.000 | 3531 ms |
| lexical only | 0.683 | 0.568 | 0.553 | 0.000 | 0.000 | 7041 ms |
| hybrid (RRF) | 0.577 | 0.460 | 0.428 | 0.000 | 0.000 | 5236 ms |
| hybrid + rerank | 0.703 | 0.603 | 0.589 | 1.000 | 0.140 | 21644 ms |
| hybrid (weighted) + rerank | 0.703 | 0.585 | 0.565 | 1.000 | 0.140 | 11765 ms |

`false refusal` is the fraction of answerable questions that were refused. It is
reported next to refusal accuracy because a system can score perfectly on one by
failing the other: refusing everything gives refusal accuracy 1.000.

## dense only

Run `d6521217-recursive_structural`, 60 questions, 1657.3s wall clock.

### Metrics

| metric | value |
|---|---|
| answer_similarity | 0.672 |
| citation_precision | 0.210 |
| false_refusal_rate | 0.000 |
| grounded_rate | 0.980 |
| hit_rate@5 | 0.420 |
| mrr | 0.302 |
| ndcg@5 | 0.318 |
| precision@5 | 0.088 |
| recall@5 | 0.400 |
| refusal_accuracy | 0.000 |

### By category

| category | n | recall@5 | nDCG@5 | MRR |
|---|---|---|---|---|
| ambiguous | 8 | 0.125 | 0.054 | 0.031 |
| factual | 32 | 0.547 | 0.445 | 0.418 |
| multi_hop | 10 | 0.150 | 0.124 | 0.150 |
| negative | 10 | 0.000 | 0.000 | 0.000 |

### By difficulty

| difficulty | n | recall@5 | nDCG@5 | MRR |
|---|---|---|---|---|
| easy | 9 | 0.556 | 0.491 | 0.481 |
| hard | 13 | 0.269 | 0.206 | 0.212 |
| medium | 28 | 0.411 | 0.315 | 0.287 |

### Stage latency

| stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| expansion | 0.0 ms | 0.0 ms | 0.0 ms | 1.3 ms |
| dense | 92.5 ms | 447.1 ms | 261.1 ms | 3193.9 ms |
| lexical | 92.5 ms | 447.1 ms | 261.1 ms | 3193.9 ms |
| fusion | 0.4 ms | 0.9 ms | 0.6 ms | 8.9 ms |
| rerank | 0.0 ms | 0.1 ms | 0.0 ms | 0.1 ms |
| retrieval | 93.1 ms | 447.7 ms | 261.8 ms | 3201.2 ms |
| generation | 846.8 ms | 1741.6 ms | 921.3 ms | 2628.5 ms |
| grounding | 663.4 ms | 1089.9 ms | 699.1 ms | 2414.2 ms |
| total | 1687.3 ms | 3531.0 ms | 1883.1 ms | 4820.6 ms |

### Exact configuration

```json
{
  "use_dense": true,
  "use_lexical": false,
  "top_k_dense": 50,
  "top_k_lexical": 50,
  "fusion": "rrf",
  "rrf_k": 60,
  "dense_weight": 0.5,
  "use_rerank": false,
  "rerank_candidates": 20,
  "final_top_k": 5,
  "expansion": "none",
  "num_paraphrases": 3,
  "chunk_strategy": "recursive_structural",
  "use_query_instruction": true,
  "hnsw_ef_search": 64
}
```

## lexical only

Run `766ad146-recursive_structural`, 60 questions, 127.3s wall clock.

### Metrics

| metric | value |
|---|---|
| answer_similarity | 0.695 |
| citation_precision | 0.337 |
| false_refusal_rate | 0.000 |
| grounded_rate | 0.960 |
| hit_rate@5 | 0.740 |
| mrr | 0.553 |
| ndcg@5 | 0.568 |
| precision@5 | 0.168 |
| recall@5 | 0.683 |
| refusal_accuracy | 0.000 |

### By category

| category | n | recall@5 | nDCG@5 | MRR |
|---|---|---|---|---|
| ambiguous | 8 | 0.375 | 0.266 | 0.229 |
| factual | 32 | 0.802 | 0.697 | 0.676 |
| multi_hop | 10 | 0.550 | 0.398 | 0.420 |
| negative | 10 | 0.000 | 0.000 | 0.000 |

### By difficulty

| difficulty | n | recall@5 | nDCG@5 | MRR |
|---|---|---|---|---|
| easy | 9 | 0.852 | 0.829 | 0.833 |
| hard | 13 | 0.615 | 0.479 | 0.477 |
| medium | 28 | 0.661 | 0.526 | 0.498 |

### Stage latency

| stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| expansion | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| dense | 100.1 ms | 367.0 ms | 140.0 ms | 1026.3 ms |
| lexical | 100.1 ms | 367.0 ms | 140.0 ms | 1026.3 ms |
| fusion | 0.4 ms | 0.7 ms | 0.5 ms | 1.1 ms |
| rerank | 0.0 ms | 0.0 ms | 0.0 ms | 0.2 ms |
| retrieval | 100.5 ms | 367.7 ms | 140.6 ms | 1026.9 ms |
| generation | 1888.0 ms | 4306.6 ms | 2269.0 ms | 7510.7 ms |
| grounding | 1007.6 ms | 3245.3 ms | 1305.0 ms | 3594.0 ms |
| total | 3027.7 ms | 7040.7 ms | 3715.0 ms | 11088.2 ms |

### Exact configuration

```json
{
  "use_dense": false,
  "use_lexical": true,
  "top_k_dense": 50,
  "top_k_lexical": 50,
  "fusion": "rrf",
  "rrf_k": 60,
  "dense_weight": 0.5,
  "use_rerank": false,
  "rerank_candidates": 20,
  "final_top_k": 5,
  "expansion": "none",
  "num_paraphrases": 3,
  "chunk_strategy": "recursive_structural",
  "use_query_instruction": true,
  "hnsw_ef_search": 64
}
```

## hybrid (RRF)

Run `52c645b2-recursive_structural`, 60 questions, 93.7s wall clock.

### Metrics

| metric | value |
|---|---|
| answer_similarity | 0.690 |
| citation_precision | 0.283 |
| false_refusal_rate | 0.000 |
| grounded_rate | 0.980 |
| hit_rate@5 | 0.620 |
| mrr | 0.428 |
| ndcg@5 | 0.460 |
| precision@5 | 0.136 |
| recall@5 | 0.577 |
| refusal_accuracy | 0.000 |

### By category

| category | n | recall@5 | nDCG@5 | MRR |
|---|---|---|---|---|
| ambiguous | 8 | 0.375 | 0.236 | 0.192 |
| factual | 32 | 0.698 | 0.586 | 0.558 |
| multi_hop | 10 | 0.350 | 0.238 | 0.200 |
| negative | 10 | 0.000 | 0.000 | 0.000 |

### By difficulty

| difficulty | n | recall@5 | nDCG@5 | MRR |
|---|---|---|---|---|
| easy | 9 | 0.593 | 0.548 | 0.537 |
| hard | 13 | 0.500 | 0.366 | 0.323 |
| medium | 28 | 0.607 | 0.476 | 0.442 |

### Stage latency

| stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| expansion | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| dense | 314.4 ms | 863.2 ms | 423.9 ms | 1590.5 ms |
| lexical | 314.4 ms | 863.2 ms | 423.9 ms | 1590.5 ms |
| fusion | 0.9 ms | 1.5 ms | 0.9 ms | 2.0 ms |
| rerank | 0.0 ms | 0.0 ms | 0.0 ms | 0.1 ms |
| retrieval | 315.1 ms | 864.6 ms | 424.9 ms | 1591.4 ms |
| generation | 1252.9 ms | 2559.9 ms | 1398.4 ms | 4501.1 ms |
| grounding | 866.2 ms | 1724.3 ms | 956.2 ms | 2391.5 ms |
| total | 2463.8 ms | 5235.5 ms | 2780.0 ms | 5816.2 ms |

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
  "use_rerank": false,
  "rerank_candidates": 20,
  "final_top_k": 5,
  "expansion": "none",
  "num_paraphrases": 3,
  "chunk_strategy": "recursive_structural",
  "use_query_instruction": true,
  "hnsw_ef_search": 64
}
```

## hybrid + rerank

Run `2dfc90a5-recursive_structural`, 60 questions, 451.8s wall clock.

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
| dense | 497.7 ms | 967.3 ms | 502.6 ms | 1300.9 ms |
| lexical | 497.7 ms | 967.3 ms | 502.6 ms | 1300.9 ms |
| fusion | 1.2 ms | 2.8 ms | 1.6 ms | 10.3 ms |
| rerank | 10031.4 ms | 15167.1 ms | 10733.3 ms | 26157.1 ms |
| retrieval | 10601.9 ms | 15695.5 ms | 11237.6 ms | 26385.2 ms |
| generation | 1384.9 ms | 6244.5 ms | 1821.8 ms | 10150.2 ms |
| grounding | 1169.9 ms | 3481.3 ms | 1360.5 ms | 4369.7 ms |
| total | 13317.3 ms | 21643.7 ms | 14420.6 ms | 31914.1 ms |

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

## hybrid (weighted) + rerank

Run `c5de53fe-recursive_structural`, 60 questions, 218.7s wall clock.

### Metrics

| metric | value |
|---|---|
| answer_similarity | 0.710 |
| citation_precision | 0.330 |
| false_refusal_rate | 0.140 |
| grounded_rate | 0.980 |
| hit_rate@5 | 0.760 |
| mrr | 0.565 |
| ndcg@5 | 0.585 |
| precision@5 | 0.176 |
| recall@5 | 0.703 |
| refusal_accuracy | 1.000 |

### By category

| category | n | recall@5 | nDCG@5 | MRR |
|---|---|---|---|---|
| ambiguous | 8 | 0.375 | 0.375 | 0.375 |
| factual | 32 | 0.818 | 0.683 | 0.651 |
| multi_hop | 10 | 0.600 | 0.439 | 0.440 |
| negative | 10 | 0.000 | 0.000 | 0.000 |

### By difficulty

| difficulty | n | recall@5 | nDCG@5 | MRR |
|---|---|---|---|---|
| easy | 9 | 0.963 | 0.846 | 0.815 |
| hard | 13 | 0.692 | 0.569 | 0.569 |
| medium | 28 | 0.625 | 0.509 | 0.482 |

### Stage latency

| stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| expansion | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| dense | 600.6 ms | 1862.8 ms | 760.9 ms | 2361.4 ms |
| lexical | 600.6 ms | 1862.8 ms | 760.9 ms | 2361.4 ms |
| fusion | 1.4 ms | 4.3 ms | 1.9 ms | 11.5 ms |
| rerank | 1937.3 ms | 5421.5 ms | 2142.9 ms | 6180.7 ms |
| retrieval | 2547.2 ms | 6222.1 ms | 2905.8 ms | 6923.9 ms |
| generation | 2093.8 ms | 5062.5 ms | 2154.9 ms | 9316.0 ms |
| grounding | 1545.0 ms | 4496.9 ms | 1689.0 ms | 5337.5 ms |
| total | 6859.2 ms | 11764.9 ms | 6750.4 ms | 13921.7 ms |

### Exact configuration

```json
{
  "use_dense": true,
  "use_lexical": true,
  "top_k_dense": 50,
  "top_k_lexical": 50,
  "fusion": "weighted",
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

