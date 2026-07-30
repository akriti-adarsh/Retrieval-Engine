"""Combining ranked candidate lists.

Reciprocal rank fusion is the default because it consumes ranks rather than scores, which
makes it scale free: cosine similarity and BM25 live on completely different scales, and RRF
needs no calibration between them. It also degrades gracefully when one retriever's score
distribution shifts, since only the ordering it produces is used.

Weighted score fusion is implemented as the alternative so the ablation can measure the
difference rather than assert it. It min-max normalises each list before the weighted sum,
which is the step that makes two incomparable scales comparable, and also the step that
makes the result sensitive to a single outlier score.

``k`` in RRF controls how sharply top ranks dominate. Small ``k`` makes rank one nearly
decisive; large ``k`` flattens the curve until fusion is effectively counting how many lists
a document appears in. Both ends are pinned by tests.
"""

from __future__ import annotations

from collections.abc import Sequence

from retrieval_engine.models import FusionMethod, RetrievalConfig, ScoredChunk, StageScores


def _merge_stages(first: StageScores, second: StageScores) -> StageScores:
    """Combine two stage records for the same chunk, keeping whichever value is present.

    A chunk retrieved by both retrievers must end up carrying both scores, so the debug
    view can show what each stage contributed.
    """
    return StageScores(
        dense_score=first.dense_score if first.dense_score is not None else second.dense_score,
        lexical_score=(
            first.lexical_score if first.lexical_score is not None else second.lexical_score
        ),
        fusion_score=first.fusion_score if first.fusion_score is not None else second.fusion_score,
        rerank_score=first.rerank_score if first.rerank_score is not None else second.rerank_score,
        dense_rank=first.dense_rank if first.dense_rank is not None else second.dense_rank,
        lexical_rank=first.lexical_rank if first.lexical_rank is not None else second.lexical_rank,
        fused_rank=first.fused_rank if first.fused_rank is not None else second.fused_rank,
        final_rank=first.final_rank if first.final_rank is not None else second.final_rank,
    )


def _finalise(scores: dict[str, float], candidates: dict[str, ScoredChunk]) -> list[ScoredChunk]:
    """Order by fused score, breaking ties on chunk_id, and stamp the fused rank."""
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    fused: list[ScoredChunk] = []
    for rank, (chunk_id, score) in enumerate(ordered, start=1):
        candidate = candidates[chunk_id]
        stages = _merge_stages(candidate.stages, StageScores())
        fused.append(
            ScoredChunk(
                chunk=candidate.chunk,
                score=score,
                stages=stages.model_copy(update={"fusion_score": score, "fused_rank": rank}),
            )
        )
    return fused


def _collect(
    result_lists: Sequence[Sequence[ScoredChunk]],
) -> dict[str, ScoredChunk]:
    """Index every candidate by chunk id, merging stage records across lists."""
    candidates: dict[str, ScoredChunk] = {}
    for results in result_lists:
        for candidate in results:
            chunk_id = candidate.chunk.chunk_id
            existing = candidates.get(chunk_id)
            if existing is None:
                candidates[chunk_id] = candidate
            else:
                candidates[chunk_id] = ScoredChunk(
                    chunk=existing.chunk,
                    score=existing.score,
                    stages=_merge_stages(existing.stages, candidate.stages),
                )
    return candidates


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[ScoredChunk]], k: int = 60
) -> list[ScoredChunk]:
    """Fuse ranked lists with ``score(d) = sum over lists of 1 / (k + rank(d))``.

    A document absent from a list contributes nothing for that list. Ranks are 1-based, as
    in the original formulation.

    Args:
        result_lists: Ranked candidate lists, best first.
        k: Damping constant. Must be positive.

    Raises:
        ValueError: ``k`` is not positive, which would divide by zero at rank ``-k``.
    """
    if k <= 0:
        msg = f"rrf k must be positive, got {k}"
        raise ValueError(msg)

    candidates = _collect(result_lists)
    scores: dict[str, float] = dict.fromkeys(candidates, 0.0)
    for results in result_lists:
        for rank, candidate in enumerate(results, start=1):
            scores[candidate.chunk.chunk_id] += 1.0 / (k + rank)
    return _finalise(scores, candidates)


def _min_max(values: Sequence[float]) -> list[float]:
    """Scale to [0, 1]. An all-equal list maps to 1.0, since every entry is jointly best."""
    if not values:
        return []
    lowest = min(values)
    highest = max(values)
    if highest == lowest:
        return [1.0] * len(values)
    span = highest - lowest
    return [(value - lowest) / span for value in values]


def weighted_score_fusion(
    result_lists: Sequence[Sequence[ScoredChunk]],
    weights: Sequence[float] | None = None,
) -> list[ScoredChunk]:
    """Min-max normalise each list, then take the weighted sum.

    Args:
        result_lists: Ranked candidate lists.
        weights: One weight per list. Defaults to equal weights.

    Raises:
        ValueError: ``weights`` is given but does not match the number of lists.
    """
    if weights is None:
        count = len(result_lists)
        weights = [1.0 / count] * count if count else []
    elif len(weights) != len(result_lists):
        msg = f"got {len(weights)} weights for {len(result_lists)} lists"
        raise ValueError(msg)

    candidates = _collect(result_lists)
    scores: dict[str, float] = dict.fromkeys(candidates, 0.0)
    for results, weight in zip(result_lists, weights, strict=True):
        normalised = _min_max([candidate.score for candidate in results])
        for candidate, value in zip(results, normalised, strict=True):
            scores[candidate.chunk.chunk_id] += weight * value
    return _finalise(scores, candidates)


def fuse(
    result_lists: Sequence[Sequence[ScoredChunk]],
    config: RetrievalConfig,
    weights: Sequence[float] | None = None,
) -> list[ScoredChunk]:
    """Fuse with whichever method ``config`` selects.

    When exactly two lists are given and no weights are supplied, the weighted method uses
    ``config.dense_weight`` for the first and the remainder for the second, matching the
    dense-then-lexical order the pipeline passes them in.
    """
    if config.fusion is FusionMethod.RRF:
        return reciprocal_rank_fusion(result_lists, config.rrf_k)
    if weights is None and len(result_lists) == 2:
        weights = [config.dense_weight, 1.0 - config.dense_weight]
    return weighted_score_fusion(result_lists, weights)


__all__ = ["fuse", "reciprocal_rank_fusion", "weighted_score_fusion"]
