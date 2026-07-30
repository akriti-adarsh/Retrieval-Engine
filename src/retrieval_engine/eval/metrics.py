"""Retrieval and answer metrics, implemented from scratch.

Written out rather than imported, for two reasons. The formulas are the thing a reader of
this repository will want to check, and a metric whose definition lives in someone else's
library is a metric nobody on the team can defend in a review. Every function below states
its formula in its docstring and has a unit test with a value computed by hand.

The most important thing in this module is not a metric. It is
:func:`chunk_matches_golden`, the predicate that decides what "relevant" means.

The golden set stores relevance as document ids plus verbatim spans, not as chunk ids,
because chunk ids change with the chunking strategy. If relevance were stored as chunk ids,
the chunking ablation would be comparing numbers computed against different ground truths,
and the comparison would be meaningless while looking perfectly reasonable. Defining
relevance once, in terms of text, is what makes the chunking rows comparable at all.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from retrieval_engine.models import Chunk, GoldenEntry

#: A chunk counts as relevant if it shares at least this many consecutive characters with a
#: golden span. Set against the 300-character span cap: large enough that an incidental
#: phrase match cannot qualify, small enough that a span split across two chunks makes both
#: halves relevant, which is the case that matters for the chunking ablation.
MIN_SPAN_OVERLAP = 50


def normalise(text: str) -> str:
    """Collapse whitespace so chunk line breaks cannot defeat a match.

    Chunks preserve the source's newlines while golden spans are written on one line, so
    without this a span would fail to match the very chunk it was copied from.
    """
    return " ".join(text.split())


def overlaps_span(chunk_text: str, span: str, min_overlap: int = MIN_SPAN_OVERLAP) -> bool:
    """Whether ``chunk_text`` contains ``span`` in full, or overlaps it by ``min_overlap``.

    The overlap test slides a window of exactly ``min_overlap`` characters across the span
    and asks whether any window appears in the chunk. That is equivalent to "shares at least
    min_overlap consecutive characters", and it avoids the quadratic longest-common-substring
    computation, which matters because this runs across every candidate of every query.
    """
    haystack = normalise(chunk_text)
    needle = normalise(span)
    if not needle or not haystack:
        return False
    if needle in haystack:
        return True
    if len(needle) < min_overlap:
        # A span shorter than the overlap threshold can only qualify by full containment,
        # which the check above already settled.
        return False
    return any(
        needle[start : start + min_overlap] in haystack
        for start in range(len(needle) - min_overlap + 1)
    )


def chunk_matches_golden(
    chunk: Chunk, entry: GoldenEntry, min_overlap: int = MIN_SPAN_OVERLAP
) -> bool:
    """Whether ``chunk`` is relevant to ``entry``.

    Relevant means the chunk overlaps at least one of the entry's spans. The document id is
    NOT required to match: a span quoted in a second document is still an answer to the
    question, and requiring the id would punish a retriever for finding a correct passage in
    an unexpected place.
    """
    return any(overlaps_span(chunk.text, span, min_overlap) for span in entry.relevant_chunk_texts)


def relevance_flags(
    chunks: Sequence[Chunk], entry: GoldenEntry, min_overlap: int = MIN_SPAN_OVERLAP
) -> list[bool]:
    """Per-position relevance for a ranked result list, best first."""
    return [chunk_matches_golden(chunk, entry, min_overlap) for chunk in chunks]


def count_relevant(
    chunks: Iterable[Chunk], entry: GoldenEntry, min_overlap: int = MIN_SPAN_OVERLAP
) -> int:
    """How many chunks in the corpus are relevant to ``entry``.

    This is recall's denominator, so it has to be the true corpus-wide count rather than the
    number of spans. Callers restrict ``chunks`` to the entry's ``relevant_doc_ids``, which is
    both far cheaper and correct: a span is verbatim text from a known document, so only that
    document's chunks can contain it.
    """
    return sum(1 for chunk in chunks if chunk_matches_golden(chunk, entry, min_overlap))


# --------------------------------------------------------------------------------------
# Ranking metrics
# --------------------------------------------------------------------------------------


def recall_at_k(flags: Sequence[bool], total_relevant: int, k: int) -> float:
    """``relevant retrieved in the top k / total relevant in the corpus``.

    Returns 0.0 when nothing is relevant, rather than 1.0. A system asked a question with no
    answer has not achieved perfect recall, and scoring it 1.0 would let a corpus with no
    relevant documents inflate the average.
    """
    if total_relevant <= 0 or k <= 0:
        return 0.0
    found = sum(1 for flag in flags[:k] if flag)
    return min(1.0, found / total_relevant)


def precision_at_k(flags: Sequence[bool], k: int) -> float:
    """``relevant retrieved in the top k / k``.

    The denominator is k, not the number actually returned. Returning three results when
    five were asked for is a retrieval failure, and dividing by three would hide it.
    """
    if k <= 0:
        return 0.0
    return sum(1 for flag in flags[:k] if flag) / k


def hit_rate_at_k(flags: Sequence[bool], k: int) -> float:
    """1.0 if any of the top k is relevant, else 0.0. Averaged over queries by the runner."""
    if k <= 0:
        return 0.0
    return 1.0 if any(flags[:k]) else 0.0


def reciprocal_rank(flags: Sequence[bool]) -> float:
    """``1 / rank of the first relevant result``, ranks being 1-based, or 0.0 if none."""
    for position, flag in enumerate(flags, start=1):
        if flag:
            return 1.0 / position
    return 0.0


def mrr(per_query_flags: Sequence[Sequence[bool]]) -> float:
    """Mean reciprocal rank across queries: ``mean(1 / rank of first relevant)``."""
    if not per_query_flags:
        return 0.0
    return sum(reciprocal_rank(flags) for flags in per_query_flags) / len(per_query_flags)


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """``sum over i of gain_i / log2(i + 1)`` for 1-based rank i, over the top k."""
    if k <= 0:
        return 0.0
    return sum(gain / math.log2(position + 1) for position, gain in enumerate(gains[:k], start=1))


def ndcg_at_k(gains: Sequence[float], k: int, total_relevant: int | None = None) -> float:
    """``DCG@k / IDCG@k``, where the ideal ordering sorts gains descending.

    The golden set is binary, so gains are 0.0 or 1.0 and this reduces to binary nDCG. The
    graded path is real code rather than decoration, and a synthetic-grades test covers it,
    so it cannot rot unnoticed.

    ``total_relevant`` extends the ideal ordering with relevant documents that were never
    retrieved. Without it, a run that retrieved one relevant document out of ten and put it
    first would score a perfect 1.0, because the ideal would be computed only from what came
    back.

    Returns 0.0 when the ideal is zero, which is the zero-relevant-documents case.
    """
    if k <= 0:
        return 0.0
    actual = dcg_at_k(gains, k)
    ideal_gains = sorted(gains, reverse=True)
    if total_relevant is not None:
        positive = sorted((gain for gain in gains if gain > 0.0), reverse=True)
        missing = max(0, total_relevant - len(positive))
        # Unretrieved relevant documents would have had the maximum gain in the ideal run.
        best = max(gains, default=1.0) or 1.0
        ideal_gains = sorted([*positive, *([best] * missing)], reverse=True)
    ideal = dcg_at_k(ideal_gains, k)
    if ideal <= 0.0:
        return 0.0
    return actual / ideal


# --------------------------------------------------------------------------------------
# Answer metrics
# --------------------------------------------------------------------------------------


def refusal_accuracy(refused: Sequence[bool]) -> float:
    """Fraction of negative-category questions correctly refused.

    The runner passes only the negative entries. This is the metric that keeps the rest of
    the table honest: a system that answers everything confidently can score well on recall
    while being useless, and this is where that shows up.
    """
    if not refused:
        return 0.0
    return sum(1 for value in refused if value) / len(refused)


def citation_precision(cited_relevance: Sequence[bool]) -> float:
    """Fraction of citations that point at a genuinely relevant chunk.

    An answer with no citations returns 0.0, not 1.0. Citing nothing is not perfect
    precision, it is an unsupported answer.
    """
    if not cited_relevance:
        return 0.0
    return sum(1 for value in cited_relevance if value) / len(cited_relevance)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, 0.0 when either vector has no magnitude."""
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def answer_similarity(generated: Sequence[float], reference: Sequence[float]) -> float:
    """Embedding cosine between a generated answer and the reference answer.

    A blunt instrument, and labelled as one: it rewards saying the same thing in the same
    words and cannot tell a correct paraphrase from a wrong one. It is reported alongside
    grounding and citation precision rather than on its own for that reason.
    """
    return cosine_similarity(generated, reference)


__all__ = [
    "MIN_SPAN_OVERLAP",
    "answer_similarity",
    "chunk_matches_golden",
    "citation_precision",
    "cosine_similarity",
    "count_relevant",
    "dcg_at_k",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "normalise",
    "overlaps_span",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "relevance_flags",
]
