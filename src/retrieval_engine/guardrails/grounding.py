"""Grounding verification: does the answer actually follow from the retrieved sources?

Verification is per sentence, not per answer. The common failure mode is a fluent answer
that mixes several supported claims with one invented detail, and a single similarity score
over the whole answer averages that detail away. Scoring sentence by sentence is what makes
the invented one visible.

Citation markers are resolved as well as scored. A ``[4]`` pointing at a source that was
never in the context is a defect, not a formatting quirk, so it is surfaced in the response
rather than quietly dropped.

Sentences are segmented with the same splitter the chunker uses, so a sentence can never be
judged against a span that was never a sentence upstream.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from retrieval_engine.config import Settings
from retrieval_engine.embed.base import Embedder
from retrieval_engine.generate.prompts import INSUFFICIENT_ANSWER
from retrieval_engine.ingest.chunker import split_sentences
from retrieval_engine.models import (
    Citation,
    GroundingReport,
    SentenceGrounding,
    SourceRef,
)

#: A citation marker such as [1] or [12].
CITATION = re.compile(r"\[(\d{1,3})\]")


def strip_citations(text: str) -> str:
    """Remove citation markers before embedding.

    The markers are punctuation to the reader but tokens to an embedder, and they appear in
    every sentence, so leaving them in adds a constant term to every similarity.
    """
    return " ".join(CITATION.sub(" ", text).split())


def resolve_citations(answer: str, sources: Sequence[SourceRef]) -> list[Citation]:
    """Map every marker in ``answer`` to the source it points at, in order of appearance.

    A marker outside the source range comes back with ``resolved=False`` rather than being
    silently discarded, because a model citing a source it was never given is exactly the
    kind of failure this service is supposed to report.
    """
    by_index = {source.index: source for source in sources}
    citations: list[Citation] = []
    seen: set[int] = set()
    for match in CITATION.finditer(answer):
        marker = int(match.group(1))
        if marker in seen:
            continue
        seen.add(marker)
        source = by_index.get(marker)
        if source is None:
            citations.append(Citation(marker=marker, resolved=False))
        else:
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=source.chunk_id,
                    doc_id=source.doc_id,
                    source_path=source.source_path,
                    resolved=True,
                )
            )
    return citations


class GroundingVerifier:
    """Scores each answer sentence against the retrieved sources."""

    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        self._settings = settings
        self._embedder = embedder

    def _is_insufficiency(self, answer: str) -> bool:
        return INSUFFICIENT_ANSWER.lower() in answer.lower()

    async def verify(self, answer: str, sources: Sequence[SourceRef]) -> GroundingReport:
        """Build the grounding report for ``answer`` against ``sources``."""
        threshold = self._settings.grounding_threshold
        citations = resolve_citations(answer, sources)
        unresolved = [citation.marker for citation in citations if not citation.resolved]

        # A statement that the sources are insufficient claims nothing about the world, so
        # there is nothing to ground. Scoring it would flag an honest refusal as a
        # hallucination, which is exactly backwards.
        if self._is_insufficiency(answer):
            return GroundingReport(
                sentences=[],
                grounded=True,
                threshold=threshold,
                unresolved_citations=unresolved,
            )

        sentences = [
            cleaned
            for sentence in split_sentences(answer)
            if (cleaned := strip_citations(sentence))
        ]
        if not sentences or not sources:
            # No sources means nothing could have supported the answer.
            return GroundingReport(
                sentences=[],
                grounded=False,
                threshold=threshold,
                flagged_sentences=sentences,
                unresolved_citations=unresolved,
            )

        sentence_vectors = np.asarray(
            await self._embedder.embed_documents(sentences), dtype=np.float64
        )
        source_vectors = np.asarray(
            await self._embedder.embed_documents([source.text for source in sources]),
            dtype=np.float64,
        )

        similarities = _cosine_matrix(sentence_vectors, source_vectors)
        best_indices = similarities.argmax(axis=1)

        graded: list[SentenceGrounding] = []
        flagged: list[str] = []
        for position, sentence in enumerate(sentences):
            best = int(best_indices[position])
            score = float(similarities[position][best])
            supported = score >= threshold
            if not supported:
                flagged.append(sentence)
            graded.append(
                SentenceGrounding(
                    sentence=sentence,
                    max_similarity=score,
                    supporting_chunk_id=sources[best].chunk_id if supported else None,
                    grounded=supported,
                )
            )

        return GroundingReport(
            sentences=graded,
            grounded=not flagged and not unresolved,
            threshold=threshold,
            flagged_sentences=flagged,
            unresolved_citations=unresolved,
        )


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity, guarding zero-magnitude rows on either side."""
    left_norms = np.linalg.norm(left, axis=1)
    right_norms = np.linalg.norm(right, axis=1)
    left_safe = np.where(left_norms == 0.0, 1.0, left_norms)
    right_safe = np.where(right_norms == 0.0, 1.0, right_norms)
    return (left / left_safe[:, None]) @ (right / right_safe[:, None]).T


__all__ = ["CITATION", "GroundingVerifier", "resolve_citations", "strip_citations"]
