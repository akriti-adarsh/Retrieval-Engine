"""Confidence-based abstention.

Refusing is a 200 with ``answer_type: "refused"`` and an explanation, never an exception. A
system that cannot say "I don't know" answers every malformed or out-of-scope question with
something confident and wrong, which is worse than returning nothing.

One subtlety that shapes this whole module: the threshold applies to the reranker's score,
not to whatever score happens to be on the top chunk. A reciprocal-rank-fusion score is
around 0.03 by construction, so comparing it against a confidence threshold of 0.3 would
refuse every single query in the no-rerank ablation rows and quietly turn them into zeros.
When no rerank score is present, the confidence check is skipped and only the
minimum-source-count rule applies, and the decision records that it was skipped.
"""

from __future__ import annotations

from retrieval_engine.config import Settings
from retrieval_engine.models import RefusalDecision, RetrievalResult, ScoredChunk


class RefusalPolicy:
    """Decides whether the retrieved evidence is strong enough to answer from."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @staticmethod
    def _confidence(chunk: ScoredChunk) -> float | None:
        """The chunk's calibrated confidence, or None when there is no such score.

        Only the cross-encoder produces a score on a comparable, roughly probabilistic
        scale. Dense cosine and fused RRF scores are relative, not calibrated.
        """
        return chunk.stages.rerank_score

    def decide(self, result: RetrievalResult) -> RefusalDecision:
        """Judge one retrieval result."""
        minimum = self._settings.min_confidence
        required = self._settings.min_sources

        if not result.chunks:
            return RefusalDecision(
                refused=True,
                reason="nothing was retrieved for this query",
                confidence=0.0,
                usable_sources=0,
            )

        scores = [self._confidence(chunk) for chunk in result.chunks]
        calibrated = [score for score in scores if score is not None]

        if not calibrated:
            # No reranker ran, so there is no calibrated score to threshold. Fall back to
            # the count rule and say plainly that the threshold was not applied.
            usable = len(result.chunks)
            refused = usable < required
            return RefusalDecision(
                refused=refused,
                reason=(
                    f"only {usable} source(s) retrieved, {required} required" if refused else ""
                ),
                confidence=result.top_score,
                usable_sources=usable,
                threshold_applied=False,
            )

        top = max(calibrated)
        usable = sum(1 for score in calibrated if score >= minimum)

        if top < minimum:
            return RefusalDecision(
                refused=True,
                reason=(
                    f"best source scored {top:.3f}, below the {minimum:.2f} confidence threshold"
                ),
                confidence=top,
                usable_sources=usable,
            )
        if usable < required:
            return RefusalDecision(
                refused=True,
                reason=(
                    f"only {usable} source(s) cleared the {minimum:.2f} threshold, "
                    f"{required} required"
                ),
                confidence=top,
                usable_sources=usable,
            )
        return RefusalDecision(
            refused=False,
            confidence=top,
            usable_sources=usable,
        )

    def explanation(self, decision: RefusalDecision) -> str:
        """Reader-facing text for a refusal, which the API returns as the answer."""
        detail = f" ({decision.reason})" if decision.reason else ""
        return f"I don't have enough information in the provided sources to answer that{detail}."


__all__ = ["RefusalPolicy"]
