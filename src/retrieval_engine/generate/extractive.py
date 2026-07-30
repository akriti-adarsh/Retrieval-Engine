"""The no-LLM answer path.

This exists so that a stopped model server is a quality reduction, not an outage. It needs
nothing beyond the base dependencies: no Ollama, no API key, no second model download, just
the embedder that already had to be loaded to retrieve anything.

It does not write prose. It selects the sentences from the retrieved passages that are
closest to the query in embedding space and returns them with their citation markers
attached. That is a deliberate limit: an extract cannot hallucinate, because every word in
the answer is quoted from a source, and the citation is exact by construction rather than
by the model's cooperation.

Selected sentences are re-ordered into source order before being joined, because ranking
order reads as a jumble while source order reads as a passage.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from retrieval_engine.config import Settings
from retrieval_engine.embed.base import Embedder
from retrieval_engine.generate.prompts import INSUFFICIENT_ANSWER
from retrieval_engine.ingest.chunker import split_sentences
from retrieval_engine.models import AnswerType, GeneratedAnswer, SourceRef

#: Recorded as the prompt version for extracted answers. There is no prompt, but eval runs
#: still have to be able to say which answer path produced a number.
EXTRACTIVE_VERSION = "extractive-v1"

MODEL_NAME = "extractive"

#: Sentences shorter than this are skipped as answer candidates. Fragments like a bare
#: heading or "See Table 2." are never a useful answer and they score deceptively well
#: against short queries.
MIN_SENTENCE_CHARS = 25


class ExtractiveAnswerer:
    """Builds a cited answer by selecting sentences, with no language model involved."""

    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        self._settings = settings
        self._embedder = embedder

    @property
    def model(self) -> str:
        return MODEL_NAME

    async def answer(self, query: str, sources: Sequence[SourceRef]) -> GeneratedAnswer:
        """Select the best sentences from ``sources`` and cite them."""
        candidates: list[tuple[int, int, str]] = []
        for source in sources:
            for position, sentence in enumerate(split_sentences(source.text)):
                cleaned = " ".join(sentence.split())
                if len(cleaned) >= MIN_SENTENCE_CHARS:
                    candidates.append((source.index, position, cleaned))

        if not candidates:
            return GeneratedAnswer(
                text=f"{INSUFFICIENT_ANSWER}.",
                answer_type=AnswerType.EXTRACTIVE,
                prompt_version=EXTRACTIVE_VERSION,
                model=MODEL_NAME,
            )

        query_vector = np.asarray(await self._embedder.embed_query(query), dtype=np.float64)
        sentence_vectors = np.asarray(
            await self._embedder.embed_documents([text for _, _, text in candidates]),
            dtype=np.float64,
        )

        query_norm = float(np.linalg.norm(query_vector))
        norms = np.linalg.norm(sentence_vectors, axis=1)
        # A zero-magnitude row would divide by zero. Leaving it unnormalised keeps its
        # similarity at zero, which is the honest score for an empty embedding.
        safe = np.where(norms == 0.0, 1.0, norms)
        if query_norm == 0.0:
            similarities = np.zeros(len(candidates))
        else:
            similarities = (sentence_vectors / safe[:, None]) @ (query_vector / query_norm)

        # Ties break on source index then position, so the same query always extracts the
        # same sentences.
        ranked = sorted(
            range(len(candidates)),
            key=lambda i: (-float(similarities[i]), candidates[i][0], candidates[i][1]),
        )
        chosen = sorted(ranked[: self._settings.extractive_sentences], key=lambda i: candidates[i])

        parts = [f"{candidates[i][2]} [{candidates[i][0]}]" for i in chosen]
        return GeneratedAnswer(
            text=" ".join(parts),
            answer_type=AnswerType.EXTRACTIVE,
            prompt_version=EXTRACTIVE_VERSION,
            model=MODEL_NAME,
            usage={"sentences_considered": len(candidates), "sentences_used": len(chosen)},
        )


__all__ = ["EXTRACTIVE_VERSION", "MIN_SENTENCE_CHARS", "MODEL_NAME", "ExtractiveAnswerer"]
