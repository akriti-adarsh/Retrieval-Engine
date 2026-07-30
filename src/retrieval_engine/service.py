"""The answer path: retrieve, decide whether to answer, generate, verify.

This lives outside the API layer on purpose. The eval harness has to measure the same
sequence the API serves, and if the orchestration lived in a route handler the harness would
either import from the web layer or reimplement it. A reimplementation would drift, and then
the published numbers would describe a code path no user ever hits.

The order of operations is the design:

1. Retrieve. Nothing else can happen without evidence.
2. Decide whether to answer at all, from the retrieval scores alone. Refusing before
   generating means a low-confidence query costs no forward pass, and it means the refusal
   cannot be talked out of by a fluent model.
3. Generate, preferring the language model and falling back to extraction when it is
   unreachable. A stopped model server is a quality reduction, not an outage.
4. Verify grounding after the fact, including citation resolution, and report it whether or
   not it passed. A grounding report that only appears when it is clean is marketing.
"""

from __future__ import annotations

import time
from collections.abc import Mapping

from retrieval_engine.config import Settings
from retrieval_engine.embed.base import Embedder
from retrieval_engine.errors import LLMUnavailableError
from retrieval_engine.generate.base import LLM
from retrieval_engine.generate.extractive import ExtractiveAnswerer
from retrieval_engine.generate.prompts import PROMPT_VERSION, build_answer_prompt, render_sources
from retrieval_engine.guardrails.grounding import GroundingVerifier, resolve_citations
from retrieval_engine.guardrails.refusal import RefusalPolicy
from retrieval_engine.logging_config import get_logger
from retrieval_engine.models import (
    AnswerType,
    DebugInfo,
    GeneratedAnswer,
    QueryResponse,
    RetrievalConfig,
    RetrievalResult,
    SourceRef,
)
from retrieval_engine.retrieve.pipeline import RetrievalPipeline

logger = get_logger(__name__)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


class AnswerService:
    """Composes retrieval, generation, and the guardrails into one answer."""

    def __init__(
        self,
        settings: Settings,
        retrieval: RetrievalPipeline,
        embedder: Embedder,
        *,
        llm: LLM | None = None,
    ) -> None:
        self._settings = settings
        self._retrieval = retrieval
        self._llm = llm
        self._extractive = ExtractiveAnswerer(settings, embedder)
        self._verifier = GroundingVerifier(settings, embedder)
        self._policy = RefusalPolicy(settings)

    async def _generate(self, query: str, sources: list[SourceRef]) -> GeneratedAnswer:
        """Generate with the model, falling back to extraction if it is unreachable."""
        if self._llm is not None:
            prompt = build_answer_prompt(query, sources)
            try:
                text = await self._llm.complete(prompt)
            except LLMUnavailableError as exc:
                logger.warning("generation_unavailable_falling_back", error=exc.message)
            else:
                stripped = text.strip()
                if stripped:
                    return GeneratedAnswer(
                        text=stripped,
                        answer_type=AnswerType.GENERATED,
                        prompt_version=PROMPT_VERSION,
                        model=self._llm.model,
                    )
                # An empty completion is a failure that did not raise. Extraction beats
                # returning a blank answer with a confident shape.
                logger.warning("generation_returned_empty_falling_back")
        return await self._extractive.answer(query, sources)

    async def answer(
        self,
        query: str,
        *,
        config: RetrievalConfig | None = None,
        filters: Mapping[str, str] | None = None,
        top_k: int | None = None,
        request_id: str = "",
        debug: bool = False,
    ) -> QueryResponse:
        """Answer ``query``, or explain why it was refused."""
        overall = time.perf_counter()
        result = await self._retrieval.retrieve(query, config=config, filters=filters, top_k=top_k)
        sources = render_sources(result.chunks)
        timings = result.timings

        decision = self._policy.decide(result)
        if decision.refused:
            timings.total_ms = _elapsed_ms(overall)
            logger.info(
                "query_refused",
                reason=decision.reason,
                confidence=round(decision.confidence, 4),
                threshold_applied=decision.threshold_applied,
            )
            return QueryResponse(
                answer=self._policy.explanation(decision),
                answer_type=AnswerType.REFUSED,
                citations=[],
                sources=sources,
                timings=timings,
                prompt_version=PROMPT_VERSION,
                model="refusal-policy",
                request_id=request_id,
                debug=_debug(result) if debug else None,
            )

        started = time.perf_counter()
        generated = await self._generate(query, sources)
        timings.generation_ms = _elapsed_ms(started)

        started = time.perf_counter()
        grounding = await self._verifier.verify(generated.text, sources)
        timings.grounding_ms = _elapsed_ms(started)
        timings.total_ms = _elapsed_ms(overall)

        logger.info(
            "query_answered",
            answer_type=generated.answer_type.value,
            grounded=grounding.grounded,
            flagged=len(grounding.flagged_sentences),
            unresolved_citations=len(grounding.unresolved_citations),
            sources=len(sources),
            total_ms=round(timings.total_ms, 2),
        )

        return QueryResponse(
            answer=generated.text,
            answer_type=generated.answer_type,
            citations=resolve_citations(generated.text, sources),
            sources=sources,
            grounding=grounding,
            timings=timings,
            prompt_version=generated.prompt_version,
            model=generated.model,
            request_id=request_id,
            debug=_debug(result) if debug else None,
        )


def _debug(result: RetrievalResult) -> DebugInfo:
    """The retrieval trail, returned only when the caller asks for it."""
    return DebugInfo(
        config=result.config,
        expanded_queries=result.expanded_queries,
        candidate_counts=result.candidate_counts,
        stages=[chunk.stages for chunk in result.chunks],
        rank_movement=[chunk.stages.rank_movement for chunk in result.chunks],
    )


__all__ = ["AnswerService"]
