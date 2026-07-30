"""The four-stage retriever: expand, generate candidates, fuse, rerank.

Every stage is individually toggleable through :class:`RetrievalConfig`, because the
ablation has to be able to remove one thing at a time. That constraint is what shaped this
module: nothing is hard-wired, and the config that produced a result travels back inside it
so a published number can always be traced to an exact configuration.

Two details that are easy to get wrong and are handled deliberately here:

* HyDE embeds its hypothetical document with the passage-side encoder, not the query-side
  one. The generated text is a passage, so applying the bge query instruction to it would
  put it in the wrong side of an asymmetric embedding space, which is the whole point of
  generating it.
* HyDE and multi-query still send the ORIGINAL query text to BM25. A generated passage's
  vocabulary is invented, and feeding invented terms to a lexical retriever manufactures
  matches on words the user never wrote.

Expansion needs a language model. If that model is unreachable, expansion is skipped with a
warning and retrieval proceeds unexpanded, because a stopped model server must degrade
quality rather than fail a request.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from retrieval_engine.config import Settings
from retrieval_engine.embed.base import Embedder
from retrieval_engine.errors import LLMUnavailableError
from retrieval_engine.generate.base import LLM
from retrieval_engine.logging_config import get_logger
from retrieval_engine.models import (
    ExpansionMode,
    RetrievalConfig,
    RetrievalResult,
    ScoredChunk,
    StageTimings,
)
from retrieval_engine.retrieve.dense import DenseRetriever
from retrieval_engine.retrieve.fusion import fuse
from retrieval_engine.retrieve.lexical import BM25Retriever
from retrieval_engine.retrieve.rerank import CrossEncoderReranker
from retrieval_engine.store.base import VectorStore

logger = get_logger(__name__)

#: Version marker for the expansion prompts, recorded in logs so a change in retrieval
#: behaviour can be traced to a prompt edit rather than to a config change.
EXPANSION_PROMPT_VERSION = "v1"

MULTI_QUERY_PROMPT = """\
Rewrite the search query below into {count} alternative phrasings that a relevant document \
might use instead. Vary the vocabulary, keep the meaning identical, and keep each rewrite on \
its own line with no numbering, quotes, or commentary.

Query: {query}
"""

HYDE_PROMPT = """\
Write a short factual passage, three sentences at most, that would directly answer the \
question below. Write it in the style of a technical paper. Do not hedge, do not mention \
that it is hypothetical, and output only the passage.

Question: {query}
"""


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


class RetrievalPipeline:
    """Orchestrates candidate generation, fusion, and reranking for one query."""

    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        embedder: Embedder,
        *,
        reranker: CrossEncoderReranker | None = None,
        lexical: BM25Retriever | None = None,
        llm: LLM | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._dense = DenseRetriever(embedder, store)
        self._lexical = (
            lexical if lexical is not None else BM25Retriever(store, Path(settings.bm25_dir))
        )
        self._reranker = reranker if reranker is not None else CrossEncoderReranker(settings)
        self._llm = llm

    # -- stage 1 ----------------------------------------------------------------------

    async def _paraphrase(self, query: str, count: int) -> list[str]:
        if self._llm is None:
            return []
        completion = await self._llm.complete(MULTI_QUERY_PROMPT.format(count=count, query=query))
        lines = [line.strip(" -\t\"'") for line in completion.splitlines()]
        return [line for line in lines if line and line.lower() != query.lower()][:count]

    async def _hypothetical(self, query: str) -> str | None:
        if self._llm is None:
            return None
        passage = (await self._llm.complete(HYDE_PROMPT.format(query=query))).strip()
        return passage or None

    async def _expand(
        self, query: str, config: RetrievalConfig
    ) -> tuple[list[str], list[Sequence[float]]]:
        """Return the query texts to search with, plus any extra passage-side vectors.

        The second element carries HyDE vectors, which are embedded as passages and so
        cannot simply be appended to the query list.
        """
        if config.expansion is ExpansionMode.NONE or self._llm is None:
            return [query], []

        try:
            if config.expansion is ExpansionMode.MULTI_QUERY:
                extra = await self._paraphrase(query, config.num_paraphrases)
                return [query, *extra], []

            hypothetical = await self._hypothetical(query)
            if hypothetical is None:
                return [query], []
            # Embedded as a passage: the generated text IS a passage, and the query-side
            # instruction would place it on the wrong side of an asymmetric space.
            vectors = await self._embedder.embed_documents([hypothetical])
            return [query], list(vectors)
        except LLMUnavailableError as exc:
            logger.warning(
                "query_expansion_unavailable",
                mode=config.expansion.value,
                error=exc.message,
                prompt_version=EXPANSION_PROMPT_VERSION,
            )
            return [query], []

    # -- stage 2 ----------------------------------------------------------------------

    async def _dense_lists(
        self,
        queries: Sequence[str],
        extra_vectors: Sequence[Sequence[float]],
        config: RetrievalConfig,
        filters: Mapping[str, str] | None,
    ) -> list[list[ScoredChunk]]:
        if not config.use_dense:
            return []
        tasks = [
            self._dense.retrieve(
                text,
                config.top_k_dense,
                filters=filters,
                ef_search=config.hnsw_ef_search,
            )
            for text in queries
        ]
        tasks.extend(
            self._dense.retrieve_vector(
                vector,
                config.top_k_dense,
                filters=filters,
                ef_search=config.hnsw_ef_search,
            )
            for vector in extra_vectors
        )
        return list(await asyncio.gather(*tasks))

    async def _lexical_lists(
        self,
        queries: Sequence[str],
        config: RetrievalConfig,
        filters: Mapping[str, str] | None,
    ) -> list[list[ScoredChunk]]:
        if not config.use_lexical:
            return []
        return list(
            await asyncio.gather(
                *(
                    self._lexical.retrieve(text, config.top_k_lexical, filters=filters)
                    for text in queries
                )
            )
        )

    # -- orchestration ----------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        config: RetrievalConfig | None = None,
        filters: Mapping[str, str] | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Run the configured stages and return the result with its full provenance."""
        active = config if config is not None else self._settings.retrieval
        final_top_k = top_k if top_k is not None else active.final_top_k
        timings = StageTimings()
        overall = time.perf_counter()

        started = time.perf_counter()
        queries, extra_vectors = await self._expand(query, active)
        timings.expansion_ms = _elapsed_ms(started)

        # Dense and lexical run concurrently: they share no state and one should not wait
        # on the other.
        started = time.perf_counter()
        dense_lists, lexical_lists = await asyncio.gather(
            self._dense_lists(queries, extra_vectors, active, filters),
            self._lexical_lists(queries, active, filters),
        )
        timings.dense_ms = timings.lexical_ms = _elapsed_ms(started)

        started = time.perf_counter()
        all_lists = [*dense_lists, *lexical_lists]
        fused = fuse(all_lists, active) if all_lists else []
        timings.fusion_ms = _elapsed_ms(started)

        counts = {
            "dense": sum(len(results) for results in dense_lists),
            "lexical": sum(len(results) for results in lexical_lists),
            "fused": len(fused),
        }

        started = time.perf_counter()
        if active.use_rerank and fused:
            shortlist = fused[: active.rerank_candidates]
            chunks = await self._reranker.rerank(query, shortlist, final_top_k)
        else:
            chunks = [
                candidate.model_copy(
                    update={"stages": candidate.stages.model_copy(update={"final_rank": rank})}
                )
                for rank, candidate in enumerate(fused[:final_top_k], start=1)
            ]
        timings.rerank_ms = _elapsed_ms(started)
        counts["reranked"] = len(chunks)

        timings.retrieval_ms = _elapsed_ms(overall)
        timings.total_ms = timings.retrieval_ms

        logger.info(
            "retrieval_finished",
            query_length=len(query),
            expanded=len(queries) + len(extra_vectors),
            dense=counts["dense"],
            lexical=counts["lexical"],
            fused=counts["fused"],
            returned=len(chunks),
            retrieval_ms=round(timings.retrieval_ms, 2),
        )

        return RetrievalResult(
            query=query,
            chunks=chunks,
            expanded_queries=list(queries),
            candidate_counts=counts,
            timings=timings,
            config=active,
        )


__all__ = [
    "EXPANSION_PROMPT_VERSION",
    "HYDE_PROMPT",
    "MULTI_QUERY_PROMPT",
    "RetrievalPipeline",
]
