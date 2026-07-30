"""The evaluation runner.

Two things here are load-bearing for whether the published numbers mean anything.

The runner answers every question through the same :class:`AnswerService` the API serves, so
the numbers describe the code path a user actually hits. If the harness had its own copy of
the orchestration, the two would drift and the table would describe a system nobody runs.

Recall's denominator is the true corpus-wide count of relevant chunks, computed by scanning
the documents each entry names with the same predicate used to score results. Using the number
of spans instead would be cheaper and wrong: it would make recall look like precision and
would move whenever the chunker split a span differently.

Indexes are cached per chunking strategy, because the chunking ablation rows need genuinely
different indexes while every other row shares one. Re-ingesting per row would multiply the
run time by six for no change in the numbers.
"""

from __future__ import annotations

import asyncio
import statistics
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from retrieval_engine.config import Settings, set_seeds
from retrieval_engine.embed import build_embedder
from retrieval_engine.embed.base import Embedder
from retrieval_engine.eval.metrics import (
    answer_similarity,
    citation_precision,
    count_relevant,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    refusal_accuracy,
    relevance_flags,
)
from retrieval_engine.generate.base import LLM
from retrieval_engine.generate.prompts import PROMPT_VERSION
from retrieval_engine.ingest.pipeline import IngestPipeline
from retrieval_engine.logging_config import get_logger
from retrieval_engine.models import (
    AnswerType,
    Chunk,
    ChunkStrategy,
    EvalRow,
    EvalRun,
    GoldenCategory,
    GoldenEntry,
    LatencyStats,
    QueryResponse,
    RetrievalConfig,
)
from retrieval_engine.retrieve.lexical import BM25Retriever
from retrieval_engine.retrieve.pipeline import RetrievalPipeline
from retrieval_engine.retrieve.rerank import CrossEncoderReranker
from retrieval_engine.service import AnswerService
from retrieval_engine.store.base import VectorStore
from retrieval_engine.store.memory import MemoryVectorStore

logger = get_logger(__name__)

#: The k every headline metric is reported at. The spec's table is recall@5 and nDCG@5, and
#: the retriever returns 5 by default, so they line up.
DEFAULT_K = 5

STAGE_NAMES = (
    "expansion",
    "dense",
    "lexical",
    "fusion",
    "rerank",
    "retrieval",
    "generation",
    "grounding",
    "total",
)


@dataclass
class _Index:
    """One ingested index plus everything derived from it."""

    store: VectorStore
    chunks: list[Chunk]
    lexical: BM25Retriever
    chunk_count: int
    doc_count: int
    #: entry qid -> corpus-wide relevant chunk count, cached because it is the same for every
    #: config that shares this index.
    relevant_counts: dict[str, int] = field(default_factory=dict)


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile. Explicit rather than inherited from a stats helper.

    With 60 queries an interpolating percentile invents a value between two measurements,
    which is not something that was observed. Nearest-rank always reports a real sample.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _latency(values: Sequence[float]) -> LatencyStats:
    if not values:
        return LatencyStats(p50_ms=0.0, p95_ms=0.0, mean_ms=0.0, max_ms=0.0)
    return LatencyStats(
        p50_ms=_percentile(values, 0.50),
        p95_ms=_percentile(values, 0.95),
        mean_ms=statistics.fmean(values),
        max_ms=max(values),
    )


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


class EvalHarness:
    """Ingests a corpus and runs golden-set evaluations against it."""

    def __init__(
        self,
        settings: Settings,
        corpus_dir: Path,
        *,
        embedder: Embedder | None = None,
        reranker: CrossEncoderReranker | None = None,
        llm: LLM | None = None,
    ) -> None:
        self._settings = settings
        self._corpus_dir = corpus_dir
        self._embedder = embedder if embedder is not None else build_embedder(settings)
        self._reranker = reranker if reranker is not None else CrossEncoderReranker(settings)
        self._llm = llm
        self._indexes: dict[ChunkStrategy, _Index] = {}

    @property
    def embedder_name(self) -> str:
        return self._embedder.info.name

    @property
    def generator_name(self) -> str:
        return self._llm.model if self._llm is not None else "extractive"

    async def index_for(self, strategy: ChunkStrategy) -> _Index:
        """Ingest the corpus under ``strategy``, or return the cached index."""
        cached = self._indexes.get(strategy)
        if cached is not None:
            return cached

        settings = self._settings.model_copy(update={"chunk_strategy": strategy})
        store: VectorStore = MemoryVectorStore(f"eval-{strategy.value}")
        pipeline = IngestPipeline(settings, store, self._embedder)
        logger.info("eval_index_building", strategy=strategy.value)
        summary = await pipeline.ingest_directory(self._corpus_dir, progress=True)
        chunks = await store.all_chunks()
        index = _Index(
            store=store,
            chunks=chunks,
            lexical=BM25Retriever(store, Path(settings.bm25_dir) / strategy.value),
            chunk_count=len(chunks),
            doc_count=summary.docs_changed + summary.docs_unchanged,
        )
        await index.lexical.ensure_index()
        self._indexes[strategy] = index
        logger.info(
            "eval_index_ready",
            strategy=strategy.value,
            docs=index.doc_count,
            chunks=index.chunk_count,
        )
        return index

    def _relevant_count(self, index: _Index, entry: GoldenEntry) -> int:
        """Corpus-wide relevant chunk count for one entry, cached per index.

        Only the documents the entry names are scanned. A span is verbatim text from a known
        document, so no other document's chunks can contain it, and scanning the whole corpus
        per entry would be thousands of times more work for the same answer.
        """
        cached = index.relevant_counts.get(entry.qid)
        if cached is not None:
            return cached
        named = set(entry.relevant_doc_ids)
        scope = [chunk for chunk in index.chunks if chunk.doc_id in named] if named else []
        count = count_relevant(scope, entry)
        index.relevant_counts[entry.qid] = count
        return count

    async def _score_entry(
        self,
        entry: GoldenEntry,
        response: QueryResponse,
        index: _Index,
        k: int,
    ) -> EvalRow:
        """Turn one answered question into a scored row."""
        chunks = list(response.sources)
        wanted = {source.chunk_id for source in chunks}
        retrieved = [chunk for chunk in index.chunks if chunk.chunk_id in wanted]
        # Preserve rank order, which the store lookup above does not.
        by_id = {chunk.chunk_id: chunk for chunk in retrieved}
        ordered = [by_id[s.chunk_id] for s in chunks if s.chunk_id in by_id]

        flags = relevance_flags(ordered, entry)
        total_relevant = self._relevant_count(index, entry)
        gains = [1.0 if flag else 0.0 for flag in flags]
        refused = response.answer_type is AnswerType.REFUSED

        metrics: dict[str, float] = {
            f"recall@{k}": recall_at_k(flags, total_relevant, k),
            f"precision@{k}": precision_at_k(flags, k),
            f"hit_rate@{k}": hit_rate_at_k(flags, k),
            "mrr": reciprocal_rank(flags),
            f"ndcg@{k}": ndcg_at_k(gains, k, total_relevant=total_relevant),
        }

        if entry.category is not GoldenCategory.NEGATIVE:
            cited = {citation.chunk_id for citation in response.citations if citation.resolved}
            cited_relevance = [
                flag for chunk, flag in zip(ordered, flags, strict=True) if chunk.chunk_id in cited
            ]
            metrics["citation_precision"] = citation_precision(cited_relevance)

            if entry.answer and response.answer and not refused:
                vectors = await self._embedder.embed_documents([response.answer, entry.answer])
                metrics["answer_similarity"] = answer_similarity(vectors[0], vectors[1])

        return EvalRow(
            qid=entry.qid,
            question=entry.question,
            category=entry.category,
            difficulty=entry.difficulty,
            retrieved_chunk_ids=[source.chunk_id for source in chunks],
            relevance_flags=flags,
            metrics=metrics,
            answer=response.answer,
            answer_type=response.answer_type,
            refused=refused,
            grounded=response.grounding.grounded,
            timings=response.timings,
        )

    async def run(
        self,
        entries: Sequence[GoldenEntry],
        config: RetrievalConfig,
        *,
        label: str | None = None,
        k: int = DEFAULT_K,
        concurrency: int = 2,
        progress: bool = True,
        run_id: str | None = None,
    ) -> tuple[EvalRun, list[EvalRow]]:
        """Evaluate ``entries`` under ``config``, returning the aggregate and the rows."""
        set_seeds(self._settings.seed)
        started = time.perf_counter()
        index = await self.index_for(config.chunk_strategy)

        retrieval = RetrievalPipeline(
            self._settings,
            index.store,
            self._embedder,
            reranker=self._reranker,
            lexical=index.lexical,
            llm=self._llm,
        )
        service = AnswerService(self._settings, retrieval, self._embedder, llm=self._llm)

        # Concurrency is modest on purpose: the embedder and reranker serialise on one lock
        # each (the HuggingFace tokenizer is not thread safe), so a large fan-out would only
        # queue up behind them while making the progress bar less honest.
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def one(entry: GoldenEntry) -> EvalRow:
            async with semaphore:
                response = await service.answer(
                    entry.question, config=config, top_k=k, request_id=entry.qid
                )
                return await self._score_entry(entry, response, index, k)

        rows: list[EvalRow] = []
        tasks = [asyncio.create_task(one(entry)) for entry in entries]
        with tqdm(
            total=len(tasks), desc=label or config.label, unit="q", disable=not progress
        ) as bar:
            for task in asyncio.as_completed(tasks):
                rows.append(await task)
                bar.update(1)

        # Restore golden-set order, so rows.jsonl is diffable between runs.
        order = {entry.qid: position for position, entry in enumerate(entries)}
        rows.sort(key=lambda row: order.get(row.qid, 0))

        aggregate = self._aggregate(
            rows,
            config,
            label=label or config.label,
            k=k,
            run_id=run_id or uuid.uuid4().hex[:12],
            elapsed=time.perf_counter() - started,
        )
        return aggregate, rows

    def _aggregate(
        self,
        rows: Sequence[EvalRow],
        config: RetrievalConfig,
        *,
        label: str,
        k: int,
        run_id: str,
        elapsed: float,
    ) -> EvalRun:
        """Collapse rows into headline metrics, breakdowns, and latency percentiles."""
        answerable = [row for row in rows if row.category is not GoldenCategory.NEGATIVE]
        negatives = [row for row in rows if row.category is GoldenCategory.NEGATIVE]

        def metric(name: str, subset: Sequence[EvalRow]) -> float:
            return _mean([row.metrics[name] for row in subset if name in row.metrics])

        # Retrieval metrics average over answerable questions only. A negative question has
        # no relevant chunk, so including it would drag every retrieval number toward zero
        # and reward a system for retrieving nothing.
        metrics = {
            f"recall@{k}": metric(f"recall@{k}", answerable),
            f"precision@{k}": metric(f"precision@{k}", answerable),
            f"hit_rate@{k}": metric(f"hit_rate@{k}", answerable),
            "mrr": metric("mrr", answerable),
            f"ndcg@{k}": metric(f"ndcg@{k}", answerable),
            "citation_precision": metric("citation_precision", answerable),
            "answer_similarity": metric("answer_similarity", answerable),
            "refusal_accuracy": refusal_accuracy([row.refused for row in negatives]),
            "grounded_rate": _mean([1.0 if row.grounded else 0.0 for row in answerable]),
            # How often an answerable question was wrongly refused. The counterpart to
            # refusal accuracy, and the number that stops a system from scoring well by
            # refusing everything.
            "false_refusal_rate": _mean([1.0 if row.refused else 0.0 for row in answerable]),
        }

        by_category: dict[str, dict[str, float]] = {}
        for category in {row.category for row in rows}:
            subset = [row for row in rows if row.category is category]
            by_category[category.value] = {
                f"recall@{k}": metric(f"recall@{k}", subset),
                f"ndcg@{k}": metric(f"ndcg@{k}", subset),
                "mrr": metric("mrr", subset),
                "refused_rate": _mean([1.0 if row.refused else 0.0 for row in subset]),
                "n": float(len(subset)),
            }

        by_difficulty: dict[str, dict[str, float]] = {}
        for difficulty in {row.difficulty for row in rows}:
            subset = [
                row
                for row in rows
                if row.difficulty is difficulty and row.category is not GoldenCategory.NEGATIVE
            ]
            if not subset:
                continue
            by_difficulty[difficulty.value] = {
                f"recall@{k}": metric(f"recall@{k}", subset),
                f"ndcg@{k}": metric(f"ndcg@{k}", subset),
                "mrr": metric("mrr", subset),
                "n": float(len(subset)),
            }

        latency = {
            stage: _latency([getattr(row.timings, f"{stage}_ms") for row in rows])
            for stage in STAGE_NAMES
        }

        return EvalRun(
            run_id=run_id,
            label=label,
            config=config,
            embedder=self.embedder_name,
            generator=self.generator_name,
            prompt_version=PROMPT_VERSION,
            n_queries=len(rows),
            metrics=metrics,
            by_category=by_category,
            by_difficulty=by_difficulty,
            latency=latency,
            elapsed_seconds=elapsed,
            seed=self._settings.seed,
        )


__all__ = ["DEFAULT_K", "STAGE_NAMES", "EvalHarness"]
