"""Orchestrated ingestion: load, chunk, embed, upsert, with change detection.

Change detection is the feature that makes this usable day to day. Re-running ingestion on
an unchanged corpus must be a no-op that embeds nothing, because embedding a few thousand
chunks on CPU takes minutes and a pipeline that redoes it on every run stops being run.

The comparison is on the document's content hash, not its mtime. A git checkout rewrites
mtimes without changing content, and mtime-based detection would then re-embed the whole
corpus for no reason.

Documents are processed concurrently under a semaphore, and each task returns its own
result rather than mutating a shared counter, so the summary cannot depend on interleaving.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from retrieval_engine.config import Settings
from retrieval_engine.embed.base import Embedder
from retrieval_engine.errors import RetrievalEngineError
from retrieval_engine.ingest.chunker import Chunker, build_chunker
from retrieval_engine.ingest.loaders import iter_source_files, load_document
from retrieval_engine.logging_config import get_logger
from retrieval_engine.models import Document, EmbeddedChunk, IngestSummary
from retrieval_engine.store.base import VectorStore

logger = get_logger(__name__)


@dataclass
class _DocResult:
    """One document's outcome, aggregated after every task finishes."""

    changed: bool = False
    chunks_created: int = 0
    chunks_skipped: int = 0
    tokens_embedded: int = 0
    error: str | None = None


class IngestPipeline:
    """Turns files into stored, embedded chunks."""

    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        embedder: Embedder,
        chunker: Chunker | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._chunker = chunker

    def _get_chunker(self) -> Chunker:
        """Build the configured chunker on first use.

        Deferred because constructing it touches ``embedder.tokenizer``, which loads the
        model, and a caller that only wants to inspect the pipeline should not pay for that.
        """
        if self._chunker is None:
            self._chunker = build_chunker(
                self._settings.chunk_strategy,
                self._settings,
                self._embedder.tokenizer,
                self._embedder,
            )
        return self._chunker

    async def _process(
        self,
        path: Path,
        known_hashes: dict[str, str],
        known_counts: dict[str, int],
        semaphore: asyncio.Semaphore,
    ) -> _DocResult:
        async with semaphore:
            try:
                # Loading is blocking file IO and parsing, so keep it off the event loop.
                document: Document = await asyncio.to_thread(load_document, path)
            except RetrievalEngineError as exc:
                logger.warning("document_load_failed", path=str(path), error=exc.message)
                return _DocResult(error=f"{path}: {exc.message}")

            if known_hashes.get(document.doc_id) == document.content_hash:
                return _DocResult(
                    changed=False,
                    chunks_skipped=known_counts.get(document.doc_id, 0),
                )

            try:
                chunks = await self._get_chunker().chunk(document)
                if not chunks:
                    logger.warning("document_produced_no_chunks", doc_id=document.doc_id)
                    return _DocResult(changed=True)

                vectors = await self._embedder.embed_documents([chunk.text for chunk in chunks])
                records = [
                    EmbeddedChunk(chunk=chunk, embedding=vector)
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ]
                written = await self._store.upsert_document(document, records)
            except RetrievalEngineError as exc:
                logger.warning("document_ingest_failed", doc_id=document.doc_id, error=exc.message)
                return _DocResult(error=f"{path}: {exc.message}")

            return _DocResult(
                changed=True,
                chunks_created=written,
                tokens_embedded=sum(chunk.token_count for chunk in chunks),
            )

    async def ingest_paths(self, paths: Sequence[Path], *, progress: bool = True) -> IngestSummary:
        """Ingest an explicit list of files and return what happened."""
        started = time.perf_counter()
        await self._store.ensure_collection(self._embedder.info)

        known_hashes = dict(await self._store.document_hashes())
        # Per-document chunk counts, so the summary can report how many chunks were skipped
        # rather than re-embedded. Two calls: one to learn the total, one to fetch it.
        known_counts: dict[str, int] = {}
        if known_hashes:
            probe = await self._store.list_documents(limit=1)
            if probe.total:
                listing = await self._store.list_documents(limit=probe.total)
                known_counts = {item.doc_id: item.chunk_count for item in listing.items}

        semaphore = asyncio.Semaphore(self._settings.ingest_concurrency)
        ordered = sorted(paths)
        tasks = [
            asyncio.create_task(self._process(path, known_hashes, known_counts, semaphore))
            for path in ordered
        ]

        results: list[_DocResult] = []
        with tqdm(total=len(tasks), desc="ingest", unit="doc", disable=not progress) as bar:
            for task in asyncio.as_completed(tasks):
                results.append(await task)
                bar.update(1)

        summary = IngestSummary(
            docs_seen=len(ordered),
            docs_changed=sum(1 for result in results if result.changed and result.error is None),
            docs_unchanged=sum(
                1 for result in results if not result.changed and result.error is None
            ),
            docs_failed=sum(1 for result in results if result.error is not None),
            chunks_created=sum(result.chunks_created for result in results),
            chunks_skipped=sum(result.chunks_skipped for result in results),
            tokens_embedded=sum(result.tokens_embedded for result in results),
            elapsed_seconds=time.perf_counter() - started,
            errors=sorted(result.error for result in results if result.error is not None),
        )
        logger.info(
            "ingest_finished",
            docs_seen=summary.docs_seen,
            docs_changed=summary.docs_changed,
            docs_unchanged=summary.docs_unchanged,
            chunks_created=summary.chunks_created,
            summary=summary.render(),
        )
        return summary

    async def ingest_directory(self, directory: Path, *, progress: bool = True) -> IngestSummary:
        """Ingest every loadable file under ``directory``."""
        return await self.ingest_paths(iter_source_files(directory), progress=progress)


__all__ = ["IngestPipeline"]
