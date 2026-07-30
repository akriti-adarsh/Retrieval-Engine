"""Lexical candidate generation with BM25.

Why lexical retrieval is here at all: a dense bi-encoder is good at paraphrase and bad at
exact rare tokens. A query naming a specific identifier, model name, or number is exactly
where embeddings blur and BM25 is precise, and that complementarity is what the hybrid
ablation is meant to measure.

The persisted index stores the tokenized corpus as JSON, not a pickled BM25 object.
Rebuilding the scorer from tokens takes well under a second, and unpickling a file to build
a search index is a code-execution surface bought for nothing.

Staleness is detected with a SHA256 fingerprint over the sorted chunk-id list, so the index
rebuilds when the chunk set changes and does not rebuild when a re-ingest produces the same
chunks in a different order.

One library behaviour worth knowing before reading the ablation: ``rank_bm25`` computes IDF
as ``log(N - df + 0.5) - log(df + 0.5)`` with no smoothing term, so a term occurring in half
or more of a small corpus gets an IDF of zero or below and contributes nothing. On a
two-chunk corpus a term in one chunk already scores exactly zero. That is correct BM25, not
a bug, but it means lexical retrieval is close to useless on a toy index and only starts
contributing at realistic corpus sizes. There is a test pinning this so it cannot surprise
anyone later.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from retrieval_engine.logging_config import get_logger
from retrieval_engine.models import Chunk, Metadata, ScoredChunk, StageScores
from retrieval_engine.store.base import VectorStore, chunk_fingerprint

logger = get_logger(__name__)

#: Lowercased alphanumeric runs. Deliberately not a stemmer: BM25's IDF term already
#: discounts common words, and stemming would collapse the rare exact tokens that are the
#: reason to run lexical retrieval next to a dense retriever.
_TOKEN = re.compile(r"[a-z0-9]+")

INDEX_FILENAME = "bm25_index.json"


def tokenize(text: str) -> list[str]:
    """Split text into BM25 terms."""
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """BM25 Okapi over the same chunk set the dense retriever searches.

    The chunk set is held in memory. That is honest for a single-node corpus of this size
    and is the first thing that would have to change for a very large one, where the index
    would need to live outside the process and support incremental updates.
    """

    def __init__(self, store: VectorStore, index_dir: Path) -> None:
        self._store = store
        self._index_dir = index_dir
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._scorer: Any | None = None
        self._fingerprint: str | None = None
        self.rebuild_count = 0

    @property
    def index_path(self) -> Path:
        return self._index_dir / INDEX_FILENAME

    @property
    def fingerprint(self) -> str | None:
        return self._fingerprint

    def _load_persisted(self, fingerprint: str) -> list[list[str]] | None:
        """Return persisted tokens when they match ``fingerprint``, else None."""
        path = self.index_path
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt or unreadable cache is not a failure, it is a cache miss.
            logger.warning("bm25_index_unreadable", path=str(path))
            return None
        if payload.get("fingerprint") != fingerprint:
            return None
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            return None
        return [[str(term) for term in document] for document in tokens]

    def _persist(self, fingerprint: str, chunk_ids: list[str]) -> None:
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(
                {"fingerprint": fingerprint, "chunk_ids": chunk_ids, "tokens": self._tokens}
            ),
            encoding="utf-8",
        )

    async def ensure_index(self) -> str:
        """Build or reuse the BM25 index, returning the current fingerprint."""
        from rank_bm25 import BM25Okapi

        chunks = await self._store.all_chunks()
        fingerprint = chunk_fingerprint(chunk.chunk_id for chunk in chunks)

        if self._scorer is not None and self._fingerprint == fingerprint:
            return fingerprint

        self._chunks = chunks
        if not chunks:
            self._scorer = None
            self._tokens = []
            self._fingerprint = fingerprint
            return fingerprint

        persisted = self._load_persisted(fingerprint)
        if persisted is not None and len(persisted) == len(chunks):
            self._tokens = persisted
        else:
            self._tokens = [tokenize(chunk.text) for chunk in chunks]
            self.rebuild_count += 1
            self._persist(fingerprint, [chunk.chunk_id for chunk in chunks])
            logger.info("bm25_index_rebuilt", chunks=len(chunks), fingerprint=fingerprint[:12])

        # rank_bm25 cannot score an empty document, so give it a single sentinel term.
        self._scorer = BM25Okapi([tokens or ["\x00empty"] for tokens in self._tokens])
        self._fingerprint = fingerprint
        return fingerprint

    @staticmethod
    def _matches(chunk: Chunk, filters: Mapping[str, str] | None) -> bool:
        if not filters:
            return True
        metadata: Metadata = chunk.metadata
        for key, wanted in filters.items():
            value = metadata.get(key)
            if isinstance(value, list):
                if wanted not in value:
                    return False
            elif value is None or str(value) != wanted:
                return False
        return True

    async def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[ScoredChunk]:
        """Score every chunk against ``query`` and return the best ``top_k``."""
        await self.ensure_index()
        if self._scorer is None or top_k <= 0:
            return []

        terms = tokenize(query)
        if not terms:
            return []

        scores: Sequence[float] = self._scorer.get_scores(terms)
        candidates = [
            (float(score), chunk)
            for score, chunk in zip(scores, self._chunks, strict=True)
            if score > 0.0 and self._matches(chunk, filters)
        ]
        # Total ordering, so a tie cannot make two identical queries disagree.
        candidates.sort(key=lambda item: (-item[0], item[1].chunk_id))

        return [
            ScoredChunk(
                chunk=chunk,
                score=score,
                stages=StageScores(lexical_score=score, lexical_rank=rank),
            )
            for rank, (score, chunk) in enumerate(candidates[:top_k], start=1)
        ]


__all__ = ["INDEX_FILENAME", "BM25Retriever", "tokenize"]
