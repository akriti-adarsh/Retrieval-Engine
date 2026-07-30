"""Cross-encoder reranking over the fused shortlist.

A cross-encoder concatenates query and passage into one sequence, so attention runs across
both and term interaction is modelled directly. That is why it outperforms a bi-encoder on
ranking, and also why nothing can be precomputed: every query and passage pair costs a
forward pass. Reranking is therefore applied to a shortlist, never to the whole corpus, and
it is the single largest latency contributor in the pipeline. The ablation exists partly to
show what that latency buys.

Scores are cached in an LRU keyed by ``(query_hash, chunk_id)``. The same query repeating
against a stable index is the common case in a demo, an eval re-run, and any UI with a
back button, and a cache hit turns a forward pass into a dict lookup.

The model shares one lock with itself across threads for the same reason the embedder does:
the underlying HuggingFace fast tokenizer is a Rust object that raises "Already borrowed"
when two threads use it at once.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

from retrieval_engine.config import Settings, set_seeds
from retrieval_engine.models import ScoredChunk


class CrossEncoderLike(Protocol):
    """The slice of the sentence-transformers CrossEncoder API used here."""

    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> Any:
        """Score query and passage pairs."""
        ...


def query_hash(query: str) -> str:
    """Short stable digest of a query, used as the cache key's first component.

    Hashing rather than storing the query keeps cache keys small and uniform, and blake2b
    rather than the builtin hash because the builtin is salted per process and would make
    the cache behave differently across restarts.
    """
    return hashlib.blake2b(query.encode("utf-8"), digest_size=8).hexdigest()


class ScoreCache:
    """Bounded LRU over ``(query_hash, chunk_id) -> score``."""

    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            msg = f"cache size must be positive, got {maxsize}"
            raise ValueError(msg)
        self._maxsize = maxsize
        self._entries: OrderedDict[tuple[str, str], float] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[str, str]) -> float | None:
        if key in self._entries:
            self._entries.move_to_end(key)
            self.hits += 1
            return self._entries[key]
        self.misses += 1
        return None

    def put(self, key: tuple[str, str], score: float) -> None:
        self._entries[key] = score
        self._entries.move_to_end(key)
        while len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from cache, 0.0 before any lookup."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self._entries)


class CrossEncoderReranker:
    """Reorders a shortlist with a cross-encoder, caching scores per query and chunk."""

    def __init__(
        self,
        settings: Settings,
        model_factory: Callable[[], CrossEncoderLike] | None = None,
    ) -> None:
        self._settings = settings
        self._factory = model_factory if model_factory is not None else self._default_factory
        self._model: CrossEncoderLike | None = None
        self._lock = threading.Lock()
        self.cache = ScoreCache(settings.reranker_cache_size)

    def _default_factory(self) -> CrossEncoderLike:
        from sentence_transformers import CrossEncoder

        # torch is imported by the line above, so this is the first point it can be seeded.
        set_seeds(self._settings.seed)
        return cast(
            "CrossEncoderLike",
            CrossEncoder(self._settings.reranker_model, device=self._settings.embedding_device),
        )

    def _load(self) -> CrossEncoderLike:
        """Load the model on first use. It is roughly 1.5 GB, so never in __init__."""
        if self._model is None:
            self._model = self._factory()
        return self._model

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._load()
        batch_size = self._settings.reranker_batch_size
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            with self._lock:
                predicted = model.predict(pairs[start : start + batch_size])
            scores.extend(float(value) for value in predicted)
        return scores

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        """Score every candidate against ``query`` and return the best ``top_k``.

        The fused rank is preserved on each result, so rank movement through this stage
        stays inspectable rather than being overwritten.
        """
        if not candidates or top_k <= 0:
            return []

        digest = query_hash(query)
        pending: list[tuple[int, tuple[str, str]]] = []
        scores: list[float | None] = []
        for index, candidate in enumerate(candidates):
            cached = self.cache.get((digest, candidate.chunk.chunk_id))
            scores.append(cached)
            if cached is None:
                pending.append((index, (query, candidate.chunk.text)))

        if pending:
            predicted = await asyncio.to_thread(self._predict, [pair for _, pair in pending])
            for (index, _), score in zip(pending, predicted, strict=True):
                scores[index] = score
                self.cache.put((digest, candidates[index].chunk.chunk_id), score)

        scored = [
            (float(score if score is not None else 0.0), candidate)
            for score, candidate in zip(scores, candidates, strict=True)
        ]
        # Total ordering, so a tie cannot make two identical queries disagree.
        scored.sort(key=lambda item: (-item[0], item[1].chunk.chunk_id))

        return [
            ScoredChunk(
                chunk=candidate.chunk,
                score=score,
                stages=candidate.stages.model_copy(
                    update={"rerank_score": score, "final_rank": rank}
                ),
            )
            for rank, (score, candidate) in enumerate(scored[:top_k], start=1)
        ]


__all__ = ["CrossEncoderLike", "CrossEncoderReranker", "ScoreCache", "query_hash"]
