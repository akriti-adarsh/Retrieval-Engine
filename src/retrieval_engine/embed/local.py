"""Local embedding via sentence-transformers, and the tokenizer the chunker slices on.

The model is loaded lazily. Constructing the FastAPI app must not block on a 130MB
download, and the eval harness builds several embedder objects while only ever using some
of them, so paying for the load in ``__init__`` would be wasteful and slow to start.

Encoding is CPU bound and synchronous, so it runs inside ``asyncio.to_thread``. Without
that, one embed call would stall every other request on the event loop.
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

from retrieval_engine.config import Settings, set_seeds
from retrieval_engine.embed.base import EmbedderInfo
from retrieval_engine.errors import ConfigurationError

#: Fallback tokenization when a tokenizer reports no usable offsets, which happens for
#: text made entirely of characters the vocabulary drops (for example lone control chars).
_WORD = re.compile(r"\S+")


class SentenceTransformerLike(Protocol):
    """The slice of the sentence-transformers API this wrapper actually uses.

    Declared as a protocol so tests can inject a stub instead of downloading a model, and
    so the type checker verifies we only rely on the three members we claim to.
    """

    tokenizer: Any

    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        """Encode a batch of texts into vectors."""
        ...


def reported_dimension(model: SentenceTransformerLike) -> int | None:
    """The model's vector width, tolerating the sentence-transformers rename.

    sentence-transformers 5.x renamed ``get_sentence_embedding_dimension`` to
    ``get_embedding_dimension`` and warns on the old name. Prefer the new one, accept the
    old one so an older pin still works, and treat neither being present as "unknown"
    rather than as an error, since the configured dimension is then authoritative.
    """
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        method = getattr(model, name, None)
        if callable(method):
            value = method()
            return int(value) if value is not None else None
    return None


class HFTokenizer:
    """Token counts and character offsets from a HuggingFace fast tokenizer.

    ``count_tokens`` reports the tokenizer's true count while ``token_spans`` returns only
    the spans that consume characters. The two deliberately differ: markdown rules and
    table separators tokenize into pieces that map to no characters, so counting only
    sliceable spans undercounts. That undercount let 733-token chunks reach a model with a
    512-token limit, where they were silently truncated and lost their tails. Budgeting on
    the true count is conservative, which is the correct direction for this error.

    Every call is serialised on a lock shared with the embedder. The fast tokenizer is a
    Rust object that cannot be borrowed mutably from two threads at once, and encoding runs
    in a worker thread while the chunker tokenizes on the event loop thread, so without
    this concurrent ingestion dies with "Already borrowed".
    """

    def __init__(self, tokenizer: Any, lock: threading.Lock | None = None) -> None:
        self._tokenizer = tokenizer
        self._lock = lock if lock is not None else threading.Lock()

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        with self._lock:
            encoded = self._tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        offsets = encoded["offset_mapping"]
        spans = [(int(start), int(end)) for start, end in offsets if int(end) > int(start)]
        if spans:
            return spans
        # A tokenizer that yields nothing usable would make the chunker drop the document
        # entirely, so fall back to whitespace spans rather than losing the text.
        return [(match.start(), match.end()) for match in _WORD.finditer(text)]

    def count_tokens(self, text: str) -> int:
        """The tokenizer's own count, including tokens that consume no characters."""
        if not text:
            return 0
        with self._lock:
            encoded = self._tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"]
        return len(ids)


class LocalEmbedder:
    """sentence-transformers embedder, local and key-free.

    ``embed_query`` applies the configured instruction prefix and ``embed_documents``
    never does. bge models document that asymmetry for retrieval, and getting it backwards
    quietly costs recall rather than raising, so the two paths are separate methods.
    """

    def __init__(
        self,
        settings: Settings,
        model_factory: Callable[[], SentenceTransformerLike] | None = None,
    ) -> None:
        self._settings = settings
        self._factory = model_factory if model_factory is not None else self._default_factory
        self._model: SentenceTransformerLike | None = None
        self._tokenizer: HFTokenizer | None = None
        self._dimension: int | None = None
        # Shared by the tokenizer wrapper and the encode path. The HuggingFace fast
        # tokenizer is a Rust object that raises "Already borrowed" if two threads use it
        # at once, and encoding runs in a worker thread while chunking tokenizes on the
        # event loop thread. Serialising is also honest about the hardware: a CPU forward
        # pass already saturates the cores, so nothing is lost by not overlapping them.
        self._lock = threading.Lock()

    def _default_factory(self) -> SentenceTransformerLike:
        from sentence_transformers import SentenceTransformer

        # torch is imported by the line above, so seeding it is only possible from here on.
        set_seeds(self._settings.seed)
        return cast(
            "SentenceTransformerLike",
            SentenceTransformer(
                self._settings.embedding_model,
                device=self._settings.embedding_device,
            ),
        )

    def _load(self) -> SentenceTransformerLike:
        """Load the model once, verifying it produces the width the store expects."""
        if self._model is None:
            model = self._factory()
            reported = reported_dimension(model)
            if reported is not None and reported != self._settings.embedding_dimension:
                msg = (
                    f"model {self._settings.embedding_model!r} produces {reported}-dim vectors "
                    f"but RE_EMBEDDING_DIMENSION is {self._settings.embedding_dimension}; "
                    "a mismatch here corrupts the whole index"
                )
                raise ConfigurationError(msg)
            self._dimension = reported or self._settings.embedding_dimension
            self._model = model
        return self._model

    @property
    def info(self) -> EmbedderInfo:
        """Identity of this embedding space.

        Readable before the model loads, using the configured width, which ``_load`` then
        checks against what the model actually reports.
        """
        return EmbedderInfo(
            name=self._settings.embedding_model,
            dimension=self._dimension or self._settings.embedding_dimension,
            normalized=self._settings.normalize_embeddings,
            query_instruction=(
                self._settings.query_instruction if self._settings.use_query_instruction else None
            ),
        )

    @property
    def tokenizer(self) -> HFTokenizer:
        model = self._load()
        if self._tokenizer is None:
            self._tokenizer = HFTokenizer(model.tokenizer, self._lock)
        return self._tokenizer

    def _rows(self, array: Any) -> list[list[float]]:
        expected = self._dimension or self._settings.embedding_dimension
        rows: list[list[float]] = []
        for row in array:
            values = [float(value) for value in row]
            if len(values) != expected:
                msg = f"embedder returned a {len(values)}-dim vector, expected {expected}"
                raise ConfigurationError(msg)
            rows.append(values)
        return rows

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        batch_size = self._settings.embedding_batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            with self._lock:
                array = model.encode(
                    texts[start : start + batch_size],
                    batch_size=batch_size,
                    normalize_embeddings=self._settings.normalize_embeddings,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            vectors.extend(self._rows(array))
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages. The query instruction is never applied here."""
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, list(texts))

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query, prefixing the bge retrieval instruction when configured."""
        prompt = (
            f"{self._settings.query_instruction}{text}"
            if self._settings.use_query_instruction
            else text
        )
        vectors = await asyncio.to_thread(self._encode, [prompt])
        return vectors[0]


__all__ = ["HFTokenizer", "LocalEmbedder", "SentenceTransformerLike", "reported_dimension"]
