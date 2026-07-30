"""Embedder and tokenizer protocols.

Two deliberate choices live here.

``embed_query`` and ``embed_documents`` are separate methods rather than one call with
a flag, because bge models document a query-side instruction prefix that must never be
applied to passages. Splitting the methods makes that asymmetry impossible to get wrong
at a call site.

``Tokenizer.token_spans`` returns character offsets rather than token ids, because the
chunker needs to slice the original text at token boundaries. Decoding token ids back
to text would lose whitespace and subword detail, so offsets are the primitive.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import Field

from retrieval_engine.models import Base


class EmbedderInfo(Base):
    """Identity of an embedding space.

    A collection records this at creation and refuses vectors that do not match, so a
    config change cannot silently mix two embedding spaces in one index.
    """

    name: str = Field(min_length=1, description="Stable id, e.g. 'BAAI/bge-small-en-v1.5'.")
    dimension: int = Field(gt=0)
    normalized: bool = Field(
        default=True, description="Whether vectors are L2-normalised, making dot product cosine."
    )
    query_instruction: str | None = Field(
        default=None, description="Prefix applied to queries only, None when unused."
    )


@runtime_checkable
class Tokenizer(Protocol):
    """The minimum a chunker needs: how many tokens, and where they sit in the text."""

    def count_tokens(self, text: str) -> int:
        """Number of tokens in ``text``, excluding any special tokens."""
        ...

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """``(start_char, end_char)`` for each token, in order, into ``text`` itself.

        Spans must be non-decreasing and lie within ``text``. Tokens that consume no
        characters (a possibility for some tokenizers) are omitted.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Async so the API can embed without blocking the loop."""

    @property
    def info(self) -> EmbedderInfo:
        """Identity of this embedding space."""
        ...

    @property
    def tokenizer(self) -> Tokenizer:
        """The model's own tokenizer, so chunk sizes are measured in real tokens."""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages. No instruction prefix is applied.

        Returns one vector per input, in input order.
        """
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query, applying the query instruction when configured."""
        ...


__all__ = ["Embedder", "EmbedderInfo", "Tokenizer"]
