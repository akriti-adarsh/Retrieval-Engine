"""The LLM protocol.

Deliberately narrow: a prompt in, text out, plus a streaming variant and a health check.
Everything that makes this service's answers trustworthy (numbered sources, citation
markers, grounding verification, refusal) is built on top of this interface rather than
inside it, so swapping Ollama for anything else cannot change those guarantees.

``stream`` is declared as a plain method returning an async iterator, not as ``async def``,
because an async generator function's call already returns an ``AsyncIterator``.

Implementations raise :class:`retrieval_engine.errors.LLMUnavailableError` when the backend
cannot be reached. Callers are expected to catch it and degrade, never to propagate a 5xx:
the extractive path exists precisely so a stopped model server is a quality reduction rather
than an outage.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    """Text generation."""

    @property
    def model(self) -> str:
        """Model identifier, recorded on every response so answers are attributable."""
        ...

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a full completion.

        Raises:
            LLMUnavailableError: the backend is unreachable or returned nothing usable.
        """
        ...

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas as they arrive.

        Raises:
            LLMUnavailableError: the backend is unreachable or the stream breaks.
        """
        ...

    async def health(self) -> bool:
        """Whether the backend is reachable right now. Must not raise."""
        ...


__all__ = ["LLM"]
