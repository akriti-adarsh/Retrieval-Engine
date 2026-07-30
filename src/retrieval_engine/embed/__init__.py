"""Turning text into vectors, locally by default."""

from __future__ import annotations

from typing import Any

from retrieval_engine.config import Settings
from retrieval_engine.embed.base import Embedder, EmbedderInfo, Tokenizer
from retrieval_engine.embed.local import HFTokenizer, LocalEmbedder
from retrieval_engine.embed.openai import ApproximateTokenizer, OpenAIEmbedder
from retrieval_engine.models import EmbedderKind


def build_embedder(settings: Settings, **kwargs: Any) -> Embedder:
    """Construct the embedder ``settings`` selects.

    ``kwargs`` passes through to the concrete class, which is how tests and
    :mod:`retrieval_engine.api.deps` inject a model factory or an httpx client without
    either of them needing to know which backend is configured.
    """
    if settings.embedder is EmbedderKind.OPENAI:
        return OpenAIEmbedder(settings, **kwargs)
    return LocalEmbedder(settings, **kwargs)


__all__ = [
    "ApproximateTokenizer",
    "Embedder",
    "EmbedderInfo",
    "HFTokenizer",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "Tokenizer",
    "build_embedder",
]
