"""Answer generation, with an extractive path that needs no model server."""

from __future__ import annotations

from typing import Any

from retrieval_engine.config import Settings
from retrieval_engine.generate.base import LLM
from retrieval_engine.generate.extractive import ExtractiveAnswerer
from retrieval_engine.generate.ollama import OllamaLLM
from retrieval_engine.generate.prompts import (
    INSUFFICIENT_ANSWER,
    PROMPT_VERSION,
    build_answer_prompt,
    render_sources,
)
from retrieval_engine.models import LLMKind


def build_llm(settings: Settings, **kwargs: Any) -> LLM | None:
    """Construct the configured language model, or ``None`` for the extractive path.

    Returning ``None`` rather than a null-object LLM keeps the fallback explicit at the call
    site: a caller holding ``None`` can see that it must extract, while a null object would
    let a silent empty answer look like a successful generation.
    """
    if settings.llm is LLMKind.EXTRACTIVE:
        return None
    return OllamaLLM(settings, **kwargs)


__all__ = [
    "INSUFFICIENT_ANSWER",
    "LLM",
    "PROMPT_VERSION",
    "ExtractiveAnswerer",
    "OllamaLLM",
    "build_answer_prompt",
    "build_llm",
    "render_sources",
]
