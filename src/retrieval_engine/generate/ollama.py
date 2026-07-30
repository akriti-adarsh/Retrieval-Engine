"""Ollama client, the default generation backend.

Ollama is the default because it needs no API key and runs on the same machine as
everything else, which is what makes the "clone it and it works" claim true. httpx talks to
its HTTP API directly; the surface used is two endpoints, so a client library would be a
dependency bought for nothing and would make the failure paths harder to test.

Every failure here becomes :class:`LLMUnavailableError`, which callers catch to fall back to
extraction. ``health`` is the one method that must never raise, because readiness checks are
supposed to report a problem, not become one.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from retrieval_engine.config import Settings
from retrieval_engine.errors import LLMUnavailableError

GENERATE_PATH = "/api/generate"
TAGS_PATH = "/api/tags"


class OllamaLLM:
    """Text generation against a local Ollama server."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client

    @property
    def model(self) -> str:
        return self._settings.ollama_model

    def _url(self, path: str) -> str:
        return f"{self._settings.ollama_base_url.rstrip('/')}{path}"

    def _payload(
        self,
        prompt: str,
        max_tokens: int | None,
        temperature: float | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                # Temperature defaults to 0 in settings, because an eval harness that
                # reports different numbers on a re-run is not measuring anything.
                "temperature": (
                    temperature
                    if temperature is not None
                    else self._settings.generation_temperature
                ),
                "num_predict": (
                    max_tokens if max_tokens is not None else self._settings.generation_max_tokens
                ),
                "seed": self._settings.seed,
            },
        }

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        url = self._url(GENERATE_PATH)
        payload = self._payload(prompt, max_tokens, temperature, stream=False)
        try:
            if self._client is not None:
                response = await self._client.post(url, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self._settings.ollama_timeout_s) as client:
                    response = await client.post(url, json=payload)
        except httpx.RequestError as exc:
            msg = f"ollama at {url} is unreachable: {type(exc).__name__}: {exc}"
            raise LLMUnavailableError(msg) from exc

        if response.status_code >= 400:
            msg = f"ollama returned HTTP {response.status_code}: {response.text[:200]}"
            raise LLMUnavailableError(msg)

        try:
            body: Any = response.json()
            text = body["response"]
        except (ValueError, KeyError, TypeError) as exc:
            msg = f"ollama returned an unusable body: {type(exc).__name__}: {exc}"
            raise LLMUnavailableError(msg) from exc

        if not isinstance(text, str):
            msg = f"ollama returned a {type(text).__name__} where text was expected"
            raise LLMUnavailableError(msg)
        return text

    async def _deltas(
        self, client: httpx.AsyncClient, url: str, payload: dict[str, Any]
    ) -> AsyncIterator[str]:
        """Yield text deltas from Ollama's newline-delimited JSON stream."""
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code >= 400:
                await response.aread()
                msg = f"ollama returned HTTP {response.status_code} on the stream"
                raise LLMUnavailableError(msg)
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    event: Any = json.loads(line)
                except ValueError as exc:
                    msg = f"ollama emitted an unparseable stream line: {exc}"
                    raise LLMUnavailableError(msg) from exc
                delta = event.get("response") if isinstance(event, dict) else None
                if isinstance(delta, str) and delta:
                    yield delta
                if isinstance(event, dict) and event.get("done"):
                    return

    async def stream(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        url = self._url(GENERATE_PATH)
        payload = self._payload(prompt, max_tokens, temperature, stream=True)
        try:
            if self._client is not None:
                async for delta in self._deltas(self._client, url, payload):
                    yield delta
            else:
                async with httpx.AsyncClient(timeout=self._settings.ollama_timeout_s) as client:
                    async for delta in self._deltas(client, url, payload):
                        yield delta
        except httpx.RequestError as exc:
            msg = f"ollama stream from {url} failed: {type(exc).__name__}: {exc}"
            raise LLMUnavailableError(msg) from exc

    async def health(self) -> bool:
        """Whether the server answers. Never raises, by contract."""
        url = self._url(TAGS_PATH)
        try:
            if self._client is not None:
                response = await self._client.get(url)
            else:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(url)
        except httpx.RequestError:
            return False
        except Exception:  # a health check must never be the thing that breaks
            return False
        return response.status_code < 400


__all__ = ["GENERATE_PATH", "TAGS_PATH", "OllamaLLM"]
