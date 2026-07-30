"""Optional remote embedding through the OpenAI-compatible embeddings API.

Two deliberate choices.

This talks to the HTTP API with httpx rather than adding the openai SDK. The whole surface
used here is one POST, so the SDK would be a dependency (and a supply-chain surface) bought
for nothing, and httpx lets every error path be tested with a MockTransport instead of a
patched client.

Token counts are approximate in this configuration, and that is stated rather than hidden.
The real BPE vocabulary is not available locally, so chunk sizes measured with
:class:`ApproximateTokenizer` will not match the server's accounting exactly. The local
embedder remains the default precisely because its token counts are exact.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import httpx

from retrieval_engine.config import Settings
from retrieval_engine.embed.base import EmbedderInfo
from retrieval_engine.errors import ConfigurationError, EmbeddingBackendError

#: Words and standalone punctuation, which tracks BPE token counts loosely but predictably.
_TOKEN = re.compile(r"\w+|[^\w\s]")

_TIMEOUT_SECONDS = 60.0


class ApproximateTokenizer:
    """Regex tokenizer used when the real vocabulary is not available locally.

    Counts words and standalone punctuation marks. This overestimates for long words that
    BPE would split and underestimates for rare ones, so chunk sizes are approximate under
    the remote embedder. Callers who need exact budgets should use the local embedder.
    """

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in _TOKEN.finditer(text)]

    def count_tokens(self, text: str) -> int:
        return len(self.token_spans(text))


class OpenAIEmbedder:
    """Embedder backed by an OpenAI-compatible ``/embeddings`` endpoint.

    No instruction prefix is applied to queries: the bge query instruction is specific to
    bge checkpoints and prepending it to an OpenAI model would only add noise.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        if settings.openai_api_key is None:
            msg = "embedder=openai requires RE_OPENAI_API_KEY"
            raise ConfigurationError(msg)
        self._settings = settings
        self._client = client
        self._tokenizer = ApproximateTokenizer()

    @property
    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            name=self._settings.openai_embedding_model,
            dimension=self._settings.embedding_dimension,
            normalized=False,
            query_instruction=None,
        )

    @property
    def tokenizer(self) -> ApproximateTokenizer:
        return self._tokenizer

    def _headers(self) -> dict[str, str]:
        key = self._settings.openai_api_key
        # Constructor rejects a missing key, so this is a type narrowing, not a check.
        secret = "" if key is None else key.get_secret_value()
        return {
            "Authorization": f"Bearer {secret}",
            "Content-Type": "application/json",
        }

    async def _post(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self._settings.openai_embedding_model, "input": texts}
        url = f"{self._settings.openai_base_url.rstrip('/')}/embeddings"
        try:
            if self._client is not None:
                response = await self._client.post(url, json=payload, headers=self._headers())
            else:
                # A client per call rather than a long-lived one, because the Embedder
                # protocol has no close hook and a leaked connection pool is worse than a
                # new handshake on an optional, non-default code path.
                async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                    response = await client.post(url, json=payload, headers=self._headers())
        except httpx.RequestError as exc:
            msg = f"embedding request to {url} failed: {type(exc).__name__}: {exc}"
            raise EmbeddingBackendError(msg) from exc

        if response.status_code >= 400:
            msg = f"embedding backend returned HTTP {response.status_code}: {response.text[:200]}"
            raise EmbeddingBackendError(msg)

        return self._parse(response, expected=len(texts))

    def _parse(self, response: httpx.Response, expected: int) -> list[list[float]]:
        try:
            body: Any = response.json()
            items = body["data"]
            # The API does not promise input order, so reorder by the index it returns.
            ordered = sorted(items, key=lambda item: int(item["index"]))
            vectors = [[float(value) for value in item["embedding"]] for item in ordered]
        except (ValueError, KeyError, TypeError) as exc:
            msg = f"embedding backend returned an unusable body: {type(exc).__name__}: {exc}"
            raise EmbeddingBackendError(msg) from exc

        if len(vectors) != expected:
            msg = f"embedding backend returned {len(vectors)} vectors for {expected} inputs"
            raise EmbeddingBackendError(msg)
        for vector in vectors:
            if len(vector) != self._settings.embedding_dimension:
                msg = (
                    f"embedding backend returned {len(vector)}-dim vectors but "
                    f"RE_EMBEDDING_DIMENSION is {self._settings.embedding_dimension}"
                )
                raise ConfigurationError(msg)
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        batch_size = self._settings.embedding_batch_size
        listed = list(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(listed), batch_size):
            vectors.extend(await self._post(listed[start : start + batch_size]))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._post([text])
        return vectors[0]


__all__ = ["ApproximateTokenizer", "OpenAIEmbedder"]
