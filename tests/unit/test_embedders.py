"""Embedders: the query/passage asymmetry, batching, and every remote error path.

No test here downloads a model or opens a socket. The local embedder takes an injected
model factory and the remote one takes an injected httpx client with a MockTransport, which
is also what lets the failure branches be exercised at all.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import httpx
import numpy as np
import pytest

from retrieval_engine.config import Settings
from retrieval_engine.embed import build_embedder
from retrieval_engine.embed.base import Embedder
from retrieval_engine.embed.local import HFTokenizer, LocalEmbedder
from retrieval_engine.embed.openai import ApproximateTokenizer, OpenAIEmbedder
from retrieval_engine.errors import ConfigurationError, EmbeddingBackendError
from retrieval_engine.models import EmbedderKind

DIMENSION = 8


class StubTokenizer:
    """Callable with the HuggingFace fast-tokenizer signature this wrapper relies on."""

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        self.last_kwargs = {
            "add_special_tokens": add_special_tokens,
            "return_offsets_mapping": return_offsets_mapping,
        }
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"offset_mapping": spans, "input_ids": list(range(len(spans)))}


class StubModel:
    """Records what it was asked to encode so batching and prefixes can be asserted."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        self._dimension = dimension
        self.tokenizer = StubTokenizer()
        self.batches: list[list[str]] = []
        self.kwargs: dict[str, Any] = {}

    def get_embedding_dimension(self) -> int | None:
        return self._dimension

    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        self.batches.append(list(sentences))
        self.kwargs = kwargs
        return np.array(
            [[float(len(text))] * self._dimension for text in sentences],
            dtype=np.float32,
        )


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "_env_file": None,
        "env": "test",
        "embedding_dimension": DIMENSION,
        "embedding_batch_size": 4,
    }
    base.update(overrides)
    return Settings(**base)


def _local(model: StubModel | None = None, **overrides: Any) -> tuple[LocalEmbedder, StubModel]:
    stub = model if model is not None else StubModel()
    return LocalEmbedder(_settings(**overrides), lambda: stub), stub


# --- protocol conformance ---------------------------------------------------------------


def test_local_embedder_satisfies_the_protocol() -> None:
    embedder, _ = _local()

    assert isinstance(embedder, Embedder)


def test_openai_embedder_satisfies_the_protocol() -> None:
    embedder = OpenAIEmbedder(
        _settings(embedder=EmbedderKind.OPENAI, openai_api_key="sk-test"),
    )

    assert isinstance(embedder, Embedder)


# --- lazy loading -----------------------------------------------------------------------


def test_model_is_not_loaded_until_it_is_needed() -> None:
    """Constructing the API app must not block on a model load."""
    calls = 0

    def factory() -> StubModel:
        nonlocal calls
        calls += 1
        return StubModel()

    embedder = LocalEmbedder(_settings(), factory)
    assert calls == 0

    # info is readable without a load, using the configured width.
    assert embedder.info.dimension == DIMENSION
    assert calls == 0

    _ = embedder.tokenizer
    assert calls == 1


async def test_model_is_loaded_only_once() -> None:
    embedder, stub = _local()

    await embedder.embed_documents(["a"])
    await embedder.embed_documents(["b"])
    _ = embedder.tokenizer

    assert len(stub.batches) == 2


# --- the query and passage asymmetry ----------------------------------------------------


async def test_query_gets_the_instruction_prefix() -> None:
    embedder, stub = _local()

    await embedder.embed_query("what is reciprocal rank fusion")

    sent = stub.batches[-1][0]
    assert sent.startswith("Represent this sentence for searching relevant passages: ")
    assert sent.endswith("what is reciprocal rank fusion")


async def test_documents_never_get_the_instruction_prefix() -> None:
    """Prefixing a passage costs recall silently, so this is asserted, not assumed."""
    embedder, stub = _local()

    await embedder.embed_documents(["a passage about bm25", "another passage"])

    for text in stub.batches[-1]:
        assert "Represent this sentence" not in text


async def test_instruction_can_be_switched_off() -> None:
    embedder, stub = _local(use_query_instruction=False)

    await embedder.embed_query("plain query")

    assert stub.batches[-1] == ["plain query"]
    assert embedder.info.query_instruction is None


def test_info_records_the_instruction_when_in_use() -> None:
    """Runs must be attributable, so the instruction's presence is part of the identity."""
    embedder, _ = _local()

    assert embedder.info.query_instruction is not None
    assert embedder.info.name == "BAAI/bge-small-en-v1.5"
    assert embedder.info.normalized is True


# --- batching ---------------------------------------------------------------------------


async def test_documents_are_encoded_in_configured_batches() -> None:
    embedder, stub = _local()

    vectors = await embedder.embed_documents([f"text {index}" for index in range(10)])

    assert [len(batch) for batch in stub.batches] == [4, 4, 2]
    assert len(vectors) == 10
    assert all(len(vector) == DIMENSION for vector in vectors)


async def test_normalisation_flag_is_forwarded_to_the_model() -> None:
    embedder, stub = _local()

    await embedder.embed_documents(["x"])

    assert stub.kwargs["normalize_embeddings"] is True
    assert stub.kwargs["show_progress_bar"] is False


async def test_empty_input_does_no_work() -> None:
    embedder, stub = _local()

    assert await embedder.embed_documents([]) == []
    assert stub.batches == []


async def test_vector_order_matches_input_order() -> None:
    embedder, _ = _local()

    vectors = await embedder.embed_documents(["a", "bbb", "cc"])

    # The stub encodes text length, so order is checkable.
    assert [vector[0] for vector in vectors] == [1.0, 3.0, 2.0]


# --- dimension guards -------------------------------------------------------------------


async def test_the_pre_rename_dimension_method_still_works() -> None:
    """sentence-transformers 5.x renamed the accessor, so both spellings are accepted."""

    class OldStyleModel:
        def __init__(self) -> None:
            self.tokenizer = StubTokenizer()

        def get_sentence_embedding_dimension(self) -> int | None:
            return DIMENSION

        def encode(self, sentences: list[str], **kwargs: Any) -> Any:
            return np.zeros((len(sentences), DIMENSION), dtype=np.float32)

    embedder = LocalEmbedder(_settings(), OldStyleModel)

    vectors = await embedder.embed_documents(["a"])

    assert len(vectors[0]) == DIMENSION


async def test_a_model_reporting_no_dimension_trusts_the_config() -> None:
    class SilentModel:
        def __init__(self) -> None:
            self.tokenizer = StubTokenizer()

        def encode(self, sentences: list[str], **kwargs: Any) -> Any:
            return np.zeros((len(sentences), DIMENSION), dtype=np.float32)

    vectors = await LocalEmbedder(_settings(), SilentModel).embed_documents(["a"])

    assert len(vectors[0]) == DIMENSION


async def test_model_dimension_mismatch_is_fatal() -> None:
    """A silent width mismatch corrupts the index invisibly, so it must raise."""
    embedder, _ = _local(model=StubModel(dimension=384), embedding_dimension=DIMENSION)

    with pytest.raises(ConfigurationError, match="corrupts the whole index"):
        await embedder.embed_documents(["x"])


async def test_row_width_mismatch_is_fatal() -> None:
    class WrongWidthModel(StubModel):
        def encode(self, sentences: list[str], **kwargs: Any) -> Any:
            self.batches.append(list(sentences))
            return np.zeros((len(sentences), 3), dtype=np.float32)

    embedder = LocalEmbedder(_settings(), WrongWidthModel)

    with pytest.raises(ConfigurationError, match="3-dim vector, expected 8"):
        await embedder.embed_documents(["x"])


# --- tokenizer --------------------------------------------------------------------------


def test_tokenizer_reports_offsets_into_the_original_text() -> None:
    tokenizer = HFTokenizer(StubTokenizer())
    text = "dense and lexical retrieval"

    spans = tokenizer.token_spans(text)

    assert [text[start:end] for start, end in spans] == [
        "dense",
        "and",
        "lexical",
        "retrieval",
    ]
    assert tokenizer.count_tokens(text) == 4


def test_tokenizer_asks_for_offsets_without_special_tokens() -> None:
    """Special tokens would report offsets that do not exist in the source text."""
    stub = StubTokenizer()

    HFTokenizer(stub).token_spans("some text")

    assert stub.last_kwargs == {"add_special_tokens": False, "return_offsets_mapping": True}


def test_tokenizer_handles_empty_text() -> None:
    tokenizer = HFTokenizer(StubTokenizer())

    assert tokenizer.token_spans("") == []
    assert tokenizer.count_tokens("") == 0


def test_tokenizer_drops_zero_width_spans() -> None:
    class ZeroWidthTokenizer:
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            return {"offset_mapping": [(0, 4), (4, 4), (5, 9)]}

    spans = HFTokenizer(ZeroWidthTokenizer()).token_spans("some text")

    assert spans == [(0, 4), (5, 9)]


def test_tokenizer_falls_back_rather_than_losing_the_document() -> None:
    class UselessTokenizer:
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            return {"offset_mapping": []}

    spans = HFTokenizer(UselessTokenizer()).token_spans("two words")

    assert spans == [(0, 3), (4, 9)]


def test_count_tokens_includes_tokens_that_consume_no_characters() -> None:
    """Regression: counting only sliceable spans undercounts and lets oversized chunks
    through. Markdown rules and table separators tokenize into pieces with zero-width
    offsets, which is how 733-token chunks reached a model with a 512-token limit and were
    silently truncated. The budget must use the tokenizer's real count.
    """

    class RuleTokenizer:
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            # Four real tokens, three that consume no characters.
            return {
                "offset_mapping": [(0, 4), (4, 4), (5, 9), (9, 9), (10, 14), (14, 14), (15, 19)],
                "input_ids": list(range(7)),
            }

    tokenizer = HFTokenizer(RuleTokenizer())

    assert len(tokenizer.token_spans("some text here nowx")) == 4
    assert tokenizer.count_tokens("some text here nowx") == 7


async def test_concurrent_encode_and_tokenize_never_overlap() -> None:
    """Regression: the Rust fast tokenizer raises "Already borrowed" when two threads use
    it at once. Encoding runs in a worker thread while the chunker tokenizes on the event
    loop thread, which killed concurrent ingestion outright until both took one lock.
    """

    class BorrowTracker:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        def use(self) -> None:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            # Long enough that an unlocked second caller would be seen inside the window.
            time.sleep(0.005)
            self.active -= 1

    tracker = BorrowTracker()

    class TrackingTokenizer:
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            tracker.use()
            spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
            return {"offset_mapping": spans, "input_ids": list(range(len(spans)))}

    class TrackingModel(StubModel):
        def __init__(self) -> None:
            super().__init__()
            self.tokenizer = TrackingTokenizer()

        def encode(self, sentences: list[str], **kwargs: Any) -> Any:
            tracker.use()
            return super().encode(sentences, **kwargs)

    embedder = LocalEmbedder(_settings(), TrackingModel)
    tokenizer = embedder.tokenizer

    await asyncio.gather(
        embedder.embed_documents([f"passage {index}" for index in range(8)]),
        asyncio.to_thread(tokenizer.count_tokens, "a chunk being measured"),
        asyncio.to_thread(tokenizer.token_spans, "another chunk being sliced"),
        embedder.embed_query("a query"),
    )

    assert tracker.max_active == 1, "tokenizer and encode must never be in use at once"


def test_approximate_tokenizer_counts_words_and_punctuation() -> None:
    tokenizer = ApproximateTokenizer()

    assert tokenizer.count_tokens("hello, world") == 3
    assert tokenizer.token_spans("ab c")[0] == (0, 2)


# --- remote embedder --------------------------------------------------------------------


def _openai(handler: Any, **overrides: Any) -> OpenAIEmbedder:
    settings = _settings(
        embedder=EmbedderKind.OPENAI,
        openai_api_key="sk-test",
        embedding_dimension=3,
        **overrides,
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAIEmbedder(settings, client)


def test_missing_key_is_rejected_at_construction() -> None:
    with pytest.raises(ConfigurationError, match="requires RE_OPENAI_API_KEY"):
        OpenAIEmbedder(_settings())


async def test_request_shape_and_auth_header() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
        )

    vectors = await _openai(handler).embed_query("what is hnsw")

    assert seen["url"] == "https://api.openai.com/v1/embeddings"
    assert seen["auth"] == "Bearer sk-test"
    assert "text-embedding-3-small" in seen["body"]
    assert "what is hnsw" in seen["body"]
    # The key must never travel in the query string.
    assert "sk-test" not in seen["url"]
    assert vectors == [0.1, 0.2, 0.3]


async def test_out_of_order_response_is_reordered_by_index() -> None:
    """The API does not promise input order, and mismatched vectors would be invisible."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 2, "embedding": [3.0, 3.0, 3.0]},
                    {"index": 0, "embedding": [1.0, 1.0, 1.0]},
                    {"index": 1, "embedding": [2.0, 2.0, 2.0]},
                ]
            },
        )

    vectors = await _openai(handler).embed_documents(["a", "b", "c"])

    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0]


async def test_remote_batches_by_configured_size() -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.read())
        sizes.append(len(payload["input"]))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [0.0, 0.0, 0.0]}
                    for index in range(len(payload["input"]))
                ]
            },
        )

    await _openai(handler).embed_documents([f"t{index}" for index in range(9)])

    assert sizes == [4, 4, 1]


async def test_empty_remote_input_does_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    assert await _openai(handler).embed_documents([]) == []


@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
async def test_non_2xx_raises_backend_error(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="upstream said no")

    with pytest.raises(EmbeddingBackendError, match=f"HTTP {status}"):
        await _openai(handler).embed_query("q")


async def test_transport_failure_raises_backend_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(EmbeddingBackendError, match="failed"):
        await _openai(handler).embed_query("q")


@pytest.mark.parametrize(
    "body",
    [
        {"nothing": "useful"},
        {"data": [{"index": 0}]},
        {"data": "not a list"},
        {"data": [{"index": 0, "embedding": "not numbers"}]},
    ],
)
async def test_malformed_body_raises_backend_error(body: dict[str, Any]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(EmbeddingBackendError, match="unusable body"):
        await _openai(handler).embed_query("q")


async def test_non_json_body_raises_backend_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with pytest.raises(EmbeddingBackendError, match="unusable body"):
        await _openai(handler).embed_query("q")


async def test_wrong_vector_count_raises_backend_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 1.0, 1.0]}]})

    with pytest.raises(EmbeddingBackendError, match="1 vectors for 2 inputs"):
        await _openai(handler).embed_documents(["a", "b"])


async def test_remote_dimension_mismatch_is_a_configuration_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0] * 1536}]})

    with pytest.raises(ConfigurationError, match="RE_EMBEDDING_DIMENSION is 3"):
        await _openai(handler).embed_query("q")


# --- factory ----------------------------------------------------------------------------


def test_factory_returns_the_local_embedder_by_default() -> None:
    assert isinstance(build_embedder(_settings()), LocalEmbedder)


def test_factory_returns_the_remote_embedder_when_selected() -> None:
    settings = _settings(embedder=EmbedderKind.OPENAI, openai_api_key="sk-test")

    assert isinstance(build_embedder(settings), OpenAIEmbedder)


def test_factory_passes_kwargs_through() -> None:
    stub = StubModel()

    embedder = build_embedder(_settings(), model_factory=lambda: stub)

    assert isinstance(embedder, LocalEmbedder)
    assert embedder.tokenizer is not None
