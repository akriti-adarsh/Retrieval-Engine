"""Prompts, the extractive fallback, and the Ollama client.

The extractive path gets the most attention here, because it is what keeps a stopped model
server from becoming an outage, and because its central claim (an extract cannot hallucinate)
is only true if every word really does come from a source. That is asserted, not assumed.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from retrieval_engine.config import Settings
from retrieval_engine.errors import LLMUnavailableError
from retrieval_engine.generate import build_llm
from retrieval_engine.generate.base import LLM
from retrieval_engine.generate.extractive import (
    EXTRACTIVE_VERSION,
    MIN_SENTENCE_CHARS,
    ExtractiveAnswerer,
)
from retrieval_engine.generate.ollama import OllamaLLM
from retrieval_engine.generate.prompts import (
    INSUFFICIENT_ANSWER,
    PROMPT_VERSION,
    build_answer_prompt,
    format_source_block,
    render_sources,
)
from retrieval_engine.models import AnswerType, Chunk, LLMKind, ScoredChunk, SourceRef
from tests.conftest import FAKE_DIMENSION, FakeEmbedder, FakeLLM


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "_env_file": None,
        "env": "test",
        "embedding_dimension": FAKE_DIMENSION,
        "ollama_base_url": "http://localhost:11434",
    }
    base.update(overrides)
    return Settings(**base)


def _scored(chunk_id: str, text: str, **metadata: Any) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="2401.12345",
            text=text,
            start_char=0,
            end_char=len(text),
            token_count=len(text.split()),
            section_path=["Methods", "Retrieval"],
            metadata=metadata,
        ),
        score=0.8,
    )


def _source(index: int, text: str) -> SourceRef:
    return SourceRef(
        index=index,
        chunk_id=f"c{index}",
        doc_id="2401.12345",
        title="A Paper",
        source_path="data/corpus/2401.12345.md",
        text=text,
        score=0.8,
    )


# --- prompts ----------------------------------------------------------------------------


def test_prompt_version_is_recorded_and_stable() -> None:
    """Prompt edits move metrics, so a number without a prompt version is not reproducible."""
    assert PROMPT_VERSION == "v1"


def test_render_sources_numbers_from_one() -> None:
    sources = render_sources([_scored("c1", "first"), _scored("c2", "second")])

    assert [source.index for source in sources] == [1, 2]
    assert [source.chunk_id for source in sources] == ["c1", "c2"]


def test_render_sources_prefers_the_title_and_falls_back_to_doc_id() -> None:
    titled = render_sources([_scored("c1", "text", title="Dense Passage Retrieval")])
    untitled = render_sources([_scored("c2", "text")])

    assert titled[0].title == "Dense Passage Retrieval"
    assert untitled[0].title == "2401.12345"


def test_render_sources_carries_the_section_path_and_score() -> None:
    sources = render_sources([_scored("c1", "text")])

    assert sources[0].section_path == ["Methods", "Retrieval"]
    assert sources[0].score == 0.8


def test_source_block_is_numbered_and_located() -> None:
    block = format_source_block(render_sources([_scored("c1", "BM25 saturates term frequency.")]))

    assert block.startswith("[1] (Methods > Retrieval)")
    assert "BM25 saturates" in block


def test_prompt_contains_the_query_the_sources_and_the_rules() -> None:
    sources = render_sources([_scored("c1", "Reciprocal rank fusion needs no calibration.")])

    prompt = build_answer_prompt("  what is rrf  ", sources)

    assert "what is rrf" in prompt
    assert "Reciprocal rank fusion needs no calibration." in prompt
    assert "[1]" in prompt
    # The exact refusal wording is asserted on by the eval harness, so the prompt must
    # contain it verbatim rather than a paraphrase.
    assert INSUFFICIENT_ANSWER in prompt


def test_prompt_with_no_sources_says_so_plainly() -> None:
    prompt = build_answer_prompt("anything", [])

    assert "(no sources retrieved)" in prompt


# --- extractive answering ---------------------------------------------------------------

PASSAGES = [
    "A hierarchical navigable small world index is a layered proximity graph structure. "
    "Search descends from the sparse top layer toward the dense bottom layer.",
    "BM25 scores a document by summing inverse document frequency weights over query terms. "
    "The saturation function keeps repeated terms from dominating the score.",
]


def _answerer(**overrides: Any) -> tuple[ExtractiveAnswerer, FakeEmbedder]:
    embedder = FakeEmbedder(dimension=FAKE_DIMENSION)
    return ExtractiveAnswerer(_settings(**overrides), embedder), embedder


async def test_extract_selects_relevant_sentences_and_cites_them() -> None:
    answerer, _ = _answerer(extractive_sentences=2)
    sources = [_source(1, PASSAGES[0]), _source(2, PASSAGES[1])]

    answer = await answerer.answer("layered proximity graph structure", sources)

    assert answer.answer_type is AnswerType.EXTRACTIVE
    assert answer.prompt_version == EXTRACTIVE_VERSION
    assert answer.model == "extractive"
    assert "[1]" in answer.text
    assert "proximity graph" in answer.text


async def test_every_word_of_an_extract_comes_from_a_source() -> None:
    """The central claim of this path: an extract cannot hallucinate."""
    answerer, _ = _answerer(extractive_sentences=3)
    sources = [_source(1, PASSAGES[0]), _source(2, PASSAGES[1])]

    answer = await answerer.answer("how does bm25 score documents", sources)

    corpus = " ".join(" ".join(source.text.split()) for source in sources)
    for fragment in answer.text.split(" ["):
        sentence = fragment.split("] ")[-1].strip()
        if sentence and not sentence.endswith("]"):
            assert sentence in corpus, f"{sentence!r} is not quoted from any source"


async def test_extract_respects_the_sentence_budget() -> None:
    answerer, _ = _answerer(extractive_sentences=1)
    sources = [_source(1, PASSAGES[0]), _source(2, PASSAGES[1])]

    answer = await answerer.answer("bm25 saturation", sources)

    assert answer.usage["sentences_used"] == 1
    assert answer.text.count("[") == 1


async def test_extract_orders_sentences_by_source_not_by_rank() -> None:
    """Ranking order reads as a jumble; source order reads as a passage."""
    answerer, _ = _answerer(extractive_sentences=4)
    sources = [_source(1, PASSAGES[0]), _source(2, PASSAGES[1])]

    answer = await answerer.answer("graph search and bm25 scoring", sources)

    markers = [part for part in answer.text.split() if part.startswith("[")]
    numbers = [int(marker.strip("[]")) for marker in markers]
    assert numbers == sorted(numbers)


async def test_extract_with_no_sources_states_insufficiency() -> None:
    answerer, _ = _answerer()

    answer = await answerer.answer("anything", [])

    assert answer.text == f"{INSUFFICIENT_ANSWER}."
    assert answer.answer_type is AnswerType.EXTRACTIVE


async def test_extract_ignores_fragments_too_short_to_be_answers() -> None:
    """A bare heading scores deceptively well against a short query."""
    answerer, _ = _answerer()

    answer = await answerer.answer("methods", [_source(1, "Methods. See Table 2. Ibid.")])

    assert answer.text == f"{INSUFFICIENT_ANSWER}."
    assert len("See Table 2.") < MIN_SENTENCE_CHARS


async def test_extract_is_deterministic() -> None:
    answerer, _ = _answerer(extractive_sentences=2)
    sources = [_source(1, PASSAGES[0]), _source(2, PASSAGES[1])]

    first = await answerer.answer("proximity graph", sources)
    second = await answerer.answer("proximity graph", sources)

    assert first.text == second.text


async def test_extract_survives_a_zero_magnitude_query() -> None:
    """An out-of-vocabulary query embeds to zeros with a hashing embedder."""
    embedder = FakeEmbedder(dimension=FAKE_DIMENSION, query_instruction=None)
    answerer = ExtractiveAnswerer(_settings(extractive_sentences=1), embedder)

    answer = await answerer.answer("   ", [_source(1, PASSAGES[0])])

    assert answer.text
    assert "[1]" in answer.text


# --- the ollama client ------------------------------------------------------------------


def _ollama(handler: Any, **overrides: Any) -> OllamaLLM:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaLLM(_settings(**overrides), client)


def test_ollama_reports_its_model() -> None:
    assert _ollama(lambda request: httpx.Response(200)).model == "llama3.1:8b"


def test_ollama_satisfies_the_llm_protocol() -> None:
    assert isinstance(_ollama(lambda request: httpx.Response(200)), LLM)


def test_fake_llm_satisfies_the_llm_protocol() -> None:
    """The fake must be a real stand-in, or tests using it prove nothing about the contract."""
    assert isinstance(FakeLLM(), LLM)


async def test_complete_sends_the_expected_request() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"response": "an answer [1]", "done": True})

    text = await _ollama(handler).complete("the prompt", max_tokens=64, temperature=0.2)

    assert text == "an answer [1]"
    assert seen["url"] == "http://localhost:11434/api/generate"
    assert seen["body"]["model"] == "llama3.1:8b"
    assert seen["body"]["prompt"] == "the prompt"
    assert seen["body"]["stream"] is False
    assert seen["body"]["options"]["num_predict"] == 64
    assert seen["body"]["options"]["temperature"] == 0.2
    # The seed is sent, because an eval that changes on a re-run measures nothing.
    assert seen["body"]["options"]["seed"] == 42


async def test_complete_defaults_come_from_settings() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"response": "x", "done": True})

    await _ollama(handler).complete("prompt")

    assert seen["body"]["options"]["temperature"] == 0.0
    assert seen["body"]["options"]["num_predict"] == 512


@pytest.mark.parametrize("status", [400, 404, 500, 503])
async def test_complete_maps_http_errors_to_unavailable(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="upstream problem")

    with pytest.raises(LLMUnavailableError, match=f"HTTP {status}"):
        await _ollama(handler).complete("prompt")


async def test_complete_maps_a_dead_server_to_unavailable() -> None:
    """This is the case the extractive fallback exists for."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMUnavailableError, match="unreachable"):
        await _ollama(handler).complete("prompt")


@pytest.mark.parametrize("body", [{"nothing": "here"}, {"response": 42}])
async def test_complete_maps_an_unusable_body_to_unavailable(body: dict[str, Any]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(LLMUnavailableError):
        await _ollama(handler).complete("prompt")


async def test_complete_maps_non_json_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with pytest.raises(LLMUnavailableError, match="unusable body"):
        await _ollama(handler).complete("prompt")


async def test_stream_yields_deltas_and_stops_at_done() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            b'{"response": "Reciprocal ", "done": false}',
            b'{"response": "rank ", "done": false}',
            b'{"response": "fusion", "done": false}',
            b'{"done": true}',
            b'{"response": "never reached", "done": false}',
        ]
        return httpx.Response(200, content=b"\n".join(lines) + b"\n")

    deltas = [delta async for delta in _ollama(handler).stream("prompt")]

    assert deltas == ["Reciprocal ", "rank ", "fusion"]
    assert "".join(deltas) == "Reciprocal rank fusion"


async def test_stream_skips_blank_lines() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b'{"response": "a", "done": false}\n\n\n{"done": true}\n'
        )

    assert [delta async for delta in _ollama(handler).stream("prompt")] == ["a"]


async def test_stream_maps_an_error_status_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model loading")

    with pytest.raises(LLMUnavailableError, match="HTTP 503"):
        _ = [delta async for delta in _ollama(handler).stream("prompt")]


async def test_stream_maps_a_broken_line_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not json at all}\n")

    with pytest.raises(LLMUnavailableError, match="unparseable"):
        _ = [delta async for delta in _ollama(handler).stream("prompt")]


async def test_stream_maps_a_dead_server_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMUnavailableError, match="failed"):
        _ = [delta async for delta in _ollama(handler).stream("prompt")]


async def test_health_is_true_when_the_server_answers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": []})

    assert await _ollama(handler).health() is True


@pytest.mark.parametrize("status", [404, 500])
async def test_health_is_false_on_an_error_status(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    assert await _ollama(handler).health() is False


async def test_health_never_raises() -> None:
    """A readiness check must report a problem, not become one."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    assert await _ollama(handler).health() is False


# --- factory ----------------------------------------------------------------------------


def test_build_llm_returns_ollama_by_default() -> None:
    assert isinstance(build_llm(_settings()), OllamaLLM)


def test_build_llm_returns_none_for_the_extractive_backend() -> None:
    """None rather than a null object, so the fallback is explicit at the call site."""
    assert build_llm(_settings(llm=LLMKind.EXTRACTIVE)) is None
