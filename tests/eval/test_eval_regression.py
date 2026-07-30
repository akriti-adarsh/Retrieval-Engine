"""The CI regression gate: retrieval quality must not silently get worse.

Read this before trusting the numbers it checks. This gate is NOT the published ablation. It
measures the twelve document fixture corpus with the deterministic fake embedder and a stubbed
cross-encoder, because the suite is forbidden from touching the network and a real model would
need a download. What it catches is a PIPELINE regression: a broken fusion, a chunker that
stops covering its input, a refusal policy that starts refusing everything. What it cannot
catch is a change in real-model retrieval quality, which is what the ablation over the arXiv
corpus in eval_results/ is for.

Both artifacts are honest about being different things, which is why they live in different
places: this floor is in src/retrieval_engine/eval/baseline.json, and the real numbers are in
eval_results/.

The floors sit a margin below the measured values, so ordinary noise does not fail a build
while a real regression does. To re-record after a deliberate change, run:

    RECORD_EVAL_BASELINE=1 uv run pytest tests/eval -q

and commit the updated baseline.json with an explanation of why the numbers moved.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from retrieval_engine.config import Settings
from retrieval_engine.eval.golden import validate_golden_set
from retrieval_engine.eval.report import write_baseline
from retrieval_engine.eval.runner import EvalHarness
from retrieval_engine.ingest.loaders import iter_source_files, load_document
from retrieval_engine.models import (
    Difficulty,
    GoldenCategory,
    GoldenEntry,
    LLMKind,
    RetrievalConfig,
    StoreKind,
)
from retrieval_engine.retrieve.rerank import CrossEncoderReranker
from tests.conftest import FAKE_DIMENSION, FakeEmbedder

BASELINE_PATH = Path("src/retrieval_engine/eval/baseline.json")

#: Metrics the gate watches. Recall and nDCG catch a retrieval regression; refusal accuracy
#: catches a guardrail regression, which no retrieval metric would notice.
WATCHED = ("recall@5", "ndcg@5", "mrr", "refusal_accuracy")

# Deliberately NOT prefixed RE_: the test suite strips every RE_ variable to isolate tests
# from the developer environment, which would swallow this flag.
RECORD_ENV = "RECORD_EVAL_BASELINE"

#: One question per fixture document, phrased to be answerable from that document alone.
QUESTIONS: dict[str, str] = {
    "doc-bm25": "how does BM25 weight a query term that appears many times in one document?",
    "doc-dense": "why can a bi-encoder compute passage vectors ahead of query time?",
    "doc-rrf": "how does reciprocal rank fusion combine two ranked lists?",
    "doc-rerank": "what does a cross encoder do that a bi-encoder cannot?",
    "doc-hnsw": "which HNSW parameter trades recall against query latency?",
    "doc-chunking": "what goes wrong when a document is split into fixed token windows?",
    "doc-grounding": "why check an answer sentence by sentence instead of as a whole?",
    "doc-ndcg": "what does normalising discounted cumulative gain make comparable?",
}

#: Questions a corpus of retrieval notes genuinely cannot answer. Their correct behaviour is
#: refusal, which is what makes refusal accuracy meaningful.
NEGATIVES: list[str] = [
    "what was the quarterly revenue of the company that funded this research?",
    "what is the mailing address of the first author?",
    "which football team won the 1998 world cup final?",
    "what dosage of ibuprofen is recommended for a child under five?",
]

_HEADING = re.compile(r"^#{1,6}\s")


def _longest_line(text: str, minimum: int = 80) -> str:
    """The longest single line of body prose in a document.

    Spans are taken from one line rather than spanning a break, and extracted from the loaded
    text rather than retyped, so they are exact substrings by construction. A hand-typed span
    is the most common way a golden set ends up quietly invalid.
    """
    candidates = [
        line.strip()
        for line in text.splitlines()
        if len(line.strip()) >= minimum and not _HEADING.match(line.strip())
    ]
    if not candidates:
        return ""
    return max(candidates, key=len)[:300]


def build_fixture_golden_set(corpus_dir: Path) -> list[GoldenEntry]:
    """Twelve questions over the fixture corpus: eight answerable, four unanswerable."""
    documents = {
        document.doc_id: document for document in map(load_document, iter_source_files(corpus_dir))
    }

    entries: list[GoldenEntry] = []
    for index, (doc_id, question) in enumerate(sorted(QUESTIONS.items()), start=1):
        document = documents[doc_id]
        span = _longest_line(document.text)
        assert span, f"no usable span found in {doc_id}"
        entries.append(
            GoldenEntry(
                qid=f"f{index:03d}",
                question=question,
                relevant_doc_ids=[doc_id],
                relevant_chunk_texts=[span],
                answer=span,
                category=GoldenCategory.FACTUAL,
                difficulty=Difficulty.MEDIUM,
            )
        )

    for index, question in enumerate(NEGATIVES, start=len(entries) + 1):
        entries.append(
            GoldenEntry(
                qid=f"f{index:03d}",
                question=question,
                relevant_doc_ids=[],
                relevant_chunk_texts=[],
                answer="The corpus does not cover this.",
                category=GoldenCategory.NEGATIVE,
                difficulty=Difficulty.EASY,
            )
        )
    return entries


def _load_baseline() -> dict[str, Any]:
    """Read the committed floors. Sync, so the blocking file read stays off the event loop."""
    assert BASELINE_PATH.is_file(), (
        f"{BASELINE_PATH} is missing. Record it with {RECORD_ENV}=1 uv run pytest tests/eval -q"
    )
    loaded: dict[str, Any] = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return loaded


class StubCrossEncoder:
    """Shared-word scoring. Deterministic, and needs no 1.5 GB download."""

    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> Any:
        return [
            len(set(query.lower().split()) & set(passage.lower().split())) / 10.0
            for query, passage in sentences
        ]


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        env="test",
        store=StoreKind.MEMORY,
        llm=LLMKind.EXTRACTIVE,
        data_dir=tmp_path / "data",
        eval_results_dir=tmp_path / "eval",
        embedding_dimension=FAKE_DIMENSION,
        chunk_size=96,
        chunk_overlap=16,
        chunk_min_tokens=24,
        # Calibrated to the STUB cross-encoder's scale, which is shared-word count over ten,
        # not to the real reranker's. At 0.01 nothing was ever refused, which made the four
        # negative questions decorative and refusal accuracy a constant zero. A gate that
        # cannot fail is not a gate. The real threshold is calibrated against the real
        # reranker's measured distribution, which CLAUDE.md records.
        min_confidence=0.25,
    )


async def _measure(tmp_path: Path, corpus_dir: Path) -> tuple[Any, list[GoldenEntry]]:
    settings = _settings(tmp_path)
    entries = build_fixture_golden_set(corpus_dir)
    harness = EvalHarness(
        settings,
        corpus_dir,
        embedder=FakeEmbedder(dimension=FAKE_DIMENSION),
        reranker=CrossEncoderReranker(settings, StubCrossEncoder),
    )
    run, _rows = await harness.run(
        entries, RetrievalConfig(), label="regression gate", progress=False
    )
    return run, entries


# --- the fixture golden set itself ------------------------------------------------------


def test_the_fixture_golden_set_is_valid(corpus_dir: Path) -> None:
    """The gate's own ground truth goes through the same validator as the real one."""
    entries = build_fixture_golden_set(corpus_dir)
    documents = [load_document(path) for path in iter_source_files(corpus_dir)]

    report = validate_golden_set(entries, documents, require_negatives=4)

    assert report.ok, report.failures
    assert report.entries_checked == 12
    assert report.negatives == 4


def test_every_fixture_span_is_verbatim(corpus_dir: Path) -> None:
    """Stated separately from the validator, because this is the property that matters."""
    documents = {
        document.doc_id: document for document in map(load_document, iter_source_files(corpus_dir))
    }

    for entry in build_fixture_golden_set(corpus_dir):
        for doc_id, span in zip(entry.relevant_doc_ids, entry.relevant_chunk_texts, strict=False):
            assert span in documents[doc_id].text


# --- the gate ---------------------------------------------------------------------------


async def test_retrieval_quality_has_not_regressed(tmp_path: Path, corpus_dir: Path) -> None:
    """The build fails if any watched metric drops below its committed floor."""
    run, _entries = await _measure(tmp_path, corpus_dir)

    if os.environ.get(RECORD_ENV):
        path = write_baseline(BASELINE_PATH, run, WATCHED, margin=0.05)
        pytest.skip(f"recorded a new baseline to {path}; commit it with a reason")

    baseline = _load_baseline()

    shortfalls = [
        f"{name}: {run.metrics.get(name, 0.0):.4f} < floor {floor:.4f}"
        for name, floor in baseline["floors"].items()
        if run.metrics.get(name, 0.0) < floor
    ]
    assert not shortfalls, "retrieval quality regressed: " + "; ".join(shortfalls)


async def test_the_gate_measures_something_rather_than_nothing(
    tmp_path: Path, corpus_dir: Path
) -> None:
    """A gate that passes on an empty result is worse than no gate.

    Without this, deleting the retriever would make every metric 0.0 and the floors could be
    lowered to 0.0 to "fix the build". This asserts the run actually retrieved and answered.
    """
    run, entries = await _measure(tmp_path, corpus_dir)

    assert run.n_queries == len(entries) == 12
    assert run.metrics["recall@5"] > 0.0
    assert run.metrics["hit_rate@5"] > 0.0
    assert run.latency["total"].p50_ms > 0.0


async def test_negative_questions_are_refused(tmp_path: Path, corpus_dir: Path) -> None:
    """Refusal accuracy is the metric that stops recall from being gamed by answering all."""
    run, _entries = await _measure(tmp_path, corpus_dir)

    assert "refusal_accuracy" in run.metrics
    assert run.by_category["negative"]["n"] == 4


async def test_the_run_records_what_produced_it(tmp_path: Path, corpus_dir: Path) -> None:
    """A metric with no embedder, prompt version, or seed attached is not reproducible."""
    run, _entries = await _measure(tmp_path, corpus_dir)

    assert run.embedder == "fake-embedder"
    assert run.generator == "extractive"
    assert run.prompt_version
    assert run.seed == 42
    assert run.config.rrf_k == 60


async def test_the_gate_is_deterministic(tmp_path: Path, corpus_dir: Path) -> None:
    """Two runs on the same corpus must give identical numbers, or the floor is noise."""
    first, _ = await _measure(tmp_path, corpus_dir)
    second, _ = await _measure(tmp_path, corpus_dir)

    assert first.metrics == second.metrics
