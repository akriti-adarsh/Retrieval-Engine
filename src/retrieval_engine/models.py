"""Every Pydantic schema and enum in the service.

Keeping the schemas in one module makes the contract between stages explicit: a
chunker produces :class:`Chunk`, a store consumes :class:`EmbeddedChunk`, a retriever
returns :class:`RetrievalResult`, and the API serialises :class:`QueryResponse`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# A document's front matter and provenance. Deliberately narrow: anything richer
# belongs in a dedicated field, not in a free-form metadata bag.
MetadataValue = str | int | float | bool | None | list[str]
Metadata = dict[str, MetadataValue]

#: Namespace for deterministic chunk ids. Fixed forever, because changing it would
#: silently invalidate every stored chunk id.
CHUNK_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

#: Maximum length of a golden-set span. Spans are matching keys, not documents: short
#: spans make the span-to-chunk relevance predicate stricter and keep quoted material
#: minimal.
MAX_GOLDEN_SPAN_CHARS = 300


def utc_now() -> datetime:
    """Timezone-aware current time, so serialised timestamps are unambiguous."""
    return datetime.now(UTC)


def make_chunk_id(doc_id: str, start_char: int) -> str:
    """Deterministic chunk id: UUID5 over ``doc_id`` and the chunk's start offset.

    Re-ingesting an unchanged document therefore produces byte-identical ids, which
    is what makes content-hash change detection and upsert semantics work.
    """
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, f"{doc_id}:{start_char}"))


class Base(BaseModel):
    """Shared model configuration: reject unknown fields, validate on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Frozen(BaseModel):
    """Immutable, hashable configuration models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------


class EmbedderKind(StrEnum):
    """Which embedding backend to use. ``local`` needs no API key."""

    LOCAL = "local"
    OPENAI = "openai"


class LLMKind(StrEnum):
    """Which generation backend to use. ``extractive`` needs no model server."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    EXTRACTIVE = "extractive"


class StoreKind(StrEnum):
    """Which vector store to use. ``memory`` needs no database."""

    PGVECTOR = "pgvector"
    MEMORY = "memory"


class ChunkStrategy(StrEnum):
    """Chunking strategies compared in docs/decisions/003-chunking-strategy.md."""

    FIXED_TOKEN = "fixed_token"
    RECURSIVE_STRUCTURAL = "recursive_structural"
    SEMANTIC = "semantic"


class FusionMethod(StrEnum):
    """How to combine the dense and lexical candidate lists."""

    RRF = "rrf"
    WEIGHTED = "weighted"


class ExpansionMode(StrEnum):
    """Optional query-side expansion, off by default."""

    NONE = "none"
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"


class AnswerType(StrEnum):
    """How an answer was produced, or that it was withheld."""

    GENERATED = "generated"
    EXTRACTIVE = "extractive"
    REFUSED = "refused"


class JobStatus(StrEnum):
    """Lifecycle of an asynchronous ingestion job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class GoldenCategory(StrEnum):
    """Golden-set question categories. ``NEGATIVE`` questions must be refused."""

    FACTUAL = "factual"
    MULTI_HOP = "multi_hop"
    NEGATIVE = "negative"
    AMBIGUOUS = "ambiguous"


class Difficulty(StrEnum):
    """Golden-set difficulty bands, used for metric breakdowns."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# --------------------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------------------


class PageSpan(Base):
    """Character range of one page or top-level section within a document's text.

    Loaders that know about pages (pdf) or headings (md, html) emit these so chunks
    can be attributed to a page number without re-parsing the source.
    """

    page_number: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    label: str | None = None

    @model_validator(mode="after")
    def _check_range(self) -> PageSpan:
        if self.end_char < self.start_char:
            msg = "end_char must not precede start_char"
            raise ValueError(msg)
        return self


class Document(Base):
    """A loaded source document, before chunking."""

    doc_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    text: str
    content_hash: str = Field(min_length=64, max_length=64)
    media_type: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)
    page_spans: list[PageSpan] = Field(default_factory=list)

    @property
    def title(self) -> str:
        """Front-matter title when present, else the file stem."""
        raw = self.metadata.get("title")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return self.source_path.replace("\\", "/").rsplit("/", 1)[-1]


class Chunk(Base):
    """One retrievable unit of text, with enough provenance to cite it."""

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    token_count: int = Field(ge=0)
    section_path: list[str] = Field(default_factory=list)
    page_number: int | None = Field(default=None, ge=1)
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE_STRUCTURAL
    metadata: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_range(self) -> Chunk:
        if self.end_char < self.start_char:
            msg = "end_char must not precede start_char"
            raise ValueError(msg)
        return self


class EmbeddedChunk(Base):
    """A chunk paired with its vector, which is what a vector store persists."""

    chunk: Chunk
    embedding: list[float] = Field(min_length=1)


class CollectionInfo(Base):
    """The embedding-space contract a store enforces on every upsert."""

    name: str = Field(min_length=1)
    embedder: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    chunk_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class IngestSummary(Base):
    """What an ingestion run did. Returned by the pipeline and the API alike."""

    docs_seen: int = Field(default=0, ge=0)
    docs_changed: int = Field(default=0, ge=0)
    docs_unchanged: int = Field(default=0, ge=0)
    docs_failed: int = Field(default=0, ge=0)
    chunks_created: int = Field(default=0, ge=0)
    chunks_skipped: int = Field(default=0, ge=0)
    tokens_embedded: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    errors: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """One-line human summary, the same wording the CLI and logs use."""
        return (
            f"{self.docs_changed} changed, {self.docs_unchanged} unchanged, "
            f"{self.chunks_created} chunks created, {self.chunks_skipped} skipped, "
            f"{self.tokens_embedded} tokens embedded in {self.elapsed_seconds:.2f}s"
        )


class IngestJob(Base):
    """Status record for an asynchronous ingestion request."""

    job_id: str = Field(min_length=1)
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    summary: IngestSummary | None = None
    error: str | None = None


class DocumentInfo(Base):
    """Document-level view for the admin listing endpoint."""

    doc_id: str
    source_path: str
    title: str
    chunk_count: int = Field(ge=0)
    metadata: Metadata = Field(default_factory=dict)


class DocumentPage(Base):
    """One page of a document listing."""

    items: list[DocumentInfo]
    total: int = Field(ge=0)
    limit: int = Field(gt=0)
    offset: int = Field(ge=0)


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------


class RetrievalConfig(Frozen):
    """Every retrieval knob, in one immutable object.

    The eval harness constructs one of these per ablation row and records it verbatim
    in the run report, so every published number is attributable to an exact config.
    """

    use_dense: bool = True
    use_lexical: bool = True
    top_k_dense: int = Field(default=50, gt=0)
    top_k_lexical: int = Field(default=50, gt=0)

    fusion: FusionMethod = FusionMethod.RRF
    rrf_k: int = Field(default=60, gt=0)
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    use_rerank: bool = True
    rerank_candidates: int = Field(default=20, gt=0)
    final_top_k: int = Field(default=5, gt=0)

    expansion: ExpansionMode = ExpansionMode.NONE
    num_paraphrases: int = Field(default=3, ge=1)

    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE_STRUCTURAL
    use_query_instruction: bool = True
    hnsw_ef_search: int = Field(default=64, gt=0)

    @model_validator(mode="after")
    def _at_least_one_retriever(self) -> RetrievalConfig:
        if not (self.use_dense or self.use_lexical):
            msg = "at least one of use_dense or use_lexical must be enabled"
            raise ValueError(msg)
        return self

    @property
    def label(self) -> str:
        """Short human label used as the ablation table's row name."""
        if self.use_dense and self.use_lexical:
            base = f"hybrid ({self.fusion.value.upper()})"
        elif self.use_dense:
            base = "dense only"
        else:
            base = "lexical only"
        parts = [base]
        if self.use_rerank:
            parts.append("rerank")
        if self.expansion is not ExpansionMode.NONE:
            parts.append(self.expansion.value)
        if self.chunk_strategy is not ChunkStrategy.RECURSIVE_STRUCTURAL:
            parts.append(f"{self.chunk_strategy.value} chunking")
        return " + ".join(parts)


class StageScores(Base):
    """Per-stage evidence for one candidate, so debugging never needs a rerun."""

    dense_score: float | None = None
    lexical_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    lexical_rank: int | None = Field(default=None, ge=1)
    fused_rank: int | None = Field(default=None, ge=1)
    final_rank: int | None = Field(default=None, ge=1)

    @property
    def rank_movement(self) -> int | None:
        """Places gained (positive) or lost by reranking, ``None`` if not reranked."""
        if self.fused_rank is None or self.final_rank is None:
            return None
        return self.fused_rank - self.final_rank


class ScoredChunk(Base):
    """A chunk with its final score and the trail of how it got there."""

    chunk: Chunk
    score: float
    stages: StageScores = Field(default_factory=StageScores)


class StageTimings(Base):
    """Millisecond breakdown of one request. Every field is measured, never estimated."""

    expansion_ms: float = Field(default=0.0, ge=0.0)
    dense_ms: float = Field(default=0.0, ge=0.0)
    lexical_ms: float = Field(default=0.0, ge=0.0)
    fusion_ms: float = Field(default=0.0, ge=0.0)
    rerank_ms: float = Field(default=0.0, ge=0.0)
    retrieval_ms: float = Field(default=0.0, ge=0.0)
    generation_ms: float = Field(default=0.0, ge=0.0)
    grounding_ms: float = Field(default=0.0, ge=0.0)
    total_ms: float = Field(default=0.0, ge=0.0)


class RetrievalResult(Base):
    """Everything stage 1 through 4 produced for one query."""

    query: str
    chunks: list[ScoredChunk] = Field(default_factory=list)
    expanded_queries: list[str] = Field(default_factory=list)
    candidate_counts: dict[str, int] = Field(default_factory=dict)
    timings: StageTimings = Field(default_factory=StageTimings)
    config: RetrievalConfig = Field(default_factory=RetrievalConfig)

    @property
    def top_score(self) -> float:
        """Score of the best candidate, or 0.0 when nothing was retrieved."""
        return self.chunks[0].score if self.chunks else 0.0


# --------------------------------------------------------------------------------------
# Generation and guardrails
# --------------------------------------------------------------------------------------


class Citation(Base):
    """A ``[n]`` marker in an answer, resolved back to the source it points at."""

    marker: int = Field(ge=1)
    chunk_id: str | None = None
    doc_id: str | None = None
    source_path: str | None = None
    resolved: bool = True


class SourceRef(Base):
    """One numbered source as shown to the model and returned to the caller."""

    index: int = Field(ge=1)
    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    text: str
    score: float
    section_path: list[str] = Field(default_factory=list)
    page_number: int | None = Field(default=None, ge=1)


class SentenceGrounding(Base):
    """Grounding evidence for a single answer sentence."""

    sentence: str
    max_similarity: float
    supporting_chunk_id: str | None = None
    grounded: bool


class GroundingReport(Base):
    """Per-sentence grounding verdicts plus the overall judgement."""

    sentences: list[SentenceGrounding] = Field(default_factory=list)
    grounded: bool = True
    threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    flagged_sentences: list[str] = Field(default_factory=list)
    unresolved_citations: list[int] = Field(default_factory=list)

    @property
    def mean_similarity(self) -> float:
        """Mean per-sentence similarity, 0.0 for an empty answer."""
        if not self.sentences:
            return 0.0
        return sum(s.max_similarity for s in self.sentences) / len(self.sentences)


class GeneratedAnswer(Base):
    """Raw output of a generation backend, before guardrails run."""

    text: str
    answer_type: AnswerType
    prompt_version: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------


class QueryRequest(Base):
    """Body of ``POST /v1/query``."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, gt=0, le=50)
    filters: dict[str, str] | None = None
    mode: ExpansionMode | None = None
    debug: bool = False

    @field_validator("query")
    @classmethod
    def _strip(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "query must not be blank"
            raise ValueError(msg)
        return stripped


class DebugInfo(Base):
    """Returned only when ``debug=true``: the retrieval trail and the exact config."""

    config: RetrievalConfig
    expanded_queries: list[str] = Field(default_factory=list)
    candidate_counts: dict[str, int] = Field(default_factory=dict)
    stages: list[StageScores] = Field(default_factory=list)
    rank_movement: list[int | None] = Field(default_factory=list)


class QueryResponse(Base):
    """Body of ``POST /v1/query``, and the terminal event of the SSE stream."""

    answer: str
    answer_type: AnswerType
    citations: list[Citation] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    grounding: GroundingReport = Field(default_factory=GroundingReport)
    timings: StageTimings = Field(default_factory=StageTimings)
    prompt_version: str
    model: str
    request_id: str
    debug: DebugInfo | None = None


class IngestRequest(Base):
    """Body of ``POST /v1/ingest`` when ingesting a server-side directory."""

    path: str | None = None
    strategy: ChunkStrategy | None = None
    async_job: bool = False


class ErrorDetail(Base):
    """The single error shape every failing route returns."""

    code: str
    message: str
    request_id: str


class ErrorResponse(Base):
    """Envelope around :class:`ErrorDetail`."""

    error: ErrorDetail


class HealthResponse(Base):
    """Body of ``GET /health``."""

    status: str = "ok"
    version: str


class ReadyResponse(Base):
    """Body of ``GET /health/ready``: per-dependency readiness, not just a ping."""

    ready: bool
    store: bool
    embedder: bool
    generator: bool
    detail: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


class GoldenEntry(Base):
    """One golden-set question.

    Relevance is stored as document ids plus verbatim spans rather than chunk ids,
    because chunk ids change with the chunking strategy and the chunking ablation has
    to compare like with like. See :mod:`retrieval_engine.eval.metrics` for the
    span-to-chunk predicate that turns these spans into per-chunk relevance.
    """

    qid: str = Field(min_length=1)
    question: str = Field(min_length=1)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    relevant_chunk_texts: list[str] = Field(default_factory=list)
    answer: str = ""
    category: GoldenCategory
    difficulty: Difficulty

    @model_validator(mode="after")
    def _check_negative_has_no_evidence(self) -> GoldenEntry:
        if self.category is GoldenCategory.NEGATIVE:
            if self.relevant_doc_ids or self.relevant_chunk_texts:
                msg = f"negative entry {self.qid} must have no relevant docs or spans"
                raise ValueError(msg)
        elif not self.relevant_chunk_texts:
            msg = f"entry {self.qid} of category {self.category} needs at least one span"
            raise ValueError(msg)
        for span in self.relevant_chunk_texts:
            if not span.strip():
                msg = f"entry {self.qid} has a blank span"
                raise ValueError(msg)
            if len(span) > MAX_GOLDEN_SPAN_CHARS:
                msg = (
                    f"entry {self.qid} has a {len(span)}-char span, over the "
                    f"{MAX_GOLDEN_SPAN_CHARS}-char cap"
                )
                raise ValueError(msg)
        return self


class EvalRow(Base):
    """Per-query eval record, one JSON line in ``eval_results/{run_id}/rows.jsonl``."""

    qid: str
    question: str
    category: GoldenCategory
    difficulty: Difficulty
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    relevance_flags: list[bool] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    answer: str = ""
    answer_type: AnswerType = AnswerType.EXTRACTIVE
    refused: bool = False
    grounded: bool = True
    timings: StageTimings = Field(default_factory=StageTimings)


class LatencyStats(Base):
    """Wall-clock summary for one stage across a run."""

    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)
    mean_ms: float = Field(ge=0.0)
    max_ms: float = Field(ge=0.0)


class EvalRun(Base):
    """Aggregate result of one eval run against one config."""

    run_id: str
    label: str
    config: RetrievalConfig
    embedder: str
    generator: str
    prompt_version: str
    n_queries: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_difficulty: dict[str, dict[str, float]] = Field(default_factory=dict)
    latency: dict[str, LatencyStats] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utc_now)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    seed: int = 42


class AblationReport(Base):
    """The full ablation matrix, which is what the README table is rendered from."""

    generated_at: datetime = Field(default_factory=utc_now)
    corpus_docs: int = Field(ge=0)
    corpus_chunks: int = Field(ge=0)
    golden_questions: int = Field(ge=0)
    runs: list[EvalRun] = Field(default_factory=list)


__all__ = [
    "CHUNK_ID_NAMESPACE",
    "MAX_GOLDEN_SPAN_CHARS",
    "AblationReport",
    "AnswerType",
    "Base",
    "Chunk",
    "ChunkStrategy",
    "Citation",
    "CollectionInfo",
    "DebugInfo",
    "Difficulty",
    "Document",
    "DocumentInfo",
    "DocumentPage",
    "EmbeddedChunk",
    "EmbedderKind",
    "ErrorDetail",
    "ErrorResponse",
    "EvalRow",
    "EvalRun",
    "ExpansionMode",
    "Frozen",
    "FusionMethod",
    "GeneratedAnswer",
    "GoldenCategory",
    "GoldenEntry",
    "GroundingReport",
    "HealthResponse",
    "IngestJob",
    "IngestRequest",
    "IngestSummary",
    "JobStatus",
    "LLMKind",
    "LatencyStats",
    "Metadata",
    "MetadataValue",
    "PageSpan",
    "QueryRequest",
    "QueryResponse",
    "ReadyResponse",
    "RetrievalConfig",
    "RetrievalResult",
    "ScoredChunk",
    "SentenceGrounding",
    "SourceRef",
    "StageScores",
    "StageTimings",
    "StoreKind",
    "make_chunk_id",
    "utc_now",
]
