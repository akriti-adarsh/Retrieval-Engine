"""Shared fixtures.

Two invariants this file enforces for the whole suite:

1. No test sees the developer's environment. Every ``RE_`` variable is stripped and
   the settings cache is cleared before each test.
2. No test touches the network. ``HF_HUB_OFFLINE`` is forced on, so a test that
   accidentally reaches for a model download fails loudly instead of hanging in CI.

The fake embedder is hash-based but not semantically inert: it hashes tokens into
dimensions, so texts sharing vocabulary really do get similar vectors. That is enough
for tests about plumbing and ranking order, and it needs no model download.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from retrieval_engine.config import DEFAULT_SEED, Settings, reset_settings_cache, set_seeds
from retrieval_engine.embed.base import EmbedderInfo
from retrieval_engine.models import Document, LLMKind, StoreKind

# --------------------------------------------------------------------------------------
# Environment isolation
# --------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip RE_* vars, force offline mode, and reseed before every test."""
    for key in list(os.environ):
        if key.startswith("RE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    reset_settings_cache()
    set_seeds(DEFAULT_SEED)
    yield
    reset_settings_cache()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a temp directory, using only backends that need no server."""
    return Settings(
        _env_file=None,
        env="test",
        store=StoreKind.MEMORY,
        llm=LLMKind.EXTRACTIVE,
        data_dir=tmp_path / "data",
        eval_results_dir=tmp_path / "eval_results",
    )


# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------

_WORD = re.compile(r"\S+")


class FakeTokenizer:
    """Whitespace tokenizer that reports real character offsets.

    Satisfies :class:`retrieval_engine.embed.base.Tokenizer` without loading a model.
    """

    def count_tokens(self, text: str) -> int:
        return len(_WORD.findall(text))

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in _WORD.finditer(text)]


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder.

    Uses blake2b rather than :func:`hash`, because the built-in hash is salted per
    process, which would make "deterministic" mean "deterministic until you restart".

    Records what it was asked to embed, so tests can assert the query instruction is
    applied to queries and never to passages.
    """

    def __init__(
        self,
        dimension: int = 32,
        name: str = "fake-embedder",
        *,
        normalized: bool = True,
        query_instruction: str | None = "query: ",
    ) -> None:
        self._dimension = dimension
        self._name = name
        self._normalized = normalized
        self._query_instruction = query_instruction
        self._tokenizer = FakeTokenizer()
        self.document_calls: list[str] = []
        self.query_calls: list[str] = []

    @property
    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            name=self._name,
            dimension=self._dimension,
            normalized=self._normalized,
            query_instruction=self._query_instruction,
        )

    @property
    def tokenizer(self) -> FakeTokenizer:
        return self._tokenizer

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _WORD.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self._dimension] += 1.0
        if self._normalized:
            norm = sum(value * value for value in vector) ** 0.5
            if norm > 0.0:
                vector = [value / norm for value in vector]
        return vector

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls.extend(texts)
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        prefixed = f"{self._query_instruction}{text}" if self._query_instruction else text
        return self._vector(prefixed)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """A dimension-correct embedder with no model download and no network."""
    return FakeEmbedder()


# --------------------------------------------------------------------------------------
# Fixture corpus
# --------------------------------------------------------------------------------------

#: Twelve short technical documents, each built around a distinctive term so a test can
#: assert that a query retrieves the document that actually discusses it.
FIXTURE_CORPUS: dict[str, str] = {
    "doc-bm25": """---
title: Lexical Retrieval with BM25
authors:
  - R. Sparck
published: '2024-01-15'
---

# Lexical Retrieval with BM25

## Scoring

BM25 scores a document by summing, over the query terms it contains, an inverse document
frequency weight multiplied by a saturating term frequency factor. The saturation is what
separates BM25 from raw term counting: the twentieth occurrence of a term adds far less
than the second.

## Length normalisation

The b parameter controls how strongly document length is normalised. Setting b to zero
disables length normalisation entirely, which favours long documents.
""",
    "doc-dense": """---
title: Dense Passage Retrieval
authors:
  - V. Karpukhin
published: '2024-02-02'
---

# Dense Passage Retrieval

## Bi-encoders

A bi-encoder embeds the query and the passage independently, so passage vectors can be
computed once at index time. This is what makes dense retrieval cheap at query time, and
it is also why a bi-encoder cannot model term interaction between query and passage.

## Training

Hard negatives mined from a lexical retriever improve a bi-encoder considerably compared
with random in-batch negatives.
""",
    "doc-rrf": """---
title: Reciprocal Rank Fusion
authors:
  - G. Cormack
published: '2024-02-20'
---

# Reciprocal Rank Fusion

## Definition

Reciprocal rank fusion assigns each document a score equal to the sum over result lists
of one divided by the quantity k plus the rank of that document in the list. Documents
absent from a list contribute nothing.

## Why ranks

Because it consumes ranks rather than scores, reciprocal rank fusion is scale free: the
two retrievers being combined need no score calibration at all.
""",
    "doc-rerank": """---
title: Cross-Encoder Reranking
authors:
  - N. Reimers
published: '2024-03-04'
---

# Cross-Encoder Reranking

## Joint encoding

A cross-encoder concatenates query and passage into a single sequence, so attention runs
across both. This models term interaction directly and consistently outperforms a
bi-encoder on ranking quality.

## Cost

The cost is that nothing can be precomputed. Every query and passage pair needs a forward
pass, so reranking is applied only to a shortlist of candidates.
""",
    "doc-hnsw": """---
title: HNSW Graph Indexes
authors:
  - Y. Malkov
published: '2024-03-11'
---

# HNSW Graph Indexes

## Structure

A hierarchical navigable small world index is a layered proximity graph. Search descends
from a sparse top layer to the dense bottom layer, greedily following edges toward the
query.

## Parameters

The m parameter sets the graph degree and dominates memory use. The ef_search parameter
sets how many candidates the search keeps in flight, trading recall against latency at
query time rather than at build time.
""",
    "doc-chunking": """---
title: Chunking Strategies for Retrieval
authors:
  - A. Adarsh
published: '2024-03-25'
---

# Chunking Strategies for Retrieval

## Fixed windows

Fixed token windows with overlap are predictable and cheap, but they cut sentences in half
and strand a claim from its subject.

## Structural splitting

Splitting on document structure keeps a heading with the text beneath it, which preserves
the section path a citation needs.

## Semantic splitting

Semantic chunking embeds sentences and cuts where consecutive sentences diverge, which
finds topic boundaries that no heading marks.
""",
    "doc-grounding": """---
title: Grounding Verification
authors:
  - S. Min
published: '2024-04-02'
---

# Grounding Verification

## Sentence level checks

Verifying an answer sentence by sentence catches the common failure where a fluent answer
mixes supported claims with one invented detail. A single similarity score over the whole
answer hides exactly that case.

## Citation resolution

Every citation marker in an answer must resolve to a source that was actually in the
context window. An unresolvable marker is a defect, not a formatting quirk.
""",
    "doc-ndcg": """---
title: Normalised Discounted Cumulative Gain
authors:
  - K. Jarvelin
published: '2024-04-18'
---

# Normalised Discounted Cumulative Gain

## Discounting

Discounted cumulative gain divides each result's gain by the logarithm of its rank, so a
relevant document at rank one counts for more than the same document at rank ten.

## Normalisation

Dividing by the ideal ordering's gain bounds the metric to the unit interval, which makes
scores comparable across queries with different numbers of relevant documents.
""",
    "doc-mrr": """---
title: Mean Reciprocal Rank
authors:
  - E. Voorhees
published: '2024-04-29'
---

# Mean Reciprocal Rank

## Definition

Mean reciprocal rank averages one divided by the rank of the first relevant result. A
query with no relevant result in the returned list contributes zero.

## Interpretation

The metric only looks at the first hit, which makes it the right choice when a user reads
one answer and the wrong choice when recall across many documents matters.
""",
    "doc-expansion": """---
title: Query Expansion and HyDE
authors:
  - L. Gao
published: '2024-05-06'
---

# Query Expansion and HyDE

## Multi-query

Generating several paraphrases of a query and retrieving for each covers vocabulary the
original phrasing missed, at the cost of one language model call plus extra searches.

## Hypothetical documents

HyDE instead asks a model to write the answer it expects, then embeds that hypothetical
document. The generated text sits closer to real passages in embedding space than a short
question does.
""",
    "doc-pgvector": """---
title: Vectors in Postgres with pgvector
authors:
  - A. Kane
published: '2024-05-19'
---

# Vectors in Postgres with pgvector

## Operators

The extension adds a vector column type and distance operators, including cosine distance
written as the angle bracket equals operator pair.

## Indexes

Both IVFFlat and HNSW indexes are available. An index must be created against the same
distance operator class the query uses, or the planner will ignore it and fall back to a
sequential scan.
""",
    "doc-normalisation": """---
title: Normalising Embeddings
authors:
  - J. Johnson
published: '2024-06-01'
---

# Normalising Embeddings

## Unit vectors

Scaling every embedding to unit length makes the dot product equal to cosine similarity,
which lets an index that only implements inner product serve cosine queries.

## Consequences

Normalisation discards vector magnitude. Where magnitude carries signal, such as term
count in a sparse representation, normalising it away loses information.
""",
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """The fixture corpus written to disk as markdown files with YAML front matter."""
    directory = tmp_path / "corpus"
    directory.mkdir(parents=True, exist_ok=True)
    for doc_id, text in FIXTURE_CORPUS.items():
        (directory / f"{doc_id}.md").write_text(text, encoding="utf-8")
    return directory


@pytest.fixture
def sample_documents() -> list[Document]:
    """The fixture corpus as :class:`Document` objects, skipping the loaders entirely."""
    return [
        Document(
            doc_id=doc_id,
            source_path=f"data/corpus/{doc_id}.md",
            text=text,
            content_hash=_sha256(text),
            media_type="text/markdown",
            metadata={"title": doc_id.replace("doc-", "").replace("-", " ").title()},
        )
        for doc_id, text in FIXTURE_CORPUS.items()
    ]
