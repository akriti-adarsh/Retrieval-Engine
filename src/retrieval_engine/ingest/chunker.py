"""Three chunking strategies behind one interface, so the ADR can compare them.

Shared design decisions, which are what make the three comparable:

* Every strategy produces candidate character ranges, and one shared finaliser turns those
  into :class:`Chunk` objects. The finaliser is where the hard token cap is enforced, so no
  strategy can emit an oversized chunk by forgetting to check, and where ids, page numbers,
  and metadata are attached, so those cannot drift between strategies.
* Ranges are always slices of ``document.text``. Nothing is reconstructed from tokens, so
  ``text``, ``start_char``, and ``end_char`` agree exactly and a citation can be resolved
  back to the source by offset.
* ``chunk`` is async on all three because the semantic strategy has to embed sentences.
  One protocol that sometimes does not await beats two protocols.
* Structural chunking never merges across a heading boundary. A chunk spanning two sections
  cannot carry an honest ``section_path``, and section_path is what a citation shows the
  reader, so a short section stays short instead.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from retrieval_engine.config import Settings
from retrieval_engine.embed.base import Embedder, Tokenizer
from retrieval_engine.errors import ConfigurationError
from retrieval_engine.models import Chunk, ChunkStrategy, Document, PageSpan, make_chunk_id

#: A markdown ATX heading line, captured for its level and title.
_HEADING_LINE = re.compile(r"^(#{1,6})[ \t]+(\S.*?)[ \t]*$")

#: Sentence boundary: terminal punctuation followed by whitespace. Deliberately simple, no
#: NLP model. It over-splits abbreviations, which costs a little chunk quality but keeps the
#: chunker dependency-free and its behaviour predictable across languages we do not test.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])[ \t]*\n+|(?<=[.!?])[ \t]+")

_WHITESPACE = re.compile(r"\s")


@runtime_checkable
class Chunker(Protocol):
    """Splits one document into retrievable chunks."""

    async def chunk(self, document: Document) -> list[Chunk]:
        """Return the document's chunks, in document order."""
        ...


@dataclass(frozen=True)
class _Candidate:
    """A proposed chunk: a character range plus the section it came from."""

    start: int
    end: int
    section_path: tuple[str, ...] = ()


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink a range past leading and trailing whitespace."""
    while start < end and _WHITESPACE.match(text[start]):
        start += 1
    while end > start and _WHITESPACE.match(text[end - 1]):
        end -= 1
    return start, end


def _span_at(spans: Sequence[PageSpan], position: int) -> PageSpan | None:
    """The page or section span containing ``position``, if any."""
    for span in spans:
        if span.start_char <= position < span.end_char:
            return span
    return None


def _sentence_ranges(text: str, offset: int = 0) -> list[tuple[int, int]]:
    """Absolute character ranges of the sentences in ``text``."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_BREAK.finditer(text):
        start, end = _trim(text, cursor, match.start())
        if end > start:
            ranges.append((offset + start, offset + end))
        cursor = match.end()
    start, end = _trim(text, cursor, len(text))
    if end > start:
        ranges.append((offset + start, offset + end))
    return ranges


def _token_windows(span_count: int, size: int, stride: int) -> list[tuple[int, int]]:
    """Index windows over ``span_count`` tokens, stopping once the tail is covered."""
    windows: list[tuple[int, int]] = []
    position = 0
    while position < span_count:
        windows.append((position, min(position + size, span_count)))
        if position + size >= span_count:
            break
        position += stride
    return windows


def _hard_split(
    document: Document, start: int, end: int, tokenizer: Tokenizer, max_tokens: int
) -> list[tuple[int, int]]:
    """Split one oversized range into token-bounded pieces with no overlap.

    This is the backstop that makes the "no chunk exceeds max_tokens" property true for
    every strategy, including a single unbroken sentence longer than the budget.
    """
    fragment = document.text[start:end]
    # The decision uses the tokenizer's true count and the slicing uses sliceable spans.
    # These differ for text with tokens that consume no characters (markdown rules, table
    # separators), and using the span count to decide would let oversized chunks through.
    if tokenizer.count_tokens(fragment) <= max_tokens:
        return [(start, end)]
    spans = tokenizer.token_spans(fragment)
    if not spans:
        return [(start, end)]
    pieces: list[tuple[int, int]] = []
    for first, last in _token_windows(len(spans), max_tokens, max_tokens):
        piece_start = start + spans[first][0]
        piece_end = start + spans[last - 1][1]
        pieces.append((piece_start, piece_end))
    return pieces


def _finalise(
    document: Document,
    candidates: Sequence[_Candidate],
    strategy: ChunkStrategy,
    tokenizer: Tokenizer,
    max_tokens: int,
) -> list[Chunk]:
    """Turn candidate ranges into chunks, enforcing the token cap and attaching provenance.

    Ranges whose token count exceeds ``max_tokens`` are split here rather than in each
    strategy, so the cap cannot be forgotten in one of three places.
    """
    chunks: list[Chunk] = []
    used_starts: set[int] = set()
    for candidate in candidates:
        start, end = _trim(document.text, candidate.start, candidate.end)
        if end <= start:
            continue
        for piece_start, piece_end in _hard_split(document, start, end, tokenizer, max_tokens):
            begin, finish = _trim(document.text, piece_start, piece_end)
            text = document.text[begin:finish]
            if not text or begin in used_starts:
                continue
            used_starts.add(begin)
            span = _span_at(document.page_spans, begin)
            section_path = list(candidate.section_path)
            if not section_path and span is not None and span.label:
                section_path = [span.label]
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.doc_id, begin),
                    doc_id=document.doc_id,
                    text=text,
                    start_char=begin,
                    end_char=finish,
                    token_count=tokenizer.count_tokens(text),
                    section_path=section_path,
                    page_number=span.page_number if span is not None else None,
                    strategy=strategy,
                    metadata=dict(document.metadata),
                )
            )
    return chunks


# --------------------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------------------


class FixedTokenChunker:
    """Overlapping token windows. Predictable and cheap, but it cuts sentences in half.

    Windows are measured with the embedder's own tokenizer and sliced from the original
    text by character offset, never by decoding token ids back to text, which would lose
    whitespace and subword detail.
    """

    strategy = ChunkStrategy.FIXED_TOKEN

    def __init__(self, tokenizer: Tokenizer, size: int = 512, overlap: int = 64) -> None:
        if size <= 0:
            msg = f"chunk size must be positive, got {size}"
            raise ConfigurationError(msg)
        if overlap >= size:
            msg = f"chunk overlap ({overlap}) must be smaller than size ({size})"
            raise ConfigurationError(msg)
        self._tokenizer = tokenizer
        self._size = size
        self._stride = size - overlap

    async def chunk(self, document: Document) -> list[Chunk]:
        spans = self._tokenizer.token_spans(document.text)
        if not spans:
            return []
        candidates = [
            _Candidate(start=spans[first][0], end=spans[last - 1][1])
            for first, last in _token_windows(len(spans), self._size, self._stride)
        ]
        return _finalise(document, candidates, self.strategy, self._tokenizer, self._size)


class RecursiveStructuralChunker:
    """Headings, then paragraphs, then sentences, merging fragments up toward the target.

    Headings are kept with the text beneath them, and merging stops at a heading boundary
    so that every chunk's ``section_path`` is literally true of all its text.
    """

    strategy = ChunkStrategy.RECURSIVE_STRUCTURAL

    def __init__(self, tokenizer: Tokenizer, target_size: int = 512, min_tokens: int = 96) -> None:
        if target_size <= 0:
            msg = f"target size must be positive, got {target_size}"
            raise ConfigurationError(msg)
        self._tokenizer = tokenizer
        self._target = target_size
        self._min = min_tokens

    def _blocks(self, document: Document) -> list[_Candidate]:
        """Paragraph-level blocks, each tagged with its heading stack."""
        text = document.text
        blocks: list[_Candidate] = []
        stack: list[str] = []
        offset = 0
        block_start: int | None = None
        block_end = 0

        def flush() -> None:
            nonlocal block_start
            if block_start is not None and block_end > block_start:
                blocks.append(
                    _Candidate(start=block_start, end=block_end, section_path=tuple(stack))
                )
            block_start = None

        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            heading = _HEADING_LINE.match(stripped)
            if heading is not None:
                flush()
                level = len(heading.group(1))
                stack = stack[: level - 1]
                stack.append(heading.group(2))
                # The heading starts the next block, so a section's title travels with it.
                block_start = offset + (len(line) - len(line.lstrip()))
                block_end = offset + len(line.rstrip())
            elif not stripped:
                flush()
            else:
                if block_start is None:
                    block_start = offset + (len(line) - len(line.lstrip()))
                block_end = offset + len(line.rstrip())
            offset += len(line)

        flush()
        return blocks

    def _split_large(self, document: Document, block: _Candidate) -> list[_Candidate]:
        """Break a block that is over target into sentence groups that are not."""
        sentences = _sentence_ranges(document.text[block.start : block.end], block.start)
        if len(sentences) <= 1:
            return [block]
        groups: list[_Candidate] = []
        group_start: int | None = None
        group_end = 0
        tokens = 0
        for start, end in sentences:
            size = self._tokenizer.count_tokens(document.text[start:end])
            if group_start is not None and tokens + size > self._target:
                groups.append(
                    _Candidate(start=group_start, end=group_end, section_path=block.section_path)
                )
                group_start, tokens = None, 0
            if group_start is None:
                group_start = start
            group_end = end
            tokens += size
        if group_start is not None:
            groups.append(
                _Candidate(start=group_start, end=group_end, section_path=block.section_path)
            )
        return groups

    async def chunk(self, document: Document) -> list[Chunk]:
        blocks: list[_Candidate] = []
        for block in self._blocks(document):
            size = self._tokenizer.count_tokens(document.text[block.start : block.end])
            blocks.extend(self._split_large(document, block) if size > self._target else [block])

        merged: list[_Candidate] = []
        for block in blocks:
            if not merged:
                merged.append(block)
                continue
            current = merged[-1]
            combined = self._tokenizer.count_tokens(document.text[current.start : block.end])
            same_section = current.section_path == block.section_path
            if same_section and combined <= self._target:
                merged[-1] = _Candidate(
                    start=current.start, end=block.end, section_path=current.section_path
                )
            else:
                merged.append(block)

        return _finalise(document, merged, self.strategy, self._tokenizer, self._target)


class SemanticChunker:
    """Splits where consecutive sentences diverge in embedding space.

    Finds topic boundaries that no heading marks, at the cost of embedding every sentence
    at ingest time. Sentences are embedded in one batched call rather than one call each,
    because per-sentence calls dominate ingest wall-clock on a real corpus.
    """

    strategy = ChunkStrategy.SEMANTIC

    def __init__(
        self,
        embedder: Embedder,
        tokenizer: Tokenizer,
        threshold_percentile: float = 95.0,
        max_tokens: int = 512,
    ) -> None:
        if not 0.0 < threshold_percentile <= 100.0:
            msg = f"threshold percentile must be in (0, 100], got {threshold_percentile}"
            raise ConfigurationError(msg)
        self._embedder = embedder
        self._tokenizer = tokenizer
        self._percentile = threshold_percentile
        self._max_tokens = max_tokens

    @staticmethod
    def _distances(vectors: Sequence[Sequence[float]]) -> list[float]:
        """Cosine distance between each consecutive pair, guarding zero-magnitude rows."""
        matrix = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1)
        safe = np.where(norms == 0.0, 1.0, norms)
        unit = matrix / safe[:, None]
        similarities = np.sum(unit[:-1] * unit[1:], axis=1)
        return [float(1.0 - value) for value in similarities]

    async def chunk(self, document: Document) -> list[Chunk]:
        sentences = _sentence_ranges(document.text)
        if not sentences:
            return []
        if len(sentences) == 1:
            candidate = _Candidate(start=sentences[0][0], end=sentences[0][1])
            return _finalise(
                document, [candidate], self.strategy, self._tokenizer, self._max_tokens
            )

        texts = [document.text[start:end] for start, end in sentences]
        vectors = await self._embedder.embed_documents(texts)
        distances = self._distances(vectors)
        # With identical sentences every distance is equal, so a strict comparison against
        # the percentile yields no boundary and the document stays whole, which the token
        # cap in _finalise then handles.
        threshold = float(np.percentile(distances, self._percentile)) if distances else 0.0

        candidates: list[_Candidate] = []
        group_start = sentences[0][0]
        group_end = sentences[0][1]
        for index, (start, end) in enumerate(sentences[1:]):
            if distances[index] > threshold:
                candidates.append(_Candidate(start=group_start, end=group_end))
                group_start = start
            group_end = end
        candidates.append(_Candidate(start=group_start, end=group_end))

        return _finalise(document, candidates, self.strategy, self._tokenizer, self._max_tokens)


def build_chunker(
    strategy: ChunkStrategy,
    settings: Settings,
    tokenizer: Tokenizer,
    embedder: Embedder | None = None,
) -> Chunker:
    """Construct the chunker ``strategy`` names, configured from ``settings``.

    Raises:
        ConfigurationError: semantic chunking was requested with no embedder to use.
    """
    if strategy is ChunkStrategy.FIXED_TOKEN:
        return FixedTokenChunker(tokenizer, settings.chunk_size, settings.chunk_overlap)
    if strategy is ChunkStrategy.RECURSIVE_STRUCTURAL:
        return RecursiveStructuralChunker(tokenizer, settings.chunk_size, settings.chunk_min_tokens)
    if embedder is None:
        msg = "semantic chunking needs an embedder to compare sentences with"
        raise ConfigurationError(msg)
    return SemanticChunker(
        embedder,
        tokenizer,
        settings.semantic_threshold_percentile,
        settings.chunk_size,
    )


__all__ = [
    "Chunker",
    "FixedTokenChunker",
    "RecursiveStructuralChunker",
    "SemanticChunker",
    "build_chunker",
]
