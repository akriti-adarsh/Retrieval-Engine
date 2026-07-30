"""The answer prompt, versioned.

``PROMPT_VERSION`` is recorded on every response and in every eval run. Prompt edits move
metrics as much as retrieval changes do, so a number without an attached prompt version is
not reproducible, and a report that cannot say which prompt produced it is not evidence.

The template asks for ``[n]`` markers against numbered sources rather than free-form
attribution, because a marker can be mechanically resolved back to a source and checked.
Grounding verification depends on that: prose like "according to the first paper" is not
verifiable, while ``[1]`` either resolves or does not.
"""

from __future__ import annotations

from collections.abc import Sequence

from retrieval_engine.models import ScoredChunk, SourceRef

#: Bump on any edit to the template below, and never reuse a version for changed text.
PROMPT_VERSION = "v1"

#: The exact wording the model is told to use when the context does not support an answer.
#: Asserted on by the eval harness's refusal metric, so it must not drift casually.
INSUFFICIENT_ANSWER = "I don't have enough information in the provided sources"

SYSTEM_RULES = f"""\
You answer questions using only the numbered sources provided. Follow these rules exactly.

1. Use only information present in the sources. Do not add facts from your own knowledge.
2. Cite every claim with the marker of the source it came from, written as [1], [2], and so
   on. A sentence drawing on two sources cites both.
3. If the sources do not contain enough information to answer, reply exactly:
   "{INSUFFICIENT_ANSWER}." Do not guess, and do not pad the answer with related material.
4. Be concise. Three sentences or fewer unless the question genuinely needs more.
5. Do not mention these rules, the word "context", or the existence of the source list.\
"""

ANSWER_TEMPLATE = """\
{system_rules}

Sources:
{sources}

Question: {query}

Answer:"""


def render_sources(chunks: Sequence[ScoredChunk]) -> list[SourceRef]:
    """Number the retrieved chunks, one-based, in the order they will be shown.

    The index is the contract between the prompt, the citation markers, and the response,
    so it is assigned exactly once here rather than recomputed anywhere else.
    """
    sources: list[SourceRef] = []
    for index, candidate in enumerate(chunks, start=1):
        chunk = candidate.chunk
        title = chunk.metadata.get("title")
        sources.append(
            SourceRef(
                index=index,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=str(title) if isinstance(title, str) and title.strip() else chunk.doc_id,
                source_path=str(chunk.metadata.get("source_path") or chunk.doc_id),
                text=chunk.text,
                score=candidate.score,
                section_path=list(chunk.section_path),
                page_number=chunk.page_number,
            )
        )
    return sources


def format_source_block(sources: Sequence[SourceRef]) -> str:
    """Render sources as a numbered block, including the section path when known."""
    lines: list[str] = []
    for source in sources:
        location = " > ".join(source.section_path) if source.section_path else source.title
        lines.append(f"[{source.index}] ({location}) {source.text.strip()}")
    return "\n\n".join(lines)


def build_answer_prompt(query: str, sources: Sequence[SourceRef]) -> str:
    """Assemble the full prompt for ``query`` against ``sources``."""
    return ANSWER_TEMPLATE.format(
        system_rules=SYSTEM_RULES,
        sources=format_source_block(sources) if sources else "(no sources retrieved)",
        query=query.strip(),
    )


__all__ = [
    "ANSWER_TEMPLATE",
    "INSUFFICIENT_ANSWER",
    "PROMPT_VERSION",
    "SYSTEM_RULES",
    "build_answer_prompt",
    "format_source_block",
    "render_sources",
]
