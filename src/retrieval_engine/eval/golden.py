"""Golden-set loading and validation.

The validator is the only thing standing between a plausible-looking evaluation and a
meaningless one. Its central rule is that every span in the golden set must be an EXACT
substring of some document in the corpus. That rule is what makes the retrieval metrics
trustworthy, because it guarantees the ground truth is text that actually exists rather than
a paraphrase somebody wrote from memory.

When a span fails to match, the fix is always to correct the golden entry against the corpus.
Loosening the validator to fuzzy matching would make every subsequent number
unfalsifiable, which is worse than having no evaluation at all.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import Field

from retrieval_engine.errors import GoldenSetValidationError
from retrieval_engine.models import (
    MAX_GOLDEN_SPAN_CHARS,
    Base,
    Document,
    GoldenCategory,
    GoldenEntry,
)

#: The spec requires at least this many questions the corpus genuinely cannot answer, whose
#: correct behaviour is refusal. Without them, a system that answers everything confidently
#: scores well on every other metric while being untrustworthy.
MIN_NEGATIVE_ENTRIES = 8


class ValidationReport(Base):
    """What the validator found. ``ok`` is the gate; the lists explain a failure."""

    ok: bool
    entries_checked: int = 0
    spans_checked: int = 0
    negatives: int = 0
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """Human summary, used by the CLI."""
        verdict = "PASS" if self.ok else "FAIL"
        return (
            f"{verdict}: {self.entries_checked} entries, {self.spans_checked} spans, "
            f"{self.negatives} negative, {len(self.failures)} failures, "
            f"{len(self.warnings)} warnings"
        )


def load_golden_set(path: Path) -> list[GoldenEntry]:
    """Read a JSONL golden set.

    Raises:
        GoldenSetValidationError: the file is missing, or a line is not a valid entry. The
            line number is included, because a 60-line file with one bad row is otherwise
            tedious to debug.
    """
    if not path.is_file():
        msg = f"golden set not found at {path}"
        raise GoldenSetValidationError(msg)

    entries: list[GoldenEntry] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entries.append(GoldenEntry.model_validate_json(stripped))
        except ValueError as exc:
            msg = f"{path}:{number} is not a valid golden entry: {exc}"
            raise GoldenSetValidationError(msg) from exc
    return entries


def write_golden_set(path: Path, entries: Iterable[GoldenEntry]) -> int:
    """Write entries as JSONL, one compact object per line. Returns how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [entry.model_dump_json() for entry in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def validate_golden_set(
    entries: Sequence[GoldenEntry],
    documents: Sequence[Document],
    *,
    require_negatives: int = MIN_NEGATIVE_ENTRIES,
) -> ValidationReport:
    """Check every entry against the corpus and the golden-set contract.

    Checks, in the order a failure is most likely:

    1. Every span is an exact substring of some document's text. Not normalised, not fuzzy.
    2. Every span is within the character cap, so spans stay matching keys rather than
       becoming quoted documents.
    3. Question ids are unique, since the runner keys results by them and a duplicate would
       silently overwrite a result.
    4. Every referenced document id exists in the corpus.
    5. There are enough negative entries to make refusal accuracy meaningful.
    """
    failures: list[str] = []
    warnings: list[str] = []
    by_id = {document.doc_id: document for document in documents}
    texts = [document.text for document in documents]

    seen_qids: set[str] = set()
    spans_checked = 0
    negatives = 0

    for entry in entries:
        if entry.qid in seen_qids:
            failures.append(f"{entry.qid}: duplicate question id")
        seen_qids.add(entry.qid)

        if entry.category is GoldenCategory.NEGATIVE:
            negatives += 1

        for doc_id in entry.relevant_doc_ids:
            if doc_id not in by_id:
                failures.append(f"{entry.qid}: references unknown document {doc_id!r}")

        for span in entry.relevant_chunk_texts:
            spans_checked += 1
            if len(span) > MAX_GOLDEN_SPAN_CHARS:
                failures.append(
                    f"{entry.qid}: span is {len(span)} chars, over the {MAX_GOLDEN_SPAN_CHARS} cap"
                )
                continue
            # Prefer the documents the entry points at, then fall back to the whole corpus:
            # a span quoted in another document is still a real span.
            named = [by_id[d].text for d in entry.relevant_doc_ids if d in by_id]
            if any(span in text for text in named):
                continue
            if any(span in text for text in texts):
                warnings.append(
                    f"{entry.qid}: span matches the corpus but not the documents it names"
                )
                continue
            preview = span[:60].replace("\n", " ")
            failures.append(f"{entry.qid}: span is not an exact substring: {preview!r}")

    if negatives < require_negatives:
        failures.append(
            f"only {negatives} negative entries, at least {require_negatives} are required "
            "for refusal accuracy to mean anything"
        )

    return ValidationReport(
        ok=not failures,
        entries_checked=len(entries),
        spans_checked=spans_checked,
        negatives=negatives,
        failures=failures,
        warnings=warnings,
    )


def summarise_categories(entries: Sequence[GoldenEntry]) -> dict[str, int]:
    """Count entries per category, for the report and the README."""
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.category.value] = counts.get(entry.category.value, 0) + 1
    return counts


def summarise_difficulty(entries: Sequence[GoldenEntry]) -> dict[str, int]:
    """Count entries per difficulty band."""
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.difficulty.value] = counts.get(entry.difficulty.value, 0) + 1
    return counts


def as_jsonl(entries: Iterable[GoldenEntry]) -> str:
    """Render entries as a JSONL string without touching the filesystem."""
    return "\n".join(json.dumps(json.loads(entry.model_dump_json())) for entry in entries)


__all__ = [
    "MIN_NEGATIVE_ENTRIES",
    "ValidationReport",
    "as_jsonl",
    "load_golden_set",
    "summarise_categories",
    "summarise_difficulty",
    "validate_golden_set",
    "write_golden_set",
]
