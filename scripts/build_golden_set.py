"""Golden-set tooling: generate candidates, and validate the committed set.

``--validate`` is the important half. It checks that every span in the committed golden set is
an EXACT substring of some corpus document. When a span fails, the fix is always to correct the
golden entry against the corpus, never to relax the validator: exact matching is the only
reason the retrieval metrics can be trusted, and a fuzzy validator would make every number
downstream unfalsifiable.

``--generate`` drafts candidate questions with the configured local model and writes them to a
REVIEW file, never to the committed set. Generated questions are a starting point for a human
pass, and writing them straight into the golden set would let the system grade itself against
its own guesses.

Usage:
    uv run python scripts/build_golden_set.py --validate
    uv run python scripts/build_golden_set.py --stats
    uv run python scripts/build_golden_set.py --generate --count 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from retrieval_engine.config import Settings, get_settings
from retrieval_engine.errors import GoldenSetValidationError, LLMUnavailableError
from retrieval_engine.eval.golden import (
    load_golden_set,
    summarise_categories,
    summarise_difficulty,
    validate_golden_set,
)
from retrieval_engine.generate import build_llm
from retrieval_engine.ingest.loaders import iter_source_files, load_document
from retrieval_engine.models import Document

CANDIDATE_PROMPT = """\
Read the passage below from a research paper and write one specific factual question that the
passage answers, plus the exact sentence from the passage that answers it.

Rules:
- The question must be answerable from this passage alone.
- The question must not mention "the passage", "the paper", or "the text".
- Copy the answering sentence VERBATIM. Do not paraphrase it, do not fix its punctuation.
- Keep the copied sentence under 300 characters.

Reply as exactly two lines:
Q: <the question>
S: <the verbatim sentence>

Passage:
{passage}
"""


def _write_candidates(path: Path, candidates: list[dict[str, object]]) -> None:
    """Write drafts to the review file. Sync, so the blocking IO stays off the event loop."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(candidate) for candidate in candidates) + "\n", encoding="utf-8"
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load every corpus document, skipping the ones that will not parse."""
    documents: list[Document] = []
    for path in iter_source_files(corpus_dir):
        try:
            documents.append(load_document(path))
        except Exception as exc:  # a broken file must not stop validation of the rest
            print(f"  skipping {path.name}: {type(exc).__name__}: {exc}")
    return documents


def command_validate(settings: Settings, args: argparse.Namespace) -> int:
    golden_path = Path(args.golden) if args.golden else settings.golden_set_path
    corpus_dir = Path(args.corpus) if args.corpus else settings.corpus_dir

    if not corpus_dir.is_dir():
        print(f"corpus directory not found: {corpus_dir}")
        print("run: uv run python scripts/download_corpus.py --limit 300 --with-html")
        return 1

    try:
        entries = load_golden_set(golden_path)
    except GoldenSetValidationError as exc:
        print(f"FAIL: {exc.message}")
        return 1

    print(f"Loaded {len(entries)} entries from {golden_path}")
    documents = load_corpus(corpus_dir)
    print(f"Loaded {len(documents)} corpus documents from {corpus_dir}")

    report = validate_golden_set(entries, documents)
    print(report.render())
    for warning in report.warnings:
        print(f"  warning: {warning}")
    for failure in report.failures:
        print(f"  FAIL: {failure}")
    return 0 if report.ok else 1


def command_stats(settings: Settings, args: argparse.Namespace) -> int:
    golden_path = Path(args.golden) if args.golden else settings.golden_set_path
    entries = load_golden_set(golden_path)

    print(f"{len(entries)} entries in {golden_path}\n")
    print("| category | n |")
    print("|---|---|")
    for name, count in sorted(summarise_categories(entries).items()):
        print(f"| {name} | {count} |")
    print("\n| difficulty | n |")
    print("|---|---|")
    for name, count in sorted(summarise_difficulty(entries).items()):
        print(f"| {name} | {count} |")

    spans = [len(span) for entry in entries for span in entry.relevant_chunk_texts]
    if spans:
        print(f"\nspans: {len(spans)}, longest {max(spans)} chars, mean {sum(spans) // len(spans)}")
    return 0


async def command_generate(settings: Settings, args: argparse.Namespace) -> int:
    """Draft candidate questions into a review file, using the configured local model."""
    corpus_dir = Path(args.corpus) if args.corpus else settings.corpus_dir
    review_path = Path(args.review)

    llm = build_llm(settings)
    if llm is None or not await llm.health():
        print("No reachable generation backend, so candidates cannot be drafted.")
        print("Start Ollama (or set RE_LLM=ollama with a running server) and try again.")
        print("Nothing was written; a golden set is not something to fake.")
        return 1

    documents = load_corpus(corpus_dir)
    if not documents:
        print(f"no documents in {corpus_dir}")
        return 1

    from retrieval_engine.embed import build_embedder
    from retrieval_engine.ingest.chunker import build_chunker

    embedder = build_embedder(settings)
    chunker = build_chunker(settings.chunk_strategy, settings, embedder.tokenizer, embedder)

    candidates: list[dict[str, object]] = []
    # Spread across documents rather than taking many from one, so the question set is not
    # dominated by whichever paper happens to chunk into the most passages.
    for position, document in enumerate(documents):
        if len(candidates) >= args.count:
            break
        chunks = await chunker.chunk(document)
        usable = [chunk for chunk in chunks if len(chunk.text) > 400]
        if not usable:
            continue
        chunk = usable[position % len(usable)]
        try:
            reply = await llm.complete(CANDIDATE_PROMPT.format(passage=chunk.text[:2000]))
        except LLMUnavailableError as exc:
            print(f"  model became unreachable: {exc.message}")
            break

        question = span = ""
        for line in reply.splitlines():
            if line.startswith("Q:"):
                question = line[2:].strip()
            elif line.startswith("S:"):
                span = line[2:].strip()
        if not question or not span:
            continue
        # Only keep a candidate whose span really is verbatim. A generated "quote" that does
        # not appear in the source is exactly what the validator exists to reject, so there is
        # no point writing it to the review file.
        if span not in document.text:
            print(f"  dropped a candidate from {document.doc_id}: span was not verbatim")
            continue
        candidates.append(
            {
                "qid": f"q{len(candidates) + 1:03d}",
                "question": question,
                "relevant_doc_ids": [document.doc_id],
                "relevant_chunk_texts": [span[:300]],
                "answer": "",
                "category": "factual",
                "difficulty": "medium",
            }
        )
        print(f"  [{len(candidates)}/{args.count}] {document.doc_id}: {question[:70]}")

    _write_candidates(review_path, candidates)
    print(f"\nWrote {len(candidates)} candidates to {review_path}")
    print("These are DRAFTS. Review them, add negative and multi-hop entries by hand, then")
    print("copy the reviewed set to data/golden/golden_set.jsonl and run --validate.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="Validate the committed set.")
    parser.add_argument("--stats", action="store_true", help="Print composition statistics.")
    parser.add_argument("--generate", action="store_true", help="Draft candidates to review.")
    parser.add_argument("--corpus", default=None, help="Corpus directory.")
    parser.add_argument("--golden", default=None, help="Golden set JSONL.")
    parser.add_argument(
        "--review",
        default="data/golden/candidates.jsonl",
        help="Where generated drafts go. Never the committed set.",
    )
    parser.add_argument("--count", type=int, default=60, help="How many candidates to draft.")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.generate:
        return asyncio.run(command_generate(settings, args))
    if args.stats:
        return command_stats(settings, args)
    if args.validate:
        return command_validate(settings, args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
