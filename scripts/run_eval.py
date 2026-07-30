"""Run the golden set against one configuration, or the whole ablation matrix.

The matrix removes one thing at a time, which is the only way a row means anything. Rows that
need a language model (multi-query and HyDE expansion) are included ONLY when a reachable
model is configured. A row produced with expansion silently disabled would be identical to
the row above it while claiming to measure something, which is worse than an absent row.

Every run writes machine-readable JSON next to the markdown, because a number in a README
with no artifact behind it is a claim rather than evidence.

Usage:
    uv run python scripts/run_eval.py --ablate
    uv run python scripts/run_eval.py --limit 12
    uv run python scripts/run_eval.py --ablate --write-baseline
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from retrieval_engine.config import Settings, configure_event_loop, get_settings
from retrieval_engine.eval.golden import load_golden_set
from retrieval_engine.eval.report import (
    render_ablation_table,
    write_ablation,
    write_baseline,
    write_rows,
    write_run,
)
from retrieval_engine.eval.runner import EvalHarness
from retrieval_engine.generate import build_llm
from retrieval_engine.models import (
    AblationReport,
    ChunkStrategy,
    ExpansionMode,
    FusionMethod,
    GoldenEntry,
    RetrievalConfig,
)

#: Metrics the CI regression gate watches. Recall and nDCG catch a retrieval regression;
#: refusal accuracy catches a guardrail regression, which a retrieval metric would miss.
BASELINE_METRICS = ("recall@5", "ndcg@5", "mrr", "refusal_accuracy")

#: How far below the measured value each floor sits. Wide enough that ordinary noise does not
#: fail a build, narrow enough that a real regression does.
BASELINE_MARGIN = 0.05


@dataclass(frozen=True)
class Row:
    """One ablation row: a label, a config, and whether it needs a language model."""

    label: str
    config: RetrievalConfig
    needs_llm: bool = False


def ablation_matrix() -> list[Row]:
    """The predefined matrix. Order matters: it reads as an argument, one change per row."""
    return [
        Row(
            "dense only",
            RetrievalConfig(use_lexical=False, use_rerank=False),
        ),
        Row(
            "lexical only",
            RetrievalConfig(use_dense=False, use_rerank=False),
        ),
        Row(
            "hybrid (RRF)",
            RetrievalConfig(use_rerank=False),
        ),
        Row(
            "hybrid + rerank",
            RetrievalConfig(),
        ),
        Row(
            "hybrid (weighted) + rerank",
            RetrievalConfig(fusion=FusionMethod.WEIGHTED),
        ),
        Row(
            "hybrid + rerank, fixed-token chunking",
            RetrievalConfig(chunk_strategy=ChunkStrategy.FIXED_TOKEN),
        ),
        Row(
            "hybrid + rerank, semantic chunking",
            RetrievalConfig(chunk_strategy=ChunkStrategy.SEMANTIC),
        ),
        Row(
            "hybrid + rerank + multi-query",
            RetrievalConfig(expansion=ExpansionMode.MULTI_QUERY),
            needs_llm=True,
        ),
        Row(
            "hybrid + rerank + HyDE",
            RetrievalConfig(expansion=ExpansionMode.HYDE),
            needs_llm=True,
        ),
    ]


async def _llm_available(settings: Settings) -> bool:
    """Whether a generation backend is actually reachable, not merely configured."""
    llm = build_llm(settings)
    if llm is None:
        return False
    return await llm.health()


async def run(args: argparse.Namespace) -> int:
    settings: Settings = get_settings()
    corpus_dir = Path(args.corpus) if args.corpus else settings.corpus_dir
    golden_path = Path(args.golden) if args.golden else settings.golden_set_path
    results_root = Path(args.out) if args.out else settings.eval_results_dir

    if not corpus_dir.is_dir():
        print(f"corpus directory not found: {corpus_dir}")
        print("run: uv run python scripts/download_corpus.py --limit 300 --with-html")
        return 1

    entries: list[GoldenEntry] = load_golden_set(golden_path)
    if args.limit:
        entries = entries[: args.limit]
    print(f"Loaded {len(entries)} golden questions from {golden_path}")

    has_llm = await _llm_available(settings)
    llm = build_llm(settings) if has_llm else None
    print(f"Generation backend: {'reachable' if has_llm else 'unavailable, using extraction'}")

    harness = EvalHarness(settings, corpus_dir, llm=llm)

    rows = ablation_matrix() if args.ablate else [Row("hybrid + rerank", settings.retrieval)]
    skipped = [row.label for row in rows if row.needs_llm and not has_llm]
    rows = [row for row in rows if not (row.needs_llm and not has_llm)]
    if skipped:
        print(f"Skipping {len(skipped)} row(s) that need a language model: {', '.join(skipped)}")
        print("  These rows are omitted rather than run with expansion silently disabled,")
        print("  which would duplicate the row above while claiming to measure something.")

    report = AblationReport(corpus_docs=0, corpus_chunks=0, golden_questions=len(entries))

    for position, row in enumerate(rows, start=1):
        print(f"\n[{position}/{len(rows)}] {row.label}")
        run_id = f"{uuid.uuid4().hex[:8]}-{row.config.chunk_strategy.value}"
        aggregate, per_query = await harness.run(
            entries,
            row.config,
            label=row.label,
            concurrency=args.concurrency,
            run_id=run_id,
        )
        write_rows(results_root, aggregate.run_id, per_query)
        write_run(results_root, aggregate)
        report.runs.append(aggregate)

        index = await harness.index_for(row.config.chunk_strategy)
        report.corpus_docs = max(report.corpus_docs, index.doc_count)
        report.corpus_chunks = max(report.corpus_chunks, index.chunk_count)

        headline = ", ".join(
            f"{name}={aggregate.metrics[name]:.3f}"
            for name in ("recall@5", "ndcg@5", "mrr", "refusal_accuracy")
            if name in aggregate.metrics
        )
        print(f"    {headline}")

        # Rewrite the report after every row. A full-corpus ablation takes over an hour, and
        # results that only land at the end are results you lose to an interrupted run.
        write_ablation(results_root, report)

    paths = write_ablation(results_root, report)
    print("\n" + render_ablation_table(report))
    print(f"\nWrote {paths['markdown']} and {paths['json']}")

    if args.write_baseline and report.runs:
        # The gate is set from the default configuration, which is what a user gets.
        default = next(
            (run_ for run_ in report.runs if run_.label == "hybrid + rerank"), report.runs[0]
        )
        baseline_path = write_baseline(
            Path(results_root) / "baseline_real_corpus.json",
            default,
            BASELINE_METRICS,
            BASELINE_MARGIN,
        )
        print(f"Wrote regression baseline from '{default.label}' to {baseline_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablate", action="store_true", help="Run the full matrix.")
    parser.add_argument("--corpus", default=None, help="Corpus directory.")
    parser.add_argument("--golden", default=None, help="Golden set JSONL.")
    parser.add_argument("--out", default=None, help="Results directory.")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N questions.")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent questions.")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Also write the CI regression baseline from the default configuration.",
    )
    args = parser.parse_args(argv)
    # Before the loop exists: psycopg's async pool cannot use Windows' default loop.
    configure_event_loop()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
