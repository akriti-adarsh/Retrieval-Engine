"""Writing eval results to disk as both JSON and markdown.

Every number that appears in the README has to exist in a committed artifact, so this module
is the only thing that produces those numbers and it always writes the machine-readable JSON
alongside the human-readable table. A markdown table with no JSON beside it is a claim; the
pair is evidence.

The rendered table records the embedder, the generator, the prompt version, and the seed on
every run. A metric without those is not reproducible, and an ablation whose rows were
produced under different conditions is not a comparison.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from retrieval_engine.models import AblationReport, EvalRow, EvalRun

#: Columns of the headline ablation table, in the order the spec's table lists them.
TABLE_COLUMNS = (
    ("recall@5", "recall@5"),
    ("ndcg@5", "nDCG@5"),
    ("mrr", "MRR"),
    ("refusal_accuracy", "refusal acc"),
)


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def run_directory(root: Path, run_id: str) -> Path:
    """Where one run's artifacts live."""
    return root / run_id


def write_rows(root: Path, run_id: str, rows: Sequence[EvalRow]) -> Path:
    """Write per-query rows as JSONL. One line per question, in golden-set order."""
    directory = run_directory(root, run_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "rows.jsonl"
    path.write_text("\n".join(row.model_dump_json() for row in rows) + "\n", encoding="utf-8")
    return path


def write_run(root: Path, run: EvalRun) -> Path:
    """Write one run's aggregate metrics as JSON."""
    directory = run_directory(root, run.run_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run.json"
    path.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def render_ablation_table(report: AblationReport) -> str:
    """The headline comparison table, one row per configuration."""
    header = "| config | " + " | ".join(label for _, label in TABLE_COLUMNS)
    header += " | false refusal | p95 latency |"
    divider = "|---" * (len(TABLE_COLUMNS) + 3) + "|"

    lines = [header, divider]
    for run in report.runs:
        cells = [_fmt(run.metrics.get(key, 0.0)) for key, _ in TABLE_COLUMNS]
        false_refusal = _fmt(run.metrics.get("false_refusal_rate", 0.0))
        p95 = run.latency.get("total")
        latency = f"{p95.p95_ms:.0f} ms" if p95 is not None else "n/a"
        lines.append(f"| {run.label} | " + " | ".join(cells) + f" | {false_refusal} | {latency} |")
    return "\n".join(lines)


def render_stage_latency_table(run: EvalRun) -> str:
    """Per-stage latency for one run, which is where the cost actually sits."""
    lines = ["| stage | p50 | p95 | mean | max |", "|---|---|---|---|---|"]
    for stage, stats in run.latency.items():
        lines.append(
            f"| {stage} | {stats.p50_ms:.1f} ms | {stats.p95_ms:.1f} ms | "
            f"{stats.mean_ms:.1f} ms | {stats.max_ms:.1f} ms |"
        )
    return "\n".join(lines)


def render_breakdown_table(run: EvalRun, breakdown: str) -> str:
    """A by-category or by-difficulty breakdown for one run."""
    data = run.by_category if breakdown == "category" else run.by_difficulty
    if not data:
        return "_no breakdown recorded_"
    lines = [f"| {breakdown} | n | recall@5 | nDCG@5 | MRR |", "|---|---|---|---|---|"]
    for name in sorted(data):
        values = data[name]
        lines.append(
            f"| {name} | {int(values.get('n', 0))} | {_fmt(values.get('recall@5', 0.0))} | "
            f"{_fmt(values.get('ndcg@5', 0.0))} | {_fmt(values.get('mrr', 0.0))} |"
        )
    return "\n".join(lines)


def render_report(report: AblationReport) -> str:
    """The full markdown report for an ablation run."""
    if not report.runs:
        return "# Ablation\n\n_no runs recorded_\n"

    first = report.runs[0]
    parts = [
        "# Ablation results",
        "",
        f"Generated {report.generated_at.isoformat()}",
        "",
        "| property | value |",
        "|---|---|",
        f"| corpus documents | {report.corpus_docs} |",
        f"| corpus chunks | {report.corpus_chunks} |",
        f"| golden questions | {report.golden_questions} |",
        f"| embedder | `{first.embedder}` |",
        f"| generator | `{first.generator}` |",
        f"| prompt version | `{first.prompt_version}` |",
        f"| seed | {first.seed} |",
        "",
        "## Headline comparison",
        "",
        render_ablation_table(report),
        "",
        "`false refusal` is the fraction of answerable questions that were refused. It is",
        "reported next to refusal accuracy because a system can score perfectly on one by",
        "failing the other: refusing everything gives refusal accuracy 1.000.",
        "",
    ]

    for run in report.runs:
        parts.extend(
            [
                f"## {run.label}",
                "",
                f"Run `{run.run_id}`, {run.n_queries} questions, "
                f"{run.elapsed_seconds:.1f}s wall clock.",
                "",
                "### Metrics",
                "",
                "| metric | value |",
                "|---|---|",
                *[f"| {name} | {_fmt(value)} |" for name, value in sorted(run.metrics.items())],
                "",
                "### By category",
                "",
                render_breakdown_table(run, "category"),
                "",
                "### By difficulty",
                "",
                render_breakdown_table(run, "difficulty"),
                "",
                "### Stage latency",
                "",
                render_stage_latency_table(run),
                "",
                "### Exact configuration",
                "",
                "```json",
                run.config.model_dump_json(indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(parts)


def write_ablation(root: Path, report: AblationReport, name: str = "ablation") -> dict[str, Path]:
    """Write the ablation report as both JSON and markdown. Returns both paths."""
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{name}.json"
    markdown_path = root / f"{name}.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_report(report) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def write_baseline(path: Path, run: EvalRun, metrics: Sequence[str], margin: float) -> Path:
    """Write the regression baseline the CI gate compares against.

    Floors are set a margin below the measured value, so ordinary run-to-run noise does not
    fail the build while a real regression still does. The measured value is stored next to
    the floor, so a reader can see how much headroom was allowed and why.
    """
    payload = {
        "run_id": run.run_id,
        "label": run.label,
        "embedder": run.embedder,
        "generator": run.generator,
        "prompt_version": run.prompt_version,
        "seed": run.seed,
        "margin": margin,
        "measured": {name: run.metrics.get(name, 0.0) for name in metrics},
        "floors": {
            name: round(max(0.0, run.metrics.get(name, 0.0) - margin), 4) for name in metrics
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "TABLE_COLUMNS",
    "render_ablation_table",
    "render_breakdown_table",
    "render_report",
    "render_stage_latency_table",
    "run_directory",
    "write_ablation",
    "write_baseline",
    "write_rows",
    "write_run",
]
