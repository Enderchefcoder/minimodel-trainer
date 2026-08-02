"""Charts for benchmark results.

Every function degrades to an ASCII rendering when matplotlib is missing, so
that inspecting results over SSH or in CI never fails for want of a plotting
backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from minimodel.core.io_utils import ensure_dir, human_count
from minimodel.core.logging_utils import get_logger

__all__ = [
    "ascii_bar_chart",
    "plot_scaling_curve",
    "plot_task_comparison",
    "plot_throughput",
]

logger = get_logger(__name__)


def _matplotlib():
    """Import matplotlib with the Agg backend, or return ``None``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def ascii_bar_chart(
    values: Mapping[str, float], *, width: int = 40, title: str = "", value_format: str = "{:.4g}"
) -> str:
    """Render a labelled horizontal bar chart as text."""
    if not values:
        return f"{title}: no data"
    label_width = max(len(str(k)) for k in values)
    largest = max(abs(float(v)) for v in values.values()) or 1.0
    lines = [title] if title else []
    for label, value in values.items():
        filled = int(abs(float(value)) / largest * width)
        bar = "#" * filled
        lines.append(f"{str(label):<{label_width}} | {bar:<{width}} {value_format.format(value)}")
    return "\n".join(lines)


def plot_task_comparison(
    results: Sequence[Mapping[str, Any]],
    output: str | Path | None = None,
    *,
    title: str = "Benchmark comparison",
    chance_lines: bool = True,
) -> str | Path:
    """Grouped bar chart of task scores across models."""
    models = [str(r.get("model", f"model{i}")) for i, r in enumerate(results)]
    task_names: list[str] = []
    for result in results:
        for name in (result.get("tasks") or {}):
            if name not in task_names:
                task_names.append(name)

    def _score(result: Mapping[str, Any], task: str) -> float:
        metrics = (result.get("tasks") or {}).get(task) or {}
        for key in ("accuracy_norm", "accuracy", "solve_rate"):
            if key in metrics:
                return float(metrics[key])
        return float("nan")

    plt = _matplotlib()
    if plt is None:
        flat = {
            f"{model}/{task}": _score(result, task)
            for model, result in zip(models, results, strict=True)
            for task in task_names
        }
        return ascii_bar_chart(flat, title=title)

    figure, axis = plt.subplots(figsize=(max(7, 1.6 * len(task_names)), 5), dpi=140)
    bar_width = 0.8 / max(1, len(models))
    for index, (model, result) in enumerate(zip(models, results, strict=True)):
        positions = [i + index * bar_width for i in range(len(task_names))]
        scores = [_score(result, task) for task in task_names]
        axis.bar(positions, scores, width=bar_width, label=model)

    if chance_lines:
        for i, task in enumerate(task_names):
            for result in results:
                chance = (result.get("tasks") or {}).get(task, {}).get("chance")
                if chance:
                    axis.hlines(
                        float(chance),
                        i - 0.1,
                        i + 0.8,
                        colors="grey",
                        linestyles="dotted",
                        linewidth=1,
                    )
                    break

    axis.set_xticks([i + 0.4 - bar_width / 2 for i in range(len(task_names))])
    axis.set_xticklabels(task_names, rotation=20, ha="right")
    axis.set_ylabel("score")
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.25, linestyle="--")
    axis.legend(frameon=False)
    figure.tight_layout()

    output = Path(output or "benchmark_comparison.png")
    ensure_dir(output.parent)
    figure.savefig(output)
    plt.close(figure)
    logger.info("wrote %s", output)
    return output


def plot_scaling_curve(
    points: Sequence[Mapping[str, Any]],
    output: str | Path | None = None,
    *,
    x_key: str = "parameters",
    y_key: str = "perplexity",
    label_key: str = "model",
    title: str = "Scaling",
) -> str | Path:
    """Plot a metric against model size on log-log axes.

    A straight line here is the expected power-law behaviour; a knee usually
    means one of the points was under-trained rather than that scaling broke.
    """
    usable = [p for p in points if x_key in p and y_key in p]
    if not usable:
        return "no data"

    plt = _matplotlib()
    if plt is None:
        return ascii_bar_chart(
            {str(p.get(label_key, i)): float(p[y_key]) for i, p in enumerate(usable)},
            title=f"{title} ({y_key})",
        )

    figure, axis = plt.subplots(figsize=(7, 5), dpi=140)
    xs = [float(p[x_key]) for p in usable]
    ys = [float(p[y_key]) for p in usable]
    axis.plot(xs, ys, marker="o", linewidth=1.6)
    for point, x, y in zip(usable, xs, ys, strict=True):
        axis.annotate(
            str(point.get(label_key, "")),
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    axis.set_xscale("log")
    if all(y > 0 for y in ys):
        axis.set_yscale("log")
    axis.set_xlabel(f"{x_key} (log)")
    axis.set_ylabel(f"{y_key} (log)")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25, linestyle="--")
    figure.tight_layout()

    output = Path(output or "scaling_curve.png")
    ensure_dir(output.parent)
    figure.savefig(output)
    plt.close(figure)
    logger.info("wrote %s", output)
    return output


def plot_throughput(
    results: Sequence[Mapping[str, Any]],
    output: str | Path | None = None,
    *,
    title: str = "Throughput",
) -> str | Path:
    """Bar chart of prefill and decode speed per model."""
    usable = [r for r in results if r.get("throughput")]
    if not usable:
        return "no throughput data"

    labels = [str(r.get("model", f"model{i}")) for i, r in enumerate(usable)]
    prefill = [float(r["throughput"].get("prefill_tokens_per_second", 0)) for r in usable]
    decode = [float(r["throughput"].get("decode_tokens_per_second", 0)) for r in usable]

    plt = _matplotlib()
    if plt is None:
        merged = {f"{label} prefill": p for label, p in zip(labels, prefill, strict=True)}
        merged.update({f"{label} decode": d for label, d in zip(labels, decode, strict=True)})
        return ascii_bar_chart(merged, title=title, value_format="{:.0f}")

    figure, axis = plt.subplots(figsize=(max(6, 1.5 * len(labels)), 5), dpi=140)
    positions = range(len(labels))
    axis.bar([p - 0.2 for p in positions], prefill, width=0.4, label="prefill")
    axis.bar([p + 0.2 for p in positions], decode, width=0.4, label="decode")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, rotation=15, ha="right")
    axis.set_ylabel("tokens / second")
    axis.set_title(title)
    axis.set_yscale("log")
    axis.grid(True, axis="y", alpha=0.25, linestyle="--")
    axis.legend(frameon=False)
    figure.tight_layout()

    output = Path(output or "throughput.png")
    ensure_dir(output.parent)
    figure.savefig(output)
    plt.close(figure)
    logger.info("wrote %s", output)
    return output


def render_report(
    results: Sequence[Mapping[str, Any]], output_dir: str | Path, *, title: str = "Benchmark report"
) -> Path:
    """Write a Markdown report with charts into ``output_dir``."""
    from minimodel.benchmarking.compare import compare_results, summarize_comparison

    output_dir = ensure_dir(output_dir)
    table = compare_results(results)
    chart = plot_task_comparison(results, output_dir / "tasks.png", title=title)
    throughput = plot_throughput(results, output_dir / "throughput.png")

    lines = [summarize_comparison(table, title=title)]
    if isinstance(chart, Path):
        lines.append(f"![tasks]({chart.name})\n")
    if isinstance(throughput, Path):
        lines.append(f"![throughput]({throughput.name})\n")
    lines.append("### Models\n")
    for result in results:
        lines.append(
            f"- **{result.get('model', '?')}** - "
            f"{human_count(result.get('parameters', 0))} parameters"
        )

    report = output_dir / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", report)
    return report
