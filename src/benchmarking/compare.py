"""Compare benchmark results and training runs.

Two comparisons matter when developing a small model, and they are different:

* **Between models** - which architecture or size wins on the same tasks. That
  is :func:`compare_results`.
* **Between runs of the same model** - did this hyperparameter change help?
  That is :func:`compare_runs`, which reads ``metrics.jsonl`` rather than
  benchmark JSON.

Both render Markdown, because the output is usually going straight into a model
card or a pull request.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from minimodel.checkpointing.loss_visualization import summarize_run
from minimodel.core.io_utils import human_count, read_json
from minimodel.core.logging_utils import get_logger

__all__ = [
    "ComparisonTable",
    "compare_results",
    "compare_runs",
    "load_results",
    "markdown_table",
    "pareto_frontier",
]

logger = get_logger(__name__)

#: Metrics where a smaller value is better.
LOWER_IS_BETTER = {"perplexity", "loss", "val_loss", "bits_per_token", "ms_per_token"}


class ComparisonTable:
    """A small table of rows with a stable column order."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None):
        self.rows = [dict(row) for row in rows]
        if columns:
            self.columns = list(columns)
        else:
            seen: list[str] = []
            for row in self.rows:
                for key in row:
                    if key not in seen:
                        seen.append(key)
            self.columns = seen

    def __len__(self) -> int:
        return len(self.rows)

    def to_markdown(self, *, highlight_best: bool = True, float_format: str = "{:.4g}") -> str:
        """Render as a GitHub-flavoured Markdown table."""
        return markdown_table(
            self.rows,
            self.columns,
            highlight_best=highlight_best,
            float_format=float_format,
        )

    def to_csv(self) -> str:
        """Render as CSV."""
        lines = [",".join(self.columns)]
        for row in self.rows:
            lines.append(",".join(str(row.get(column, "")) for column in self.columns))
        return "\n".join(lines)

    def to_dict(self) -> list[dict[str, Any]]:
        """Return the raw rows."""
        return [dict(row) for row in self.rows]

    def __repr__(self) -> str:
        return f"ComparisonTable(rows={len(self.rows)}, columns={self.columns})"


def _best_index(values: list[Any], column: str) -> int | None:
    """Index of the best value in ``values``, or ``None`` if not comparable."""
    numeric = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
    if len(numeric) < 2:
        return None
    lower_better = any(token in column.lower() for token in LOWER_IS_BETTER)
    picker = min if lower_better else max
    return picker(numeric, key=lambda pair: pair[1])[0]


def markdown_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
    *,
    highlight_best: bool = True,
    float_format: str = "{:.4g}",
) -> str:
    """Render rows as Markdown, optionally bolding the best cell per column."""
    rows = [dict(row) for row in rows]
    if not rows:
        return "_no results_"
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))

    formatted: list[list[str]] = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(float_format.format(value))
            elif isinstance(value, int) and column in {"parameters", "active_parameters", "tokens"}:
                cells.append(human_count(value))
            else:
                cells.append(str(value))
        formatted.append(cells)

    if highlight_best:
        for index, column in enumerate(columns):
            best = _best_index([row.get(column) for row in rows], column)
            if best is not None:
                formatted[best][index] = f"**{formatted[best][index]}**"

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(cells) + " |" for cells in formatted]
    return "\n".join([header, divider, *body])


def load_results(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load benchmark JSON files written by :meth:`BenchmarkResult.save`."""
    results: list[dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            candidates = sorted(path.glob("*.json"))
            if not candidates:
                logger.warning("no benchmark JSON in %s", path)
                continue
            path = candidates[0]
        results.append(read_json(path))
    return results


def compare_results(
    results: Sequence[Mapping[str, Any] | str | Path],
    *,
    metric_filter: Sequence[str] | None = None,
    include_throughput: bool = True,
) -> ComparisonTable:
    """Build a model-by-task comparison table.

    Each row is a model; each column is a task's headline metric.
    """
    loaded: list[dict[str, Any]] = []
    for entry in results:
        if isinstance(entry, (str, Path)):
            loaded.extend(load_results([entry]))
        else:
            loaded.append(dict(entry))

    rows: list[dict[str, Any]] = []
    for result in loaded:
        row: dict[str, Any] = {
            "model": result.get("model", "?"),
            "params": result.get("parameters", 0),
        }
        for task_name, metrics in (result.get("tasks") or {}).items():
            if metric_filter and task_name not in metric_filter:
                continue
            for key in ("accuracy_norm", "accuracy", "perplexity", "solve_rate"):
                if key in metrics:
                    row[task_name] = round(float(metrics[key]), 5)
                    break
        if include_throughput and result.get("throughput"):
            decode = result["throughput"].get("decode_tokens_per_second")
            if decode:
                row["decode_tok_s"] = round(float(decode), 1)
        rows.append(row)
    return ComparisonTable(rows)


def compare_runs(
    runs: Sequence[str | Path],
    *,
    names: Sequence[str] | None = None,
) -> ComparisonTable:
    """Compare training runs by reading each one's ``metrics.jsonl``."""
    rows: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        run_path = Path(run)
        metrics_path = run_path / "metrics.jsonl" if run_path.is_dir() else run_path
        try:
            summary = summarize_run(metrics_path)
        except FileNotFoundError:
            logger.warning("no metrics found for %s", run)
            continue
        name = names[index] if names and index < len(names) else run_path.name
        row: dict[str, Any] = {"run": name}
        for key in (
            "steps",
            "final_loss",
            "best_loss",
            "best_val_loss",
            "final_perplexity",
            "total_tokens",
            "elapsed_seconds",
        ):
            if key in summary:
                row[key] = summary[key]
        rows.append(row)
    return ComparisonTable(rows)


def pareto_frontier(
    results: Sequence[Mapping[str, Any]],
    *,
    quality_key: str = "accuracy",
    cost_key: str = "parameters",
) -> list[dict[str, Any]]:
    """Return the entries not dominated on both quality and cost.

    For a size ladder this answers the only question that matters: which of
    these models is actually worth training, given that a bigger one exists?
    """
    entries = [dict(r) for r in results if quality_key in r and cost_key in r]
    frontier: list[dict[str, Any]] = []
    for candidate in sorted(entries, key=lambda r: float(r[cost_key])):
        dominated = any(
            float(other[quality_key]) >= float(candidate[quality_key])
            and float(other[cost_key]) <= float(candidate[cost_key])
            and other is not candidate
            for other in frontier
        )
        if not dominated:
            frontier.append(candidate)
    return frontier


def summarize_comparison(table: ComparisonTable, *, title: str = "Comparison") -> str:
    """Wrap a table in a Markdown section with a generated timestamp."""
    import datetime

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"## {title}\n\n_generated {stamp}_\n\n{table.to_markdown()}\n"


def write_comparison(
    table: ComparisonTable, path: str | Path, *, title: str = "Comparison"
) -> Path:
    """Write a comparison to Markdown (``.md``), JSON (``.json``) or CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(table.to_dict(), indent=2) + "\n", encoding="utf-8")
    elif path.suffix == ".csv":
        path.write_text(table.to_csv() + "\n", encoding="utf-8")
    else:
        path.write_text(summarize_comparison(table, title=title), encoding="utf-8")
    return path
