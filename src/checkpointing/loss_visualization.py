"""Loss curves and training diagnostics.

Reads the ``metrics.jsonl`` written by the trainer and renders it either as a
PNG (when matplotlib is installed) or as an ASCII sparkline plot in the
terminal. The ASCII path is not a fallback of last resort - it is genuinely the
fastest way to check on a run over SSH.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from minimodel.core.io_utils import ensure_dir, read_jsonl
from minimodel.core.logging_utils import get_logger

__all__ = [
    "ascii_plot",
    "load_metrics",
    "plot_loss_curve",
    "smooth",
    "summarize_run",
]

logger = get_logger(__name__)

_BLOCKS = " ▁▂▃▄▅▆▇█"


def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Read a ``metrics.jsonl`` file, tolerating a truncated final line."""
    path = Path(path)
    if path.is_dir():
        path = path / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no metrics file at {path}")
    return list(read_jsonl(path, skip_bad_lines=True))


def _series(
    rows: Iterable[dict[str, Any]], key: str, x_key: str = "step"
) -> tuple[list[float], list[float]]:
    """Extract ``(x, y)`` pairs where both fields are present and numeric."""
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        if key not in row or x_key not in row:
            continue
        try:
            y = float(row[key])
            x = float(row[x_key])
        except (TypeError, ValueError):
            continue
        if math.isnan(y) or math.isinf(y):
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def smooth(values: Sequence[float], weight: float = 0.9) -> list[float]:
    """Exponential moving average, as used by TensorBoard's smoothing slider.

    >>> smooth([1.0, 0.0], weight=0.5)
    [1.0, 0.5]
    """
    if not values:
        return []
    weight = min(max(float(weight), 0.0), 0.999)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(out[-1] * weight + float(value) * (1.0 - weight))
    return out


def ascii_plot(
    values: Sequence[float],
    *,
    width: int = 72,
    height: int = 12,
    label: str = "loss",
) -> str:
    """Render a series as a text plot.

    Values are bucketed into ``width`` columns (averaging within a bucket) and
    drawn with block characters, with the y-range annotated on the left.
    """
    numeric = [float(v) for v in values if v == v and abs(v) != float("inf")]
    if not numeric:
        return f"{label}: no data"
    if len(numeric) == 1:
        return f"{label}: {numeric[0]:.4f} (single point)"

    buckets: list[float] = []
    per_bucket = max(1, len(numeric) / float(width))
    for i in range(min(width, len(numeric))):
        start = int(i * per_bucket)
        end = max(start + 1, int((i + 1) * per_bucket))
        chunk = numeric[start:end]
        buckets.append(sum(chunk) / len(chunk))

    low, high = min(buckets), max(buckets)
    span = high - low or 1.0

    if height <= 1:
        scaled = [
            _BLOCKS[min(len(_BLOCKS) - 1, int((v - low) / span * (len(_BLOCKS) - 1)))]
            for v in buckets
        ]
        return f"{label} [{low:.4f} .. {high:.4f}]\n" + "".join(scaled)

    grid = [[" "] * len(buckets) for _ in range(height)]
    for col, value in enumerate(buckets):
        row = height - 1 - int((value - low) / span * (height - 1))
        row = min(height - 1, max(0, row))
        grid[row][col] = "*"
        for below in range(row + 1, height):
            if grid[below][col] == " ":
                grid[below][col] = "." if below == row + 1 else " "

    lines = [f"{label}  min={low:.4f}  max={high:.4f}  n={len(numeric)}"]
    for index, row in enumerate(grid):
        if index == 0:
            axis = f"{high:9.4f} |"
        elif index == height - 1:
            axis = f"{low:9.4f} |"
        else:
            axis = " " * 9 + " |"
        lines.append(axis + "".join(row))
    lines.append(" " * 10 + "+" + "-" * len(buckets))
    return "\n".join(lines)


def plot_loss_curve(
    metrics: str | Path | Iterable[dict[str, Any]],
    output: str | Path | None = None,
    *,
    keys: Sequence[str] = ("loss", "val_loss"),
    x_key: str = "step",
    smoothing: float = 0.9,
    title: str = "Training loss",
    log_y: bool = False,
    ascii_fallback: bool = True,
) -> str | Path:
    """Plot one or more metric series.

    Returns the PNG path when matplotlib is available and ``output`` is given,
    otherwise an ASCII plot string.
    """
    rows = load_metrics(metrics) if isinstance(metrics, (str, Path)) else list(metrics)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if not ascii_fallback:
            raise
        logger.info("matplotlib not installed, rendering an ASCII plot instead")
        blocks = []
        for key in keys:
            _, ys = _series(rows, key, x_key)
            if ys:
                blocks.append(ascii_plot(smooth(ys, smoothing), label=key))
        return "\n\n".join(blocks) if blocks else "no data"

    figure, axis = plt.subplots(figsize=(9, 5), dpi=140)
    plotted = False
    for key in keys:
        xs, ys = _series(rows, key, x_key)
        if not ys:
            continue
        plotted = True
        if smoothing > 0 and len(ys) > 2:
            axis.plot(xs, ys, alpha=0.22, linewidth=1.0)
            axis.plot(xs, smooth(ys, smoothing), linewidth=1.8, label=key)
        else:
            axis.plot(xs, ys, linewidth=1.6, label=key)

    if not plotted:
        plt.close(figure)
        return "no data"

    axis.set_xlabel(x_key)
    axis.set_ylabel("loss")
    axis.set_title(title)
    if log_y:
        axis.set_yscale("log")
    axis.grid(True, alpha=0.25, linestyle="--")
    axis.legend(frameon=False)
    figure.tight_layout()

    if output is None:
        output = "loss_curve.png"
    output = Path(output)
    ensure_dir(output.parent)
    figure.savefig(output)
    plt.close(figure)
    logger.info("wrote %s", output)
    return output


def plot_learning_rate(
    metrics: str | Path | Iterable[dict[str, Any]],
    output: str | Path | None = None,
) -> str | Path:
    """Plot the learning-rate schedule that was actually applied."""
    return plot_loss_curve(
        metrics, output, keys=("lr",), title="Learning rate", smoothing=0.0
    )


def summarize_run(metrics: str | Path | Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarise a run: best/final losses, step count, throughput, duration.

    Feeds both ``minimodel bench compare`` and the model-card generator.
    """
    rows = load_metrics(metrics) if isinstance(metrics, (str, Path)) else list(metrics)
    if not rows:
        return {"steps": 0}

    summary: dict[str, Any] = {"records": len(rows)}
    steps = [int(r["step"]) for r in rows if "step" in r]
    if steps:
        summary["steps"] = max(steps)

    for key in ("loss", "val_loss", "grad_norm", "tokens_per_second"):
        _, values = _series(rows, key)
        if not values:
            continue
        summary[f"final_{key}"] = round(values[-1], 6)
        summary[f"best_{key}"] = round(min(values) if "loss" in key else max(values), 6)
        summary[f"mean_{key}"] = round(sum(values) / len(values), 6)

    if "final_loss" in summary:
        summary["final_perplexity"] = round(math.exp(min(summary["final_loss"], 20.0)), 4)
    if "best_val_loss" in summary:
        summary["best_val_perplexity"] = round(math.exp(min(summary["best_val_loss"], 20.0)), 4)

    elapsed = [float(r["elapsed_s"]) for r in rows if "elapsed_s" in r]
    if elapsed:
        summary["elapsed_seconds"] = round(max(elapsed), 1)

    total_tokens = [int(r["total_tokens"]) for r in rows if "total_tokens" in r]
    if total_tokens:
        summary["total_tokens"] = max(total_tokens)
    return summary
