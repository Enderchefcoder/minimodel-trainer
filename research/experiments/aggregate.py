"""Aggregate experiment result JSONs into Markdown tables and plots.

Reads research/data/results/*.json, groups by the experiment name prefix, and
writes a Markdown summary plus (if matplotlib is present) bar/line charts under
research/reports/figures/. The Glint-2 baseline is always included as the line
to beat.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULTS = Path("research/data/results")
FIGURES = Path("research/reports/figures")

METRICS = ["params", "val_loss", "wikitext_byte_ppl", "blimp_acc", "arc_easy_acc", "arc_easy_acc_norm"]
LOWER_BETTER = {"val_loss", "wikitext_byte_ppl", "wikitext_ppl", "final_loss"}


def load_results() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(RESULTS.glob("*.json")):
        if path.name in {"loops_scaling.json", "loop_robustness.json"}:
            continue  # nested structures, handled separately
        data = json.loads(path.read_text())
        data.setdefault("name", data.get("model", path.stem))
        out[data["name"]] = data
    return out


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], *, highlight: bool = True) -> str:
    if not rows:
        return "_no results_"
    cells = [[_fmt(r.get(c, "")) for c in columns] for r in rows]
    if highlight:
        for j, col in enumerate(columns):
            vals = [(i, r.get(col)) for i, r in enumerate(rows) if isinstance(r.get(col), (int, float))]
            if len(vals) < 2 or col in {"params"}:
                continue
            best = (min if col in LOWER_BETTER else max)(vals, key=lambda t: t[1])[0]
            cells[best][j] = f"**{cells[best][j]}**"
    header = "| " + " | ".join(columns) + " |"
    div = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in cells]
    return "\n".join([header, div, *body])


def group_by_prefix(results: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for name, data in results.items():
        prefix = name.split("_")[0]
        groups.setdefault(prefix, []).append(data)
    return groups


def bar_chart(rows: list[dict[str, Any]], metric: str, path: Path, *, baseline: float | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    usable = [(r["name"], r[metric]) for r in rows if isinstance(r.get(metric), (int, float))]
    if not usable:
        return
    names, vals = zip(*usable, strict=True)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(names)), 4), dpi=140)
    ax.bar(range(len(names)), vals, color="#4C72B0")
    if baseline is not None:
        ax.axhline(baseline, color="crimson", linestyle="--", linewidth=1.4, label="glint-2")
        ax.legend(frameon=False)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    results = load_results()
    if not results:
        print("no results yet")
        return
    baseline = results.get("glint-2 (loops=8)") or _load_baseline()
    groups = group_by_prefix(results)

    lines = ["# Experiment results (auto-aggregated)\n"]
    lines.append(f"_Generated from {len(results)} result files. Glint-2 baseline included where relevant._\n")
    if baseline:
        lines.append("## Baseline\n")
        lines.append(markdown_table([_flatten_baseline(baseline)],
                                    ["name", "params", "blimp_acc", "arc_easy_acc",
                                     "wikitext_byte_ppl"], highlight=False))
        lines.append("")

    for prefix, rows in sorted(groups.items()):
        if prefix == "glint-2":
            continue
        rows = sorted(rows, key=lambda r: r["name"])
        lines.append(f"## {prefix}\n")
        cols = ["name", "params", "val_loss", "wikitext_byte_ppl", "blimp_acc",
                "arc_easy_acc", "arc_easy_acc_norm", "tokens_per_second"]
        cols = [c for c in cols if any(c in r for r in rows)]
        lines.append(markdown_table(rows, cols))
        lines.append("")
        if baseline:
            bl = _flatten_baseline(baseline)
            for metric in ("wikitext_byte_ppl", "blimp_acc", "val_loss"):
                bar_chart(rows, metric, FIGURES / f"{prefix}_{metric}.png",
                          baseline=bl.get(metric) if metric != "val_loss" else None)

    out = Path("research/reports/RESULTS.md")
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(results)} results, {len(groups)} groups)")


def _load_baseline() -> dict[str, Any] | None:
    p = RESULTS / "glint2.json"
    return json.loads(p.read_text()) if p.exists() else None


def _flatten_baseline(b: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "glint-2", "params": b.get("params"), "blimp_acc": b.get("blimp_acc"),
        "arc_easy_acc": b.get("arc_easy_acc"), "arc_easy_acc_norm": b.get("arc_easy_acc_norm"),
        "wikitext_byte_ppl": b.get("wikitext_byte_ppl"), "val_loss": None,
    }


if __name__ == "__main__":
    main()
