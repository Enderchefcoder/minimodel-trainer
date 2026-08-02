"""Automatic model card generation.

A model card written by hand at the end of a project is written from memory and
is wrong. This module builds one from artifacts the run already produced: the
architecture config, ``run_metadata.json``, ``metrics.jsonl``, benchmark JSON
and the dataset registry entries that were actually used.

Anything that cannot be derived is emitted as an explicit ``[TODO: ...]`` marker
rather than being quietly omitted, so the gaps are visible in review.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minimodel.checkpointing.loss_visualization import summarize_run
from minimodel.core.io_utils import ensure_dir, human_bytes, human_count, human_duration, read_json
from minimodel.core.logging_utils import get_logger

__all__ = [
    "TEMPLATE_DIR",
    "ModelCard",
    "ModelCardData",
    "collect_card_data",
    "generate_model_card",
    "render_card",
]

logger = get_logger(__name__)

#: Directory holding the bundled card templates.
TEMPLATE_DIR = Path(__file__).parent / "templates"

TODO = "[TODO: {}]"


@dataclass
class ModelCardData:
    """Everything the card renderer needs, gathered from a run."""

    name: str = "model"
    organization: str = ""
    description: str = ""
    license: str = "apache-2.0"
    architecture: str = ""
    parameters: int = 0
    active_parameters: int | None = None
    vocab_size: int = 0
    max_seq_len: int = 0
    model_config: dict[str, Any] = field(default_factory=dict)

    training: dict[str, Any] = field(default_factory=dict)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    benchmarks: dict[str, dict[str, Any]] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    throughput: dict[str, Any] = field(default_factory=dict)
    languages: list[str] = field(default_factory=lambda: ["en"])
    tags: list[str] = field(default_factory=list)
    base_model: str = ""
    stage: str = "base"

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view."""
        from dataclasses import asdict

        return asdict(self)


def _read_optional_json(path: Path) -> dict[str, Any]:
    """Read a JSON file, returning ``{}`` when absent or unparseable."""
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return {}


def collect_card_data(
    run_dir: str | Path | None = None,
    *,
    model_dir: str | Path | None = None,
    benchmark_paths: Sequence[str | Path] = (),
    name: str = "",
    organization: str = "",
    license: str = "apache-2.0",
    datasets: Sequence[Mapping[str, Any]] = (),
    stage: str = "base",
    base_model: str = "",
) -> ModelCardData:
    """Gather card data from a run directory and/or an exported model."""
    data = ModelCardData(
        name=name or "model",
        organization=organization,
        license=license,
        stage=stage,
        base_model=base_model,
        datasets=[dict(d) for d in datasets],
    )

    if model_dir is not None:
        config = _read_optional_json(Path(model_dir) / "config.json")
        if config:
            data.architecture = str(config.get("architecture", ""))
            data.vocab_size = int(config.get("vocab_size", 0) or 0)
            data.max_seq_len = int(config.get("max_seq_len", 0) or 0)
            data.model_config = {k: v for k, v in config.items() if k != "architecture"}
        weights = Path(model_dir) / "model.pt"
        if weights.exists():
            data.metrics["weights_size"] = human_bytes(weights.stat().st_size)

    if run_dir is not None:
        run_path = Path(run_dir)
        metadata = _read_optional_json(run_path / "run_metadata.json")
        if metadata:
            model_info = metadata.get("model") or {}
            data.architecture = data.architecture or str(model_info.get("architecture", ""))
            data.parameters = int(model_info.get("parameters", 0) or 0)
            if not data.model_config:
                data.model_config = dict(model_info.get("config") or {})
                data.vocab_size = int(data.model_config.get("vocab_size", 0) or 0)
                data.max_seq_len = int(data.model_config.get("max_seq_len", 0) or 0)
            data.training = dict(metadata.get("config") or {})
            data.hardware = dict(metadata.get("device") or {})
            data.training["dtype"] = metadata.get("dtype")
            data.training["planned_tokens"] = metadata.get("planned_tokens")
            data.training["world_size"] = (metadata.get("distributed") or {}).get("world_size", 1)

        metrics_path = run_path / "metrics.jsonl"
        if metrics_path.exists():
            data.metrics.update(summarize_run(metrics_path))
        result = _read_optional_json(run_path / "result.json")
        if result:
            data.metrics.setdefault("final_loss", result.get("final_loss"))
            data.metrics.setdefault("total_tokens", result.get("total_tokens"))
            data.metrics.setdefault("elapsed_seconds", result.get("elapsed_seconds"))

    for path in benchmark_paths:
        payload = _read_optional_json(Path(path))
        if not payload:
            continue
        data.parameters = data.parameters or int(payload.get("parameters", 0) or 0)
        for task, metrics in (payload.get("tasks") or {}).items():
            data.benchmarks[task] = dict(metrics)
        if payload.get("throughput"):
            data.throughput = dict(payload["throughput"])

    if data.architecture:
        data.tags = sorted({"minimodel", data.architecture, stage, *data.tags})
    return data


def _benchmark_table(benchmarks: Mapping[str, Mapping[str, Any]], model_name: str) -> str:
    """Render the benchmark section as a Markdown table."""
    if not benchmarks:
        return (
            "| Benchmark | Metric | Result |\n| --- | --- | --- |\n"
            f"| {TODO.format('benchmark name')} | | |\n"
        )
    rows = ["| Benchmark | Metric | " + (model_name or "Model") + " |", "| --- | --- | --- |"]
    for task, metrics in benchmarks.items():
        for key in ("accuracy_norm", "accuracy", "solve_rate", "perplexity", "loss"):
            if key in metrics:
                value = float(metrics[key])
                rendered = f"{value:.4f}" if key != "perplexity" else f"{value:.2f}"
                extra = f" (n={metrics['n']})" if "n" in metrics else ""
                rows.append(f"| **{task}** | {key} | {rendered}{extra} |")
                break
    return "\n".join(rows)


def _dataset_table(datasets: Sequence[Mapping[str, Any]]) -> str:
    """Render the training-data section as a Markdown table."""
    if not datasets:
        return f"| {TODO.format('dataset')} | | | |\n"
    rows = ["| Dataset | Stage | Tokens | License |", "| --- | --- | --- | --- |"]
    for dataset in datasets:
        rows.append(
            f"| {dataset.get('name', '?')} | {dataset.get('stage', '?')} | "
            f"{dataset.get('tokens', '?')} | {dataset.get('license', '?')} |"
        )
    return "\n".join(rows)


class ModelCard:
    """Renders :class:`ModelCardData` into Markdown."""

    def __init__(self, data: ModelCardData):
        self.data = data

    def frontmatter(self) -> str:
        """Hugging Face model-card YAML header."""
        data = self.data
        lines = ["---", f"license: {data.license}", "library_name: minimodel-trainer"]
        if data.languages:
            lines.append("language:")
            lines.extend(f"  - {lang}" for lang in data.languages)
        if data.tags:
            lines.append("tags:")
            lines.extend(f"  - {tag}" for tag in data.tags)
        if data.datasets:
            lines.append("datasets:")
            for dataset in data.datasets:
                repo = dataset.get("repo") or dataset.get("name")
                if repo:
                    lines.append(f"  - {repo}")
        if data.base_model:
            lines.append(f"base_model: {data.base_model}")
        lines.append("---")
        return "\n".join(lines)

    def render(self) -> str:
        """Render the full card."""
        data = self.data
        name = data.name
        org = data.organization or TODO.format("organization")
        params = human_count(data.parameters) if data.parameters else TODO.format("parameters")
        tokens = data.metrics.get("total_tokens")
        token_text = human_count(tokens) if tokens else TODO.format("training tokens")

        sections: list[str] = [self.frontmatter(), ""]
        sections.append(f"# {name}\n")

        description = data.description or (
            f"{name} is a compact {params}-parameter "
            f"{'instruct' if data.stage != 'base' else 'base'} language model built by {org}. "
            f"It was trained {'from scratch ' if data.stage == 'base' else ''}"
            f"on {token_text} tokens."
        )
        sections.append(description + "\n")

        sections.append("## Model details\n")
        detail_rows = [
            ("Architecture", data.architecture or TODO.format("architecture")),
            ("Parameters", f"{data.parameters:,}" if data.parameters else TODO.format("count")),
        ]
        if data.active_parameters:
            detail_rows.append(("Active parameters", f"{data.active_parameters:,}"))
        detail_rows.extend(
            [
                ("Vocabulary", f"{data.vocab_size:,}" if data.vocab_size else TODO.format("vocab")),
                ("Context length", f"{data.max_seq_len:,}" if data.max_seq_len else TODO.format("context")),
                ("Training stage", data.stage),
            ]
        )
        if data.base_model:
            detail_rows.append(("Base model", data.base_model))
        sections.append("| | |\n| --- | --- |")
        sections.extend(f"| **{label}** | {value} |" for label, value in detail_rows)
        sections.append("")

        sections.append("## Benchmarks\n")
        sections.append(
            "Evaluated with `minimodel bench`. Scores are log-likelihood based unless noted.\n"
        )
        sections.append(_benchmark_table(data.benchmarks, name))
        sections.append("")

        sections.append("## Usage\n")
        sections.append("```python")
        sections.append("from minimodel.inference import load_for_inference, complete")
        sections.append("")
        sections.append(f'model = load_for_inference("{org}/{name}")')
        sections.append('print(complete(model, "The river runs", max_new_tokens=64))')
        sections.append("```\n")
        sections.append("```bash")
        sections.append(f"minimodel chat --model {org}/{name}")
        sections.append("```\n")

        sections.append("## Training\n")
        sections.append("### Data\n")
        sections.append(_dataset_table(data.datasets))
        sections.append("")

        sections.append("### Hardware\n")
        hardware_name = data.hardware.get("name") or TODO.format("GPU model")
        device_count = data.hardware.get("device_count", 1)
        elapsed = data.metrics.get("elapsed_seconds")
        sections.append(f"- Compute: {device_count}x {hardware_name}")
        sections.append(f"- Tokens: {token_text}")
        sections.append(
            f"- Wall-clock: {human_duration(elapsed) if elapsed else TODO.format('duration')}"
        )
        if data.throughput.get("decode_tokens_per_second"):
            sections.append(
                f"- Decode throughput: "
                f"{human_count(data.throughput['decode_tokens_per_second'])} tok/s"
            )
        sections.append("")

        sections.append("### Hyperparameters\n")
        training = data.training
        hyper_rows = [
            ("Optimizer", training.get("optimizer", TODO.format("optimizer"))),
            ("Learning rate", training.get("lr", TODO.format("lr"))),
            ("Schedule", training.get("lr_schedule", TODO.format("schedule"))),
            ("Warmup", training.get("warmup", "-")),
            ("Weight decay", training.get("weight_decay", "-")),
            ("Gradient clip", training.get("grad_clip", "-")),
            ("Batch size", training.get("batch_size", TODO.format("batch size"))),
            ("Sequence length", training.get("seq_len", TODO.format("seq len"))),
            ("Gradient accumulation", training.get("grad_accum_steps", 1)),
            ("Steps", data.metrics.get("steps", training.get("max_steps", "-"))),
            ("Precision", training.get("dtype", TODO.format("precision"))),
            ("Seed", training.get("seed", "-")),
        ]
        sections.append("| Hyperparameter | Value |\n| --- | --- |")
        sections.extend(f"| {label} | {value} |" for label, value in hyper_rows)
        sections.append("")

        if data.metrics:
            sections.append("### Results\n")
            for key in ("final_loss", "best_loss", "best_val_loss", "final_perplexity"):
                if data.metrics.get(key) is not None:
                    sections.append(f"- {key.replace('_', ' ')}: {data.metrics[key]}")
            sections.append("")

        sections.append("## Limitations\n")
        sections.append(
            "- Small models hallucinate confidently. Treat every factual claim as unverified."
        )
        sections.append(
            f"- Context is limited to {data.max_seq_len or TODO.format('N')} tokens; "
            "longer inputs are truncated."
        )
        sections.append(
            f"- Trained primarily on {', '.join(data.languages)}; other languages are unsupported."
        )
        sections.append(
            "- No safety tuning was applied unless stated above. Filter outputs before "
            "user-facing deployment."
        )
        sections.append("")

        sections.append("## Reproducing\n")
        sections.append("```bash")
        sections.append("minimodel train --config configs/pretrain/your_recipe.yaml")
        sections.append("minimodel posttrain --config configs/sft/your_recipe.yaml")
        sections.append("minimodel bench --model runs/your_run/model")
        sections.append("```\n")

        stamp = datetime.datetime.now().strftime("%Y-%m-%d")
        sections.append(f"---\n\n_Card generated by minimodel-trainer on {stamp}._")
        return "\n".join(sections)


def render_card(data: ModelCardData) -> str:
    """Render a card from already-collected data."""
    return ModelCard(data).render()


def generate_model_card(
    output: str | Path,
    *,
    run_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    benchmark_paths: Sequence[str | Path] = (),
    name: str = "",
    organization: str = "",
    license: str = "apache-2.0",
    datasets: Sequence[Mapping[str, Any]] = (),
    stage: str = "base",
    base_model: str = "",
    description: str = "",
) -> Path:
    """Collect data from a run and write ``README.md``.

    Returns the path that was written.
    """
    data = collect_card_data(
        run_dir=run_dir,
        model_dir=model_dir,
        benchmark_paths=benchmark_paths,
        name=name,
        organization=organization,
        license=license,
        datasets=datasets,
        stage=stage,
        base_model=base_model,
    )
    if description:
        data.description = description

    output = Path(output)
    if output.is_dir() or not output.suffix:
        output = Path(output) / "README.md"
    ensure_dir(output.parent)
    output.write_text(render_card(data), encoding="utf-8")
    logger.info("wrote model card to %s", output)
    return output
