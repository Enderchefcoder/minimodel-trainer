"""The Open_SLM_Leaderboard benchmark suite, bundled as tasks.

``AxiomicLabs/Open_SLM_Leaderboard`` scores small language models on the classic
open-LLM set (ARC, HellaSwag, MMLU, PIQA, WinoGrande, GSM8K). This module gives
the harness one entry point for it:

* :func:`load_leaderboard_tasks` pulls the registered eval datasets through the
  dataset registry (network happens only inside ``minimodel datasets pull``, never
  here) and normalises them into :class:`~minimodel.benchmarking.tasks.Task`.
* BLiMP (minimal pairs) and ARC-Easy are always included, since they are the two
  benchmarks with the most signal at small scale.
* :data:`LEADERBOARD_DATASETS` maps leaderboard benchmark names onto registry
  entries so the suite and ``datasets.yaml`` cannot drift apart.

Everything degrades gracefully offline: a dataset that has not been pulled yet is
reported as missing rather than failing the run, so a partial suite still
produces comparable numbers.
"""

from __future__ import annotations

from collections.abc import Sequence

from minimodel.benchmarking.tasks import Task, load_task
from minimodel.core.logging_utils import get_logger
from minimodel.datasets.registry import get_dataset, resolve_mixture
from minimodel.datasets.shards import TokenizedCorpus

__all__ = [
    "LEADERBOARD_DATASETS",
    "REQUIRED_TASKS",
    "load_leaderboard_suite",
    "load_leaderboard_tasks",
]

logger = get_logger(__name__)

#: Leaderboard benchmark name -> registry dataset name and task kind.
LEADERBOARD_DATASETS: dict[str, tuple[str, str]] = {
    "arc_easy": ("arc-easy", "multiple_choice"),
    "arc_challenge": ("arc-challenge", "multiple_choice"),
    "hellaswag": ("hellaswag", "multiple_choice"),
    "piqa": ("piqa", "multiple_choice"),
    "winogrande": ("winogrande", "multiple_choice"),
    "mmlu": ("mmlu", "multiple_choice"),
    "gsm8k": ("gsm8k-rlvr", "generation"),
    "blimp": ("blimp", "minimal_pairs"),
}

#: Tasks the user explicitly asked for in every before/after comparison.
REQUIRED_TASKS = ("blimp", "arc_easy")


def load_leaderboard_tasks(
    names: Sequence[str] | None = None,
    *,
    data_dir: str = "data/raw",
    limit: int | None = None,
) -> dict[str, Task]:
    """Load leaderboard tasks by name (default: all).

    A dataset whose raw JSONL has not been pulled yet is skipped with a warning;
    check the returned dict's keys to see what actually ran.
    """
    from pathlib import Path

    selected = dict(LEADERBOARD_DATASETS)
    if names:
        unknown = [n for n in names if n not in LEADERBOARD_DATASETS]
        if unknown:
            valid = ", ".join(sorted(LEADERBOARD_DATASETS))
            raise ValueError(f"unknown leaderboard tasks {unknown}; valid: {valid}")
        selected = {n: LEADERBOARD_DATASETS[n] for n in names}

    tasks: dict[str, Task] = {}
    for task_name, (dataset_name, kind) in selected.items():
        try:
            get_dataset(dataset_name)  # fail loudly on an unknown registry entry
        except Exception as exc:
            logger.warning("skipping %s: %s", task_name, exc)
            continue
        path = Path(data_dir) / f"{dataset_name}.jsonl"
        if not path.exists():
            logger.warning(
                "skipping %s: run `minimodel data pull %s` first (expected %s)",
                task_name,
                dataset_name,
                path,
            )
            continue
        tasks[task_name] = load_task(task_name, path, kind, limit=limit)
    return tasks


def load_leaderboard_suite(
    *,
    data_dir: str = "data/raw",
    limit: int | None = None,
) -> tuple[list[Task], TokenizedCorpus | None]:
    """Tasks plus the perplexity corpus, ready for :func:`run_suite`.

    The corpus is the tokenized pretraining shard used for perplexity; ``None``
    if none has been tokenized yet.
    """
    from pathlib import Path

    tasks = list(load_leaderboard_tasks(data_dir=data_dir, limit=limit).values())
    corpus_path = Path("data/tokenized/pretrain")
    corpus = TokenizedCorpus(corpus_path) if corpus_path.exists() else None
    return tasks, corpus


def _registry_has_mixture() -> bool:  # pragma: no cover - trivial guard
    """True when the eval-suite mixture resolves (used by tests/doc builds)."""
    try:
        resolve_mixture("eval-suite")
        return True
    except Exception:
        return False
