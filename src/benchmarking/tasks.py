"""Benchmark task definitions and loaders.

Four task shapes cover everything the harness needs to score:

``perplexity``
    Average negative log-likelihood per token over held-out text. The only
    metric that is meaningful at every model size.
``multiple_choice``
    Score each candidate continuation by length-normalised log-likelihood and
    pick the best. This is how ARC, HellaSwag, PIQA and WinoGrande are scored -
    no generation, no parsing, no prompt-format sensitivity.
``minimal_pairs``
    Two sentences differing in one grammatical feature; the model is correct if
    it assigns higher likelihood to the grammatical one. BLiMP's format, and by
    far the most informative benchmark for models under 50M parameters, which
    are at chance on knowledge tasks but well above it on syntax.
``generation``
    Free-form generation checked by a verifier function, for maths and code.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minimodel.core.logging_utils import get_logger

__all__ = [
    "BUILTIN_TASKS",
    "GenerationItem",
    "MinimalPairItem",
    "MultipleChoiceItem",
    "Task",
    "load_task",
    "normalize_multiple_choice",
]

logger = get_logger(__name__)


@dataclass
class MultipleChoiceItem:
    """One multiple-choice question."""

    context: str
    choices: list[str]
    label: int
    #: Optional shared prefix used to normalise by unconditional likelihood.
    unconditional_context: str = ""


@dataclass
class MinimalPairItem:
    """A grammatical/ungrammatical sentence pair."""

    good: str
    bad: str


@dataclass
class GenerationItem:
    """A prompt with a checkable answer."""

    prompt: str
    answer: str
    verifier: str = "numeric"


@dataclass
class Task:
    """A named benchmark plus its items."""

    name: str
    kind: str
    items: list[Any] = field(default_factory=list)
    description: str = ""
    metric: str = "accuracy"

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f"Task({self.name!r}, kind={self.kind!r}, items={len(self.items)})"


def normalize_multiple_choice(record: dict[str, Any]) -> MultipleChoiceItem | None:
    """Map a raw record onto :class:`MultipleChoiceItem`.

    Handles the ARC, HellaSwag, PIQA and WinoGrande layouts, which all express
    the same task with different field names.
    """
    # ARC / generic: {"question": ..., "choices": {"text": [...], "label": [...]}, "answerKey": "B"}
    if "choices" in record and isinstance(record["choices"], dict):
        texts = list(record["choices"].get("text", []))
        labels = [str(x) for x in record["choices"].get("label", [])]
        answer = str(record.get("answerKey", ""))
        if not texts or answer not in labels:
            return None
        return MultipleChoiceItem(
            context=str(record.get("question", "")), choices=texts, label=labels.index(answer)
        )

    # HellaSwag: {"ctx": ..., "endings": [...], "label": "2"}
    if "endings" in record:
        try:
            label = int(record.get("label", 0))
        except (TypeError, ValueError):
            return None
        return MultipleChoiceItem(
            context=str(record.get("ctx") or record.get("ctx_a", "")),
            choices=[str(e) for e in record["endings"]],
            label=label,
        )

    # PIQA: {"goal": ..., "sol1": ..., "sol2": ..., "label": 0}
    if "sol1" in record and "sol2" in record:
        return MultipleChoiceItem(
            context=str(record.get("goal", "")),
            choices=[str(record["sol1"]), str(record["sol2"])],
            label=int(record.get("label", 0)),
        )

    # WinoGrande: {"sentence": ..., "option1": ..., "option2": ..., "answer": "1"}
    if "option1" in record and "option2" in record:
        try:
            label = int(record.get("answer", 1)) - 1
        except (TypeError, ValueError):
            return None
        return MultipleChoiceItem(
            context=str(record.get("sentence", "")),
            choices=[str(record["option1"]), str(record["option2"])],
            label=label,
        )

    # Already normalised.
    if "choices" in record and isinstance(record["choices"], list):
        return MultipleChoiceItem(
            context=str(record.get("context") or record.get("question", "")),
            choices=[str(c) for c in record["choices"]],
            label=int(record.get("label", 0)),
        )
    return None


def _iter_jsonl(path: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def load_task(
    name: str,
    source: str | Path | Sequence[dict[str, Any]],
    kind: str = "multiple_choice",
    *,
    limit: int | None = None,
) -> Task:
    """Build a :class:`Task` from a JSONL file or an in-memory sequence."""
    records: Iterator[dict[str, Any]]
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"benchmark data not found: {path}")
        records = _iter_jsonl(path, limit)
    else:
        records = iter(list(source)[:limit] if limit else list(source))

    items: list[Any] = []
    skipped = 0
    for record in records:
        if kind == "multiple_choice":
            item = normalize_multiple_choice(record)
            if item is None:
                skipped += 1
                continue
            items.append(item)
        elif kind == "minimal_pairs":
            good = record.get("sentence_good") or record.get("good")
            bad = record.get("sentence_bad") or record.get("bad")
            if not good or not bad:
                skipped += 1
                continue
            items.append(MinimalPairItem(good=str(good), bad=str(bad)))
        elif kind == "generation":
            prompt = record.get("prompt") or record.get("question")
            answer = record.get("answer") or record.get("solution")
            if prompt is None or answer is None:
                skipped += 1
                continue
            items.append(
                GenerationItem(
                    prompt=str(prompt),
                    answer=str(answer),
                    verifier=str(record.get("verifier", "numeric")),
                )
            )
        elif kind == "perplexity":
            text = record.get("text") or record.get("content")
            if not text:
                skipped += 1
                continue
            items.append(str(text))
        else:
            raise ValueError(f"unknown task kind {kind!r}")

    if skipped:
        logger.debug("skipped %d unparseable records in task %s", skipped, name)
    return Task(name=name, kind=kind, items=items)


#: Tiny built-in tasks so `minimodel bench` works offline. They are sanity
#: checks, not benchmarks - a real evaluation needs the datasets in the
#: registry.
BUILTIN_TASKS: dict[str, Task] = {
    "demo-syntax": Task(
        name="demo-syntax",
        kind="minimal_pairs",
        description="Six hand-written subject/verb agreement pairs.",
        items=[
            MinimalPairItem("The cat sleeps in the window.", "The cat sleep in the window."),
            MinimalPairItem("The books are on the table.", "The books is on the table."),
            MinimalPairItem("She walks to the library.", "She walk to the library."),
            MinimalPairItem("They were waiting outside.", "They was waiting outside."),
            MinimalPairItem("The river runs east.", "The river run east."),
            MinimalPairItem("Two people can carry it.", "Two people carries it."),
        ],
    ),
    "demo-choice": Task(
        name="demo-choice",
        kind="multiple_choice",
        description="Four completion questions over the bundled corpus.",
        items=[
            MultipleChoiceItem("The river runs", ["east through the valley.", "under the oven."], 0),
            MultipleChoiceItem("Bread needs flour, water,", ["salt and time.", "iron and rust."], 0),
            MultipleChoiceItem("Iron rusts because", ["it prefers to be an oxide.", "it is asleep."], 0),
            MultipleChoiceItem("The train leaves at seven and", ["arrives at ten.", "eats the bridge."], 0),
        ],
    ),
    "demo-math": Task(
        name="demo-math",
        kind="generation",
        description="Eight one-step arithmetic problems.",
        items=[
            GenerationItem("What is 5 plus 8?", "13"),
            GenerationItem("What is 12 plus 30?", "42"),
            GenerationItem("What is 20 minus 7?", "13"),
            GenerationItem("What is 6 times 7?", "42"),
            GenerationItem("What is 12 divided by 4?", "3"),
            GenerationItem("What is 9 plus 9?", "18"),
            GenerationItem("What is 100 minus 58?", "42"),
            GenerationItem("What is 3 times 5?", "15"),
        ],
    ),
}
