"""Soft-label next-token triples for the crush-glint2 QA corpus.

The bundled ``research/data/corpus/slm_next_token_dataset.json`` stores, for
each prompt, a gold continuation and a 3-way soft distribution over candidate
next tokens at every step. This module turns that schema into:

1. plain ``{text}`` documents for ordinary CE pretraining (prompt+completion);
2. per-step soft targets that can be trained with KL / CE against the model's
   logits after re-encoding each candidate with *our* tokenizer.

Probabilities in the file are heuristic teaching targets, not corpus
frequencies — re-encode every token; if a candidate splits into multiple
subwords, mass is assigned to the first sub-token (see the JSON ``caveat``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "DEFAULT_CORPUS_PATH",
    "SoftStep",
    "entries_to_plain_texts",
    "iter_plain_texts",
    "load_soft_label_dataset",
    "soft_kl_loss",
    "write_plain_jsonl",
]

#: Canonical location of the soft-label JSON inside this repository.
DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "data"
    / "corpus"
    / "slm_next_token_dataset.json"
)


def load_soft_label_dataset(path: str | Path | None = None) -> dict[str, Any]:
    """Load the soft-label JSON; defaults to the bundled corpus."""
    target = Path(path) if path is not None else DEFAULT_CORPUS_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"soft-label corpus not found at {target}; expected the bundled "
            "slm_next_token_dataset.json under research/data/corpus/"
        )
    return json.loads(target.read_text(encoding="utf-8"))


def entries_to_plain_texts(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten entries into ``{id, category, text}`` CE documents.

    ``text`` is ``prompt + completion``. Prefer the provided ``completion_text``;
    otherwise concatenate each step's ``chosen`` token.
    """
    docs: list[dict[str, Any]] = []
    for entry in payload.get("entries", []):
        prompt = str(entry.get("prompt", ""))
        completion = entry.get("completion_text")
        if not completion:
            completion = "".join(str(step.get("chosen", "")) for step in entry.get("steps", []))
        docs.append(
            {
                "id": entry.get("id"),
                "category": entry.get("category"),
                "text": f"{prompt}{completion}",
            }
        )
    return docs


def iter_plain_texts(path: str | Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield plain CE documents from the soft-label corpus."""
    yield from entries_to_plain_texts(load_soft_label_dataset(path))


def write_plain_jsonl(
    out_path: str | Path,
    path: str | Path | None = None,
) -> int:
    """Write plain CE documents as JSONL; returns the number of rows."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    docs = entries_to_plain_texts(load_soft_label_dataset(path))
    with out.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return len(docs)


class SoftStep:
    """One soft-label prediction step after tokenizer alignment."""

    __slots__ = ("context_ids", "determinism", "probs", "token_ids")

    def __init__(
        self,
        context_ids: list[int],
        token_ids: list[int],
        probs: list[float],
        determinism: str = "medium",
    ) -> None:
        self.context_ids = context_ids
        self.token_ids = token_ids
        self.probs = probs
        self.determinism = determinism


def align_steps_to_tokenizer(
    entry: Mapping[str, Any],
    encode,
    *,
    min_prob: float = 1e-4,
) -> list[SoftStep]:
    """Re-encode an entry's soft steps with ``encode(str) -> list[int]``.

    Candidates that tokenize to multiple ids keep only the first id and keep
    their probability mass (JSON caveat). Context is ``prompt + chosen so far``.
    """
    prompt = str(entry.get("prompt", ""))
    chosen_so_far = ""
    aligned: list[SoftStep] = []
    for step in entry.get("steps", []):
        context = prompt + chosen_so_far
        context_ids = list(encode(context))
        token_ids: list[int] = []
        probs: list[float] = []
        for cand in step.get("candidates", []):
            piece = str(cand.get("token", ""))
            ids = list(encode(piece))
            if not ids:
                continue
            token_ids.append(ids[0])
            probs.append(max(float(cand.get("p", 0.0)), min_prob))
        if token_ids:
            total = sum(probs)
            probs = [p / total for p in probs]
            aligned.append(
                SoftStep(
                    context_ids=context_ids,
                    token_ids=token_ids,
                    probs=probs,
                    determinism=str(step.get("determinism", "medium")),
                )
            )
        chosen_so_far += str(step.get("chosen", ""))
    return aligned


def soft_kl_loss(
    logits: Tensor,
    token_ids: Sequence[int],
    probs: Sequence[float],
    *,
    temperature: float = 1.0,
) -> Tensor:
    """KL(teacher || student) over a sparse 3-way soft target on one position.

    ``logits`` is a 1-D vector over the vocabulary (the next-token distribution
    at the last context position). Teacher mass outside ``token_ids`` is zero.
    """
    if logits.ndim != 1:
        raise ValueError(f"logits must be 1-D vocab vector, got shape {tuple(logits.shape)}")
    if len(token_ids) != len(probs) or not token_ids:
        raise ValueError("token_ids and probs must be non-empty and aligned")
    teacher = torch.zeros_like(logits)
    idx = torch.tensor(list(token_ids), device=logits.device, dtype=torch.long)
    vals = torch.tensor(list(probs), device=logits.device, dtype=logits.dtype)
    teacher.index_add_(0, idx, vals)
    teacher = teacher / teacher.sum().clamp_min(1e-8)
    log_student = F.log_softmax(logits / temperature, dim=-1)
    return F.kl_div(log_student, teacher, reduction="sum")
