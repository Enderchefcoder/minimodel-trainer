"""Model-agnostic evaluation harness for the SLM research program.

One code path scores Glint-2 and our own models identically. Metrics:

* BLiMP  - minimal-pair accuracy (sum log-prob of good > bad), macro-averaged
           over the 67 paradigms. This is the standard BLiMP protocol.
* ARC-Easy - multiple choice, both `acc` (raw continuation log-prob) and
           `acc_norm` (per-character normalised), lm-eval-harness style.
* WikiText-2 - token perplexity (tokenizer-dependent, to check against Glint-2's
           reported 3.09) AND bits-per-byte (tokenizer-INDEPENDENT, the only
           fair cross-model perplexity metric).

Batched right-padded scoring is exact for causal models: position t never
attends to t+1, so right padding cannot change the log-prob of real tokens.

A `ModelAdapter` wraps any model with `encode(text)->ids` and
`forward(tokens_2d)->logits_3d`.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

EVAL_DIR = Path("research/data/eval")


@dataclass
class ModelAdapter:
    """Uniform interface over Glint-2 and our own models."""

    name: str
    encode: Callable[[str], list[int]]
    forward: Callable[[torch.Tensor], torch.Tensor]
    #: Bytes per decoded string, for bits-per-byte. Defaults to UTF-8 length.
    n_bytes: Callable[[list[int]], int] | None = None
    max_len: int = 512
    batch_size: int = 16
    params: int = 0

    @torch.no_grad()
    def score_sequences(self, sequences: Sequence[list[int]]) -> list[tuple[float, int]]:
        """Return (sum log-prob, n_scored_tokens) for each token sequence.

        Log-prob is over positions 1..T-1 (position t predicts token t+1).
        """
        out: list[tuple[float, int]] = []
        for start in range(0, len(sequences), self.batch_size):
            batch = sequences[start : start + self.batch_size]
            width = max(len(s) for s in batch)
            padded = torch.zeros(len(batch), width, dtype=torch.long)
            for i, seq in enumerate(batch):
                padded[i, : len(seq)] = torch.tensor(seq[: self.max_len], dtype=torch.long)
            logits = self.forward(padded).float()
            logp = torch.log_softmax(logits, dim=-1)
            for i, seq in enumerate(batch):
                length = min(len(seq), self.max_len)
                if length < 2:
                    out.append((0.0, 0))
                    continue
                targets = torch.tensor(seq[1:length])
                gathered = logp[i, : length - 1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                out.append((float(gathered.sum()), length - 1))
        return out


@dataclass
class EvalResult:
    model: str
    params: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "params": self.params, **self.metrics, "timing": self.timing}


def _load_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (EVAL_DIR / f"{name}.jsonl").read_text().splitlines() if line]


def eval_blimp(adapter: ModelAdapter, *, per_paradigm: int | None = None) -> dict[str, Any]:
    """Macro-averaged BLiMP accuracy. `per_paradigm` subsamples for speed."""
    rows = _load_jsonl("blimp")
    by_paradigm: dict[str, list[dict]] = {}
    for row in rows:
        by_paradigm.setdefault(row["paradigm"], []).append(row)

    paradigm_acc: dict[str, float] = {}
    for paradigm, items in by_paradigm.items():
        if per_paradigm is not None:
            items = items[:per_paradigm]
        good = adapter.score_sequences([adapter.encode(x["good"]) for x in items])
        bad = adapter.score_sequences([adapter.encode(x["bad"]) for x in items])
        correct = sum(1 for (g, _), (b, _) in zip(good, bad, strict=True) if g > b)
        paradigm_acc[paradigm] = correct / len(items)

    macro = sum(paradigm_acc.values()) / len(paradigm_acc)
    return {
        "blimp_acc": round(100 * macro, 2),
        "blimp_n_paradigms": len(paradigm_acc),
        "blimp_per_paradigm": {k: round(100 * v, 1) for k, v in sorted(paradigm_acc.items())},
    }


def eval_arc_easy(adapter: ModelAdapter, *, limit: int | None = None) -> dict[str, Any]:
    """ARC-Easy acc and acc_norm, lm-eval-harness prompt format."""
    rows = _load_jsonl("arc_easy")
    if limit:
        rows = rows[:limit]
    correct_raw = correct_norm = usable = 0
    for row in rows:
        labels = row["labels"]
        if row["answerKey"] not in labels:
            continue
        usable += 1
        gold = labels.index(row["answerKey"])
        prompt = f"Question: {row['question']}\nAnswer:"
        raw_scores, norm_scores = [], []
        for choice in row["choices"]:
            continuation = f" {choice}"
            prompt_ids = adapter.encode(prompt)
            full_ids = adapter.encode(prompt + continuation)
            (total, _), = adapter.score_sequences([full_ids])
            # Continuation-only log-prob: subtract the prompt's own log-prob.
            (prompt_total, _), = adapter.score_sequences([prompt_ids])
            cont_lp = total - prompt_total
            raw_scores.append(cont_lp)
            norm_scores.append(cont_lp / max(1, len(continuation)))
        if int(max(range(len(raw_scores)), key=raw_scores.__getitem__)) == gold:
            correct_raw += 1
        if int(max(range(len(norm_scores)), key=norm_scores.__getitem__)) == gold:
            correct_norm += 1
    return {
        "arc_easy_acc": round(100 * correct_raw / usable, 2),
        "arc_easy_acc_norm": round(100 * correct_norm / usable, 2),
        "arc_easy_n": usable,
    }


def eval_wikitext(
    adapter: ModelAdapter, *, stride: int | None = None, max_tokens: int | None = None
) -> dict[str, Any]:
    """WikiText-2 token perplexity + bits-per-byte over the model's context.

    Reports three numbers:
      * ``wikitext_ppl`` - token perplexity (tokenizer-dependent; only comparable
        between models sharing a tokenizer).
      * ``wikitext_bits_per_byte`` - NLL in bits over the exact UTF-8 byte count
        of the scored span (tokenizer-INDEPENDENT).
      * ``wikitext_byte_ppl`` = 2 ** bits_per_byte - the byte-normalised
        perplexity. Glint-2's "wikitext-2 ppl" leaderboard number is this metric
        (validated: matches our measurement), so it is the fair head-to-head.
    """
    text = (EVAL_DIR / "wikitext2_test.txt").read_text(encoding="utf-8")
    if max_tokens is None and adapter.params and adapter.params < 5_000_000:
        max_tokens = 60_000  # keep CPU eval of tiny models bounded during dev
    ids = adapter.encode(text)
    if max_tokens:
        ids = ids[:max_tokens]
    if adapter.n_bytes is not None:
        n_bytes = adapter.n_bytes(ids)
    else:
        n_bytes = len(text.encode("utf-8"))
        if max_tokens:
            n_bytes = int(n_bytes * len(ids) / max(1, len(adapter.encode(text))))

    window = adapter.max_len
    stride = stride or window
    total_nll = 0.0
    total_tokens = 0
    position = 0
    while position < len(ids) - 1:
        chunk = ids[position : position + window]
        if len(chunk) < 2:
            break
        (nll_sum, n) = _chunk_nll(adapter, chunk, skip=0 if position == 0 else window - stride)
        total_nll += nll_sum
        total_tokens += n
        position += stride
    mean_nll = total_nll / max(1, total_tokens)
    bits_per_byte = (total_nll / math.log(2)) / max(1, n_bytes)
    return {
        "wikitext_ppl": round(math.exp(min(mean_nll, 20)), 3),
        "wikitext_bits_per_byte": round(bits_per_byte, 4),
        "wikitext_byte_ppl": round(2.0**bits_per_byte, 3),
        "wikitext_n_tokens": total_tokens,
        "wikitext_n_bytes": n_bytes,
    }


@torch.no_grad()
def _chunk_nll(adapter: ModelAdapter, chunk: list[int], skip: int) -> tuple[float, int]:
    """Summed NLL (nats) for a chunk, skipping the first `skip` predictions."""
    tokens = torch.tensor([chunk], dtype=torch.long)
    logits = adapter.forward(tokens).float()
    logp = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = torch.tensor(chunk[1:])
    token_nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    token_nll = token_nll[skip:]
    return float(token_nll.sum()), token_nll.numel()


def run_all(
    adapter: ModelAdapter,
    *,
    blimp_per_paradigm: int | None = None,
    arc_limit: int | None = None,
    wikitext_max_tokens: int | None = None,
) -> EvalResult:
    """Run BLiMP, ARC-Easy and WikiText, returning one result object."""
    result = EvalResult(model=adapter.name, params=adapter.params)
    t0 = time.perf_counter()
    result.metrics.update(eval_blimp(adapter, per_paradigm=blimp_per_paradigm))
    result.timing["blimp_s"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    result.metrics.update(eval_arc_easy(adapter, limit=arc_limit))
    result.timing["arc_s"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    result.metrics.update(eval_wikitext(adapter, max_tokens=wikitext_max_tokens))
    result.timing["wikitext_s"] = round(time.perf_counter() - t0, 1)
    return result
