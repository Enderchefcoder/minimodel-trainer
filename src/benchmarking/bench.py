"""The evaluation harness.

Everything is scored by log-likelihood where possible, and only by generation
where the task genuinely requires it. That choice matters a lot at small scale:
a 20M-parameter model may know that "the cat sleeps" is more likely than "the
cat sleep" while being completely unable to answer the same question when it is
posed as an instruction. Likelihood scoring measures what the model learned;
generation scoring measures what it learned *plus* whether it can follow a
prompt format, which for a base model is a different question.

Two normalisations are reported for multiple choice:

``accuracy``
    Raw sum of log-probabilities. Biased toward short options.
``accuracy_norm``
    Divided by the number of characters, which is the standard correction and
    the number usually quoted in papers.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from minimodel.benchmarking.tasks import (
    BUILTIN_TASKS,
    GenerationItem,
    MinimalPairItem,
    MultipleChoiceItem,
    Task,
)
from minimodel.core.devices import describe_device, device_memory_stats, resolve_device
from minimodel.core.io_utils import human_count, write_json
from minimodel.core.logging_utils import get_logger
from minimodel.datasets.shards import TokenizedCorpus

__all__ = [
    "BenchmarkResult",
    "evaluate_generation",
    "evaluate_minimal_pairs",
    "evaluate_multiple_choice",
    "evaluate_perplexity",
    "measure_throughput",
    "run_suite",
    "sequence_logprob",
]

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Everything one evaluation run produced."""

    model: str = ""
    parameters: int = 0
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    throughput: dict[str, Any] = field(default_factory=dict)
    device: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view."""
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        """Write the result as JSON."""
        return write_json(path, self.to_dict())

    def headline(self) -> dict[str, float]:
        """The primary metric of each task, for tables."""
        out: dict[str, float] = {}
        for name, metrics in self.tasks.items():
            for key in ("accuracy_norm", "accuracy", "perplexity", "solve_rate"):
                if key in metrics:
                    out[name] = float(metrics[key])
                    break
        return out

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={v:.4g}" for k, v in self.headline().items())
        return f"BenchmarkResult({self.model!r}, {parts})"


@torch.no_grad()
def sequence_logprob(
    model: nn.Module,
    tokenizer: Any,
    prefix: str,
    continuation: str,
    *,
    device: torch.device | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[float, int]:
    """Log-probability of ``continuation`` given ``prefix``.

    Returns ``(total_logprob, n_continuation_tokens)``. Only the continuation
    tokens are scored, so the value is comparable across candidates that share
    a prefix.
    """
    device = device or next(model.parameters()).device
    prefix_ids = tokenizer.encode(prefix, add_bos=True, allow_special=False)
    continuation_ids = tokenizer.encode(continuation, allow_special=False)
    if not continuation_ids:
        return 0.0, 0

    tokens = torch.tensor([prefix_ids + continuation_ids], dtype=torch.long, device=device)
    logits = model(tokens, **(model_kwargs or {}))
    logprobs = torch.log_softmax(logits[0].float(), dim=-1)

    total = 0.0
    # Position i predicts token i+1, so the first continuation token is
    # predicted from the last prefix position.
    for offset, token_id in enumerate(continuation_ids):
        position = len(prefix_ids) + offset - 1
        total += float(logprobs[position, token_id])
    return total, len(continuation_ids)


@torch.no_grad()
def evaluate_multiple_choice(
    model: nn.Module,
    tokenizer: Any,
    task: Task,
    *,
    device: torch.device | None = None,
    limit: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a multiple-choice task by continuation likelihood."""
    device = device or next(model.parameters()).device
    items: Sequence[MultipleChoiceItem] = task.items[:limit] if limit else task.items
    if not items:
        return {"accuracy": 0.0, "accuracy_norm": 0.0, "n": 0}

    correct = 0
    correct_norm = 0
    was_training = model.training
    model.eval()

    for item in items:
        raw_scores: list[float] = []
        norm_scores: list[float] = []
        for choice in item.choices:
            total, _ = sequence_logprob(
                model,
                tokenizer,
                item.context,
                " " + choice.strip() if not choice.startswith(" ") else choice,
                device=device,
                model_kwargs=model_kwargs,
            )
            raw_scores.append(total)
            norm_scores.append(total / max(1, len(choice)))
        if int(max(range(len(raw_scores)), key=raw_scores.__getitem__)) == item.label:
            correct += 1
        if int(max(range(len(norm_scores)), key=norm_scores.__getitem__)) == item.label:
            correct_norm += 1

    model.train(was_training)
    n = len(items)
    return {
        "accuracy": correct / n,
        "accuracy_norm": correct_norm / n,
        "n": n,
        "chance": 1.0 / max(1, len(items[0].choices)),
    }


@torch.no_grad()
def evaluate_minimal_pairs(
    model: nn.Module,
    tokenizer: Any,
    task: Task,
    *,
    device: torch.device | None = None,
    limit: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a minimal-pairs task: is the grammatical sentence more likely?"""
    device = device or next(model.parameters()).device
    items: Sequence[MinimalPairItem] = task.items[:limit] if limit else task.items
    if not items:
        return {"accuracy": 0.0, "n": 0}

    correct = 0
    margins: list[float] = []
    was_training = model.training
    model.eval()

    for item in items:
        good, good_n = sequence_logprob(
            model, tokenizer, "", item.good, device=device, model_kwargs=model_kwargs
        )
        bad, bad_n = sequence_logprob(
            model, tokenizer, "", item.bad, device=device, model_kwargs=model_kwargs
        )
        # Length-normalise: the two sentences usually differ by a token or two.
        good_avg = good / max(1, good_n)
        bad_avg = bad / max(1, bad_n)
        if good_avg > bad_avg:
            correct += 1
        margins.append(good_avg - bad_avg)

    model.train(was_training)
    return {
        "accuracy": correct / len(items),
        "mean_margin": sum(margins) / len(margins),
        "n": len(items),
        "chance": 0.5,
    }


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    corpus: TokenizedCorpus | str | Path,
    *,
    seq_len: int = 512,
    max_batches: int = 50,
    batch_size: int = 4,
    device: torch.device | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Token-level perplexity over a tokenized corpus.

    Windows are taken sequentially and without overlap, so every token is scored
    exactly once and the number is comparable across models with the same
    tokenizer.
    """
    device = device or next(model.parameters()).device
    corpus = corpus if isinstance(corpus, TokenizedCorpus) else TokenizedCorpus(corpus)
    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    position = 0
    batches = 0

    while batches < max_batches and position + (seq_len + 1) * batch_size <= corpus.n_tokens:
        rows = []
        for _ in range(batch_size):
            window = corpus.read(position, seq_len + 1)
            rows.append(torch.as_tensor(window.astype("int64")))
            position += seq_len
        tokens = torch.stack(rows).to(device)
        logits = model(tokens[:, :-1], **(model_kwargs or {}))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            tokens[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_nll += float(loss)
        total_tokens += int(tokens[:, 1:].numel())
        batches += 1

    model.train(was_training)
    if total_tokens == 0:
        return {"perplexity": float("inf"), "loss": float("inf"), "n_tokens": 0}
    mean_nll = total_nll / total_tokens
    return {
        "loss": mean_nll,
        "perplexity": math.exp(min(mean_nll, 20.0)),
        "bits_per_token": mean_nll / math.log(2),
        "n_tokens": total_tokens,
    }


@torch.no_grad()
def evaluate_generation(
    model: nn.Module,
    tokenizer: Any,
    task: Task,
    *,
    device: torch.device | None = None,
    limit: int | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    prompt_template: str = "{prompt}",
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate an answer per item and score it with the item's verifier."""
    from minimodel.inference.sampling import SamplingConfig, generate
    from minimodel.training.rl.rlvr import VERIFIERS

    device = device or next(model.parameters()).device
    items: Sequence[GenerationItem] = task.items[:limit] if limit else task.items
    if not items:
        return {"solve_rate": 0.0, "n": 0}

    sampling = SamplingConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        stop_token_ids=[tokenizer.eos_id],
        model_kwargs=dict(model_kwargs or {}),
    )
    was_training = model.training
    model.eval()

    solved = 0.0
    samples: list[dict[str, str]] = []
    for item in items:
        prompt = prompt_template.format(prompt=item.prompt)
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        output = generate(
            model, torch.tensor([prompt_ids], dtype=torch.long), sampling, device=device
        )
        completion = tokenizer.decode(output[0].tolist()[len(prompt_ids) :])
        reward = float(VERIFIERS.get(item.verifier)(completion, item.answer))
        solved += reward
        if len(samples) < 5:
            samples.append(
                {"prompt": item.prompt, "completion": completion[:200], "expected": item.answer}
            )

    model.train(was_training)
    return {"solve_rate": solved / len(items), "n": len(items), "samples": samples}


@torch.no_grad()
def measure_throughput(
    model: nn.Module,
    *,
    batch_size: int = 1,
    prompt_len: int = 128,
    generate_tokens: int = 64,
    warmup: int = 2,
    repeats: int = 3,
    device: torch.device | str | None = None,
    vocab_size: int | None = None,
) -> dict[str, Any]:
    """Measure prefill and decode speed.

    Prefill and decode are reported separately because they are bound by
    different things: prefill is compute-bound, decode is memory-bandwidth-bound.
    A model can be fast at one and slow at the other.
    """
    device = resolve_device(device) if device is not None else next(model.parameters()).device
    vocab_size = vocab_size or getattr(model, "vocab_size", 1024)
    was_training = model.training
    model.eval()

    tokens = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)

    for _ in range(warmup):
        cache = model.new_cache() if hasattr(model, "new_cache") else None
        model(tokens, cache=cache)

    prefill_times: list[float] = []
    decode_times: list[float] = []
    for _ in range(repeats):
        cache = model.new_cache() if hasattr(model, "new_cache") else None
        if device.type == "cuda":  # pragma: no cover - hardware dependent
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(tokens, cache=cache)
        if device.type == "cuda":  # pragma: no cover - hardware dependent
            torch.cuda.synchronize()
        prefill_times.append(time.perf_counter() - start)

        next_token = logits[:, -1:].argmax(dim=-1)
        start = time.perf_counter()
        for _ in range(generate_tokens):
            logits = model(next_token, cache=cache)
            next_token = logits[:, -1:].argmax(dim=-1)
        if device.type == "cuda":  # pragma: no cover - hardware dependent
            torch.cuda.synchronize()
        decode_times.append(time.perf_counter() - start)

    model.train(was_training)
    prefill = min(prefill_times)
    decode = min(decode_times)
    return {
        "prefill_tokens_per_second": batch_size * prompt_len / prefill,
        "decode_tokens_per_second": batch_size * generate_tokens / decode,
        "prefill_seconds": round(prefill, 6),
        "decode_seconds": round(decode, 6),
        "ms_per_token": round(decode / generate_tokens * 1000, 4),
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        **device_memory_stats(device),
    }


def run_suite(
    model: nn.Module,
    tokenizer: Any,
    *,
    tasks: Sequence[Task] | None = None,
    perplexity_corpus: TokenizedCorpus | str | Path | None = None,
    device: torch.device | str | None = None,
    limit: int | None = None,
    include_throughput: bool = True,
    model_name: str = "",
    model_kwargs: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Run a set of tasks and collect the results.

    With no ``tasks`` the bundled demo tasks are used, which makes
    ``minimodel bench`` work with no downloads.
    """
    device = resolve_device(device) if device is not None else next(model.parameters()).device
    tasks = list(tasks) if tasks is not None else list(BUILTIN_TASKS.values())

    result = BenchmarkResult(
        model=model_name or getattr(model, "architecture_name", type(model).__name__),
        parameters=sum(p.numel() for p in model.parameters()),
        device=describe_device(device),
    )

    for task in tasks:
        started = time.perf_counter()
        if task.kind == "multiple_choice":
            metrics = evaluate_multiple_choice(
                model, tokenizer, task, device=device, limit=limit, model_kwargs=model_kwargs
            )
        elif task.kind == "minimal_pairs":
            metrics = evaluate_minimal_pairs(
                model, tokenizer, task, device=device, limit=limit, model_kwargs=model_kwargs
            )
        elif task.kind == "generation":
            metrics = evaluate_generation(
                model, tokenizer, task, device=device, limit=limit, model_kwargs=model_kwargs
            )
        elif task.kind == "perplexity":
            corpus = perplexity_corpus
            if corpus is None:
                continue
            metrics = evaluate_perplexity(model, corpus, device=device, model_kwargs=model_kwargs)
        else:
            logger.warning("skipping task %s with unknown kind %s", task.name, task.kind)
            continue
        metrics["seconds"] = round(time.perf_counter() - started, 3)
        result.tasks[task.name] = metrics
        logger.info("%s: %s", task.name, {k: v for k, v in metrics.items() if k != "samples"})

    if perplexity_corpus is not None and "perplexity" not in result.tasks:
        result.tasks["perplexity"] = evaluate_perplexity(
            model, perplexity_corpus, device=device, model_kwargs=model_kwargs
        )

    if include_throughput:
        result.throughput = measure_throughput(
            model,
            device=device,
            vocab_size=getattr(tokenizer, "vocab_size", None),
            prompt_len=min(128, getattr(model, "max_seq_len", 128) or 128),
        )
        logger.info(
            "throughput: %s prefill tok/s, %s decode tok/s",
            human_count(result.throughput["prefill_tokens_per_second"]),
            human_count(result.throughput["decode_tokens_per_second"]),
        )

    return result
