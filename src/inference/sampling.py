"""Token sampling and text generation.

Generation uses the incremental KV cache every architecture in this package
implements, so decoding cost is linear in the number of tokens rather than
quadratic.

Sampling knobs, in the order they are applied to the logits:

1. ``repetition_penalty`` / ``presence_penalty`` - discourage tokens already
   produced. Small models loop far more than large ones, so these matter more
   here than they would for a 7B model.
2. ``temperature`` - flatten or sharpen the distribution.
3. ``top_k`` - keep only the k most likely tokens.
4. ``top_p`` (nucleus) - keep the smallest set whose mass exceeds p.
5. ``min_p`` - keep tokens with probability at least ``min_p * p_max``. This
   adapts to how confident the model is, which behaves better than a fixed
   ``top_p`` at low parameter counts where confidence varies wildly.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from minimodel.core.devices import resolve_device

__all__ = [
    "SamplingConfig",
    "apply_penalties",
    "filter_logits",
    "generate",
    "generate_batch",
    "generate_text",
    "stream_generate",
]


@dataclass
class SamplingConfig:
    """Decoding parameters.

    ``do_sample=False`` makes decoding greedy and ignores every other knob.
    """

    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 0.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    do_sample: bool = True
    stop_token_ids: list[int] = field(default_factory=list)
    seed: int | None = None
    #: Forwarded to the model; the looped architecture uses it to pick a depth.
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")
        if self.temperature == 0:
            self.do_sample = False


def apply_penalties(
    logits: Tensor,
    generated: Tensor,
    *,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
) -> Tensor:
    """Down-weight tokens that already appear in ``generated``.

    ``repetition_penalty`` divides positive logits and multiplies negative ones
    (the CTRL formulation), while ``presence_penalty`` subtracts a flat amount.
    """
    if repetition_penalty == 1.0 and presence_penalty == 0.0:
        return logits
    logits = logits.clone()
    for row in range(logits.shape[0]):
        seen = torch.unique(generated[row])
        if seen.numel() == 0:
            continue
        if repetition_penalty != 1.0:
            values = logits[row, seen]
            logits[row, seen] = torch.where(
                values > 0, values / repetition_penalty, values * repetition_penalty
            )
        if presence_penalty != 0.0:
            logits[row, seen] -= presence_penalty
    return logits


def filter_logits(
    logits: Tensor, *, top_k: int = 0, top_p: float = 0.0, min_p: float = 0.0
) -> Tensor:
    """Mask out logits excluded by top-k / top-p / min-p filtering."""
    if top_k > 0:
        k = min(int(top_k), logits.shape[-1])
        threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if 0.0 < top_p < 1.0:
        ordered, indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        remove = cumulative - torch.softmax(ordered, dim=-1) >= top_p
        remove[..., 0] = False  # always keep the most likely token
        mask = torch.zeros_like(remove).scatter(-1, indices, remove)
        logits = logits.masked_fill(mask, float("-inf"))

    if min_p > 0.0:
        probabilities = torch.softmax(logits, dim=-1)
        threshold = min_p * probabilities.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probabilities < threshold, float("-inf"))

    return logits


def _sample_from(logits: Tensor, config: SamplingConfig, generator: torch.Generator | None) -> Tensor:
    """Pick the next token id for each row of ``logits``."""
    if not config.do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    scaled = logits / max(config.temperature, 1e-5)
    scaled = filter_logits(scaled, top_k=config.top_k, top_p=config.top_p, min_p=config.min_p)
    probabilities = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: Tensor,
    config: SamplingConfig | None = None,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Generate a continuation for a batch of prompts.

    Parameters
    ----------
    input_ids:
        ``[B, T]`` prompt tokens. All rows must be the same length; use
        :func:`generate_batch` for ragged prompts.

    Returns
    -------
    Tensor
        ``[B, T + n_new]`` including the prompt.
    """
    config = config or SamplingConfig()
    device = resolve_device(device) if device is not None else next(model.parameters()).device
    input_ids = input_ids.to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    generator: torch.Generator | None = None
    if config.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(config.seed))

    was_training = model.training
    model.eval()

    cache = model.new_cache() if hasattr(model, "new_cache") else None
    tokens = input_ids
    logits = model(tokens, cache=cache, **config.model_kwargs)[:, -1, :].float()

    stop_ids = set(int(t) for t in config.stop_token_ids)
    finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=device)

    for _ in range(config.max_new_tokens):
        step_logits = apply_penalties(
            logits,
            tokens,
            repetition_penalty=config.repetition_penalty,
            presence_penalty=config.presence_penalty,
        )
        next_token = _sample_from(step_logits, config, generator)
        # Once a row has hit a stop token it keeps emitting it, so the batch can
        # continue without the finished rows drifting.
        if finished.any() and stop_ids:
            filler = torch.full_like(next_token, next(iter(stop_ids)))
            next_token = torch.where(finished.unsqueeze(-1), filler, next_token)

        tokens = torch.cat([tokens, next_token], dim=1)
        if stop_ids:
            hit = torch.tensor(
                [int(t) in stop_ids for t in next_token.flatten().tolist()], device=device
            )
            finished = finished | hit
            if bool(finished.all()):
                break

        if cache is not None:
            logits = model(next_token, cache=cache, **config.model_kwargs)[:, -1, :].float()
        else:  # pragma: no cover - all bundled models provide a cache
            logits = model(tokens, **config.model_kwargs)[:, -1, :].float()

    model.train(was_training)
    return tokens


@torch.no_grad()
def stream_generate(
    model: nn.Module,
    input_ids: Tensor,
    config: SamplingConfig | None = None,
    *,
    device: torch.device | str | None = None,
) -> Iterator[int]:
    """Yield generated token ids one at a time for a single prompt.

    Streaming matters for interactive use even at these model sizes, because
    time-to-first-token is what makes a chat feel responsive.
    """
    config = config or SamplingConfig()
    device = resolve_device(device) if device is not None else next(model.parameters()).device
    tokens = input_ids.to(device)
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.shape[0] != 1:
        raise ValueError("stream_generate handles a single sequence; use generate() for batches")

    generator: torch.Generator | None = None
    if config.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(config.seed))

    was_training = model.training
    model.eval()
    cache = model.new_cache() if hasattr(model, "new_cache") else None
    logits = model(tokens, cache=cache, **config.model_kwargs)[:, -1, :].float()
    stop_ids = {int(t) for t in config.stop_token_ids}

    try:
        for _ in range(config.max_new_tokens):
            step_logits = apply_penalties(
                logits,
                tokens,
                repetition_penalty=config.repetition_penalty,
                presence_penalty=config.presence_penalty,
            )
            next_token = _sample_from(step_logits, config, generator)
            token_id = int(next_token.item())
            if token_id in stop_ids:
                return
            yield token_id
            tokens = torch.cat([tokens, next_token], dim=1)
            if cache is not None:
                logits = model(next_token, cache=cache, **config.model_kwargs)[:, -1, :].float()
            else:  # pragma: no cover - all bundled models provide a cache
                logits = model(tokens, **config.model_kwargs)[:, -1, :].float()
    finally:
        model.train(was_training)


def generate_text(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    device: torch.device | str | None = None,
    include_prompt: bool = True,
    stop_token_ids: Sequence[int] | None = None,
    **model_kwargs: Any,
) -> str:
    """Convenience wrapper: text in, text out."""
    config = SamplingConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        stop_token_ids=list(stop_token_ids or []),
        model_kwargs=dict(model_kwargs),
    )
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    tokens = torch.tensor([prompt_ids], dtype=torch.long)
    output = generate(model, tokens, config, device=device)
    ids = output[0].tolist()
    if not include_prompt:
        ids = ids[len(prompt_ids) :]
    return tokenizer.decode(ids)


def generate_batch(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    config: SamplingConfig | None = None,
    *,
    device: torch.device | str | None = None,
) -> list[str]:
    """Generate completions for several prompts.

    Prompts are left-padded to a common length so the whole batch shares one
    decoding loop. Left padding (rather than right) keeps the final prompt token
    at the last position of every row, which is where generation continues from.
    """
    config = config or SamplingConfig()
    encoded = [tokenizer.encode(p, add_bos=True) for p in prompts]
    max_len = max(len(ids) for ids in encoded)
    pad_id = getattr(tokenizer, "pad_id", 0)
    padded = [[pad_id] * (max_len - len(ids)) + ids for ids in encoded]
    tokens = torch.tensor(padded, dtype=torch.long)

    output = generate(model, tokens, config, device=device)
    results: list[str] = []
    for row, original in zip(output.tolist(), encoded, strict=True):
        completion = row[max_len:]
        results.append(tokenizer.decode(original) + tokenizer.decode(completion))
    return results
