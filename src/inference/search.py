"""Effort-scaled generation: spend inference compute to reach the model's ceiling.

Small models have a fixed quality ceiling, but they hit it only occasionally.
Search — proposing several continuations and keeping the best under a rerank
score — makes them hit it far more often, at no training cost. This is the
technique behind Glint-2's "effort ladder", generalised to any model in this
toolkit and wired to our own :class:`~minimodel.inference.quality_probe.QualityProbe`.

The ladder scales *search*, never the model:

======  ==============================================================
level   what it does
======  ==============================================================
low     one careful sample
medium  one sample, best single-shot settings
high    best-of-N complete continuations, reranked
xhigh   chunked beam: N instances propose short chunks, resync onto the top-K
max     same as xhigh with a wider beam
ultra   several independent `max` searches, best final wins
======  ==============================================================

The rerank score is
``mean token log-prob − rep_weight·(4-gram repetition) − shortfall + probe_weight·P(real)``.
The repetition and length terms are cheap, general anti-degeneracy penalties;
the probe term (when a quality probe is supplied) stops self-log-prob reranking
from Goodharting toward confident boilerplate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = [
    "EFFORT_LEVELS",
    "EffortConfig",
    "effort_generate",
    "score_continuation",
]


@dataclass
class EffortConfig:
    """One rung of the effort ladder."""

    instances: int = 1
    chunk: int = 0  # 0 = single-shot full continuations; >0 = chunked beam
    beams: int = 1
    runs: int = 1  # >1 = N independent searches, best final wins
    temperature: float | None = 0.3
    top_k: int = 10
    repetition_penalty: float = 1.2
    #: rerank score weights
    rep_weight: float = 2.0
    shortfall_weight: float = 1.5
    probe_weight: float = 2.0


#: The six-rung ladder. Values follow Glint-2's `effort.py` so behaviour is
#: comparable; `instances`/`beams`/`runs` scale the compute.
EFFORT_LEVELS: dict[str, EffortConfig] = {
    "low": EffortConfig(instances=1, temperature=0.25, top_k=8, repetition_penalty=1.15),
    "medium": EffortConfig(instances=1, temperature=0.3, top_k=8, repetition_penalty=1.2),
    "high": EffortConfig(instances=6, temperature=None, top_k=10, repetition_penalty=1.2),
    "xhigh": EffortConfig(instances=8, chunk=24, beams=2, temperature=None, top_k=10),
    "max": EffortConfig(instances=8, chunk=24, beams=4, temperature=None, top_k=10),
    "ultra": EffortConfig(instances=8, chunk=24, beams=4, runs=10, temperature=None, top_k=10),
}

#: Temperatures cycled across instances when a level leaves `temperature=None`.
_INSTANCE_TEMPS = (0.3, 0.35, 0.4, 0.45)


def _instance_temp(fixed: float | None, index: int) -> float:
    return fixed if fixed is not None else _INSTANCE_TEMPS[index % len(_INSTANCE_TEMPS)]


@torch.no_grad()
def _sample_next(
    logits: Tensor, temp: float, top_k: int, rep: float, recent: Sequence[int],
    generator: torch.Generator | None,
) -> int:
    """Top-k temperature sample with a CTRL-style repetition penalty."""
    logits = logits.clone()
    for token in set(recent):
        logits[token] = logits[token] / rep if logits[token] > 0 else logits[token] * rep
    values, indices = logits.topk(min(top_k, logits.numel()))
    probs = F.softmax(values / max(temp, 1e-5), dim=-1)
    return int(indices[torch.multinomial(probs, 1, generator=generator)])


@torch.no_grad()
def _sample_continuation(
    model: nn.Module, tokens: list[int], n: int, cfg: EffortConfig, temp: float,
    eos_id: int, generator: torch.Generator | None, no_eos_until: int, model_kwargs: dict,
    rep_window: int = 128,
) -> list[int]:
    """Autoregressively extend `tokens` by up to `n` tokens (no KV cache: search
    rewinds and re-scores prefixes, so a cache would be repeatedly invalidated)."""
    out = list(tokens)
    for _ in range(n):
        logits = model(torch.tensor([out]), **model_kwargs)[0, -1]
        if len(out) < no_eos_until:
            logits = logits.clone()
            logits[eos_id] = float("-inf")
        nxt = _sample_next(logits, temp, cfg.top_k, cfg.repetition_penalty,
                           out[-rep_window:], generator)
        if nxt == eos_id:
            break
        out.append(nxt)
    return out


@torch.no_grad()
def score_continuation(
    model: nn.Module, tokens: list[int], prompt_len: int, *, cfg: EffortConfig,
    target_len: int = 0, probe: Any = None, model_kwargs: dict | None = None,
) -> float:
    """Rerank score for a full sequence (see module docstring for the formula)."""
    if len(tokens) <= prompt_len:
        return float("-inf")
    model_kwargs = model_kwargs or {}
    logits = model(torch.tensor([tokens]), **model_kwargs)
    logp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = torch.tensor(tokens[1:])
    token_lp = logp[torch.arange(len(targets)), targets][prompt_len - 1 :]
    cont = tokens[prompt_len:]
    grams = [tuple(cont[i : i + 4]) for i in range(len(cont) - 3)]
    rep_frac = 1.0 - len(set(grams)) / len(grams) if grams else 0.0
    shortfall = max(0.0, (target_len - len(cont)) / target_len) if target_len else 0.0
    score = token_lp.mean().item() - cfg.rep_weight * rep_frac - cfg.shortfall_weight * shortfall
    if probe is not None:
        score += cfg.probe_weight * probe.p_real(model, tokens, prompt_len, model_kwargs)
    return score


@torch.no_grad()
def _search(
    model: nn.Module, prompt: list[int], cfg: EffortConfig, max_new: int, eos_id: int,
    seed: int, probe: Any, model_kwargs: dict,
) -> tuple[list[int], float]:
    """One search pass (single-shot best-of-N or chunked beam)."""
    prompt_len = len(prompt)
    no_eos_until = prompt_len + int(max_new * 0.6)
    generator = torch.Generator().manual_seed(seed)

    if cfg.chunk == 0:  # best-of-N complete continuations
        best, best_score = None, float("-inf")
        for i in range(cfg.instances):
            cand = _sample_continuation(
                model, prompt, max_new, cfg, _instance_temp(cfg.temperature, i),
                eos_id, generator, no_eos_until, model_kwargs,
            )
            s = score_continuation(model, cand, prompt_len, cfg=cfg, target_len=max_new,
                                   probe=probe, model_kwargs=model_kwargs) if cfg.instances > 1 else 0.0
            if s > best_score:
                best, best_score = cand, s
        return best, best_score

    frontier = [prompt]
    for round_idx in range(max(1, max_new // cfg.chunk)):
        pool: list[list[int]] = []
        for beam in frontier:
            if len(beam) - prompt_len >= max_new:
                pool.append(beam)
                continue
            for i in range(cfg.instances):
                pool.append(_sample_continuation(
                    model, beam, cfg.chunk, cfg, _instance_temp(cfg.temperature, i),
                    eos_id, generator, no_eos_until, model_kwargs,
                ))
        target = min(max_new, (round_idx + 1) * cfg.chunk)
        unique = {tuple(c): c for c in pool}
        scored = sorted(
            unique.values(),
            key=lambda c: score_continuation(model, c, prompt_len, cfg=cfg, target_len=target,
                                              probe=probe, model_kwargs=model_kwargs),
            reverse=True,
        )
        frontier = scored[: cfg.beams]
        if all(len(b) - prompt_len >= max_new for b in frontier):
            break
    best = frontier[0]
    return best, score_continuation(model, best, prompt_len, cfg=cfg, target_len=max_new,
                                    probe=probe, model_kwargs=model_kwargs)


@torch.no_grad()
def effort_generate(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    level: str = "high",
    max_new_tokens: int = 96,
    probe: Any = None,
    seed: int = 0,
    return_score: bool = False,
    **model_kwargs: Any,
) -> str | tuple[str, float]:
    """Generate with the effort ladder.

    Parameters
    ----------
    level:
        One of :data:`EFFORT_LEVELS`.
    probe:
        Optional :class:`~minimodel.inference.quality_probe.QualityProbe` used in
        reranking from `high` upward.
    model_kwargs:
        Forwarded to the model (e.g. ``loops=8`` for the looped architecture).
    """
    if level not in EFFORT_LEVELS:
        raise ValueError(f"unknown effort level {level!r}; choose from {list(EFFORT_LEVELS)}")
    cfg = EFFORT_LEVELS[level]
    eos_id = getattr(tokenizer, "eos_id", 0)
    prompt_ids = tokenizer.encode(prompt, add_bos=False) if _accepts_bos(tokenizer) else tokenizer.encode(prompt)
    prompt_ids = list(prompt_ids)

    was_training = model.training
    model.eval()
    try:
        if cfg.runs > 1:
            single = EffortConfig(**{**cfg.__dict__, "runs": 1})
            best, best_score = None, float("-inf")
            for r in range(cfg.runs):
                cand, _ = _search(model, prompt_ids, single, max_new_tokens, eos_id,
                                  seed + 1000 * (r + 1), probe, model_kwargs)
                s = score_continuation(model, cand, len(prompt_ids), cfg=cfg,
                                       target_len=max_new_tokens, probe=probe,
                                       model_kwargs=model_kwargs)
                if s > best_score:
                    best, best_score = cand, s
        else:
            best, best_score = _search(model, prompt_ids, cfg, max_new_tokens, eos_id,
                                       seed, probe, model_kwargs)
    finally:
        model.train(was_training)

    text = tokenizer.decode(best)
    return (text, best_score) if return_score else text


def _accepts_bos(tokenizer: Any) -> bool:
    """Whether ``tokenizer.encode`` takes an ``add_bos`` keyword (our tokenizer does)."""
    import inspect

    try:
        return "add_bos" in inspect.signature(tokenizer.encode).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic tokenizers
        return False
