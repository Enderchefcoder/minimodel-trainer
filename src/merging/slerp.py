"""Weight-space model merging.

Merging combines two or more fine-tunes of the same base model into one set of
weights without any further training. It works because fine-tuning tends to move
weights within a single loss basin, so points between the fine-tunes are also
low-loss - and often better than either endpoint, because independent fine-tunes
make independent errors.

Methods
-------
``linear``
    Weighted average. Simple, and the right choice for more than two models.
``slerp``
    Spherical interpolation. Preserves the *norm* of each weight tensor, which
    linear interpolation does not; averaging two vectors of similar magnitude
    but different direction shrinks the result, and shrinking every weight
    matrix by a few percent measurably degrades a model.
``task_arithmetic``
    Add scaled task vectors ``(fine-tuned - base)`` to the base. Lets you dial a
    capability up or down, or subtract one with a negative coefficient.
``ties``
    Task arithmetic that first trims small deltas, then resolves sign conflicts
    by elected majority, then averages only the agreeing entries. This is the
    method to use when merging three or more fine-tunes, where naive averaging
    lets opposing updates cancel.
``dare``
    Randomly drops a fraction of the delta entries and rescales the rest.
    Surprisingly lossless up to ~90% drop, and composes with TIES.

All functions operate on plain ``state_dict`` mappings, so they work on any
architecture in this package - and on checkpoints from outside it, as long as
the keys match.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from minimodel.core.logging_utils import get_logger

__all__ = [
    "MERGE_METHODS",
    "dare_merge",
    "linear_merge",
    "load_state_dicts",
    "merge_models",
    "slerp",
    "slerp_merge",
    "task_arithmetic_merge",
    "ties_merge",
]

logger = get_logger(__name__)


def load_state_dicts(paths: Sequence[str | Path]) -> list[dict[str, Tensor]]:
    """Load ``model.pt`` from each path (file or checkpoint directory)."""
    states: list[dict[str, Tensor]] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            path = path / "model.pt"
        if not path.exists():
            raise FileNotFoundError(f"model weights not found: {path}")
        states.append(torch.load(path, map_location="cpu", weights_only=True))
    return states


def _check_compatible(states: Sequence[Mapping[str, Tensor]]) -> list[str]:
    """Return the keys common to every state dict, warning about the rest."""
    if not states:
        raise ValueError("merging needs at least one state dict")
    common = set(states[0])
    for state in states[1:]:
        common &= set(state)
    for index, state in enumerate(states):
        extra = set(state) - common
        if extra:
            logger.warning("model %d has %d keys not shared by all inputs", index, len(extra))
    shapes = {}
    for key in list(common):
        for state in states:
            shape = tuple(state[key].shape)
            if key in shapes and shapes[key] != shape:
                logger.warning("dropping %s: shape mismatch %s vs %s", key, shapes[key], shape)
                common.discard(key)
                break
            shapes[key] = shape
    return sorted(common)


def slerp(a: Tensor, b: Tensor, t: float, *, eps: float = 1e-8, dot_threshold: float = 0.9995) -> Tensor:
    """Spherical linear interpolation between two tensors.

    Falls back to a linear blend when the two are nearly parallel, where the
    spherical formula is numerically unstable and the two agree anyway.
    """
    a32, b32 = a.float(), b.float()
    a_norm = a32 / (a32.norm() + eps)
    b_norm = b32 / (b32.norm() + eps)
    dot = float((a_norm * b_norm).sum().clamp(-1.0, 1.0))

    if abs(dot) > dot_threshold:
        return torch.lerp(a32, b32, t).to(a.dtype)

    theta = torch.acos(torch.tensor(dot))
    sin_theta = torch.sin(theta)
    weight_a = torch.sin((1.0 - t) * theta) / sin_theta
    weight_b = torch.sin(t * theta) / sin_theta
    return (weight_a * a32 + weight_b * b32).to(a.dtype)


def linear_merge(
    states: Sequence[Mapping[str, Tensor]], weights: Sequence[float] | None = None
) -> dict[str, Tensor]:
    """Weighted average of several state dicts."""
    keys = _check_compatible(states)
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("merge weights must sum to a positive number")
    weights = [w / total for w in weights]

    merged: dict[str, Tensor] = {}
    for key in keys:
        accumulator = torch.zeros_like(states[0][key], dtype=torch.float32)
        for state, weight in zip(states, weights, strict=True):
            accumulator += state[key].float() * weight
        merged[key] = accumulator.to(states[0][key].dtype)
    return merged


def slerp_merge(
    states: Sequence[Mapping[str, Tensor]], t: float = 0.5
) -> dict[str, Tensor]:
    """Spherically interpolate exactly two state dicts."""
    if len(states) != 2:
        raise ValueError(f"slerp merges exactly two models, got {len(states)}")
    keys = _check_compatible(states)
    merged: dict[str, Tensor] = {}
    for key in keys:
        a, b = states[0][key], states[1][key]
        # 0-d and 1-d tensors (norm gains, biases, gates) have no meaningful
        # direction to interpolate, so they are blended linearly.
        merged[key] = torch.lerp(a.float(), b.float(), t).to(a.dtype) if a.dim() < 2 else slerp(a, b, t)
    return merged


def task_arithmetic_merge(
    base: Mapping[str, Tensor],
    fine_tuned: Sequence[Mapping[str, Tensor]],
    coefficients: Sequence[float] | None = None,
) -> dict[str, Tensor]:
    """``base + sum(coefficient_i * (fine_tuned_i - base))``.

    Negative coefficients subtract a capability, which is the cleanest way to
    remove a behaviour that a fine-tune introduced.
    """
    keys = _check_compatible([base, *fine_tuned])
    if coefficients is None:
        coefficients = [1.0 / len(fine_tuned)] * len(fine_tuned)
    if len(coefficients) != len(fine_tuned):
        raise ValueError("coefficients and fine_tuned must have the same length")

    merged: dict[str, Tensor] = {}
    for key in keys:
        result = base[key].float().clone()
        for state, coefficient in zip(fine_tuned, coefficients, strict=True):
            result += coefficient * (state[key].float() - base[key].float())
        merged[key] = result.to(base[key].dtype)
    return merged


def ties_merge(
    base: Mapping[str, Tensor],
    fine_tuned: Sequence[Mapping[str, Tensor]],
    *,
    density: float = 0.2,
    coefficients: Sequence[float] | None = None,
) -> dict[str, Tensor]:
    """TIES merging: trim, elect a sign, then average the agreeing deltas.

    Parameters
    ----------
    density:
        Fraction of delta entries to keep per tensor, by magnitude. 0.2 is the
        value the TIES paper recommends and it holds up well in practice.
    """
    if not 0 < density <= 1:
        raise ValueError(f"density must be in (0, 1], got {density}")
    keys = _check_compatible([base, *fine_tuned])
    if coefficients is None:
        coefficients = [1.0] * len(fine_tuned)

    merged: dict[str, Tensor] = {}
    for key in keys:
        base_tensor = base[key].float()
        deltas = []
        for state, coefficient in zip(fine_tuned, coefficients, strict=True):
            delta = (state[key].float() - base_tensor) * coefficient
            if density < 1.0 and delta.numel() > 1:
                keep = max(1, int(delta.numel() * density))
                threshold = delta.abs().flatten().kthvalue(delta.numel() - keep + 1).values
                delta = torch.where(delta.abs() >= threshold, delta, torch.zeros_like(delta))
            deltas.append(delta)

        stacked = torch.stack(deltas)
        # Elect the sign whose total magnitude is larger, then keep only the
        # entries that agree with it.
        elected_sign = torch.sign(stacked.sum(dim=0))
        elected_sign[elected_sign == 0] = 1.0
        agrees = torch.sign(stacked) == elected_sign.unsqueeze(0)
        agreeing = stacked * agrees.float()
        count = agrees.float().sum(dim=0).clamp(min=1.0)
        merged[key] = (base_tensor + agreeing.sum(dim=0) / count).to(base[key].dtype)
    return merged


def dare_merge(
    base: Mapping[str, Tensor],
    fine_tuned: Sequence[Mapping[str, Tensor]],
    *,
    drop_rate: float = 0.9,
    coefficients: Sequence[float] | None = None,
    seed: int = 0,
) -> dict[str, Tensor]:
    """DARE: randomly drop delta entries and rescale the survivors.

    Rescaling by ``1/(1 - drop_rate)`` keeps the delta's expectation unchanged,
    which is why dropping 90% of it costs so little.
    """
    if not 0 <= drop_rate < 1:
        raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")
    keys = _check_compatible([base, *fine_tuned])
    if coefficients is None:
        coefficients = [1.0 / len(fine_tuned)] * len(fine_tuned)

    generator = torch.Generator().manual_seed(int(seed))
    merged: dict[str, Tensor] = {}
    for key in keys:
        base_tensor = base[key].float()
        result = base_tensor.clone()
        for state, coefficient in zip(fine_tuned, coefficients, strict=True):
            delta = state[key].float() - base_tensor
            mask = (torch.rand(delta.shape, generator=generator) >= drop_rate).float()
            result += coefficient * delta * mask / (1.0 - drop_rate)
        merged[key] = result.to(base[key].dtype)
    return merged


#: Name -> merge function, for the CLI and config files.
MERGE_METHODS = {
    "linear": linear_merge,
    "slerp": slerp_merge,
    "task_arithmetic": task_arithmetic_merge,
    "ties": ties_merge,
    "dare": dare_merge,
}


def merge_models(
    paths: Sequence[str | Path],
    *,
    method: str = "linear",
    weights: Sequence[float] | None = None,
    base: str | Path | None = None,
    output: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Tensor]:
    """Merge checkpoints from disk and optionally write the result.

    ``task_arithmetic``, ``ties`` and ``dare`` require ``base``; ``linear`` and
    ``slerp`` do not.
    """
    normalized = method.strip().lower()
    if normalized not in MERGE_METHODS:
        raise ValueError(
            f"unknown merge method {method!r}; available: {', '.join(sorted(MERGE_METHODS))}"
        )
    states = load_state_dicts(paths)

    if normalized == "linear":
        merged = linear_merge(states, weights)
    elif normalized == "slerp":
        merged = slerp_merge(states, float(kwargs.get("t", weights[0] if weights else 0.5)))
    else:
        if base is None:
            raise ValueError(f"{method} merging requires a `base` model")
        base_state = load_state_dicts([base])[0]
        if normalized == "task_arithmetic":
            merged = task_arithmetic_merge(base_state, states, weights)
        elif normalized == "ties":
            merged = ties_merge(
                base_state, states, density=float(kwargs.get("density", 0.2)), coefficients=weights
            )
        else:
            merged = dare_merge(
                base_state,
                states,
                drop_rate=float(kwargs.get("drop_rate", 0.9)),
                coefficients=weights,
                seed=int(kwargs.get("seed", 0)),
            )

    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(merged, output / "model.pt")
        # Carry the architecture config across so the merge is loadable.
        source = Path(paths[0])
        config_source = (source if source.is_dir() else source.parent) / "config.json"
        if config_source.exists():
            (output / "config.json").write_text(
                config_source.read_text(encoding="utf-8"), encoding="utf-8"
            )
        logger.info("wrote merged model to %s", output)
    return merged
