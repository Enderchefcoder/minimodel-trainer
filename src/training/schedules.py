"""Learning-rate schedules.

All schedules are expressed as a plain function ``step -> multiplier`` and
applied through :class:`torch.optim.lr_scheduler.LambdaLR`, which keeps them
composable, easy to test, and correct across a resume (the multiplier depends
only on the step number, never on accumulated state).

Available schedules
-------------------
``cosine``
    Warmup then cosine decay to ``min_lr_ratio``. The default.
``wsd``
    Warmup-Stable-Decay: warmup, a long constant plateau, then a short decay.
    Its advantage is that the plateau can be extended indefinitely - you can
    decide to train longer *after* starting, and any checkpoint from the plateau
    is a valid branch point. That makes it the right choice when the total token
    budget is not known up front.
``linear``
    Warmup then linear decay.
``constant``
    Warmup then flat.
``inverse_sqrt``
    Warmup then ``1/sqrt(step)``; scale-free, good for very long runs.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from minimodel.core.registry import Registry

__all__ = [
    "SCHEDULES",
    "build_scheduler",
    "constant_schedule",
    "cosine_schedule",
    "inverse_sqrt_schedule",
    "linear_schedule",
    "wsd_schedule",
]

#: Registry of ``(total_steps, warmup_steps, **kwargs) -> (step -> multiplier)``.
SCHEDULES: Registry[Callable[..., Callable[[int], float]]] = Registry("lr_schedule")


def _warmup_factor(step: int, warmup_steps: int) -> float:
    """Linear warmup multiplier in ``[0, 1]``."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / float(warmup_steps))


@SCHEDULES.register("cosine")
def cosine_schedule(
    total_steps: int, warmup_steps: int = 0, *, min_lr_ratio: float = 0.1, **_: object
) -> Callable[[int], float]:
    """Warmup then cosine decay to ``min_lr_ratio`` of the peak.

    >>> fn = cosine_schedule(100, warmup_steps=10)
    >>> round(fn(9), 3)
    1.0
    """
    decay_steps = max(1, total_steps - warmup_steps)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return schedule


@SCHEDULES.register("wsd", aliases=("warmup_stable_decay", "trapezoid"))
def wsd_schedule(
    total_steps: int,
    warmup_steps: int = 0,
    *,
    decay_ratio: float = 0.1,
    min_lr_ratio: float = 0.0,
    decay_shape: str = "linear",
    **_: object,
) -> Callable[[int], float]:
    """Warmup, constant plateau, then a short decay over the final fraction.

    Parameters
    ----------
    decay_ratio:
        Fraction of the total run spent decaying. 0.1 is a good default;
        shorter decays leave quality on the table, longer ones waste plateau.
    decay_shape:
        ``linear``, ``cosine`` or ``sqrt``. ``sqrt`` (i.e. ``1 - sqrt(x)``)
        drops fastest at the start of the decay and tends to edge out linear.
    """
    decay_steps = max(1, int(total_steps * decay_ratio))
    decay_start = max(warmup_steps, total_steps - decay_steps)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps)
        if step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
        if decay_shape == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif decay_shape == "sqrt":
            factor = 1.0 - math.sqrt(progress)
        else:
            factor = 1.0 - progress
        return min_lr_ratio + (1.0 - min_lr_ratio) * factor

    return schedule


@SCHEDULES.register("linear")
def linear_schedule(
    total_steps: int, warmup_steps: int = 0, *, min_lr_ratio: float = 0.0, **_: object
) -> Callable[[int], float]:
    """Warmup then linear decay."""
    decay_steps = max(1, total_steps - warmup_steps)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)

    return schedule


@SCHEDULES.register("constant", aliases=("flat", "none"))
def constant_schedule(
    total_steps: int = 0, warmup_steps: int = 0, **_: object
) -> Callable[[int], float]:
    """Warmup then a flat learning rate."""

    def schedule(step: int) -> float:
        return _warmup_factor(step, warmup_steps)

    return schedule


@SCHEDULES.register("inverse_sqrt", aliases=("isqrt", "noam"))
def inverse_sqrt_schedule(
    total_steps: int = 0, warmup_steps: int = 1, *, min_lr_ratio: float = 0.0, **_: object
) -> Callable[[int], float]:
    """Warmup then ``sqrt(warmup / step)`` decay."""
    warmup_steps = max(1, warmup_steps)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps)
        return max(min_lr_ratio, math.sqrt(warmup_steps / float(step + 1)))

    return schedule


def build_scheduler(
    optimizer: Optimizer,
    name: str = "cosine",
    *,
    total_steps: int = 1000,
    warmup_steps: int = 0,
    last_epoch: int = -1,
    **kwargs: object,
) -> LambdaLR:
    """Create a ``LambdaLR`` from a registered schedule.

    Parameters
    ----------
    warmup_steps:
        Absolute step count, or a fraction of ``total_steps`` when less than 1.
    """
    if 0 < warmup_steps < 1:
        warmup_steps = int(total_steps * warmup_steps)
    factory = SCHEDULES.get(name)
    schedule = factory(int(total_steps), int(warmup_steps), **kwargs)
    return LambdaLR(optimizer, lr_lambda=schedule, last_epoch=last_epoch)


def resolve_warmup(warmup: float | int, total_steps: int) -> int:
    """Interpret a warmup value as either a fraction or an absolute count.

    >>> resolve_warmup(0.02, 10000)
    200
    >>> resolve_warmup(500, 10000)
    500
    """
    if warmup <= 0:
        return 0
    if warmup < 1:
        return max(1, int(total_steps * float(warmup)))
    return int(warmup)
