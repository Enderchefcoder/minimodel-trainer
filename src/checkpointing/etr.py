"""Estimated time remaining and throughput tracking.

``ETR`` = estimated time remaining. The estimator keeps a sliding window of
recent step durations rather than a running average over the whole run, because
the first steps of a run are always unrepresentative: compilation, cuDNN
autotuning and cold page caches make them several times slower than steady
state, and averaging them in makes the ETA wrong for hours.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from minimodel.core.io_utils import human_count, human_duration

__all__ = ["ETREstimator", "ThroughputMeter"]


@dataclass
class _Sample:
    step: int
    timestamp: float


class ETREstimator:
    """Predicts when a run will finish from recent step timings.

    Parameters
    ----------
    total_steps:
        Target step count. May be updated later via :meth:`set_total`.
    window:
        Number of recent samples used for the rate estimate.
    warmup_steps:
        Samples from before this step are ignored entirely.

    Examples
    --------
    >>> etr = ETREstimator(total_steps=100, warmup_steps=0)
    >>> _ = etr.update(0); _ = etr.update(10)
    >>> etr.total_steps
    100
    """

    def __init__(self, total_steps: int, *, window: int = 50, warmup_steps: int = 2):
        self.total_steps = int(total_steps)
        self.window = max(2, int(window))
        self.warmup_steps = int(warmup_steps)
        self._samples: deque[_Sample] = deque(maxlen=self.window)
        self._start_time = time.time()
        self._start_step: int | None = None

    def set_total(self, total_steps: int) -> None:
        """Update the target step count (e.g. after resuming)."""
        self.total_steps = int(total_steps)

    def update(self, step: int, *, now: float | None = None) -> None:
        """Record that ``step`` has just completed."""
        timestamp = time.time() if now is None else now
        if self._start_step is None:
            self._start_step = int(step)
            self._start_time = timestamp
        if step < self.warmup_steps:
            return
        self._samples.append(_Sample(int(step), timestamp))

    @property
    def steps_per_second(self) -> float:
        """Rate over the sliding window; ``0.0`` before enough samples exist."""
        if len(self._samples) < 2:
            return 0.0
        first, last = self._samples[0], self._samples[-1]
        elapsed = last.timestamp - first.timestamp
        steps = last.step - first.step
        if elapsed <= 0 or steps <= 0:
            return 0.0
        return steps / elapsed

    @property
    def seconds_per_step(self) -> float:
        """Inverse of :attr:`steps_per_second`; ``inf`` when unknown."""
        rate = self.steps_per_second
        return 1.0 / rate if rate > 0 else float("inf")

    def remaining_steps(self, current_step: int | None = None) -> int:
        """Steps left until :attr:`total_steps`."""
        step = (
            current_step
            if current_step is not None
            else (self._samples[-1].step if self._samples else 0)
        )
        return max(0, self.total_steps - int(step))

    def remaining_seconds(self, current_step: int | None = None) -> float:
        """Seconds left, or ``inf`` while the rate is still unknown."""
        rate = self.steps_per_second
        if rate <= 0:
            return float("inf")
        return self.remaining_steps(current_step) / rate

    def elapsed_seconds(self) -> float:
        """Wall-clock seconds since the first :meth:`update`."""
        return time.time() - self._start_time

    def eta_timestamp(self, current_step: int | None = None) -> float | None:
        """Absolute Unix time of the predicted finish, or ``None`` if unknown."""
        remaining = self.remaining_seconds(current_step)
        if remaining == float("inf"):
            return None
        return time.time() + remaining

    def format(self, current_step: int | None = None) -> str:
        """Compact ``"2h 14m 3s (1.42 it/s)"`` string for progress logs."""
        remaining = self.remaining_seconds(current_step)
        if remaining == float("inf"):
            return "estimating..."
        rate = self.steps_per_second
        unit = f"{rate:.2f} it/s" if rate >= 1 else f"{1 / rate:.2f} s/it"
        return f"{human_duration(remaining)} ({unit})"

    def stats(self, current_step: int | None = None) -> dict[str, Any]:
        """Machine-readable view for metric logs."""
        return {
            "steps_per_second": round(self.steps_per_second, 4),
            "remaining_steps": self.remaining_steps(current_step),
            "remaining_seconds": (
                None
                if self.remaining_seconds(current_step) == float("inf")
                else round(self.remaining_seconds(current_step), 1)
            ),
            "elapsed_seconds": round(self.elapsed_seconds(), 1),
            "eta": self.format(current_step),
        }


class ThroughputMeter:
    """Tracks tokens (or samples) processed per second.

    Reported as a sliding-window rate so a slow checkpoint write or a validation
    pass does not permanently depress the number.
    """

    def __init__(self, *, window: int = 50):
        self.window = max(2, int(window))
        self._samples: deque[tuple[float, int]] = deque(maxlen=self.window)
        self.total_tokens = 0
        self._start = time.time()

    def update(self, tokens: int, *, now: float | None = None) -> None:
        """Record that ``tokens`` more tokens have been processed."""
        timestamp = time.time() if now is None else now
        self.total_tokens += int(tokens)
        self._samples.append((timestamp, self.total_tokens))

    @property
    def tokens_per_second(self) -> float:
        """Sliding-window throughput."""
        if len(self._samples) < 2:
            return 0.0
        (t0, n0), (t1, n1) = self._samples[0], self._samples[-1]
        elapsed = t1 - t0
        if elapsed <= 0:
            return 0.0
        return (n1 - n0) / elapsed

    @property
    def average_tokens_per_second(self) -> float:
        """Throughput averaged over the whole run."""
        elapsed = time.time() - self._start
        return self.total_tokens / elapsed if elapsed > 0 else 0.0

    def format(self) -> str:
        """``"12.4K tok/s"`` style string."""
        return f"{human_count(self.tokens_per_second)} tok/s"

    def stats(self) -> dict[str, Any]:
        """Machine-readable view for metric logs."""
        return {
            "tokens_per_second": round(self.tokens_per_second, 1),
            "total_tokens": self.total_tokens,
            "avg_tokens_per_second": round(self.average_tokens_per_second, 1),
        }


def estimate_training_time(
    total_steps: int, tokens_per_step: int, tokens_per_second: float
) -> dict[str, Any]:
    """Predict a run's duration before starting it.

    Useful for answering "can I finish this overnight?" from a short benchmark.

    >>> result = estimate_training_time(1000, 8192, 40000.0)
    >>> result["total_tokens"]
    8192000
    """
    total_tokens = int(total_steps) * int(tokens_per_step)
    if tokens_per_second <= 0:
        return {"total_tokens": total_tokens, "seconds": None, "formatted": "unknown"}
    seconds = total_tokens / tokens_per_second
    return {
        "total_tokens": total_tokens,
        "seconds": round(seconds, 1),
        "formatted": human_duration(seconds),
    }
