"""Trainer callbacks.

Callbacks are how optional behaviour attaches to the training loop without the
loop growing branches for it. Every hook is optional; subclass :class:`Callback`
and implement only what you need.

Hook order within one step::

    on_step_begin -> (forward/backward/optimizer) -> on_step_end
                                                  -> on_evaluate (periodically)
                                                  -> on_checkpoint (periodically)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from minimodel.core.io_utils import human_count
from minimodel.core.logging_utils import MetricFormatter, get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from minimodel.training.trainer import Trainer

__all__ = [
    "Callback",
    "CallbackList",
    "ConsoleLogger",
    "EarlyStopping",
    "GradientMonitor",
    "SampleGenerator",
]

logger = get_logger(__name__)


class Callback:
    """Base class. Every hook defaults to doing nothing."""

    def on_train_begin(self, trainer: Trainer) -> None:
        """Called once before the first step."""

    def on_train_end(self, trainer: Trainer) -> None:
        """Called once after the loop finishes or is stopped."""

    def on_step_begin(self, trainer: Trainer, step: int) -> None:
        """Called before each optimizer step."""

    def on_step_end(self, trainer: Trainer, step: int, metrics: Mapping[str, Any]) -> None:
        """Called after each optimizer step, with that step's metrics."""

    def on_evaluate(self, trainer: Trainer, step: int, metrics: Mapping[str, Any]) -> None:
        """Called after each evaluation pass."""

    def on_checkpoint(self, trainer: Trainer, step: int, path: Path) -> None:
        """Called after a checkpoint is written."""


class CallbackList(Callback):
    """Fans every hook out to a list of callbacks."""

    def __init__(self, callbacks: list[Callback] | None = None):
        self.callbacks = list(callbacks or [])

    def append(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)

    def __iter__(self):
        return iter(self.callbacks)

    def __len__(self) -> int:
        return len(self.callbacks)

    def on_train_begin(self, trainer: Trainer) -> None:
        """Forward to every child."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer: Trainer) -> None:
        """Forward to every child."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_step_begin(self, trainer: Trainer, step: int) -> None:
        """Forward to every child."""
        for callback in self.callbacks:
            callback.on_step_begin(trainer, step)

    def on_step_end(self, trainer: Trainer, step: int, metrics: Mapping[str, Any]) -> None:
        """Forward to every child."""
        for callback in self.callbacks:
            callback.on_step_end(trainer, step, metrics)

    def on_evaluate(self, trainer: Trainer, step: int, metrics: Mapping[str, Any]) -> None:
        """Forward to every child."""
        for callback in self.callbacks:
            callback.on_evaluate(trainer, step, metrics)

    def on_checkpoint(self, trainer: Trainer, step: int, path: Path) -> None:
        """Forward to every child."""
        for callback in self.callbacks:
            callback.on_checkpoint(trainer, step, path)


class ConsoleLogger(Callback):
    """Prints a metric line every ``every`` steps, with an ETA."""

    def __init__(self, every: int = 10, formatter: MetricFormatter | None = None):
        self.every = max(1, int(every))
        self.formatter = formatter or MetricFormatter()

    def on_train_begin(self, trainer: Trainer) -> None:
        """Announce the run configuration."""
        model_params = trainer.model_params
        logger.info(
            "training %s (%s params) for %s steps on %s",
            trainer.model_name,
            human_count(model_params),
            human_count(trainer.config.max_steps),
            trainer.device,
        )

    def on_step_end(self, trainer: Trainer, step: int, metrics: Mapping[str, Any]) -> None:
        """Log the step's metrics on the configured cadence."""
        if step % self.every != 0 and step != trainer.config.max_steps:
            return
        payload = dict(metrics)
        payload["eta"] = trainer.etr.format(step)
        logger.info(self.formatter.format(payload))

    def on_evaluate(self, trainer: Trainer, step: int, metrics: Mapping[str, Any]) -> None:
        """Log evaluation results."""
        logger.info("eval | %s", self.formatter.format(metrics))


class EarlyStopping(Callback):
    """Stops training when a monitored metric stops improving.

    Parameters
    ----------
    monitor:
        Metric key, usually ``"val_loss"``.
    patience:
        Number of evaluations without improvement before stopping.
    min_delta:
        Improvements smaller than this do not count.
    mode:
        ``"min"`` or ``"max"``.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        *,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best: float | None = None
        self.waited = 0

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def on_evaluate(self, trainer: Trainer, step: int, metrics: Mapping[str, Any]) -> None:
        """Update the patience counter and request a stop when exhausted."""
        if self.monitor not in metrics:
            return
        value = float(metrics[self.monitor])
        if self._improved(value):
            self.best = value
            self.waited = 0
            return
        self.waited += 1
        if self.waited >= self.patience:
            logger.info(
                "early stopping at step %d: %s has not improved for %d evaluations (best %.4f)",
                step,
                self.monitor,
                self.waited,
                self.best if self.best is not None else float("nan"),
            )
            trainer.should_stop = True


class GradientMonitor(Callback):
    """Watches for divergence and can halt a run before it wastes GPU hours.

    A loss that has become NaN or has grown far above its running best is not
    going to recover; stopping immediately preserves the last good checkpoint.
    """

    def __init__(self, *, max_loss_ratio: float = 4.0, stop_on_nan: bool = True):
        self.max_loss_ratio = float(max_loss_ratio)
        self.stop_on_nan = bool(stop_on_nan)
        self.best_loss: float | None = None

    def on_step_end(self, trainer: Trainer, step: int, metrics: Mapping[str, Any]) -> None:
        """Check this step's loss for NaN or explosion."""
        loss = metrics.get("loss")
        if loss is None:
            return
        loss = float(loss)
        if math.isnan(loss) or math.isinf(loss):
            logger.error("loss is %s at step %d", loss, step)
            if self.stop_on_nan:
                trainer.should_stop = True
            return
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            return
        if self.best_loss > 0 and loss > self.best_loss * self.max_loss_ratio:
            logger.error(
                "loss diverged at step %d: %.4f is %.1fx the best seen (%.4f)",
                step,
                loss,
                loss / self.best_loss,
                self.best_loss,
            )
            trainer.should_stop = True


class SampleGenerator(Callback):
    """Generates a sample completion at each evaluation.

    Loss curves tell you the model is improving; samples tell you *how*. For
    small models this is the difference between noticing at step 500 and
    noticing at step 50000 that the data pipeline is feeding garbage.
    """

    def __init__(
        self,
        prompt: str = "The",
        *,
        max_new_tokens: int = 48,
        temperature: float = 0.8,
        every: int = 1,
    ):
        self.prompt = prompt
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.every = max(1, int(every))
        self._calls = 0

    def on_evaluate(self, trainer: Trainer, step: int, metrics: Mapping[str, Any]) -> None:
        """Sample from the model and log the text."""
        self._calls += 1
        if self._calls % self.every != 0:
            return
        tokenizer = getattr(trainer, "tokenizer", None)
        if tokenizer is None:
            return
        from minimodel.inference.sampling import generate_text

        was_training = trainer.model.training
        trainer.model.eval()
        try:
            text = generate_text(
                trainer.model,
                tokenizer,
                self.prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                device=trainer.device,
            )
            logger.info("sample @ step %d | %s", step, text.replace("\n", "\\n")[:300])
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
            logger.warning("sample generation failed: %s", exc)
        finally:
            trainer.model.train(was_training)
