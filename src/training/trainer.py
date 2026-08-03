"""The core training loop.

:class:`Trainer` runs pretraining and is the base for every post-training
variant in this package: SFT, chain-of-thought distillation, DPO, SPIN and RLVR
all subclass it and override :meth:`Trainer.compute_loss`. Everything else -
gradient accumulation, mixed precision, clipping, scheduling, logging,
checkpointing, resume, evaluation, early stopping - is shared.

The loop is *step-driven*, not epoch-driven. A pretraining corpus is effectively
infinite relative to the compute budget, so "how many optimizer steps" is the
meaningful unit; :func:`~minimodel.datasets.loader.infinite_loader` cycles the
data as needed.

Example
-------
.. code-block:: python

    trainer = Trainer(
        model=build_model("dense_3m"),
        config=TrainerConfig(max_steps=1000, batch_size=8, seq_len=256),
        train_dataset=PackedTextDataset("data/tokenized/demo", seq_len=256),
    )
    result = trainer.fit()
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from minimodel.checkpointing.checkpointing import CheckpointManager
from minimodel.checkpointing.etr import ETREstimator, ThroughputMeter
from minimodel.core.devices import (
    autocast_context,
    describe_device,
    device_memory_stats,
    resolve_device,
    resolve_dtype,
)
from minimodel.core.distributed import DistInfo, setup_distributed
from minimodel.core.io_utils import ensure_dir, human_count, write_json
from minimodel.core.logging_utils import JsonlLogger, get_logger, setup_logging
from minimodel.core.seeding import set_seed
from minimodel.datasets.loader import build_dataloader, infinite_loader
from minimodel.training.callbacks import (
    Callback,
    CallbackList,
    ConsoleLogger,
    GradientMonitor,
)
from minimodel.training.optim import build_optimizer
from minimodel.training.schedules import build_scheduler, resolve_warmup

__all__ = ["Trainer", "TrainerConfig", "TrainingResult", "count_batch_tokens"]

logger = get_logger(__name__)


def count_batch_tokens(batch: Mapping[str, Tensor]) -> int:
    """Count the tokens in a batch, whatever shape the objective uses.

    Pretraining batches have ``input_ids``; preference batches have
    ``chosen_ids``/``rejected_ids`` instead. Throughput accounting should work
    for both without the loop knowing which trainer it is running.
    """
    total = 0
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and key.endswith("_ids"):
            total += int(value.numel())
    return total


@dataclass
class TrainerConfig:
    """Everything the training loop needs to know.

    Most fields map 1:1 onto keys in the ``training:`` section of a recipe YAML.
    """

    # --- run identity -------------------------------------------------
    run_name: str = "run"
    output_dir: str = "runs"
    seed: int = 1337

    # --- schedule -----------------------------------------------------
    max_steps: int = 1000
    batch_size: int = 8
    seq_len: int = 256
    grad_accum_steps: int = 1

    # --- optimization -------------------------------------------------
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    lr_schedule: str = "cosine"
    warmup: float = 0.02
    min_lr_ratio: float = 0.1
    schedule_kwargs: dict[str, Any] = field(default_factory=dict)

    # --- hardware -----------------------------------------------------
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False
    num_workers: int = 0
    pin_memory: bool = False

    # --- cadence ------------------------------------------------------
    log_every: int = 10
    eval_every: int = 0
    eval_batches: int = 20
    save_every: int = 0
    keep_last: int = 3
    keep_best: int = 1
    monitor: str = "val_loss"

    # --- behaviour ----------------------------------------------------
    resume: bool = True
    ignore_index: int = -100
    max_grad_norm_log: bool = True
    early_stopping_patience: int = 0

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.grad_accum_steps <= 0:
            raise ValueError(f"grad_accum_steps must be positive, got {self.grad_accum_steps}")
        self.betas = tuple(self.betas)  # type: ignore[assignment]

    @property
    def tokens_per_step(self) -> int:
        """Tokens consumed by one optimizer step, ignoring data parallelism."""
        return self.batch_size * self.seq_len * self.grad_accum_steps

    @property
    def run_dir(self) -> Path:
        """Directory holding this run's logs and checkpoints."""
        return Path(self.output_dir) / self.run_name

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view, written into the run metadata."""
        return asdict(self)


@dataclass
class TrainingResult:
    """What :meth:`Trainer.fit` returns."""

    steps: int
    final_loss: float
    best_loss: float
    best_val_loss: float | None
    elapsed_seconds: float
    total_tokens: int
    run_dir: Path
    metrics_path: Path
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view."""
        payload = asdict(self)
        payload["run_dir"] = str(self.run_dir)
        payload["metrics_path"] = str(self.metrics_path)
        return payload

    @property
    def final_perplexity(self) -> float:
        """``exp(final_loss)``, clamped so a diverged run does not overflow."""
        return math.exp(min(self.final_loss, 20.0))


class Trainer:
    """Trains a causal language model.

    Parameters
    ----------
    model:
        Any :class:`~minimodel.architectures.base.BaseLanguageModel`.
    config:
        A :class:`TrainerConfig`.
    train_dataset / train_loader:
        Provide either. A dataset is wrapped in a dataloader automatically.
    eval_dataset / eval_loader:
        Optional; enables periodic validation.
    tokenizer:
        Optional, only used by callbacks that generate samples and to record the
        vocabulary in run metadata.
    callbacks:
        Extra callbacks. A console logger and a divergence monitor are always
        added.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig | None = None,
        *,
        train_dataset: Dataset | None = None,
        train_loader: DataLoader | None = None,
        eval_dataset: Dataset | None = None,
        eval_loader: DataLoader | None = None,
        tokenizer: Any = None,
        callbacks: list[Callback] | None = None,
        model_forward_kwargs: Mapping[str, Any] | None = None,
    ):
        self.config = config or TrainerConfig()
        self.tokenizer = tokenizer
        self.model_forward_kwargs = dict(model_forward_kwargs or {})
        self.should_stop = False

        self.dist: DistInfo = setup_distributed()
        set_seed(self.config.seed + self.dist.rank)

        self.run_dir = ensure_dir(self.config.run_dir)
        setup_logging(
            log_file=self.run_dir / "train.log" if self.dist.is_main else None,
            rank=self.dist.rank,
            force=True,
        )

        self.device = resolve_device(self.config.device)
        self.dtype = resolve_dtype(self.config.dtype, self.device)
        self.model = model.to(self.device)
        self.model_name = getattr(model, "architecture_name", type(model).__name__)
        self.model_params = sum(p.numel() for p in model.parameters())

        if self.config.compile and hasattr(torch, "compile"):
            logger.info("compiling model (first step will be slow)")
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        if self.dist.world_size > 1:  # pragma: no cover - requires multi-process launch
            from torch.nn.parallel import DistributedDataParallel

            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank] if self.device.type == "cuda" else None,
            )

        self.optimizer = build_optimizer(
            self.raw_model,
            self.config.optimizer,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            **self.config.optimizer_kwargs,
        )
        warmup_steps = resolve_warmup(self.config.warmup, self.config.max_steps)
        self.scheduler = build_scheduler(
            self.optimizer,
            self.config.lr_schedule,
            total_steps=self.config.max_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=self.config.min_lr_ratio,
            **self.config.schedule_kwargs,
        )
        # A GradScaler is only needed for fp16; bf16 has enough exponent range.
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.dtype == torch.float16)

        self.train_loader = train_loader or self._make_loader(train_dataset, shuffle=False)
        self.eval_loader = eval_loader or (
            self._make_loader(eval_dataset, shuffle=False) if eval_dataset is not None else None
        )
        self._train_iter: Iterator[dict[str, Tensor]] | None = None

        self.checkpoints = CheckpointManager(
            self.run_dir / "checkpoints",
            keep_last=self.config.keep_last,
            keep_best=self.config.keep_best,
            monitor=self.config.monitor,
        )
        self.metrics_logger = JsonlLogger(self.run_dir / "metrics.jsonl", enabled=self.dist.is_main)
        self.etr = ETREstimator(self.config.max_steps)
        self.throughput = ThroughputMeter()

        self.callbacks = CallbackList(
            [ConsoleLogger(self.config.log_every), GradientMonitor(), *(callbacks or [])]
        )
        if self.config.early_stopping_patience > 0:
            from minimodel.training.callbacks import EarlyStopping

            self.callbacks.append(
                EarlyStopping(self.config.monitor, patience=self.config.early_stopping_patience)
            )

        self.step = 0
        self.best_loss = float("inf")
        self.best_val_loss: float | None = None
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Properties and small helpers
    # ------------------------------------------------------------------
    @property
    def raw_model(self) -> nn.Module:
        """The underlying module, unwrapping DDP and ``torch.compile``."""
        model = self.model
        for attribute in ("module", "_orig_mod"):
            inner = getattr(model, attribute, None)
            if inner is not None:
                model = inner
        return model

    def _make_loader(self, dataset: Dataset | None, *, shuffle: bool) -> DataLoader | None:
        if dataset is None:
            return None
        return build_dataloader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            seed=self.config.seed,
            pin_memory=self.config.pin_memory and self.device.type == "cuda",
        )

    def _next_batch(self) -> dict[str, Tensor]:
        if self.train_loader is None:
            raise ValueError("Trainer needs a train_dataset or train_loader")
        if self._train_iter is None:
            self._train_iter = infinite_loader(self.train_loader)
        batch = next(self._train_iter)
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def compute_loss(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Compute the loss for one micro-batch.

        Subclasses override this to implement a different objective. Returns the
        loss tensor plus a dictionary of extra scalars to log.
        """
        output = self.raw_model.forward_with_loss(
            batch["input_ids"],
            batch["labels"],
            ignore_index=self.config.ignore_index,
            **self.model_forward_kwargs,
        )
        assert output.loss is not None
        return output.loss, {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_step(self) -> dict[str, Any]:
        """Run one optimizer step, including gradient accumulation."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        extras: dict[str, float] = {}
        tokens = 0

        for _ in range(self.config.grad_accum_steps):
            batch = self._next_batch()
            tokens += count_batch_tokens(batch)
            with autocast_context(self.device, self.dtype):
                loss, step_extras = self.compute_loss(batch)
                scaled = loss / self.config.grad_accum_steps

            if self.scaler.is_enabled():
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

            total_loss += float(loss.detach())
            for key, value in step_extras.items():
                extras[key] = extras.get(key, 0.0) + float(value)
            del batch, loss

        grad_norm = float("nan")
        if self.config.grad_clip > 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            )
        elif self.config.max_grad_norm_log:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf")))

        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()

        accum = self.config.grad_accum_steps
        metrics: dict[str, Any] = {
            "loss": total_loss / accum,
            "lr": self.scheduler.get_last_lr()[0],
            "grad_norm": grad_norm,
        }
        metrics.update({key: value / accum for key, value in extras.items()})
        self.throughput.update(tokens)
        return metrics

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader | None = None, max_batches: int | None = None
    ) -> dict[str, float]:
        """Run a validation pass and return averaged metrics."""
        loader = loader or self.eval_loader
        if loader is None:
            return {}
        max_batches = max_batches or self.config.eval_batches
        was_training = self.model.training
        self.model.eval()

        total = 0.0
        count = 0
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with autocast_context(self.device, self.dtype):
                loss, _ = self.compute_loss(batch)
            total += float(loss.detach())
            count += 1

        self.model.train(was_training)
        if count == 0:
            return {}
        mean_loss = total / count
        return {
            "val_loss": mean_loss,
            "val_perplexity": math.exp(min(mean_loss, 20.0)),
            "val_batches": count,
        }

    def fit(self) -> TrainingResult:
        """Run the training loop and return a summary."""
        if self.config.resume:
            self.maybe_resume()

        self._write_run_metadata()
        self.callbacks.on_train_begin(self)
        self._start_time = time.time()
        stopped_early = False
        last_loss = float("nan")

        try:
            while self.step < self.config.max_steps and not self.should_stop:
                self.step += 1
                self.callbacks.on_step_begin(self, self.step)

                metrics = self.train_step()
                metrics["step"] = self.step
                metrics["total_tokens"] = self.throughput.total_tokens
                metrics["tokens_per_second"] = round(self.throughput.tokens_per_second, 1)
                if self.device.type == "cuda":  # pragma: no cover - hardware dependent
                    metrics.update(device_memory_stats(self.device))

                last_loss = float(metrics["loss"])
                self.best_loss = min(self.best_loss, last_loss)
                self.etr.update(self.step)
                self.metrics_logger.log(metrics)
                self.callbacks.on_step_end(self, self.step, metrics)

                if self._due(self.config.eval_every):
                    eval_metrics = self.evaluate()
                    if eval_metrics:
                        eval_metrics["step"] = self.step
                        value = eval_metrics.get("val_loss")
                        if value is not None:
                            self.best_val_loss = (
                                value
                                if self.best_val_loss is None
                                else min(self.best_val_loss, value)
                            )
                        self.metrics_logger.log(eval_metrics)
                        self.callbacks.on_evaluate(self, self.step, eval_metrics)

                if self._due(self.config.save_every):
                    self.save_checkpoint()

        except KeyboardInterrupt:  # pragma: no cover - interactive
            logger.warning("interrupted at step %d, saving a checkpoint", self.step)
            self.save_checkpoint()
            stopped_early = True

        if self.should_stop:
            stopped_early = True

        if self.dist.is_main:
            self.save_checkpoint(final=True)
        self.callbacks.on_train_end(self)
        self.metrics_logger.close()

        result = TrainingResult(
            steps=self.step,
            final_loss=last_loss,
            best_loss=self.best_loss,
            best_val_loss=self.best_val_loss,
            elapsed_seconds=time.time() - self._start_time,
            total_tokens=self.throughput.total_tokens,
            run_dir=self.run_dir,
            metrics_path=self.run_dir / "metrics.jsonl",
            stopped_early=stopped_early,
        )
        if self.dist.is_main:
            write_json(self.run_dir / "result.json", result.to_dict())
            logger.info(
                "finished %s steps in %.1fs, final loss %.4f (%s tokens)",
                result.steps,
                result.elapsed_seconds,
                result.final_loss,
                human_count(result.total_tokens),
            )
        return result

    def _due(self, every: int) -> bool:
        """Whether a periodic action should fire on the current step."""
        return every > 0 and self.step % every == 0

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, *, final: bool = False) -> Path | None:
        """Write a checkpoint for the current step."""
        if not self.dist.is_main:
            return None
        metrics: dict[str, Any] = {"train_loss": self.best_loss}
        if self.best_val_loss is not None:
            metrics[self.config.monitor] = self.best_val_loss
        checkpoint = self.checkpoints.save(
            self.step,
            model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.scaler.is_enabled() else None,
            metrics=metrics,
            extra={"total_tokens": self.throughput.total_tokens, "final": final},
        )
        self.callbacks.on_checkpoint(self, self.step, checkpoint.path)
        return checkpoint.path

    def maybe_resume(self) -> bool:
        """Resume from the latest checkpoint in this run directory, if any."""
        latest = self.checkpoints.latest()
        if latest is None:
            return False
        state = self.checkpoints.load(
            latest,
            model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.scaler.is_enabled() else None,
            map_location=str(self.device),
        )
        self.step = int(state.get("step", 0))
        extra = state.get("extra") or {}
        self.throughput.total_tokens = int(extra.get("total_tokens", 0))
        logger.info("resumed at step %d", self.step)
        return True

    def _write_run_metadata(self) -> None:
        """Record the configuration and environment for reproducibility."""
        if not self.dist.is_main:
            return
        metadata: dict[str, Any] = {
            "run_name": self.config.run_name,
            "trainer": type(self).__name__,
            "config": self.config.to_dict(),
            "model": {
                "architecture": self.model_name,
                "parameters": self.model_params,
                "config": dict(getattr(self.raw_model, "config", {})),
            },
            "device": describe_device(self.device),
            "dtype": str(self.dtype),
            "distributed": self.dist.to_dict(),
            "tokens_per_step": self.config.tokens_per_step,
            "planned_tokens": self.config.tokens_per_step * self.config.max_steps,
        }
        if self.tokenizer is not None:
            metadata["tokenizer"] = {
                "vocab_size": getattr(self.tokenizer, "vocab_size", None),
                "type": type(self.tokenizer).__name__,
            }
        write_json(self.run_dir / "run_metadata.json", metadata)

    def export(self, destination: str | Path | None = None) -> Path:
        """Export the best checkpoint's weights and config for publishing."""
        destination = Path(destination) if destination else self.run_dir / "model"
        return self.checkpoints.export_model(destination)
