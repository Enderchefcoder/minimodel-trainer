"""Diffusion objectives and the diffusion trainer.

Two formulations are supported.

**Rectified flow / flow matching** (default). The forward process is a straight
line between data and noise::

    x_t = (1 - t) * x_0 + t * noise

and the model predicts the constant velocity ``noise - x_0``. This is the
default because the probability path is straight, which means far fewer sampling
steps are needed to integrate it accurately - 20-50 Euler steps produce what
DDPM needs several hundred for - and because the loss has no timestep-dependent
weighting to tune.

**DDPM / epsilon prediction**. The classic variance-preserving formulation with
a cosine schedule, provided for comparison and for loading external checkpoints.

Timestep sampling uses a logit-normal distribution by default. Uniform sampling
wastes capacity: the very low and very high noise levels are easy and contribute
little gradient, while the middle of the trajectory is where the model actually
has to work.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from minimodel.checkpointing.checkpointing import CheckpointManager
from minimodel.checkpointing.etr import ETREstimator, ThroughputMeter
from minimodel.core.devices import autocast_context, describe_device, resolve_device, resolve_dtype
from minimodel.core.io_utils import ensure_dir, write_json
from minimodel.core.logging_utils import JsonlLogger, MetricFormatter, get_logger, setup_logging
from minimodel.core.seeding import set_seed
from minimodel.training.optim import build_optimizer
from minimodel.training.schedules import build_scheduler, resolve_warmup
from minimodel.vision.data.datasets import build_image_dataloader

__all__ = [
    "DiffusionConfig",
    "DiffusionTrainer",
    "cosine_alpha_bar",
    "flow_matching_targets",
    "sample_timesteps",
]

logger = get_logger(__name__)


def sample_timesteps(
    batch_size: int,
    *,
    device: torch.device,
    distribution: str = "logit_normal",
    mean: float = 0.0,
    std: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample timesteps in ``(0, 1)``.

    ``logit_normal`` concentrates samples in the middle of the trajectory, where
    the denoising problem is hardest and the gradient is most informative.
    """
    if distribution == "uniform":
        return torch.rand(batch_size, device=device, generator=generator)
    if distribution == "logit_normal":
        normal = torch.randn(batch_size, device=device, generator=generator) * std + mean
        return torch.sigmoid(normal)
    if distribution == "cosine":
        uniform = torch.rand(batch_size, device=device, generator=generator)
        return 1.0 - torch.cos(uniform * math.pi / 2.0)
    raise ValueError(f"unknown timestep distribution {distribution!r}")


def flow_matching_targets(
    x0: Tensor, noise: Tensor, t: Tensor
) -> tuple[Tensor, Tensor]:
    """Return ``(x_t, velocity_target)`` for rectified flow.

    ``t`` is broadcast over the trailing dimensions so it works for any tensor
    rank.
    """
    shape = (t.shape[0],) + (1,) * (x0.dim() - 1)
    t_expanded = t.view(shape)
    x_t = (1.0 - t_expanded) * x0 + t_expanded * noise
    return x_t, noise - x0


def cosine_alpha_bar(t: Tensor, *, offset: float = 0.008) -> Tensor:
    """Cumulative alpha for the DDPM cosine schedule."""
    numerator = torch.cos((t + offset) / (1.0 + offset) * math.pi / 2.0) ** 2
    denominator = math.cos(offset / (1.0 + offset) * math.pi / 2.0) ** 2
    return (numerator / denominator).clamp(1e-5, 1.0)


@dataclass
class DiffusionConfig:
    """Configuration for :class:`DiffusionTrainer`."""

    run_name: str = "diffusion"
    output_dir: str = "runs"
    seed: int = 1337

    max_steps: int = 5000
    batch_size: int = 32
    grad_accum_steps: int = 1

    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.99)
    grad_clip: float = 1.0
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    lr_schedule: str = "cosine"
    warmup: float = 0.05
    min_lr_ratio: float = 0.1

    device: str = "auto"
    dtype: str = "auto"
    num_workers: int = 0
    compile: bool = False

    log_every: int = 50
    eval_every: int = 0
    sample_every: int = 0
    save_every: int = 1000
    keep_last: int = 3
    keep_best: int = 1

    #: ``flow`` (rectified flow) or ``ddpm``.
    objective: str = "flow"
    timestep_distribution: str = "logit_normal"
    #: Exponential moving average decay for the sampling weights. EMA is close
    #: to mandatory for diffusion: the raw weights produce visibly noisier
    #: samples than their own moving average.
    ema_decay: float = 0.999
    #: Horizontal flip augmentation, which is free extra data for most corpora
    #: (but wrong for anything with text or handedness in it).
    horizontal_flip: bool = False
    resume: bool = True


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone().float()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the current weights into the shadow copy."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> dict[str, Tensor]:
        """Install the EMA weights, returning the originals for restoration."""
        backup: dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.detach().clone()
                param.copy_(self.shadow[name].to(param.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: Mapping[str, Tensor]) -> None:
        """Undo :meth:`copy_to`."""
        for name, param in model.named_parameters():
            if name in backup:
                param.copy_(backup[name])

    def state_dict(self) -> dict[str, Tensor]:
        """Serialisable shadow weights."""
        return dict(self.shadow)

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore shadow weights."""
        self.shadow = {k: v.clone().float() for k, v in state.items()}


class DiffusionTrainer:
    """Trains a DiT or UNet with a flow-matching or DDPM objective.

    Parameters
    ----------
    model:
        A model whose ``forward(x, t, ...)`` returns a prediction with the same
        shape as ``x``.
    dataset:
        Yields dicts with ``image`` and optionally ``label``/``text_tokens``/
        ``source``.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DiffusionConfig | None = None,
        *,
        dataset: Dataset | None = None,
        loader: DataLoader | None = None,
        eval_dataset: Dataset | None = None,
        sampler_callback: Any = None,
    ):
        self.config = config or DiffusionConfig()
        set_seed(self.config.seed)

        self.run_dir = ensure_dir(Path(self.config.output_dir) / self.config.run_name)
        setup_logging(log_file=self.run_dir / "train.log", force=True)

        self.device = resolve_device(self.config.device)
        self.dtype = resolve_dtype(self.config.dtype, self.device)
        self.model = model.to(self.device)
        if self.config.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        self.optimizer = build_optimizer(
            self.raw_model,
            self.config.optimizer,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            **self.config.optimizer_kwargs,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            self.config.lr_schedule,
            total_steps=self.config.max_steps,
            warmup_steps=resolve_warmup(self.config.warmup, self.config.max_steps),
            min_lr_ratio=self.config.min_lr_ratio,
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.dtype == torch.float16)

        self.loader = loader or (
            build_image_dataloader(
                dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                seed=self.config.seed,
            )
            if dataset is not None
            else None
        )
        if self.loader is None:
            raise ValueError("DiffusionTrainer needs a dataset or a loader")
        self.eval_loader = (
            build_image_dataloader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
                seed=self.config.seed,
            )
            if eval_dataset is not None
            else None
        )
        self._iterator = None

        self.ema = EMA(self.raw_model, self.config.ema_decay) if self.config.ema_decay > 0 else None
        self.checkpoints = CheckpointManager(
            self.run_dir / "checkpoints",
            keep_last=self.config.keep_last,
            keep_best=self.config.keep_best,
            monitor="val_loss",
        )
        self.metrics_logger = JsonlLogger(self.run_dir / "metrics.jsonl")
        self.formatter = MetricFormatter()
        self.etr = ETREstimator(self.config.max_steps)
        self.throughput = ThroughputMeter()
        self.sampler_callback = sampler_callback
        self.step = 0
        self.best_loss = float("inf")

    @property
    def raw_model(self) -> nn.Module:
        """The underlying module, unwrapping ``torch.compile``."""
        return getattr(self.model, "_orig_mod", self.model)

    def _next_batch(self) -> dict[str, Tensor]:
        if self._iterator is None:
            self._iterator = iter(self.loader)
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            batch = next(self._iterator)
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def compute_loss(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Diffusion loss for one batch."""
        x0 = batch["image"]
        noise = torch.randn_like(x0)
        t = sample_timesteps(
            x0.shape[0], device=x0.device, distribution=self.config.timestep_distribution
        )

        model_kwargs: dict[str, Any] = {}
        if "label" in batch:
            model_kwargs["labels"] = batch["label"]
        if "text_tokens" in batch:
            model_kwargs["text_tokens"] = batch["text_tokens"]
        if "source" in batch:
            model_kwargs["reference"] = batch["source"]

        if self.config.objective == "flow":
            x_t, target = flow_matching_targets(x0, noise, t)
        elif self.config.objective == "ddpm":
            alpha_bar = cosine_alpha_bar(t).view((-1,) + (1,) * (x0.dim() - 1))
            x_t = alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise
            target = noise
        else:
            raise ValueError(f"unknown diffusion objective {self.config.objective!r}")

        prediction = self.model(x_t, t, **model_kwargs)
        loss = F.mse_loss(prediction.float(), target.float())
        return loss, {"t_mean": float(t.mean())}

    def train_step(self) -> dict[str, Any]:
        """Run one optimizer step."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total = 0.0
        extras: dict[str, float] = {}
        images = 0
        for _ in range(self.config.grad_accum_steps):
            batch = self._next_batch()
            images += int(batch["image"].shape[0])
            with autocast_context(self.device, self.dtype):
                loss, step_extras = self.compute_loss(batch)
                scaled = loss / self.config.grad_accum_steps
            if self.scaler.is_enabled():
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()
            total += float(loss.detach())
            for key, value in step_extras.items():
                extras[key] = extras.get(key, 0.0) + value

        grad_norm = float("nan")
        if self.config.grad_clip > 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            )
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.raw_model)

        self.throughput.update(images)
        accum = self.config.grad_accum_steps
        return {
            "loss": total / accum,
            "lr": self.scheduler.get_last_lr()[0],
            "grad_norm": grad_norm,
            **{k: v / accum for k, v in extras.items()},
        }

    @torch.no_grad()
    def evaluate(self, max_batches: int = 20) -> dict[str, float]:
        """Average loss over held-out batches."""
        if self.eval_loader is None:
            return {}
        was_training = self.model.training
        self.model.eval()
        total = 0.0
        count = 0
        for index, batch in enumerate(self.eval_loader):
            if index >= max_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with autocast_context(self.device, self.dtype):
                loss, _ = self.compute_loss(batch)
            total += float(loss)
            count += 1
        self.model.train(was_training)
        return {"val_loss": total / count} if count else {}

    def fit(self) -> dict[str, Any]:
        """Run training and return a summary."""
        if self.config.resume:
            latest = self.checkpoints.latest()
            if latest is not None:
                state = self.checkpoints.load(
                    latest,
                    model=self.raw_model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    map_location=str(self.device),
                )
                self.step = int(state.get("step", 0))
                if self.ema is not None and (state.get("extra") or {}).get("ema"):
                    self.ema.load_state_dict(state["extra"]["ema"])

        self._write_metadata()
        import time

        started = time.time()
        last_loss = float("nan")

        while self.step < self.config.max_steps:
            self.step += 1
            metrics = self.train_step()
            metrics["step"] = self.step
            metrics["images_per_second"] = round(self.throughput.tokens_per_second, 1)
            last_loss = float(metrics["loss"])
            self.best_loss = min(self.best_loss, last_loss)
            self.etr.update(self.step)
            self.metrics_logger.log(metrics)

            if self.step % max(1, self.config.log_every) == 0:
                logger.info(
                    "%s | eta %s", self.formatter.format(metrics), self.etr.format(self.step)
                )
            if self.config.eval_every and self.step % self.config.eval_every == 0:
                evaluation = self.evaluate()
                if evaluation:
                    evaluation["step"] = self.step
                    self.metrics_logger.log(evaluation)
                    logger.info("eval | %s", self.formatter.format(evaluation))
            if (
                self.config.sample_every
                and self.sampler_callback is not None
                and self.step % self.config.sample_every == 0
            ):
                self.sampler_callback(self, self.step)
            if self.config.save_every and self.step % self.config.save_every == 0:
                self.save_checkpoint()

        self.save_checkpoint(final=True)
        self.metrics_logger.close()
        elapsed = time.time() - started
        summary = {
            "steps": self.step,
            "final_loss": last_loss,
            "best_loss": self.best_loss,
            "elapsed_seconds": round(elapsed, 2),
            "run_dir": str(self.run_dir),
        }
        write_json(self.run_dir / "result.json", summary)
        logger.info("finished %d steps in %.1fs, final loss %.4f", self.step, elapsed, last_loss)
        return summary

    def save_checkpoint(self, *, final: bool = False) -> Path:
        """Write a checkpoint, including the EMA weights."""
        extra: dict[str, Any] = {"final": final}
        if self.ema is not None:
            extra["ema"] = self.ema.state_dict()
        checkpoint = self.checkpoints.save(
            self.step,
            model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metrics={"train_loss": self.best_loss},
            extra=extra,
        )
        return checkpoint.path

    def export(self, destination: str | Path | None = None, *, use_ema: bool = True) -> Path:
        """Export sampling weights (EMA by default) plus the config."""
        destination = ensure_dir(destination or self.run_dir / "model")
        backup = None
        if use_ema and self.ema is not None:
            backup = self.ema.copy_to(self.raw_model)
        try:
            self.raw_model.save_pretrained(destination)
        finally:
            if backup is not None and self.ema is not None:
                self.ema.restore(self.raw_model, backup)
        return destination

    def _write_metadata(self) -> None:
        """Record configuration and environment for reproducibility."""
        from dataclasses import asdict

        write_json(
            self.run_dir / "run_metadata.json",
            {
                "run_name": self.config.run_name,
                "trainer": type(self).__name__,
                "config": asdict(self.config),
                "model": {
                    "architecture": getattr(self.raw_model, "architecture_name", "unknown"),
                    "parameters": sum(p.numel() for p in self.raw_model.parameters()),
                    "config": dict(getattr(self.raw_model, "config", {})),
                },
                "device": describe_device(self.device),
                "dtype": str(self.dtype),
            },
        )
