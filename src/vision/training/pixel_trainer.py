"""Training loops for PixelGPT and for the latent autoencoder.

:class:`PixelGPTTrainer` is a thin specialisation of the language-model
:class:`~minimodel.training.trainer.Trainer`: generating pixel art
autoregressively *is* language modelling, over a palette vocabulary instead of a
subword one, so it reuses the same loop, checkpointing and scheduling.

:class:`VAETrainer` is separate because its objective (reconstruction + KL) does
not fit the next-token framing at all.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from minimodel.core.logging_utils import get_logger
from minimodel.training.trainer import Trainer, TrainerConfig
from minimodel.vision.data.datasets import build_image_dataloader

__all__ = ["PixelGPTConfig", "PixelGPTTrainer", "VAETrainer", "VAETrainerConfig"]

logger = get_logger(__name__)


@dataclass
class PixelGPTConfig(TrainerConfig):
    """Trainer config with pixel-art defaults."""

    run_name: str = "pixelgpt"
    max_steps: int = 5000
    batch_size: int = 32
    lr: float = 6e-4
    weight_decay: float = 0.05
    warmup: float = 0.03
    lr_schedule: str = "cosine"
    log_every: int = 50
    save_every: int = 1000
    eval_every: int = 500
    #: ``seq_len`` is derived from the model's image size; the field is kept so
    #: the shared config machinery works unchanged.
    seq_len: int = 576
    #: Horizontal flip augmentation. Sprites are usually symmetric, so this
    #: roughly doubles an already-small corpus at no cost.
    horizontal_flip: bool = True


class PixelGPTTrainer(Trainer):
    """Autoregressive training over palette indices.

    Examples
    --------
    Batches come from
    :class:`~minimodel.vision.data.datasets.PixelSequenceDataset` and hold
    ``pixels`` (and optionally ``label``) rather than ``input_ids``.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PixelGPTConfig | None = None,
        *,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        **kwargs: Any,
    ):
        config = config or PixelGPTConfig()
        train_loader = (
            build_image_dataloader(
                train_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                seed=config.seed,
            )
            if train_dataset is not None
            else None
        )
        eval_loader = (
            build_image_dataloader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                seed=config.seed,
            )
            if eval_dataset is not None
            else None
        )
        super().__init__(
            model, config, train_loader=train_loader, eval_loader=eval_loader, **kwargs
        )

    def compute_loss(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Cross-entropy over palette indices, plus pixel accuracy."""
        pixels = batch["pixels"]
        labels = batch.get("label")
        logits = self.raw_model(pixels, labels=labels)[:, :-1]

        from minimodel.vision.architectures.pixelgpt import N_SPECIAL_TOKENS

        targets = pixels + N_SPECIAL_TOKENS
        flat_logits = logits.reshape(-1, logits.size(-1)).float()
        flat_targets = targets.reshape(-1)
        loss = torch.nn.functional.cross_entropy(flat_logits, flat_targets)

        with torch.no_grad():
            accuracy = float((flat_logits.argmax(dim=-1) == flat_targets).float().mean())
        return loss, {
            "pixel_accuracy": accuracy,
            "bits_per_pixel": float(loss.detach()) / math.log(2),
        }

    @torch.no_grad()
    def evaluate(self, loader=None, max_batches: int | None = None) -> dict[str, float]:
        """Held-out cross-entropy and pixel accuracy."""
        loader = loader or self.eval_loader
        if loader is None:
            return {}
        max_batches = max_batches or self.config.eval_batches
        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        count = 0
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss, extras = self.compute_loss(batch)
            total_loss += float(loss)
            total_accuracy += extras["pixel_accuracy"]
            count += 1

        self.model.train(was_training)
        if count == 0:
            return {}
        mean_loss = total_loss / count
        return {
            "val_loss": mean_loss,
            "val_pixel_accuracy": total_accuracy / count,
            "val_bits_per_pixel": mean_loss / math.log(2),
        }


@dataclass
class VAETrainerConfig(TrainerConfig):
    """Trainer config for the latent autoencoder."""

    run_name: str = "vae"
    max_steps: int = 10000
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup: float = 0.02
    log_every: int = 50
    save_every: int = 2000
    #: Weight on the KL term. Small on purpose: the goal is a bounded latent
    #: scale, not a generative prior.
    kl_weight: float = 1e-6


class VAETrainer(Trainer):
    """Trains the latent autoencoder used for latent diffusion."""

    def __init__(
        self,
        model: nn.Module,
        config: VAETrainerConfig | None = None,
        *,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        **kwargs: Any,
    ):
        config = config or VAETrainerConfig()
        train_loader = (
            build_image_dataloader(
                train_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                seed=config.seed,
            )
            if train_dataset is not None
            else None
        )
        eval_loader = (
            build_image_dataloader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                seed=config.seed,
            )
            if eval_dataset is not None
            else None
        )
        super().__init__(
            model, config, train_loader=train_loader, eval_loader=eval_loader, **kwargs
        )
        self.vae_config = config

    def compute_loss(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Reconstruction + KL."""
        return self.raw_model.loss(batch["image"], kl_weight=self.vae_config.kl_weight)

    @torch.no_grad()
    def evaluate(self, loader=None, max_batches: int | None = None) -> dict[str, float]:
        """Held-out reconstruction error and latent statistics."""
        loader = loader or self.eval_loader
        if loader is None:
            return {}
        max_batches = max_batches or self.config.eval_batches
        was_training = self.model.training
        self.model.eval()
        totals: dict[str, float] = {}
        count = 0
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss, extras = self.compute_loss(batch)
            totals["val_loss"] = totals.get("val_loss", 0.0) + float(loss)
            for key, value in extras.items():
                totals[f"val_{key}"] = totals.get(f"val_{key}", 0.0) + value
            count += 1
        self.model.train(was_training)
        return {k: v / count for k, v in totals.items()} if count else {}

    def export(self, destination: str | Path | None = None) -> Path:
        """Export the autoencoder weights and config."""
        destination = Path(destination) if destination else self.run_dir / "model"
        self.raw_model.save_pretrained(destination)
        return destination
