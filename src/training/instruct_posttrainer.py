"""Supervised fine-tuning for instruction following.

SFT differs from pretraining in three ways, and all three are easy to get wrong:

1. **Loss masking.** Only assistant tokens are supervised. Training on the
   prompt teaches the model to generate user turns, which wastes capacity a
   small model does not have.
2. **Learning rate.** SFT runs at roughly a tenth of the pretraining rate. A
   high rate on a small dataset erases pretrained knowledge within a few hundred
   steps - the model gets fluent at the SFT format and forgets everything else.
3. **Epochs, not steps.** SFT sets are small and are usually seen 2-3 times.
   Beyond that, memorisation sets in and validation loss climbs while training
   loss keeps falling.

The class below is a thin specialisation of :class:`~minimodel.training.trainer.Trainer`
that sets those defaults, reports token-level accuracy on supervised positions,
and can optionally hold part of the pretraining corpus in the mixture to limit
forgetting.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from minimodel.core.logging_utils import get_logger
from minimodel.datasets.loader import MixtureDataset
from minimodel.training.trainer import Trainer, TrainerConfig

__all__ = ["InstructTrainer", "InstructTrainerConfig", "build_sft_mixture"]

logger = get_logger(__name__)


@dataclass
class InstructTrainerConfig(TrainerConfig):
    """Trainer config with SFT-appropriate defaults."""

    lr: float = 2e-5
    warmup: float = 0.03
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.0
    max_steps: int = 500
    batch_size: int = 8
    seq_len: int = 512
    eval_every: int = 50
    save_every: int = 250
    monitor: str = "val_loss"
    run_name: str = "sft"

    #: Fraction of batches drawn from the pretraining corpus to limit
    #: catastrophic forgetting. 0 disables the replay mixture.
    replay_fraction: float = 0.0
    #: Label smoothing. A little (0.05-0.1) helps small models stop being
    #: overconfident on the handful of answer templates in an SFT set.
    label_smoothing: float = 0.0
    #: Log the fraction of supervised positions predicted correctly.
    track_accuracy: bool = True
    extra_metrics: dict[str, Any] = field(default_factory=dict)


def build_sft_mixture(
    sft_dataset: Dataset,
    pretrain_dataset: Dataset | None,
    replay_fraction: float,
    *,
    seed: int = 0,
) -> Dataset:
    """Blend an SFT dataset with pretraining replay.

    Replay is the cheapest defence against catastrophic forgetting: keeping
    5-15% of the batches from the original corpus preserves most of the
    pretrained ability at negligible cost to instruction quality.
    """
    if not pretrain_dataset or replay_fraction <= 0:
        return sft_dataset
    if replay_fraction >= 1:
        raise ValueError(f"replay_fraction must be < 1, got {replay_fraction}")
    return MixtureDataset(
        [sft_dataset, pretrain_dataset],
        [1.0 - replay_fraction, replay_fraction],
        names=["sft", "replay"],
        seed=seed,
    )


class InstructTrainer(Trainer):
    """Fine-tunes a pretrained model to follow instructions.

    Expects a dataset that yields ``(input_ids, labels)`` where prompt positions
    are :data:`~minimodel.datasets.shards.IGNORE_INDEX`, which is what
    :class:`~minimodel.datasets.loader.SupervisedDataset` produces from a corpus
    tokenized with ``format="chat"``.
    """

    def __init__(
        self,
        model: nn.Module,
        config: InstructTrainerConfig | None = None,
        *,
        replay_dataset: Dataset | None = None,
        train_dataset: Dataset | None = None,
        **kwargs: Any,
    ):
        config = config or InstructTrainerConfig()
        if train_dataset is not None and config.replay_fraction > 0:
            train_dataset = build_sft_mixture(
                train_dataset, replay_dataset, config.replay_fraction, seed=config.seed
            )
        super().__init__(model, config, train_dataset=train_dataset, **kwargs)
        self.sft_config = config

    def compute_loss(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Cross-entropy over supervised positions only, plus token accuracy."""
        logits = self.raw_model(batch["input_ids"], **self.model_forward_kwargs)
        labels = batch["labels"]
        flat_logits = logits.reshape(-1, logits.size(-1)).float()
        flat_labels = labels.reshape(-1)

        loss = torch.nn.functional.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=self.config.ignore_index,
            label_smoothing=self.sft_config.label_smoothing,
        )

        extras: dict[str, float] = {}
        mask = flat_labels != self.config.ignore_index
        supervised = int(mask.sum())
        extras["supervised_frac"] = supervised / max(1, flat_labels.numel())
        if self.sft_config.track_accuracy and supervised > 0:
            with torch.no_grad():
                predictions = flat_logits[mask].argmax(dim=-1)
                extras["token_accuracy"] = float((predictions == flat_labels[mask]).float().mean())
        return loss, extras

    @torch.no_grad()
    def evaluate(self, loader=None, max_batches: int | None = None) -> dict[str, float]:
        """Validation pass that also reports supervised-token accuracy."""
        loader = loader or self.eval_loader
        if loader is None:
            return {}
        max_batches = max_batches or self.config.eval_batches
        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_supervised = 0
        batches = 0
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.raw_model(batch["input_ids"], **self.model_forward_kwargs)
            labels = batch["labels"]
            flat_logits = logits.reshape(-1, logits.size(-1)).float()
            flat_labels = labels.reshape(-1)
            mask = flat_labels != self.config.ignore_index
            if int(mask.sum()) == 0:
                continue
            total_loss += float(
                torch.nn.functional.cross_entropy(
                    flat_logits, flat_labels, ignore_index=self.config.ignore_index
                )
            )
            predictions = flat_logits[mask].argmax(dim=-1)
            total_correct += int((predictions == flat_labels[mask]).sum())
            total_supervised += int(mask.sum())
            batches += 1

        self.model.train(was_training)
        if batches == 0:
            return {}
        mean_loss = total_loss / batches
        return {
            "val_loss": mean_loss,
            "val_perplexity": math.exp(min(mean_loss, 20.0)),
            "val_token_accuracy": total_correct / max(1, total_supervised),
            "val_batches": batches,
        }
