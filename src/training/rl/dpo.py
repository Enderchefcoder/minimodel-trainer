"""Direct Preference Optimization.

DPO skips the reward model that RLHF needs. The insight is that the optimal
policy for a given reward has a closed form, so the reward can be re-expressed
in terms of the policy itself; fitting preferences then becomes a supervised
binary classification problem over pairs.

The loss is::

    L = -log sigmoid(beta * (log pi(chosen)/pi_ref(chosen)
                             - log pi(rejected)/pi_ref(rejected)))

The reference model ``pi_ref`` is a frozen copy of the starting policy. It is
what stops the model from drifting arbitrarily far to satisfy the preference
signal: without it, the objective is trivially maximised by degenerate outputs.

Variants
--------
``sigmoid``
    Standard DPO.
``ipo``
    Replaces the log-sigmoid with a squared loss around ``1/(2 beta)``. It does
    not saturate, so it overfits less on small preference sets - which is the
    common case for small models.
``hinge``
    Margin loss; the most robust to noisy labels.
``cpo``
    Drops the reference model entirely and adds an SFT term on the chosen
    response. Halves memory, and works surprisingly well when the policy has
    just been SFT'd on related data.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minimodel.core.logging_utils import get_logger
from minimodel.datasets.shards import IGNORE_INDEX
from minimodel.training.trainer import Trainer, TrainerConfig

__all__ = ["DPOConfig", "DPOTrainer", "dpo_loss"]

logger = get_logger(__name__)


@dataclass
class DPOConfig(TrainerConfig):
    """Configuration for :class:`DPOTrainer`."""

    run_name: str = "dpo"
    lr: float = 5e-7
    max_steps: int = 500
    batch_size: int = 4
    seq_len: int = 512
    warmup: float = 0.1
    lr_schedule: str = "cosine"
    weight_decay: float = 0.0

    #: Strength of the KL constraint toward the reference policy. Lower values
    #: allow more movement; 0.1 is the usual starting point.
    beta: float = 0.1
    #: ``sigmoid`` | ``ipo`` | ``hinge`` | ``cpo``.
    loss_type: str = "sigmoid"
    #: Conservative-DPO label smoothing, for preference data with label noise.
    label_smoothing: float = 0.0
    #: Weight of an auxiliary SFT term on the chosen response. Non-zero values
    #: keep the model from getting worse at generating *anything* while it
    #: learns to prefer one thing over another.
    sft_weight: float = 0.0
    #: Normalise sequence log-probabilities by length. Prevents the model from
    #: learning "shorter is better" when chosen responses are systematically
    #: shorter than rejected ones.
    length_normalize: bool = False
    max_pair_length: int = 512


def dpo_loss(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    *,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    label_smoothing: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute a preference loss from sequence log-probabilities.

    Parameters
    ----------
    policy_chosen, policy_rejected:
        ``[B]`` sequence log-probabilities under the trained policy.
    reference_chosen, reference_rejected:
        The same under the frozen reference.

    Returns
    -------
    tuple
        ``(loss, chosen_rewards, rejected_rewards)``. The rewards are the
        implicit DPO rewards ``beta * log(pi/pi_ref)``; their difference is the
        quantity the loss is trying to make positive.
    """
    policy_logratio = policy_chosen - policy_rejected
    reference_logratio = reference_chosen - reference_rejected
    logits = policy_logratio - reference_logratio

    normalized = loss_type.strip().lower()
    if normalized == "sigmoid":
        loss = (
            -F.logsigmoid(beta * logits) * (1.0 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    elif normalized == "ipo":
        loss = (logits - 1.0 / (2.0 * beta)) ** 2
    elif normalized == "hinge":
        loss = torch.relu(1.0 - beta * logits)
    elif normalized == "cpo":
        # Reference-free: the reference terms are already zero when the caller
        # passes zeros, so this is just the log-sigmoid of the policy ratio.
        loss = -F.logsigmoid(beta * policy_logratio)
    else:
        raise ValueError(
            f"unknown DPO loss_type {loss_type!r}; expected sigmoid, ipo, hinge or cpo"
        )

    chosen_rewards = beta * (policy_chosen - reference_chosen).detach()
    rejected_rewards = beta * (policy_rejected - reference_rejected).detach()
    return loss.mean(), chosen_rewards, rejected_rewards


def _sequence_logprob(
    model: nn.Module, input_ids: Tensor, labels: Tensor, *, average: bool = False
) -> Tensor:
    """Sum of log-probabilities of the supervised tokens in each sequence."""
    logits = model(input_ids)
    # Shift so position t predicts token t+1.
    logits = logits[:, :-1]
    targets = labels[:, 1:]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    mask = targets != IGNORE_INDEX
    safe = targets.masked_fill(~mask, 0)
    gathered = logprobs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * mask.float()
    total = gathered.sum(dim=-1)
    if average:
        return total / mask.float().sum(dim=-1).clamp(min=1.0)
    return total


class _PairIterator:
    """Cycles a preference JSONL file, yielding padded batches."""

    def __init__(self, path: str | Path, batch_size: int, max_length: int, pad_id: int = 0):
        self.path = Path(path)
        if self.path.is_dir():
            self.path = self.path / "pairs.jsonl"
        if not self.path.exists():
            raise FileNotFoundError(f"preference pairs not found: {self.path}")
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.pad_id = int(pad_id)
        self.n_pairs = sum(1 for line in self.path.open(encoding="utf-8") if line.strip())
        if self.n_pairs == 0:
            raise ValueError(f"{self.path} contains no preference pairs")

    def _rows(self) -> Iterator[dict[str, Any]]:
        while True:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    def batches(self) -> Iterator[dict[str, Tensor]]:
        """Yield batches forever."""
        buffer: list[dict[str, Any]] = []
        for row in self._rows():
            buffer.append(row)
            if len(buffer) < self.batch_size:
                continue
            yield self._collate(buffer)
            buffer = []

    def _collate(self, rows: list[dict[str, Any]]) -> dict[str, Tensor]:
        keys = ("chosen_ids", "chosen_labels", "rejected_ids", "rejected_labels")
        lengths = [
            min(self.max_length, len(row[key])) for row in rows for key in keys
        ]
        width = max(lengths)
        batch: dict[str, Tensor] = {}
        for key in keys:
            pad = IGNORE_INDEX if key.endswith("labels") else self.pad_id
            stacked = []
            for row in rows:
                values = row[key][: self.max_length]
                stacked.append(values + [pad] * (width - len(values)))
            batch[key] = torch.tensor(stacked, dtype=torch.long)
        return batch


class DPOTrainer(Trainer):
    """Trains a policy on preference pairs.

    Parameters
    ----------
    model:
        The policy. A frozen deep copy becomes the reference unless
        ``loss_type="cpo"``.
    pairs_path:
        JSONL file written by
        :func:`~minimodel.datasets.tokenize_datasets.tokenize_preference_records`.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DPOConfig | None = None,
        *,
        pairs_path: str | Path,
        reference_model: nn.Module | None = None,
        tokenizer: Any = None,
        **kwargs: Any,
    ):
        config = config or DPOConfig()
        super().__init__(model, config, tokenizer=tokenizer, **kwargs)
        self.dpo_config = config

        self.reference: nn.Module | None = None
        if config.loss_type.strip().lower() != "cpo":
            self.reference = reference_model or copy.deepcopy(self.raw_model)
            self.reference.to(self.device).eval()
            for param in self.reference.parameters():
                param.requires_grad_(False)

        pad_id = getattr(tokenizer, "pad_id", 0) if tokenizer else 0
        self.pairs = _PairIterator(
            pairs_path, config.batch_size, config.max_pair_length, pad_id=pad_id
        )
        self._pair_iter = self.pairs.batches()
        logger.info("DPO over %d preference pairs (%s loss)", self.pairs.n_pairs, config.loss_type)

    def _next_batch(self) -> dict[str, Tensor]:
        """Preference batches replace the standard token batches."""
        batch = next(self._pair_iter)
        return {k: v.to(self.device) for k, v in batch.items()}

    def compute_loss(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """DPO loss plus reward/accuracy diagnostics."""
        average = self.dpo_config.length_normalize
        policy_chosen = _sequence_logprob(
            self.raw_model, batch["chosen_ids"], batch["chosen_labels"], average=average
        )
        policy_rejected = _sequence_logprob(
            self.raw_model, batch["rejected_ids"], batch["rejected_labels"], average=average
        )

        if self.reference is None:
            reference_chosen = torch.zeros_like(policy_chosen)
            reference_rejected = torch.zeros_like(policy_rejected)
        else:
            with torch.no_grad():
                reference_chosen = _sequence_logprob(
                    self.reference, batch["chosen_ids"], batch["chosen_labels"], average=average
                )
                reference_rejected = _sequence_logprob(
                    self.reference,
                    batch["rejected_ids"],
                    batch["rejected_labels"],
                    average=average,
                )

        loss, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=self.dpo_config.beta,
            loss_type=self.dpo_config.loss_type,
            label_smoothing=self.dpo_config.label_smoothing,
        )

        extras: dict[str, float] = {
            "reward_chosen": float(chosen_rewards.mean()),
            "reward_rejected": float(rejected_rewards.mean()),
            "reward_margin": float((chosen_rewards - rejected_rewards).mean()),
            # The single most useful DPO diagnostic: the fraction of pairs the
            # implicit reward ranks correctly. It should climb past 0.6 quickly.
            "reward_accuracy": float((chosen_rewards > rejected_rewards).float().mean()),
            "logp_chosen": float(policy_chosen.mean().detach()),
            "logp_rejected": float(policy_rejected.mean().detach()),
        }

        if self.dpo_config.sft_weight > 0:
            chosen_tokens = (batch["chosen_labels"][:, 1:] != IGNORE_INDEX).float().sum(-1)
            sft_loss = -(policy_chosen / chosen_tokens.clamp(min=1.0)).mean()
            loss = loss + self.dpo_config.sft_weight * sft_loss
            extras["sft_loss"] = float(sft_loss.detach())

        return loss, extras

    @torch.no_grad()
    def evaluate(self, loader=None, max_batches: int | None = None) -> dict[str, float]:
        """Evaluate on held-out preference batches drawn from the same file."""
        max_batches = max_batches or self.config.eval_batches
        was_training = self.model.training
        self.model.eval()
        totals: dict[str, float] = {}
        count = 0
        for _ in range(max_batches):
            batch = self._next_batch()
            loss, extras = self.compute_loss(batch)
            totals["val_loss"] = totals.get("val_loss", 0.0) + float(loss)
            for key, value in extras.items():
                totals[f"val_{key}"] = totals.get(f"val_{key}", 0.0) + value
            count += 1
        self.model.train(was_training)
        if count == 0:
            return {}
        return {key: value / count for key, value in totals.items()}
