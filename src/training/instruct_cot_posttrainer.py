"""Chain-of-thought post-training (reasoning distillation).

Training a small model on long reasoning traces is not just SFT with a different
dataset. Three problems appear that do not appear at 7B:

**The trace dominates the loss.** A 900-token trace followed by a 20-token
answer means 98% of the gradient is spent learning to imitate reasoning style.
:class:`CoTTrainerConfig` exposes ``reasoning_loss_weight`` so the trace can be
down-weighted relative to the answer - at ``0.0`` the model learns only to
produce correct answers, at ``1.0`` it is plain SFT.

**Traces exceed what the model can hold.** ``max_reasoning_tokens`` truncates
traces at tokenization time; below roughly 30M parameters, training on
1000-token traces produces a model that generates 1000 tokens of plausible-
looking nonsense and never reaches an answer.

**The model learns to open a thought and never close it.** ``enforce_think_close``
adds an auxiliary term on the closing marker's logit so the ``<|/think|>``
transition stays sharp, which is what makes budget-forced decoding work later.

The corresponding inference mode lives in
:func:`minimodel.inference.run.generate_with_reasoning`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from minimodel.core.logging_utils import get_logger
from minimodel.training.instruct_posttrainer import InstructTrainer, InstructTrainerConfig

__all__ = ["CoTTrainer", "CoTTrainerConfig"]

logger = get_logger(__name__)


@dataclass
class CoTTrainerConfig(InstructTrainerConfig):
    """SFT config specialised for reasoning traces."""

    run_name: str = "cot"
    seq_len: int = 1024
    lr: float = 1e-5
    max_steps: int = 1000

    #: Weight applied to loss on tokens inside ``<|think|>...<|/think|>``.
    #: Below 1.0 the model spends more of its capacity on the final answer.
    reasoning_loss_weight: float = 1.0
    #: Extra weight on the answer span after the trace closes.
    answer_loss_weight: float = 1.0
    #: Token id of ``<|think|>``; filled in from the tokenizer when omitted.
    think_open_id: int | None = None
    #: Token id of ``<|/think|>``.
    think_close_id: int | None = None
    #: Auxiliary weight encouraging a confident close-of-thought prediction.
    enforce_think_close: float = 0.0


class CoTTrainer(InstructTrainer):
    """Distils reasoning traces into a small model.

    The reasoning span is located at runtime from the ``<|think|>`` /
    ``<|/think|>`` marker ids, so the same packed dataset can be trained with
    different reasoning weights without re-tokenizing.
    """

    def __init__(
        self,
        model: nn.Module,
        config: CoTTrainerConfig | None = None,
        *,
        tokenizer: Any = None,
        **kwargs: Any,
    ):
        config = config or CoTTrainerConfig()
        if tokenizer is not None:
            if config.think_open_id is None:
                config.think_open_id = tokenizer.token_to_id("<|think|>")
            if config.think_close_id is None:
                config.think_close_id = tokenizer.token_to_id("<|/think|>")
        super().__init__(model, config, tokenizer=tokenizer, **kwargs)
        self.cot_config = config
        if config.think_open_id is None or config.think_close_id is None:
            logger.warning(
                "no <|think|> markers in the tokenizer; reasoning weighting is disabled "
                "and this run behaves like plain SFT"
            )

    def reasoning_mask(self, input_ids: Tensor) -> Tensor:
        """Boolean ``[B, T]`` mask that is true inside a reasoning span.

        Implemented with a cumulative sum of open/close events so it stays
        vectorised: the span is open where (opens seen) > (closes seen).
        """
        open_id = self.cot_config.think_open_id
        close_id = self.cot_config.think_close_id
        if open_id is None or close_id is None:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        opens = (input_ids == open_id).to(torch.int32).cumsum(dim=1)
        closes = (input_ids == close_id).to(torch.int32).cumsum(dim=1)
        return opens > closes

    def compute_loss(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Per-token cross-entropy, re-weighted by span."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        logits = self.raw_model(input_ids, **self.model_forward_kwargs)

        flat_logits = logits.reshape(-1, logits.size(-1)).float()
        flat_labels = labels.reshape(-1)
        per_token = torch.nn.functional.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=self.config.ignore_index,
            reduction="none",
            label_smoothing=self.sft_config.label_smoothing,
        )

        supervised = (flat_labels != self.config.ignore_index).float()
        in_reasoning = self.reasoning_mask(input_ids).reshape(-1).float()
        weights = supervised * (
            in_reasoning * self.cot_config.reasoning_loss_weight
            + (1.0 - in_reasoning) * self.cot_config.answer_loss_weight
        )
        denominator = weights.sum().clamp(min=1.0)
        loss = (per_token * weights).sum() / denominator

        extras: dict[str, float] = {
            "reasoning_frac": float(in_reasoning.mean()),
            "supervised_frac": float(supervised.mean()),
        }

        if self.cot_config.enforce_think_close > 0 and self.cot_config.think_close_id is not None:
            close_positions = flat_labels == self.cot_config.think_close_id
            if bool(close_positions.any()):
                close_loss = per_token[close_positions].mean()
                loss = loss + self.cot_config.enforce_think_close * close_loss
                extras["think_close_loss"] = float(close_loss.detach())

        if self.sft_config.track_accuracy:
            with torch.no_grad():
                mask = supervised.bool()
                if bool(mask.any()):
                    predictions = flat_logits[mask].argmax(dim=-1)
                    extras["token_accuracy"] = float(
                        (predictions == flat_labels[mask]).float().mean()
                    )
                    answer_mask = mask & (in_reasoning == 0)
                    if bool(answer_mask.any()):
                        answer_predictions = flat_logits[answer_mask].argmax(dim=-1)
                        extras["answer_accuracy"] = float(
                            (answer_predictions == flat_labels[answer_mask]).float().mean()
                        )
        return loss, extras
