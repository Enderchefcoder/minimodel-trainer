"""Common base class and metadata for all architectures.

Anything that consumes a model (trainers, evaluators, samplers, mergers) does so
through :class:`BaseLanguageModel`. Adding a new architecture therefore means
implementing three things: ``forward``, ``from_config`` and the
``architecture_name`` class attribute.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minimodel.architectures.layers import KVCache
from minimodel.core.io_utils import write_json

__all__ = ["BaseLanguageModel", "ModelOutput"]


@dataclass
class ModelOutput:
    """Container returned by :meth:`BaseLanguageModel.forward_with_loss`."""

    logits: Tensor
    loss: Tensor | None = None
    hidden_states: Tensor | None = None

    def __iter__(self):
        """Allow ``logits, loss = output`` style unpacking."""
        yield self.logits
        yield self.loss


class BaseLanguageModel(nn.Module):
    """Shared behaviour for causal language models in this repository.

    Subclasses must set :attr:`architecture_name`, accept a config mapping in
    ``from_config`` and implement ``forward``.
    """

    #: Registry key, also written into checkpoints so they can be reloaded.
    architecture_name: str = "base"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.config: dict[str, Any] = dict(config)
        self.vocab_size: int = int(self.config.get("vocab_size", 0))
        self.max_seq_len: int = int(self.config.get("max_seq_len", 0))

    # ------------------------------------------------------------------
    # Construction / persistence
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> BaseLanguageModel:
        """Instantiate the model from a plain config mapping."""
        return cls(config)  # pragma: no cover - overridden by every subclass

    def save_pretrained(
        self, directory: str | Path, *, extra: Mapping[str, Any] | None = None
    ) -> Path:
        """Write ``model.pt`` and ``config.json`` into ``directory``.

        The saved config always carries ``architecture`` so
        :func:`minimodel.architectures.builder.load_model` can round-trip it
        without extra arguments.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = dict(self.config)
        payload["architecture"] = self.architecture_name
        if extra:
            payload.update(dict(extra))
        write_json(directory / "config.json", payload)
        torch.save(self.state_dict(), directory / "model.pt")
        return directory

    @classmethod
    def load_pretrained(
        cls, directory: str | Path, *, map_location: str | torch.device = "cpu", strict: bool = True
    ) -> BaseLanguageModel:
        """Recreate a model previously written by :meth:`save_pretrained`."""
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        model = cls.from_config(config)
        state = torch.load(directory / "model.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(state, strict=strict)
        return model

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def num_parameters(self, *, trainable_only: bool = False) -> int:
        """Total parameter count, counting tied tensors once."""
        seen: set[int] = set()
        total = 0
        for param in self.parameters():
            if trainable_only and not param.requires_grad:
                continue
            if id(param) in seen:
                continue
            seen.add(id(param))
            total += param.numel()
        return total

    def parameter_breakdown(self) -> dict[str, int]:
        """Parameter counts grouped by top-level module name.

        Useful for checking an implementation against the ``param_budget``
        section of an architecture template.
        """
        breakdown: dict[str, int] = {}
        seen: set[int] = set()
        for name, param in self.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            group = name.split(".")[0]
            breakdown[group] = breakdown.get(group, 0) + param.numel()
        breakdown["total"] = sum(breakdown.values())
        return breakdown

    @property
    def device(self) -> torch.device:
        """Device of the first parameter."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the first parameter."""
        return next(self.parameters()).dtype

    def new_cache(self, *, max_length: int | None = None) -> KVCache:
        """Create an empty decoding cache sized for this model."""
        return KVCache(max_length=max_length)

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------
    def forward_with_loss(
        self,
        tokens: Tensor,
        targets: Tensor | None = None,
        *,
        ignore_index: int = -100,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> ModelOutput:
        """Run the model and, when ``targets`` are given, the cross-entropy loss.

        ``targets`` are expected to be already shifted by the caller (the data
        pipeline emits ``(input_ids, labels)`` pairs), which keeps the loss
        masking logic for instruction tuning in one place.
        """
        logits = self(tokens, **kwargs)
        loss: Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=ignore_index,
                reduction=reduction,
            )
        return ModelOutput(logits=logits, loss=loss)

    def token_log_probs(
        self,
        tokens: Tensor,
        targets: Tensor,
        *,
        ignore_index: int = -100,
        **kwargs: Any,
    ) -> Tensor:
        """Per-token log probabilities of ``targets``, zero where masked.

        Shape ``[B, T]``. This is the primitive used by DPO, SPIN and the
        multiple-choice evaluation harness.
        """
        logits = self(tokens, **kwargs)
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        safe_targets = targets.clone()
        mask = safe_targets != ignore_index
        safe_targets[~mask] = 0
        gathered = logprobs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        return gathered * mask.to(gathered.dtype)

    def sequence_log_prob(
        self,
        tokens: Tensor,
        targets: Tensor,
        *,
        ignore_index: int = -100,
        average: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """Sum (or mean) of :meth:`token_log_probs` over the time axis."""
        per_token = self.token_log_probs(tokens, targets, ignore_index=ignore_index, **kwargs)
        mask = (targets != ignore_index).to(per_token.dtype)
        total = (per_token * mask).sum(dim=-1)
        if average:
            return total / mask.sum(dim=-1).clamp(min=1.0)
        return total

    def extra_repr(self) -> str:
        return f"architecture={self.architecture_name}, params={self.num_parameters():,}"
