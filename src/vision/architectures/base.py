"""Base class shared by the image models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from minimodel.core.io_utils import write_json

__all__ = ["BaseImageModel"]


class BaseImageModel(nn.Module):
    """Common persistence and introspection for image models.

    Deliberately mirrors :class:`~minimodel.architectures.base.BaseLanguageModel`
    so that checkpointing, merging and card generation work identically for both
    modalities.
    """

    #: Registry key, written into checkpoints.
    architecture_name: str = "image-base"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.config: dict[str, Any] = dict(config)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> BaseImageModel:
        """Instantiate from a config mapping."""
        return cls(config)  # pragma: no cover - overridden by every subclass

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
        """Parameter counts grouped by top-level module name."""
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

    def save_pretrained(
        self, directory: str | Path, *, extra: Mapping[str, Any] | None = None
    ) -> Path:
        """Write ``model.pt`` and ``config.json`` into ``directory``."""
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
    ) -> BaseImageModel:
        """Recreate a model written by :meth:`save_pretrained`."""
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        model = cls.from_config(config)
        state = torch.load(directory / "model.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(state, strict=strict)
        return model

    def extra_repr(self) -> str:
        return f"architecture={self.architecture_name}, params={self.num_parameters():,}"
