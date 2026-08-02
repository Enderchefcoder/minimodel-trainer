"""Diffusion transformer for image generation.

This is the general-purpose image model of the toolkit. It predicts a velocity
field under rectified flow (see
:mod:`minimodel.vision.training.diffusion`) from a noisy image, a timestep and
an optional condition.

Why a transformer rather than a UNet at this scale: the UNet's inductive bias
helps most when data is scarce, but a DiT scales more predictably, handles
class and text conditioning through one uniform mechanism (AdaLN), and reaches
better sample quality per parameter once the training set is more than a few
tens of thousands of images. Both are provided; :mod:`minimodel.vision.architectures.unet`
is the better choice below roughly 20K training images.

Conditioning modes
------------------
``none``
    Unconditional generation.
``class``
    A label embedding with a null class for classifier-free guidance.
``text``
    A small jointly-trained text encoder over the same BPE tokenizer used by the
    language models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from minimodel.vision.architectures.base import BaseImageModel
from minimodel.vision.architectures.layers import (
    DiTBlock,
    FinalLayer,
    LabelEmbedding,
    PatchEmbed,
    TextConditioner,
    TimestepEmbedding,
    sincos_position_embedding,
    unpatchify,
)

__all__ = ["DiT", "DiTConfig"]

#: Defaults for every key the DiT understands.
DiTConfig: dict[str, Any] = {
    "image_size": 32,
    "patch_size": 2,
    "in_channels": 3,
    "out_channels": None,
    "dim": 384,
    "depth": 12,
    "n_heads": 6,
    "mlp_ratio": 4.0,
    "qk_norm": True,
    "condition": "none",
    "num_classes": 0,
    "class_dropout": 0.1,
    "text_vocab_size": 0,
    "text_max_len": 32,
    "text_layers": 2,
    "learned_position": False,
    #: Extra channels concatenated to the input, used by the edit model to pass
    #: a reference image.
    "extra_in_channels": 0,
}


class DiT(BaseImageModel):
    """Patch-based diffusion transformer.

    Examples
    --------
    >>> model = DiT({"image_size": 8, "patch_size": 2, "dim": 32, "depth": 2, "n_heads": 2})
    >>> x = torch.randn(2, 3, 8, 8)
    >>> t = torch.rand(2)
    >>> tuple(model(x, t).shape)
    (2, 3, 8, 8)
    """

    architecture_name = "dit"

    def __init__(self, config: Mapping[str, Any] | None = None):
        merged = {**DiTConfig, **dict(config or {})}
        super().__init__(merged)
        cfg = self.config

        self.image_size = int(cfg["image_size"])
        self.patch_size = int(cfg["patch_size"])
        self.in_channels = int(cfg["in_channels"])
        self.extra_in_channels = int(cfg["extra_in_channels"])
        self.out_channels = int(cfg["out_channels"] or cfg["in_channels"])
        self.dim = int(cfg["dim"])
        self.condition_mode = str(cfg["condition"]).lower()

        self.patch_embed = PatchEmbed(
            self.image_size,
            self.patch_size,
            self.in_channels + self.extra_in_channels,
            self.dim,
        )
        self.grid_size = self.patch_embed.grid_size

        if bool(cfg["learned_position"]):
            self.position_embedding = nn.Parameter(
                torch.zeros(1, self.patch_embed.n_patches, self.dim)
            )
            nn.init.normal_(self.position_embedding, std=0.02)
        else:
            self.register_buffer(
                "position_embedding",
                sincos_position_embedding(self.dim, self.grid_size),
                persistent=False,
            )

        self.time_embedding = TimestepEmbedding(self.dim)

        self.label_embedding: LabelEmbedding | None = None
        self.text_conditioner: TextConditioner | None = None
        if self.condition_mode == "class":
            if int(cfg["num_classes"]) <= 0:
                raise ValueError("condition='class' requires num_classes > 0")
            self.label_embedding = LabelEmbedding(
                int(cfg["num_classes"]), self.dim, dropout_prob=float(cfg["class_dropout"])
            )
        elif self.condition_mode == "text":
            if int(cfg["text_vocab_size"]) <= 0:
                raise ValueError("condition='text' requires text_vocab_size > 0")
            self.text_conditioner = TextConditioner(
                int(cfg["text_vocab_size"]),
                self.dim,
                n_layers=int(cfg["text_layers"]),
                n_heads=max(1, int(cfg["n_heads"]) // 2),
                max_len=int(cfg["text_max_len"]),
                dropout_prob=float(cfg["class_dropout"]),
            )
        elif self.condition_mode not in {"none", ""}:
            raise ValueError(f"unknown condition mode {self.condition_mode!r}")

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    self.dim,
                    int(cfg["n_heads"]),
                    mlp_ratio=float(cfg["mlp_ratio"]),
                    qk_norm=bool(cfg["qk_norm"]),
                )
                for _ in range(int(cfg["depth"]))
            ]
        )
        self.final = FinalLayer(self.dim, self.patch_size, self.out_channels)
        self.init_weights()

    def init_weights(self) -> None:
        """Xavier init for projections; AdaLN gates stay at zero."""

        def _init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for block in self.blocks:
            block.attention.apply(_init)
            block.mlp.apply(_init)
        nn.init.xavier_uniform_(self.patch_embed.projection.weight.flatten(1).view_as(
            self.patch_embed.projection.weight.flatten(1)
        ).view(self.patch_embed.projection.weight.shape))
        nn.init.zeros_(self.patch_embed.projection.bias)

    def build_condition(
        self,
        t: Tensor,
        *,
        labels: Tensor | None = None,
        text_tokens: Tensor | None = None,
    ) -> Tensor:
        """Combine the timestep with the class/text condition into one vector."""
        condition = self.time_embedding(t)
        if self.label_embedding is not None:
            if labels is None:
                labels = torch.full(
                    (t.shape[0],), self.label_embedding.null_index, dtype=torch.long, device=t.device
                )
            condition = condition + self.label_embedding(labels)
        if self.text_conditioner is not None:
            condition = condition + self.text_conditioner(
                text_tokens, batch_size=t.shape[0], device=t.device
            )
        return condition

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        *,
        labels: Tensor | None = None,
        text_tokens: Tensor | None = None,
        reference: Tensor | None = None,
    ) -> Tensor:
        """Predict the velocity field for noisy images ``x`` at timestep ``t``.

        Parameters
        ----------
        x:
            ``[B, C, H, W]`` noisy images.
        t:
            ``[B]`` timesteps in ``[0, 1]``.
        reference:
            Optional ``[B, C_extra, H, W]`` conditioning image, concatenated
            channel-wise. Used by the image-edit model.
        """
        if reference is not None:
            x = torch.cat([x, reference], dim=1)
        elif self.extra_in_channels:
            zeros = torch.zeros(
                x.shape[0], self.extra_in_channels, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, zeros], dim=1)

        tokens = self.patch_embed(x) + self.position_embedding.to(x.dtype)
        condition = self.build_condition(t, labels=labels, text_tokens=text_tokens)
        for block in self.blocks:
            tokens = block(tokens, condition)
        patches = self.final(tokens, condition)
        return unpatchify(patches, self.patch_size, self.out_channels, self.grid_size)

    def forward_with_guidance(
        self,
        x: Tensor,
        t: Tensor,
        *,
        labels: Tensor | None = None,
        text_tokens: Tensor | None = None,
        reference: Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        """Classifier-free guidance in a single batched forward pass.

        The conditional and unconditional predictions are stacked into one batch
        rather than run sequentially, which halves the number of kernel launches
        and matters a lot at these model sizes where launch overhead dominates.
        """
        if guidance_scale == 1.0 or (self.label_embedding is None and self.text_conditioner is None):
            return self(x, t, labels=labels, text_tokens=text_tokens, reference=reference)

        batch = x.shape[0]
        x_double = torch.cat([x, x], dim=0)
        t_double = torch.cat([t, t], dim=0)

        labels_double = None
        if self.label_embedding is not None:
            null = torch.full(
                (batch,), self.label_embedding.null_index, dtype=torch.long, device=x.device
            )
            labels_double = torch.cat([labels if labels is not None else null, null], dim=0)

        text_double = None
        if self.text_conditioner is not None and text_tokens is not None:
            text_double = torch.cat([text_tokens, torch.zeros_like(text_tokens)], dim=0)

        reference_double = (
            torch.cat([reference, reference], dim=0) if reference is not None else None
        )

        prediction = self(
            x_double,
            t_double,
            labels=labels_double,
            text_tokens=text_double,
            reference=reference_double,
        )
        conditional, unconditional = prediction.chunk(2, dim=0)
        return unconditional + guidance_scale * (conditional - unconditional)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DiT:
        """Build from a config mapping, ignoring bookkeeping keys."""
        payload = {k: v for k, v in dict(config).items() if k in DiTConfig}
        return cls(payload)
