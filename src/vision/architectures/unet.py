"""Convolutional UNet for pixel-space diffusion.

The UNet's locality and multi-scale structure are a strong prior for images, and
at small data scale that prior is worth more than the DiT's flexibility. Below
roughly 20-50K training images a UNet of the same parameter count will usually
produce cleaner samples; above that the DiT pulls ahead.

The implementation is the standard one: a downsampling path of residual blocks
with optional self-attention at the lower resolutions, a middle block, and a
symmetric upsampling path with skip connections. Timestep and class conditioning
are injected into every residual block.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minimodel.vision.architectures.base import BaseImageModel
from minimodel.vision.architectures.layers import (
    AttentionBlock2d,
    Downsample2d,
    LabelEmbedding,
    ResBlock2d,
    TextConditioner,
    TimestepEmbedding,
    Upsample2d,
)

__all__ = ["UNet", "UNetConfig"]

#: Defaults for every key the UNet understands.
UNetConfig: dict[str, Any] = {
    "image_size": 32,
    "in_channels": 3,
    "out_channels": None,
    "base_channels": 96,
    "channel_multipliers": [1, 2, 2],
    "blocks_per_level": 2,
    "attention_resolutions": [16, 8],
    "n_heads": 4,
    "dropout": 0.0,
    "condition": "none",
    "num_classes": 0,
    "class_dropout": 0.1,
    "text_vocab_size": 0,
    "text_max_len": 32,
    "extra_in_channels": 0,
    "groups": 8,
}


class UNet(BaseImageModel):
    """Residual UNet with timestep and optional class/text conditioning.

    Examples
    --------
    >>> model = UNet({"image_size": 16, "base_channels": 16, "channel_multipliers": [1, 2],
    ...               "blocks_per_level": 1, "attention_resolutions": []})
    >>> tuple(model(torch.randn(2, 3, 16, 16), torch.rand(2)).shape)
    (2, 3, 16, 16)
    """

    architecture_name = "unet"

    def __init__(self, config: Mapping[str, Any] | None = None):
        merged = {**UNetConfig, **dict(config or {})}
        super().__init__(merged)
        cfg = self.config

        self.image_size = int(cfg["image_size"])
        self.in_channels = int(cfg["in_channels"])
        self.extra_in_channels = int(cfg["extra_in_channels"])
        self.out_channels = int(cfg["out_channels"] or cfg["in_channels"])
        base = int(cfg["base_channels"])
        multipliers: Sequence[int] = list(cfg["channel_multipliers"])
        blocks_per_level = int(cfg["blocks_per_level"])
        attention_resolutions = {int(r) for r in cfg["attention_resolutions"]}
        condition_dim = base * 4
        groups = int(cfg["groups"])

        self.time_embedding = TimestepEmbedding(condition_dim)
        self.condition_mode = str(cfg["condition"]).lower()
        self.label_embedding: LabelEmbedding | None = None
        self.text_conditioner: TextConditioner | None = None
        if self.condition_mode == "class":
            if int(cfg["num_classes"]) <= 0:
                raise ValueError("condition='class' requires num_classes > 0")
            self.label_embedding = LabelEmbedding(
                int(cfg["num_classes"]), condition_dim, dropout_prob=float(cfg["class_dropout"])
            )
        elif self.condition_mode == "text":
            if int(cfg["text_vocab_size"]) <= 0:
                raise ValueError("condition='text' requires text_vocab_size > 0")
            self.text_conditioner = TextConditioner(
                int(cfg["text_vocab_size"]),
                condition_dim,
                max_len=int(cfg["text_max_len"]),
                dropout_prob=float(cfg["class_dropout"]),
            )

        self.input_conv = nn.Conv2d(self.in_channels + self.extra_in_channels, base, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        skip_channels = [base]
        channels = base
        resolution = self.image_size

        for level, multiplier in enumerate(multipliers):
            out_channels = base * int(multiplier)
            for _ in range(blocks_per_level):
                self.down_blocks.append(
                    ResBlock2d(
                        channels,
                        out_channels,
                        condition_dim=condition_dim,
                        groups=groups,
                        dropout=float(cfg["dropout"]),
                    )
                )
                self.down_attentions.append(
                    AttentionBlock2d(out_channels, int(cfg["n_heads"]), groups)
                    if resolution in attention_resolutions
                    else nn.Identity()
                )
                channels = out_channels
                skip_channels.append(channels)
            if level < len(multipliers) - 1:
                self.downsamplers.append(Downsample2d(channels))
                skip_channels.append(channels)
                resolution //= 2
            else:
                self.downsamplers.append(nn.Identity())

        self.middle_block1 = ResBlock2d(
            channels, channels, condition_dim=condition_dim, groups=groups
        )
        self.middle_attention = AttentionBlock2d(channels, int(cfg["n_heads"]), groups)
        self.middle_block2 = ResBlock2d(
            channels, channels, condition_dim=condition_dim, groups=groups
        )

        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for level, multiplier in reversed(list(enumerate(multipliers))):
            out_channels = base * int(multiplier)
            for _ in range(blocks_per_level + 1):
                skip = skip_channels.pop()
                self.up_blocks.append(
                    ResBlock2d(
                        channels + skip,
                        out_channels,
                        condition_dim=condition_dim,
                        groups=groups,
                        dropout=float(cfg["dropout"]),
                    )
                )
                self.up_attentions.append(
                    AttentionBlock2d(out_channels, int(cfg["n_heads"]), groups)
                    if resolution in attention_resolutions
                    else nn.Identity()
                )
                channels = out_channels
            if level > 0:
                self.upsamplers.append(Upsample2d(channels))
                resolution *= 2
            else:
                self.upsamplers.append(nn.Identity())

        final_groups = min(groups, channels)
        while final_groups > 1 and channels % final_groups:
            final_groups -= 1
        self.output_norm = nn.GroupNorm(final_groups, channels)
        self.output_conv = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
        self.blocks_per_level = blocks_per_level
        self.n_levels = len(multipliers)

    def build_condition(
        self, t: Tensor, *, labels: Tensor | None = None, text_tokens: Tensor | None = None
    ) -> Tensor:
        """Combine the timestep with the class/text condition."""
        condition = self.time_embedding(t)
        if self.label_embedding is not None:
            if labels is None:
                labels = torch.full(
                    (t.shape[0],),
                    self.label_embedding.null_index,
                    dtype=torch.long,
                    device=t.device,
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
        """Predict the velocity field for noisy images ``x`` at timestep ``t``."""
        if reference is not None:
            x = torch.cat([x, reference], dim=1)
        elif self.extra_in_channels:
            zeros = torch.zeros(
                x.shape[0],
                self.extra_in_channels,
                x.shape[2],
                x.shape[3],
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, zeros], dim=1)

        condition = self.build_condition(t, labels=labels, text_tokens=text_tokens)

        h = self.input_conv(x)
        skips = [h]
        index = 0
        for level in range(self.n_levels):
            for _ in range(self.blocks_per_level):
                h = self.down_blocks[index](h, condition)
                h = self.down_attentions[index](h)
                skips.append(h)
                index += 1
            downsampler = self.downsamplers[level]
            if not isinstance(downsampler, nn.Identity):
                h = downsampler(h)
                skips.append(h)

        h = self.middle_block1(h, condition)
        h = self.middle_attention(h)
        h = self.middle_block2(h, condition)

        index = 0
        for level in range(self.n_levels):
            for _ in range(self.blocks_per_level + 1):
                skip = skips.pop()
                if skip.shape[-1] != h.shape[-1]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = self.up_blocks[index](torch.cat([h, skip], dim=1), condition)
                h = self.up_attentions[index](h)
                index += 1
            upsampler = self.upsamplers[level]
            if not isinstance(upsampler, nn.Identity):
                h = upsampler(h)

        return self.output_conv(F.silu(self.output_norm(h)))

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
        """Classifier-free guidance in one batched forward pass."""
        if guidance_scale == 1.0 or (
            self.label_embedding is None and self.text_conditioner is None
        ):
            return self(x, t, labels=labels, text_tokens=text_tokens, reference=reference)

        batch = x.shape[0]
        labels_double = None
        if self.label_embedding is not None:
            null = torch.full(
                (batch,), self.label_embedding.null_index, dtype=torch.long, device=x.device
            )
            labels_double = torch.cat([labels if labels is not None else null, null], dim=0)
        text_double = (
            torch.cat([text_tokens, torch.zeros_like(text_tokens)], dim=0)
            if (self.text_conditioner is not None and text_tokens is not None)
            else None
        )
        reference_double = (
            torch.cat([reference, reference], dim=0) if reference is not None else None
        )
        prediction = self(
            torch.cat([x, x], dim=0),
            torch.cat([t, t], dim=0),
            labels=labels_double,
            text_tokens=text_double,
            reference=reference_double,
        )
        conditional, unconditional = prediction.chunk(2, dim=0)
        return unconditional + guidance_scale * (conditional - unconditional)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> UNet:
        """Build from a config mapping, ignoring bookkeeping keys."""
        payload = {k: v for k, v in dict(config).items() if k in UNetConfig}
        return cls(payload)
