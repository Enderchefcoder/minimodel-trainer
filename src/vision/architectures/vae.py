"""A small convolutional autoencoder for latent diffusion.

Diffusion in pixel space is affordable up to about 64x64. Beyond that, most of
the model's capacity goes into reproducing high-frequency detail that a simple
autoencoder can reconstruct almost perfectly. Compressing to a latent grid
(typically 8x smaller per side, so 64x fewer positions) lets the diffusion model
spend its capacity on structure instead.

The encoder outputs a diagonal Gaussian. A *small* KL weight is used - large
enough to keep the latent scale bounded and roughly unit-variance, small enough
that reconstruction stays sharp. Latents are then multiplied by
``scaling_factor`` so the diffusion model sees data with approximately unit
variance, which is what its noise schedule assumes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minimodel.vision.architectures.base import BaseImageModel
from minimodel.vision.architectures.layers import (
    AttentionBlock2d,
    Downsample2d,
    ResBlock2d,
    Upsample2d,
)

__all__ = ["VAE", "VAEConfig", "VAEOutput"]

#: Defaults for every key the VAE understands.
VAEConfig: dict[str, Any] = {
    "image_size": 64,
    "in_channels": 3,
    "latent_channels": 4,
    "base_channels": 64,
    "channel_multipliers": [1, 2, 4],
    "blocks_per_level": 2,
    "attention_at_bottleneck": True,
    "groups": 8,
    "scaling_factor": 0.18215,
}


@dataclass
class VAEOutput:
    """Reconstruction plus the pieces needed for the loss."""

    reconstruction: Tensor
    latent: Tensor
    mean: Tensor
    logvar: Tensor

    def kl_divergence(self) -> Tensor:
        """KL to a standard normal, averaged over the batch."""
        return (
            0.5
            * torch.sum(
                self.mean.pow(2) + self.logvar.exp() - 1.0 - self.logvar, dim=[1, 2, 3]
            ).mean()
        )


class VAE(BaseImageModel):
    """Convolutional variational autoencoder.

    Examples
    --------
    >>> vae = VAE({"image_size": 16, "base_channels": 16, "channel_multipliers": [1, 2],
    ...            "blocks_per_level": 1})
    >>> out = vae(torch.randn(2, 3, 16, 16))
    >>> tuple(out.reconstruction.shape)
    (2, 3, 16, 16)
    """

    architecture_name = "vae"

    def __init__(self, config: Mapping[str, Any] | None = None):
        merged = {**VAEConfig, **dict(config or {})}
        super().__init__(merged)
        cfg = self.config

        self.in_channels = int(cfg["in_channels"])
        self.latent_channels = int(cfg["latent_channels"])
        self.scaling_factor = float(cfg["scaling_factor"])
        base = int(cfg["base_channels"])
        multipliers: Sequence[int] = list(cfg["channel_multipliers"])
        blocks_per_level = int(cfg["blocks_per_level"])
        groups = int(cfg["groups"])
        self.downsample_factor = 2 ** (len(multipliers) - 1)

        encoder: list[nn.Module] = [nn.Conv2d(self.in_channels, base, 3, padding=1)]
        channels = base
        for level, multiplier in enumerate(multipliers):
            out_channels = base * int(multiplier)
            for _ in range(blocks_per_level):
                encoder.append(ResBlock2d(channels, out_channels, groups=groups))
                channels = out_channels
            if level < len(multipliers) - 1:
                encoder.append(Downsample2d(channels))
        if bool(cfg["attention_at_bottleneck"]):
            encoder.append(AttentionBlock2d(channels, groups=groups))
        encoder.append(ResBlock2d(channels, channels, groups=groups))
        self.encoder = nn.Sequential(*encoder)

        bottleneck_groups = min(groups, channels)
        while bottleneck_groups > 1 and channels % bottleneck_groups:
            bottleneck_groups -= 1
        self.encoder_norm = nn.GroupNorm(bottleneck_groups, channels)
        self.to_latent = nn.Conv2d(channels, self.latent_channels * 2, 1)
        self.from_latent = nn.Conv2d(self.latent_channels, channels, 1)

        decoder: list[nn.Module] = [ResBlock2d(channels, channels, groups=groups)]
        if bool(cfg["attention_at_bottleneck"]):
            decoder.append(AttentionBlock2d(channels, groups=groups))
        for level, multiplier in reversed(list(enumerate(multipliers))):
            out_channels = base * int(multiplier)
            for _ in range(blocks_per_level):
                decoder.append(ResBlock2d(channels, out_channels, groups=groups))
                channels = out_channels
            if level > 0:
                decoder.append(Upsample2d(channels))
        self.decoder = nn.Sequential(*decoder)

        final_groups = min(groups, channels)
        while final_groups > 1 and channels % final_groups:
            final_groups -= 1
        self.decoder_norm = nn.GroupNorm(final_groups, channels)
        self.output_conv = nn.Conv2d(channels, self.in_channels, 3, padding=1)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return the ``(mean, logvar)`` of the latent distribution."""
        h = self.to_latent(F.silu(self.encoder_norm(self.encoder(x))))
        mean, logvar = h.chunk(2, dim=1)
        # Clamping keeps the KL term finite if the encoder briefly diverges.
        return mean, logvar.clamp(-30.0, 20.0)

    def sample_latent(self, mean: Tensor, logvar: Tensor, *, deterministic: bool = False) -> Tensor:
        """Reparameterised sample (or the mean when ``deterministic``)."""
        if deterministic:
            return mean
        return mean + torch.randn_like(mean) * (0.5 * logvar).exp()

    def decode(self, latent: Tensor) -> Tensor:
        """Decode a latent grid back to an image in ``[-1, 1]``."""
        h = self.decoder(self.from_latent(latent))
        return torch.tanh(self.output_conv(F.silu(self.decoder_norm(h))))

    def forward(self, x: Tensor, *, deterministic: bool = False) -> VAEOutput:
        """Encode, sample and decode."""
        mean, logvar = self.encode(x)
        latent = self.sample_latent(mean, logvar, deterministic=deterministic)
        return VAEOutput(
            reconstruction=self.decode(latent), latent=latent, mean=mean, logvar=logvar
        )

    def encode_for_diffusion(self, x: Tensor, *, deterministic: bool = True) -> Tensor:
        """Encode and rescale to roughly unit variance for a diffusion model."""
        mean, logvar = self.encode(x)
        return self.sample_latent(mean, logvar, deterministic=deterministic) * self.scaling_factor

    def decode_from_diffusion(self, latent: Tensor) -> Tensor:
        """Undo :meth:`encode_for_diffusion` and decode."""
        return self.decode(latent / self.scaling_factor)

    def loss(
        self, x: Tensor, *, kl_weight: float = 1e-6, deterministic: bool = False
    ) -> tuple[Tensor, dict[str, float]]:
        """Reconstruction + KL loss.

        L1 reconstruction is used rather than L2: it is less tolerant of the
        uniform blur that L2 accepts, which matters for small images where a
        blurry reconstruction is immediately obvious.
        """
        output = self(x, deterministic=deterministic)
        reconstruction_loss = F.l1_loss(output.reconstruction, x)
        kl = output.kl_divergence()
        total = reconstruction_loss + kl_weight * kl
        return total, {
            "reconstruction_loss": float(reconstruction_loss.detach()),
            "kl": float(kl.detach()),
            "latent_std": float(output.latent.std().detach()),
        }

    def latent_shape(self, image_size: int | None = None) -> tuple[int, int, int]:
        """Latent shape ``(C, H, W)`` for a given image size."""
        size = int(image_size or self.config["image_size"])
        grid = size // self.downsample_factor
        return (self.latent_channels, grid, grid)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> VAE:
        """Build from a config mapping, ignoring bookkeeping keys."""
        payload = {k: v for k, v in dict(config).items() if k in VAEConfig}
        return cls(payload)
