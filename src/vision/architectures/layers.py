"""Building blocks shared by the image models.

Two families of layer live here:

* transformer pieces for the diffusion transformer (patch embedding,
  bidirectional attention, adaptive layer norm conditioning);
* convolutional pieces for the UNet and the autoencoder (residual blocks,
  resampling, attention over feature maps).

The conditioning mechanism is worth calling out. Diffusion models must be told
*when* they are in the denoising process, and usually also *what* to generate.
Both signals are injected with adaptive layer norm (AdaLN-Zero): the condition
vector predicts a per-channel scale and shift for each normalisation, plus a
gate on each residual branch that starts at zero. Starting the gates at zero
means every block begins as an identity function, so a freshly initialised model
is a well-behaved identity map rather than noise - which is what makes these
models trainable without warmup tricks.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "AttentionBlock2d",
    "DiTBlock",
    "Downsample2d",
    "FinalLayer",
    "LabelEmbedding",
    "PatchEmbed",
    "ResBlock2d",
    "SelfAttention",
    "TextConditioner",
    "TimestepEmbedding",
    "Upsample2d",
    "modulate",
    "unpatchify",
]


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply an AdaLN shift and scale to ``[B, N, C]`` activations."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep features followed by a small MLP.

    Sinusoidal features give the network a smooth, multi-scale encoding of the
    noise level; the MLP lets it learn a useful basis on top of that.
    """

    def __init__(self, dim: int, *, frequency_dim: int = 256, max_period: float = 10000.0):
        super().__init__()
        self.dim = int(dim)
        self.frequency_dim = int(frequency_dim)
        self.max_period = float(max_period)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

    def timestep_features(self, t: Tensor) -> Tensor:
        """Sinusoidal encoding of ``t``, shaped ``[B, frequency_dim]``."""
        half = self.frequency_dim // 2
        exponents = torch.arange(half, dtype=torch.float32, device=t.device) / half
        frequencies = torch.exp(-math.log(self.max_period) * exponents)
        angles = t.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if self.frequency_dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        """Embed a batch of timesteps into ``[B, dim]``."""
        return self.mlp(self.timestep_features(t))


class LabelEmbedding(nn.Module):
    """Class-conditioning embedding with a dedicated "unconditional" slot.

    The extra slot is what makes classifier-free guidance possible: during
    training a fraction of labels are replaced by it, so at sampling time the
    model can produce both a conditional and an unconditional prediction and the
    two can be extrapolated apart.
    """

    def __init__(self, num_classes: int, dim: int, *, dropout_prob: float = 0.1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.dropout_prob = float(dropout_prob)
        self.embedding = nn.Embedding(self.num_classes + 1, dim)

    @property
    def null_index(self) -> int:
        """Index of the unconditional embedding."""
        return self.num_classes

    def forward(self, labels: Tensor, *, train: bool | None = None) -> Tensor:
        """Embed labels, randomly dropping some to the null class in training."""
        train = self.training if train is None else train
        if train and self.dropout_prob > 0:
            drop = torch.rand(labels.shape, device=labels.device) < self.dropout_prob
            labels = torch.where(drop, torch.full_like(labels, self.null_index), labels)
        return self.embedding(labels)


class TextConditioner(nn.Module):
    """Turns a token sequence into one conditioning vector.

    A full text encoder would dwarf the models this package trains, so the
    conditioner is deliberately tiny: token embeddings, a couple of transformer
    layers, and mean pooling. That is enough to distinguish the short
    instructions and captions these image models are trained on, and it trains
    jointly with the diffusion model instead of needing a separate pretrained
    encoder.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 64,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.dropout_prob = float(dropout_prob)
        self.token_embedding = nn.Embedding(self.vocab_size, dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_len, dim))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=n_heads,
                    dim_feedforward=dim * 4,
                    batch_first=True,
                    norm_first=True,
                    dropout=0.0,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.null_embedding = nn.Parameter(torch.zeros(dim))

    def forward(self, tokens: Tensor | None, batch_size: int = 1, device=None) -> Tensor:
        """Encode ``[B, T]`` tokens into ``[B, dim]``; ``None`` gives the null vector."""
        if tokens is None:
            device = device or self.null_embedding.device
            return self.null_embedding.unsqueeze(0).expand(batch_size, -1).to(device)

        tokens = tokens[:, : self.max_len]
        x = self.token_embedding(tokens) + self.position_embedding[:, : tokens.shape[1]]
        mask = tokens == 0
        for block in self.blocks:
            x = block(x, src_key_padding_mask=mask)
        x = self.norm(x)
        # Mean-pool over non-padding positions.
        keep = (~mask).float().unsqueeze(-1)
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)

        if self.training and self.dropout_prob > 0:
            drop = torch.rand(pooled.shape[0], device=pooled.device) < self.dropout_prob
            pooled = torch.where(drop.unsqueeze(-1), self.null_embedding.unsqueeze(0), pooled)
        return pooled


class PatchEmbed(nn.Module):
    """Split an image into patches and project each to a token."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
            )
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.n_patches = self.grid_size**2
        self.projection = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        """``[B, C, H, W]`` -> ``[B, n_patches, dim]``."""
        return self.projection(x).flatten(2).transpose(1, 2)


def unpatchify(x: Tensor, patch_size: int, out_channels: int, grid_size: int) -> Tensor:
    """Inverse of :class:`PatchEmbed`: ``[B, N, p*p*C]`` -> ``[B, C, H, W]``."""
    batch = x.shape[0]
    x = x.reshape(batch, grid_size, grid_size, patch_size, patch_size, out_channels)
    x = torch.einsum("bhwpqc->bchpwq", x)
    return x.reshape(batch, out_channels, grid_size * patch_size, grid_size * patch_size)


class SelfAttention(nn.Module):
    """Bidirectional multi-head self-attention over a token sequence."""

    def __init__(self, dim: int, n_heads: int, *, qkv_bias: bool = True, qk_norm: bool = True):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.n_heads = int(n_heads)
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.projection = nn.Linear(dim, dim)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        """``[B, N, C]`` -> ``[B, N, C]``."""
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.projection(out.transpose(1, 2).reshape(b, n, c))


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning.

    The conditioning vector produces six values per channel: shift, scale and
    gate for the attention branch and the same for the MLP branch. Both gates
    are initialised to zero.
    """

    def __init__(self, dim: int, n_heads: int, *, mlp_ratio: float = 4.0, qk_norm: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = SelfAttention(dim, n_heads, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim)
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Apply attention and MLP, modulated by ``condition``."""
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(condition).chunk(6, dim=-1)
        x = x + gate1.unsqueeze(1) * self.attention(modulate(self.norm1(x), shift1, scale1))
        return x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))


class FinalLayer(nn.Module):
    """Final AdaLN + linear projection back to patch pixels."""

    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """``[B, N, C]`` -> ``[B, N, p*p*out_channels]``."""
        shift, scale = self.adaln(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class ResBlock2d(nn.Module):
    """Convolutional residual block with optional timestep conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        condition_dim: int | None = None,
        groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        groups = min(groups, in_channels, out_channels)
        while groups > 1 and (in_channels % groups or out_channels % groups):
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.condition = (
            nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, out_channels * 2))
            if condition_dim
            else None
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: Tensor, condition: Tensor | None = None) -> Tensor:
        """Apply the block, conditioning on ``condition`` when provided."""
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        if self.condition is not None and condition is not None:
            scale, shift = self.condition(condition).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
            h = h * (1.0 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttentionBlock2d(nn.Module):
    """Self-attention over the spatial positions of a feature map."""

    def __init__(self, channels: int, n_heads: int = 4, groups: int = 8):
        super().__init__()
        groups = min(groups, channels)
        while groups > 1 and channels % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.n_heads = max(1, min(n_heads, channels // 8 or 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.projection = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: Tensor) -> Tensor:
        """``[B, C, H, W]`` -> ``[B, C, H, W]``."""
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b, 3, self.n_heads, c // self.n_heads, h * w).permute(1, 0, 2, 4, 3)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return x + self.projection(out)


class Downsample2d(nn.Module):
    """Strided convolution that halves spatial resolution."""

    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channels or channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Halve height and width."""
        return self.conv(x)


class Upsample2d(nn.Module):
    """Nearest-neighbour upsample followed by a convolution.

    Preferred over a transposed convolution because it does not produce the
    checkerboard artifacts that are very visible at small image sizes.
    """

    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channels or channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Double height and width."""
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


def sincos_position_embedding(dim: int, grid_size: int) -> Tensor:
    """Fixed 2D sine-cosine position embeddings, shaped ``[1, grid^2, dim]``.

    Fixed rather than learned so a model trained at one resolution can be
    sampled at another by regenerating the table.
    """
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D sincos embeddings, got {dim}")
    quarter = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(quarter, dtype=torch.float32) / quarter))
    coords = torch.arange(grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

    def _encode(values: Tensor) -> Tensor:
        angles = values.reshape(-1, 1) * omega.reshape(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    embedding = torch.cat([_encode(grid_y), _encode(grid_x)], dim=1)
    return embedding.unsqueeze(0)
