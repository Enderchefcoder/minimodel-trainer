"""Autoregressive pixel-art transformer.

PixelGPT generates images the way a language model generates text: one pixel at
a time, left to right and top to bottom, over a *palette* vocabulary. For pixel
art that is the right formulation rather than a compromise:

* Pixel art has a small, discrete palette. A diffusion model works in continuous
  colour space and then has to be quantised back, which smears the crisp colour
  boundaries that define the style. An autoregressive model over palette indices
  produces exactly-on-palette output by construction.
* 24x24 is 576 tokens - a perfectly ordinary sequence length. The quadratic cost
  that makes autoregressive generation impractical at 512x512 simply does not
  apply.
* Sampling temperature gives direct, interpretable control over how adventurous
  the output is, and a low temperature reliably produces clean sprites.

The default configuration is tuned for the 24x24 sprite datasets (such as
``unstonio/pixelgpt-24x24-20k``) and lands at roughly 10M parameters.

Position information is provided by *separate row and column embeddings* added
to each token. A single flat position embedding would force the model to learn
that positions 24 apart are vertically adjacent; giving it row and column
directly means the two-dimensional structure is available from step one, and it
is what makes vertical symmetry - which almost every sprite has - learnable in a
model this size.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minimodel.architectures.layers import KVCache, RMSNorm, RotaryEmbedding, TransformerBlock
from minimodel.vision.architectures.base import BaseImageModel

__all__ = ["PixelGPT", "PixelGPTConfig"]

#: Defaults for every key PixelGPT understands. The defaults are the 24x24,
#: 64-colour, ~10M-parameter configuration.
PixelGPTConfig: dict[str, Any] = {
    "image_size": 24,
    "palette_size": 64,
    "dim": 320,
    "n_layers": 8,
    "n_heads": 8,
    "head_dim": 40,
    "n_kv_heads": 4,
    "ffn_hidden": 864,
    "norm_eps": 1e-6,
    "qk_norm": True,
    "num_classes": 0,
    "class_dropout": 0.1,
    "text_vocab_size": 0,
    "text_max_len": 24,
    "init_std": 0.02,
    "scale_residual_init": True,
}

#: Reserved ids prepended to the palette vocabulary.
BOS_ID = 0
PAD_ID = 1
N_SPECIAL_TOKENS = 2


class PixelGPT(BaseImageModel):
    """Causal transformer over palette-indexed pixels.

    Examples
    --------
    >>> model = PixelGPT({"image_size": 4, "palette_size": 8, "dim": 32,
    ...                   "n_layers": 2, "n_heads": 2, "head_dim": 16,
    ...                   "n_kv_heads": 1, "ffn_hidden": 64})
    >>> pixels = torch.zeros(2, 16, dtype=torch.long)
    >>> tuple(model(pixels).shape)
    (2, 17, 10)
    """

    architecture_name = "pixelgpt"

    def __init__(self, config: Mapping[str, Any] | None = None):
        merged = {**PixelGPTConfig, **dict(config or {})}
        super().__init__(merged)
        cfg = self.config

        dim = int(cfg["dim"])
        n_heads = int(cfg["n_heads"])
        head_dim = int(cfg["head_dim"])
        if n_heads * head_dim != dim:
            raise ValueError(
                f"n_heads * head_dim must equal dim (got {n_heads} * {head_dim} != {dim})"
            )

        self.image_size = int(cfg["image_size"])
        self.palette_size = int(cfg["palette_size"])
        self.n_pixels = self.image_size**2
        self.vocab_size = self.palette_size + N_SPECIAL_TOKENS
        self.dim = dim
        self.n_layers = int(cfg["n_layers"])
        self.n_kv_heads = int(cfg["n_kv_heads"]) or n_heads
        self.num_classes = int(cfg["num_classes"])

        self.token_embedding = nn.Embedding(self.vocab_size, dim)
        # Row and column embeddings expose the 2D grid structure directly.
        self.row_embedding = nn.Embedding(self.image_size, dim)
        self.col_embedding = nn.Embedding(self.image_size, dim)
        self.start_embedding = nn.Parameter(torch.zeros(1, 1, dim))

        self.class_embedding: nn.Embedding | None = None
        if self.num_classes > 0:
            # One extra slot is the unconditional class.
            self.class_embedding = nn.Embedding(self.num_classes + 1, dim)
        self.class_dropout = float(cfg["class_dropout"])

        self.rope = RotaryEmbedding(head_dim, max_seq_len=self.n_pixels + 2)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    ffn_hidden=int(cfg["ffn_hidden"]),
                    norm_eps=float(cfg["norm_eps"]),
                    window=None,
                    value_residual=True,
                    n_kv_heads=self.n_kv_heads,
                    qk_norm=bool(cfg["qk_norm"]),
                )
                for _ in range(self.n_layers)
            ]
        )
        self.final_norm = RMSNorm(dim, eps=float(cfg["norm_eps"]))
        self.lm_head = nn.Linear(dim, self.vocab_size, bias=False)

        self.register_buffer(
            "row_index", torch.arange(self.n_pixels) // self.image_size, persistent=False
        )
        self.register_buffer(
            "col_index", torch.arange(self.n_pixels) % self.image_size, persistent=False
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Normal init with depth-scaled residual projections."""
        std = float(self.config["init_std"])

        def _init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        self.apply(_init)
        with torch.no_grad():
            if bool(self.config["scale_residual_init"]):
                scale = (2.0 * self.n_layers) ** -0.5
                for block in self.blocks:
                    block.attention.W_out.weight.mul_(scale)
                    block.ffn.down.weight.mul_(scale)
            for module in self.modules():
                if isinstance(module, RMSNorm):
                    module.weight.fill_(1.0)
                v_logit = getattr(module, "v_lambda_logit", None)
                if isinstance(v_logit, nn.Parameter):
                    v_logit.zero_()
            self.start_embedding.zero_()

    def _position_embedding(self, start: int, length: int, device: torch.device) -> Tensor:
        """Row+column embeddings for pixel positions ``[start, start + length)``."""
        indices = torch.arange(start, start + length, device=device).clamp(max=self.n_pixels - 1)
        rows = self.row_index.to(device)[indices]
        cols = self.col_index.to(device)[indices]
        return (self.row_embedding(rows) + self.col_embedding(cols)).unsqueeze(0)

    def forward(
        self,
        pixels: Tensor,
        *,
        labels: Tensor | None = None,
        cache: KVCache | None = None,
        prefix_only: bool = False,
    ) -> Tensor:
        """Predict the next palette index at every position.

        Parameters
        ----------
        pixels:
            ``[B, N]`` palette indices in ``[0, palette_size)``. The sequence is
            internally prefixed with a start token, so the returned logits have
            length ``N + 1`` and position ``i`` predicts pixel ``i``.
        labels:
            Optional ``[B]`` class labels for conditional generation.
        cache:
            Decoding cache. When present, ``pixels`` holds only the new tokens.
        """
        if pixels.dim() != 2:
            raise ValueError(f"expected pixels of shape [B, N], got {tuple(pixels.shape)}")
        batch = pixels.shape[0]
        device = pixels.device
        q_offset = cache.length if cache is not None else 0
        if cache is not None:
            cache.begin_forward()

        if q_offset == 0:
            start = self.start_embedding.expand(batch, 1, -1)
            if self.class_embedding is not None:
                if labels is None:
                    labels = torch.full((batch,), self.num_classes, dtype=torch.long, device=device)
                elif self.training and self.class_dropout > 0:
                    drop = torch.rand(batch, device=device) < self.class_dropout
                    labels = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
                start = start + self.class_embedding(labels).unsqueeze(1)
            tokens = self.token_embedding(pixels + N_SPECIAL_TOKENS)
            tokens = tokens + self._position_embedding(0, pixels.shape[1], device)
            x = torch.cat([start, tokens], dim=1)
        else:
            x = self.token_embedding(pixels + N_SPECIAL_TOKENS)
            x = x + self._position_embedding(q_offset - 1, pixels.shape[1], device)

        seq_len = x.shape[1]
        cos_full, sin_full = self.rope(q_offset + seq_len, device=device, dtype=torch.float32)
        cos = cos_full[:, :, q_offset : q_offset + seq_len]
        sin = sin_full[:, :, q_offset : q_offset + seq_len]

        v_prev: Tensor | None = None
        for block in self.blocks:
            x, v_prev = block(x, cos, sin, v_prev=v_prev, cache=cache, q_offset=q_offset)

        if cache is not None:
            cache.length = q_offset + seq_len

        hidden = self.final_norm(x)
        if prefix_only:
            hidden = hidden[:, -1:]
        return self.lm_head(hidden)

    def loss(self, pixels: Tensor, labels: Tensor | None = None) -> Tensor:
        """Cross-entropy over all pixel positions.

        ``pixels`` is ``[B, N]`` palette indices; the loss compares the model's
        ``N`` predictions against them.
        """
        logits = self(pixels, labels=labels)[:, :-1]
        targets = pixels + N_SPECIAL_TOKENS
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))

    @torch.no_grad()
    def generate(
        self,
        n_samples: int = 1,
        *,
        labels: Tensor | None = None,
        temperature: float = 0.9,
        top_k: int = 0,
        top_p: float = 0.0,
        prompt: Tensor | None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Sample complete images, returning ``[B, H, W]`` palette indices.

        ``prompt`` optionally fixes the first ``k`` pixels, which is how partial
        images are completed (give it the top half of a sprite and it finishes
        the bottom).
        """
        device = torch.device(device) if device is not None else next(self.parameters()).device
        was_training = self.training
        self.eval()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        cache = KVCache()
        prefix = (
            prompt.to(device)
            if prompt is not None
            else torch.zeros(n_samples, 0, dtype=torch.long, device=device)
        )
        if prefix.dim() == 1:
            prefix = prefix.unsqueeze(0).expand(n_samples, -1)
        n_samples = prefix.shape[0]

        generated = prefix.clone()
        logits = self(generated, labels=labels, cache=cache)[:, -1, :]

        for _ in range(self.n_pixels - generated.shape[1]):
            # Special tokens are never valid pixel values.
            step_logits = logits[:, N_SPECIAL_TOKENS:].float()
            if temperature <= 0:
                next_pixel = step_logits.argmax(dim=-1, keepdim=True)
            else:
                step_logits = step_logits / temperature
                if top_k > 0:
                    k = min(top_k, step_logits.shape[-1])
                    threshold = torch.topk(step_logits, k, dim=-1).values[..., -1:]
                    step_logits = step_logits.masked_fill(step_logits < threshold, float("-inf"))
                if 0.0 < top_p < 1.0:
                    ordered, indices = torch.sort(step_logits, descending=True, dim=-1)
                    probabilities = torch.softmax(ordered, dim=-1)
                    cumulative = probabilities.cumsum(dim=-1) - probabilities
                    remove = cumulative >= top_p
                    remove[..., 0] = False
                    mask = torch.zeros_like(remove).scatter(-1, indices, remove)
                    step_logits = step_logits.masked_fill(mask, float("-inf"))
                probabilities = torch.softmax(step_logits, dim=-1)
                next_pixel = torch.multinomial(probabilities, 1, generator=generator)

            generated = torch.cat([generated, next_pixel], dim=1)
            logits = self(next_pixel, labels=labels, cache=cache)[:, -1, :]

        self.train(was_training)
        return generated.view(n_samples, self.image_size, self.image_size)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PixelGPT:
        """Build from a config mapping, ignoring bookkeeping keys."""
        payload = {k: v for k, v in dict(config).items() if k in PixelGPTConfig}
        return cls(payload)
