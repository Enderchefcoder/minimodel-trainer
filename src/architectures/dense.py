"""A conventional decoder-only transformer.

This is the "boring" baseline: N distinct pre-norm blocks with RMSNorm, RoPE and
SwiGLU, optionally tying the LM head to the input embedding. It exists so that
looped-architecture results can be compared against a familiar reference at
matched parameter count or matched compute, and so that anyone who just wants a
small GPT can use this repository without adopting the looped design.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minimodel.architectures.base import BaseLanguageModel
from minimodel.architectures.layers import KVCache, RMSNorm, RotaryEmbedding, TransformerBlock

__all__ = ["DenseTransformer", "DenseTransformerConfig"]

#: Defaults for every key the dense architecture understands.
DenseTransformerConfig: dict[str, Any] = {
    "vocab_size": 4096,
    "dim": 256,
    "n_layers": 6,
    "n_heads": 8,
    "head_dim": 32,
    "n_kv_heads": None,
    "ffn_hidden": 704,
    "norm_eps": 1e-6,
    "bias": False,
    "window": None,
    "window_pattern": None,
    "qk_norm": True,
    "rope_base": 10000.0,
    "max_seq_len": 1024,
    "tie_embeddings": True,
    "value_residual": False,
    "init_std": 0.02,
    "scale_residual_init": True,
}


class DenseTransformer(BaseLanguageModel):
    """Standard stacked decoder-only transformer.

    Examples
    --------
    >>> model = DenseTransformer({"vocab_size": 64, "dim": 32, "n_layers": 2,
    ...                           "n_heads": 2, "head_dim": 16, "ffn_hidden": 64})
    >>> tuple(model(torch.zeros(2, 4, dtype=torch.long)).shape)
    (2, 4, 64)
    """

    architecture_name = "dense-transformer"

    def __init__(self, config: Mapping[str, Any] | None = None):
        merged = {**DenseTransformerConfig, **dict(config or {})}
        super().__init__(merged)
        cfg = self.config

        dim = int(cfg["dim"])
        n_heads = int(cfg["n_heads"])
        head_dim = int(cfg["head_dim"])
        if n_heads * head_dim != dim:
            raise ValueError(
                f"n_heads * head_dim must equal dim (got {n_heads} * {head_dim} != {dim})"
            )
        if int(cfg["n_layers"]) < 1:
            raise ValueError("n_layers must be at least 1")

        self.dim = dim
        self.n_layers = int(cfg["n_layers"])
        self.window = int(cfg["window"]) if cfg["window"] else None
        self.vocab_size = int(cfg["vocab_size"])
        self.max_seq_len = int(cfg["max_seq_len"])
        self.tie_embeddings = bool(cfg["tie_embeddings"])
        self.n_kv_heads = int(cfg["n_kv_heads"]) if cfg["n_kv_heads"] else n_heads

        self.embedding = nn.Embedding(self.vocab_size, dim)
        self.rope = RotaryEmbedding(
            head_dim, base=float(cfg["rope_base"]), max_seq_len=self.max_seq_len
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    ffn_hidden=int(cfg["ffn_hidden"]),
                    norm_eps=float(cfg["norm_eps"]),
                    bias=bool(cfg["bias"]),
                    window=self._layer_window(layer_idx),
                    value_residual=bool(cfg["value_residual"]),
                    n_kv_heads=self.n_kv_heads,
                    qk_norm=bool(cfg["qk_norm"]),
                )
                for layer_idx in range(self.n_layers)
            ]
        )
        self.final_norm = RMSNorm(dim, eps=float(cfg["norm_eps"]))
        if self.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(dim, self.vocab_size, bias=False)

        self.init_weights()

    def _layer_window(self, layer_idx: int) -> int | None:
        """Attention window for ``layer_idx``.

        ``window_pattern`` implements interleaved local/global attention: with
        ``window_pattern: 4`` every fourth layer sees the full context and the
        rest are limited to ``window`` tokens. That keeps most of the quality of
        full attention while cutting the decoding cache substantially.
        """
        pattern = self.config.get("window_pattern")
        if not pattern or self.window is None:
            return self.window
        return None if (layer_idx + 1) % int(pattern) == 0 else self.window

    def init_weights(self) -> None:
        """Normal init for all projections, with optional depth-scaled residuals.

        Scaling the output projection of each residual branch by
        ``1/sqrt(2 * n_layers)`` keeps the variance of the residual stream from
        growing with depth, which is the standard GPT-2 initialisation trick.
        """
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

    def forward(
        self,
        tokens: Tensor,
        *,
        return_hidden: bool = False,
        cache: KVCache | None = None,
        **_ignored: Any,
    ) -> Tensor:
        """Map ``[B, T]`` token ids to ``[B, T, vocab_size]`` logits.

        Extra keyword arguments (such as ``loops``) are accepted and ignored so
        that trainers can call every architecture with the same signature.
        """
        if tokens.dim() != 2:
            raise ValueError(f"expected tokens of shape [B, T], got {tuple(tokens.shape)}")
        seq_len = tokens.shape[1]
        q_offset = cache.length if cache is not None else 0
        if cache is not None:
            cache.begin_forward()

        x = self.embedding(tokens)
        cos_full, sin_full = self.rope(q_offset + seq_len, device=x.device, dtype=torch.float32)
        cos = cos_full[:, :, q_offset : q_offset + seq_len]
        sin = sin_full[:, :, q_offset : q_offset + seq_len]

        v_prev: Tensor | None = None
        for block in self.blocks:
            x, v_prev = block(x, cos, sin, v_prev=v_prev, cache=cache, q_offset=q_offset)

        if cache is not None:
            cache.length = q_offset + seq_len

        hidden = self.final_norm(x)
        if return_hidden:
            return hidden
        if self.lm_head is None:
            return F.linear(hidden, self.embedding.weight)
        return self.lm_head(hidden)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DenseTransformer:
        """Build a model from a config mapping, ignoring bookkeeping keys."""
        payload = {k: v for k, v in dict(config).items() if k in DenseTransformerConfig}
        return cls(payload)

    def expected_parameter_count(self) -> int:
        """Analytic parameter count for budget checks."""
        cfg = self.config
        dim = int(cfg["dim"])
        vocab = int(cfg["vocab_size"])
        hidden = int(cfg["ffn_hidden"])
        head_dim = int(cfg["head_dim"])
        kv_dim = self.n_kv_heads * head_dim
        per_block = (
            dim * (dim + 2 * kv_dim)  # W_qkv
            + dim * dim  # W_out
            + (self.n_kv_heads if cfg["value_residual"] else 0)
            + (2 * head_dim if cfg["qk_norm"] else 0)
            + 2 * dim  # two RMSNorm gains
            + dim * 2 * hidden
            + hidden * dim
        )
        total = vocab * dim + self.n_layers * per_block + dim
        if not self.tie_embeddings:
            total += vocab * dim
        return total
