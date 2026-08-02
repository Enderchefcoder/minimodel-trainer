"""Sparse mixture-of-experts transformer.

A MoE model has a large *total* parameter count but only activates a small
fraction of it per token. That trade is unusually good for small models trained
on a fixed token budget: memory is cheap, FLOPs are the bottleneck, and an
8-expert top-2 model gets most of the quality of a dense model 3-4x its active
size at roughly the dense model's training cost.

Two design choices differ from a textbook MoE and both matter at this scale:

* a **shared expert** is always active, so no token can be left with only a
  poorly-routed transformation;
* load balancing is **auxiliary-loss free** (a bias on the routing logits used
  for selection only), which avoids trading language-modelling quality for
  balance the way an auxiliary loss term does.

The ``moe_layer_pattern`` option keeps the first layers dense. Early layers
route poorly because their representations are still mostly positional, so
spending experts there is wasteful.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minimodel.architectures.base import BaseLanguageModel
from minimodel.architectures.layers import (
    KVCache,
    MoEFeedForward,
    RMSNorm,
    RotaryEmbedding,
    TransformerBlock,
)

__all__ = ["MoETransformer", "MoETransformerConfig"]

#: Defaults for every key the MoE architecture understands.
MoETransformerConfig: dict[str, Any] = {
    "vocab_size": 4096,
    "dim": 256,
    "n_layers": 8,
    "n_heads": 8,
    "head_dim": 32,
    "n_kv_heads": 2,
    "ffn_hidden": 352,
    "norm_eps": 1e-6,
    "bias": False,
    "window": None,
    "qk_norm": True,
    "rope_base": 10000.0,
    "max_seq_len": 1024,
    "tie_embeddings": True,
    "value_residual": True,
    "init_std": 0.02,
    "scale_residual_init": True,
    # Mixture-of-experts settings
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "top_k": 2,
    "balance_rate": 1e-3,
    "normalize_gates": True,
    #: Layers before this index stay dense; -1 or 0 makes every layer sparse.
    "first_moe_layer": 1,
}


class MoETransformer(BaseLanguageModel):
    """Decoder-only transformer whose later layers use sparse MoE FFNs.

    Examples
    --------
    >>> model = MoETransformer({"vocab_size": 64, "dim": 32, "n_layers": 2,
    ...                         "n_heads": 2, "head_dim": 16, "ffn_hidden": 48,
    ...                         "n_routed_experts": 4, "top_k": 2, "n_kv_heads": 1})
    >>> tuple(model(torch.zeros(2, 6, dtype=torch.long)).shape)
    (2, 6, 64)
    """

    architecture_name = "moe-transformer"

    def __init__(self, config: Mapping[str, Any] | None = None):
        merged = {**MoETransformerConfig, **dict(config or {})}
        super().__init__(merged)
        cfg = self.config

        dim = int(cfg["dim"])
        n_heads = int(cfg["n_heads"])
        head_dim = int(cfg["head_dim"])
        if n_heads * head_dim != dim:
            raise ValueError(
                f"n_heads * head_dim must equal dim (got {n_heads} * {head_dim} != {dim})"
            )

        self.dim = dim
        self.n_layers = int(cfg["n_layers"])
        self.vocab_size = int(cfg["vocab_size"])
        self.max_seq_len = int(cfg["max_seq_len"])
        self.window = int(cfg["window"]) if cfg["window"] else None
        self.tie_embeddings = bool(cfg["tie_embeddings"])
        self.n_kv_heads = int(cfg["n_kv_heads"]) if cfg["n_kv_heads"] else n_heads
        self.first_moe_layer = max(0, int(cfg["first_moe_layer"]))

        moe_kwargs = {
            "n_routed": int(cfg["n_routed_experts"]),
            "n_shared": int(cfg["n_shared_experts"]),
            "top_k": int(cfg["top_k"]),
            "balance_rate": float(cfg["balance_rate"]),
            "normalize_gates": bool(cfg["normalize_gates"]),
        }

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
                    window=self.window,
                    value_residual=bool(cfg["value_residual"]),
                    n_kv_heads=self.n_kv_heads,
                    qk_norm=bool(cfg["qk_norm"]),
                    moe=moe_kwargs if layer_idx >= self.first_moe_layer else None,
                )
                for layer_idx in range(self.n_layers)
            ]
        )
        self.final_norm = RMSNorm(dim, eps=float(cfg["norm_eps"]))
        self.lm_head = None if self.tie_embeddings else nn.Linear(dim, self.vocab_size, bias=False)

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
                    ffn = block.ffn
                    if isinstance(ffn, MoEFeedForward):
                        for expert in list(ffn.experts) + list(ffn.shared_experts):
                            expert.down.weight.mul_(scale)
                    else:
                        ffn.down.weight.mul_(scale)
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
        """Map ``[B, T]`` token ids to ``[B, T, vocab_size]`` logits."""
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

    # ------------------------------------------------------------------
    def active_parameters(self) -> int:
        """Parameters actually used for a single token.

        Total parameters overstate compute for a sparse model; this is the
        number to compare against a dense baseline.
        """
        total = self.num_parameters()
        for block in self.blocks:
            ffn = block.ffn
            if isinstance(ffn, MoEFeedForward):
                per_expert = sum(p.numel() for p in ffn.experts[0].parameters())
                inactive = (ffn.n_routed - ffn.top_k) * per_expert
                total -= inactive
        return total

    def routing_stats(self) -> dict[str, float]:
        """Aggregate load-balance statistics across all MoE layers."""
        stats: dict[str, float] = {}
        counts = 0
        for idx, block in enumerate(self.blocks):
            ffn = block.ffn
            if not isinstance(ffn, MoEFeedForward):
                continue
            counts += 1
            for key, value in ffn.load_balance_stats().items():
                stats[key] = stats.get(key, 0.0) + value
            stats[f"layer{idx}_max_over_mean"] = ffn.load_balance_stats()["max_over_mean"]
        if counts:
            for key in ("max_over_mean", "entropy", "dead_experts"):
                if key in stats:
                    stats[key] /= counts
        return stats

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> MoETransformer:
        """Build a model from a config mapping, ignoring bookkeeping keys."""
        payload = {k: v for k, v in dict(config).items() if k in MoETransformerConfig}
        return cls(payload)
