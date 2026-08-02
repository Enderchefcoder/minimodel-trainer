"""Hybrid recurrent/attention transformer (Griffin-style).

Full attention costs O(T^2) time and an O(T) cache per layer. For a small model
that wants long context, that cache dominates memory. Replacing most layers with
a gated linear recurrence gives O(T) compute and a *constant-size* state, while
keeping a few attention layers so the model can still do exact retrieval - which
is the one thing pure recurrent models are reliably bad at.

The layer pattern is configurable; the default ``["recurrent", "recurrent",
"attention"]`` repeats so one layer in three is attention, matching the ratio
Griffin found to retain attention-level quality.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minimodel.architectures.base import BaseLanguageModel
from minimodel.architectures.layers import (
    KVCache,
    RecurrentBlock,
    RMSNorm,
    RotaryEmbedding,
    TransformerBlock,
)

__all__ = ["HybridRecurrentTransformer", "HybridRecurrentConfig"]

#: Defaults for every key the hybrid architecture understands.
HybridRecurrentConfig: dict[str, Any] = {
    "vocab_size": 4096,
    "dim": 256,
    "n_layers": 9,
    "n_heads": 8,
    "head_dim": 32,
    "n_kv_heads": 2,
    "ffn_hidden": 704,
    "norm_eps": 1e-6,
    "bias": False,
    "window": 512,
    "qk_norm": True,
    "rope_base": 10000.0,
    "max_seq_len": 2048,
    "tie_embeddings": True,
    "value_residual": True,
    "recurrent_expansion": 1.0,
    "layer_pattern": ["recurrent", "recurrent", "attention"],
    "init_std": 0.02,
    "scale_residual_init": True,
}


class HybridRecurrentTransformer(BaseLanguageModel):
    """Interleaved gated-recurrence and local-attention language model.

    Examples
    --------
    >>> model = HybridRecurrentTransformer({"vocab_size": 64, "dim": 32, "n_layers": 3,
    ...                                     "n_heads": 2, "head_dim": 16,
    ...                                     "ffn_hidden": 64, "n_kv_heads": 1})
    >>> tuple(model(torch.zeros(2, 8, dtype=torch.long)).shape)
    (2, 8, 64)
    """

    architecture_name = "hybrid-recurrent"

    def __init__(self, config: Mapping[str, Any] | None = None):
        merged = {**HybridRecurrentConfig, **dict(config or {})}
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
        self.layer_types = self._expand_pattern(cfg["layer_pattern"], self.n_layers)

        self.embedding = nn.Embedding(self.vocab_size, dim)
        self.rope = RotaryEmbedding(
            head_dim, base=float(cfg["rope_base"]), max_seq_len=self.max_seq_len
        )

        blocks: list[nn.Module] = []
        for layer_type in self.layer_types:
            if layer_type == "attention":
                blocks.append(
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
                    )
                )
            else:
                blocks.append(
                    RecurrentBlock(
                        dim=dim,
                        ffn_hidden=int(cfg["ffn_hidden"]),
                        expansion=float(cfg["recurrent_expansion"]),
                        norm_eps=float(cfg["norm_eps"]),
                        bias=bool(cfg["bias"]),
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(dim, eps=float(cfg["norm_eps"]))
        self.lm_head = None if self.tie_embeddings else nn.Linear(dim, self.vocab_size, bias=False)

        self.init_weights()

    @staticmethod
    def _expand_pattern(pattern: Sequence[str] | str, n_layers: int) -> list[str]:
        """Tile ``pattern`` up to ``n_layers`` entries, validating the names."""
        if isinstance(pattern, str):
            pattern = [pattern]
        pattern = [str(p).strip().lower() for p in pattern]
        if not pattern:
            raise ValueError("layer_pattern must not be empty")
        unknown = set(pattern) - {"attention", "recurrent"}
        if unknown:
            raise ValueError(f"unknown layer types in layer_pattern: {sorted(unknown)}")
        return [pattern[i % len(pattern)] for i in range(n_layers)]

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
                    if isinstance(block, TransformerBlock):
                        block.attention.W_out.weight.mul_(scale)
                    elif isinstance(block, RecurrentBlock):
                        block.mixer.out_proj.weight.mul_(scale)
                    block.ffn.down.weight.mul_(scale)
            for module in self.modules():
                if isinstance(module, RMSNorm):
                    module.weight.fill_(1.0)
                v_logit = getattr(module, "v_lambda_logit", None)
                if isinstance(v_logit, nn.Parameter):
                    v_logit.zero_()
            # Restore the recurrence timescale spread that the generic pass
            # would otherwise have flattened.
            for block in self.blocks:
                if isinstance(block, RecurrentBlock):
                    block.mixer.log_lambda.copy_(
                        torch.linspace(-4.0, 0.0, block.mixer.log_lambda.numel())
                    )

    def forward(
        self,
        tokens: Tensor,
        *,
        return_hidden: bool = False,
        cache: KVCache | None = None,
        states: list[Tensor | None] | None = None,
        **_ignored: Any,
    ) -> Tensor:
        """Map ``[B, T]`` token ids to ``[B, T, vocab_size]`` logits.

        ``states`` optionally carries the recurrent state across chunks; it is
        updated in place so a caller can stream a long document through the
        model in fixed-size windows. When a ``cache`` is supplied without
        explicit ``states``, the cache's own state slots are used, so
        incremental decoding matches a full forward pass exactly.
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

        state_store = cache.recurrent_states if (cache is not None and states is None) else None

        v_prev: Tensor | None = None
        for idx, block in enumerate(self.blocks):
            if isinstance(block, TransformerBlock):
                x, v_prev = block(x, cos, sin, v_prev=v_prev, cache=cache, q_offset=q_offset)
            else:
                if states is not None:
                    prior = states[idx]
                elif state_store is not None:
                    prior = state_store.get(idx)
                else:
                    prior = None
                x, new_state = block(x, prior)
                if states is not None:
                    states[idx] = new_state
                elif state_store is not None:
                    state_store[idx] = new_state

        if cache is not None:
            cache.length = q_offset + seq_len

        hidden = self.final_norm(x)
        if return_hidden:
            return hidden
        if self.lm_head is None:
            return F.linear(hidden, self.embedding.weight)
        return self.lm_head(hidden)

    def new_states(self) -> list[Tensor | None]:
        """Fresh (empty) recurrent state slots, one per layer."""
        return [None] * len(self.blocks)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> HybridRecurrentTransformer:
        """Build a model from a config mapping, ignoring bookkeeping keys."""
        payload = {k: v for k, v in dict(config).items() if k in HybridRecurrentConfig}
        return cls(payload)
