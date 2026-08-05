"""Experimental transformer variants aimed at the ~1M Glint-2 budget.

Five *original* block compositions that stay recognisably Transformer (attention
+ FFN + residual) but change how information moves across depth or heads. They
exist so we can measure whether novel wiring beats the researched dense GQA
baseline at equal parameters — not as production defaults.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minimodel.architectures.base import BaseLanguageModel
from minimodel.architectures.layers import (
    CausalLocalAttention,
    KVCache,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUFeedForward,
    apply_rope,
    build_attention_mask,
    repeat_kv,
)

__all__ = [
    "BraidAttention",
    "EchoFeedForward",
    "ExperimentalTransformer",
    "ExperimentalTransformerConfig",
    "KVInheritAttention",
    "ResiMixBlock",
]

#: Defaults for every key the experimental family understands.
ExperimentalTransformerConfig: dict[str, Any] = {
    "vocab_size": 4096,
    "dim": 112,
    "n_layers": 5,
    "n_heads": 7,
    "head_dim": 16,
    "n_kv_heads": 1,
    "ffn_hidden": 256,
    "norm_eps": 1e-6,
    "bias": False,
    "window": 512,
    "local_window": 128,
    "qk_norm": True,
    "rope_base": 10000.0,
    "rope_base_slow": 500.0,
    "max_seq_len": 1024,
    "tie_embeddings": True,
    "value_residual": True,
    "echo_lora_rank": 8,
    "init_std": 0.02,
    "scale_residual_init": True,
    # resimix | kv_inherit | braid | echo_ffn | dual_rope
    "variant": "resimix",
}


class EchoFeedForward(nn.Module):
    """Shared-weight SwiGLU applied twice with a tiny LoRA between passes.

    The second pass reuses the same gate/up/down matrices (parameter tying) and
    injects a low-rank residual so the echo is not a pure fixed point. This is
    a *per-block* mini-loop over the FFN only — distinct from full depth-looping.
    """

    def __init__(self, dim: int, hidden: int, *, rank: int = 8, bias: bool = False):
        super().__init__()
        self.inner = SwiGLUFeedForward(dim, hidden, bias=bias)
        self.norm = RMSNorm(dim)
        self.lora_a = nn.Linear(dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Two tied FFN passes with a LoRA bridge."""
        y = self.inner(x)
        bridge = self.lora_b(self.lora_a(self.norm(y)))
        return self.inner(y + bridge)


class BraidAttention(nn.Module):
    """Odd heads attend locally; even heads see the full causal context.

    One layer therefore braids short-range and long-range retrieval without
    needing a separate global layer in the stack (original small-model wiring).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        *,
        bias: bool = False,
        local_window: int = 128,
        n_kv_heads: int | None = None,
        qk_norm: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if n_heads * head_dim != dim:
            raise ValueError(
                f"n_heads * head_dim must equal dim (got {n_heads} * {head_dim} != {dim})"
            )
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.n_kv_heads = int(n_kv_heads) if n_kv_heads else self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.local_window = int(local_window)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.W_qkv = nn.Linear(self.dim, self.dim + 2 * self.kv_dim, bias=bias)
        self.W_out = nn.Linear(self.dim, self.dim, bias=bias)
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        cache: KVCache | None = None,
        q_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(output, values)``; values are unused by braid callers."""
        q, k, v = self.W_qkv(x).split([self.dim, self.kv_dim, self.kv_dim], dim=-1)
        bsz, seq_len, _ = x.shape
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(k, v)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        kv_len = k.shape[2]
        # Per-head masks: even -> full causal, odd -> local window.
        full_mask = build_attention_mask(
            seq_len, kv_len, window=None, device=x.device, q_offset=q_offset
        )
        local_mask = build_attention_mask(
            seq_len, kv_len, window=self.local_window, device=x.device, q_offset=q_offset
        )
        head_mask = torch.stack(
            [full_mask if (h % 2 == 0) else local_mask for h in range(self.n_heads)], dim=0
        )  # [H, Tq, Tk]
        scores = scores.masked_fill(~head_mask.unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        return self.W_out(out), v


class KVInheritAttention(CausalLocalAttention):
    """Value residual generalised to a soft mix over *all* prior layer values.

    Standard value residual blends only with the previous layer. Here each layer
    keeps a learnable softmax over the history of value tensors, so early-layer
    values can skip directly to deep layers (original depth-routing idea).
    """

    def __init__(self, *args: Any, max_history: int = 8, **kwargs: Any):
        kwargs["value_residual"] = False
        super().__init__(*args, **kwargs)
        self.max_history = int(max_history)
        self.inherit_logits = nn.Parameter(torch.zeros(self.max_history))

    def mix_history(self, v: Tensor, history: list[Tensor]) -> Tensor:
        """Blend ``v`` with up to ``max_history`` prior value tensors."""
        if not history:
            return v
        usable = history[-self.max_history :]
        logits = self.inherit_logits[-len(usable) :]
        weights = torch.softmax(logits, dim=0)
        mixed = v
        for weight, prior in zip(weights, usable, strict=True):
            mixed = mixed + weight * (prior - mixed)
        return mixed


class ResiMixBlock(nn.Module):
    """Prenorm residual mixed with a learnable postnorm residual path.

    ``y = x + sigmoid(a) * F_pre(x) + (1-sigmoid(a)) * F_post(x)`` where ``a`` is a
    per-channel gate. Prenorm is the stable default; the postnorm path lets the
    block recover mean-preserving updates when they help — a convex mix rather
    than a hard switch.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        ffn_hidden: int,
        *,
        norm_eps: float = 1e-6,
        bias: bool = False,
        window: int | None = None,
        n_kv_heads: int | None = None,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = RMSNorm(dim, eps=norm_eps)
        self.post_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = CausalLocalAttention(
            dim,
            n_heads,
            head_dim,
            bias=bias,
            window=window,
            value_residual=False,
            n_kv_heads=n_kv_heads,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
        )
        self.ffn_pre = RMSNorm(dim, eps=norm_eps)
        self.ffn_post = RMSNorm(dim, eps=norm_eps)
        self.ffn = SwiGLUFeedForward(dim, ffn_hidden, bias=bias)
        self.attn_mix = nn.Parameter(torch.zeros(dim))
        self.ffn_mix = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        cache: KVCache | None = None,
        q_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(hidden, values)``."""
        # One attention call (cache-safe): mix the prenorm residual with a
        # post-normalised copy of the same output via a per-channel gate.
        a = torch.sigmoid(self.attn_mix)
        attn_pre, v = self.attention(self.pre_norm(x), cos, sin, cache=cache, q_offset=q_offset)
        attn_post = self.post_norm(attn_pre)
        x = x + a * attn_pre + (1.0 - a) * attn_post

        b = torch.sigmoid(self.ffn_mix)
        ffn_pre = self.ffn(self.ffn_pre(x))
        ffn_post = self.ffn_post(ffn_pre)
        return x + b * ffn_pre + (1.0 - b) * ffn_post, v


class _EchoBlock(nn.Module):
    """Attention + :class:`EchoFeedForward`."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        ffn_hidden: int,
        *,
        echo_rank: int,
        norm_eps: float = 1e-6,
        bias: bool = False,
        window: int | None = None,
        n_kv_heads: int | None = None,
        qk_norm: bool = True,
        value_residual: bool = True,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = CausalLocalAttention(
            dim,
            n_heads,
            head_dim,
            bias=bias,
            window=window,
            value_residual=value_residual,
            n_kv_heads=n_kv_heads,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
        )
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = EchoFeedForward(dim, ffn_hidden, rank=echo_rank, bias=bias)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        v_prev: Tensor | None = None,
        cache: KVCache | None = None,
        q_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        attn_out, v = self.attention(
            self.attn_norm(x), cos, sin, v_prev=v_prev, cache=cache, q_offset=q_offset
        )
        x = x + attn_out
        return x + self.ffn(self.ffn_norm(x)), v


class _BraidBlock(nn.Module):
    """Braid attention + SwiGLU."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        ffn_hidden: int,
        *,
        local_window: int,
        norm_eps: float = 1e-6,
        bias: bool = False,
        n_kv_heads: int | None = None,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = BraidAttention(
            dim,
            n_heads,
            head_dim,
            bias=bias,
            local_window=local_window,
            n_kv_heads=n_kv_heads,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
        )
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = SwiGLUFeedForward(dim, ffn_hidden, bias=bias)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        cache: KVCache | None = None,
        q_offset: int = 0,
        **_ignored: Any,
    ) -> tuple[Tensor, Tensor]:
        attn_out, v = self.attention(self.attn_norm(x), cos, sin, cache=cache, q_offset=q_offset)
        x = x + attn_out
        return x + self.ffn(self.ffn_norm(x)), v


class _KVInheritBlock(nn.Module):
    """Attention with multi-layer value inheritance + SwiGLU."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        ffn_hidden: int,
        *,
        max_history: int,
        norm_eps: float = 1e-6,
        bias: bool = False,
        window: int | None = None,
        n_kv_heads: int | None = None,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = KVInheritAttention(
            dim,
            n_heads,
            head_dim,
            bias=bias,
            window=window,
            n_kv_heads=n_kv_heads,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            max_history=max_history,
        )
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = SwiGLUFeedForward(dim, ffn_hidden, bias=bias)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        history: list[Tensor],
        cache: KVCache | None = None,
        q_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        attn_out, v = self.attention(
            self.attn_norm(x), cos, sin, cache=cache, q_offset=q_offset
        )
        v = self.attention.mix_history(v, history)
        x = x + attn_out
        return x + self.ffn(self.ffn_norm(x)), v


class _DualRoPEBlock(nn.Module):
    """Standard attention whose RoPE is a per-head mix of two bases."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        ffn_hidden: int,
        *,
        norm_eps: float = 1e-6,
        bias: bool = False,
        window: int | None = None,
        n_kv_heads: int | None = None,
        qk_norm: bool = True,
        value_residual: bool = True,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = CausalLocalAttention(
            dim,
            n_heads,
            head_dim,
            bias=bias,
            window=window,
            value_residual=value_residual,
            n_kv_heads=n_kv_heads,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
        )
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = SwiGLUFeedForward(dim, ffn_hidden, bias=bias)
        self.rope_mix = nn.Parameter(torch.zeros(n_heads))

    def forward(
        self,
        x: Tensor,
        cos_fast: Tensor,
        sin_fast: Tensor,
        cos_slow: Tensor,
        sin_slow: Tensor,
        *,
        v_prev: Tensor | None = None,
        cache: KVCache | None = None,
        q_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        gate = torch.sigmoid(self.rope_mix)
        if self.attention.n_kv_heads == self.attention.n_heads:
            g = gate.view(1, -1, 1, 1)
            cos = g * cos_fast + (1.0 - g) * cos_slow
            sin = g * sin_fast + (1.0 - g) * sin_slow
        else:
            # GQA keys have fewer heads than the per-head mix; fall back to a
            # single shared blend so RoPE still broadcasts cleanly.
            g = gate.mean()
            cos = g * cos_fast + (1.0 - g) * cos_slow
            sin = g * sin_fast + (1.0 - g) * sin_slow
        attn_out, v = self.attention(
            self.attn_norm(x), cos, sin, v_prev=v_prev, cache=cache, q_offset=q_offset
        )
        x = x + attn_out
        return x + self.ffn(self.ffn_norm(x)), v


class ExperimentalTransformer(BaseLanguageModel):
    """Dense decoder whose block wiring is selected by ``variant``."""

    architecture_name = "experimental-transformer"

    def __init__(self, config: Mapping[str, Any] | None = None):
        merged = {**ExperimentalTransformerConfig, **dict(config or {})}
        super().__init__(merged)
        cfg = self.config

        dim = int(cfg["dim"])
        n_heads = int(cfg["n_heads"])
        head_dim = int(cfg["head_dim"])
        if n_heads * head_dim != dim:
            raise ValueError(
                f"n_heads * head_dim must equal dim (got {n_heads} * {head_dim} != {dim})"
            )
        self.variant = str(cfg["variant"]).strip().lower()
        valid = {"resimix", "kv_inherit", "braid", "echo_ffn", "dual_rope"}
        if self.variant not in valid:
            raise ValueError(f"unknown experimental variant {self.variant!r}; valid: {sorted(valid)}")

        self.dim = dim
        self.n_layers = int(cfg["n_layers"])
        self.vocab_size = int(cfg["vocab_size"])
        self.max_seq_len = int(cfg["max_seq_len"])
        self.tie_embeddings = bool(cfg["tie_embeddings"])
        self.window = int(cfg["window"]) if cfg["window"] else None
        self.n_kv_heads = int(cfg["n_kv_heads"]) if cfg["n_kv_heads"] else n_heads

        self.embedding = nn.Embedding(self.vocab_size, dim)
        self.rope = RotaryEmbedding(
            head_dim, base=float(cfg["rope_base"]), max_seq_len=self.max_seq_len
        )
        self.rope_slow = (
            RotaryEmbedding(
                head_dim, base=float(cfg["rope_base_slow"]), max_seq_len=self.max_seq_len
            )
            if self.variant == "dual_rope"
            else None
        )

        common = {
            "dim": dim,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "ffn_hidden": int(cfg["ffn_hidden"]),
            "norm_eps": float(cfg["norm_eps"]),
            "bias": bool(cfg["bias"]),
            "n_kv_heads": self.n_kv_heads,
            "qk_norm": bool(cfg["qk_norm"]),
        }
        blocks: list[nn.Module] = []
        for _ in range(self.n_layers):
            if self.variant == "resimix":
                blocks.append(ResiMixBlock(**common, window=self.window))
            elif self.variant == "kv_inherit":
                blocks.append(
                    _KVInheritBlock(**common, window=self.window, max_history=self.n_layers)
                )
            elif self.variant == "braid":
                blocks.append(
                    _BraidBlock(**common, local_window=int(cfg["local_window"]))
                )
            elif self.variant == "echo_ffn":
                blocks.append(
                    _EchoBlock(
                        **common,
                        window=self.window,
                        echo_rank=int(cfg["echo_lora_rank"]),
                        value_residual=bool(cfg["value_residual"]),
                    )
                )
            else:
                blocks.append(
                    _DualRoPEBlock(
                        **common,
                        window=self.window,
                        value_residual=bool(cfg["value_residual"]),
                    )
                )
        self.blocks = nn.ModuleList(blocks)
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
                    attn = getattr(block, "attention", None)
                    if attn is not None and hasattr(attn, "W_out"):
                        attn.W_out.weight.mul_(scale)
                    ffn = getattr(block, "ffn", None)
                    if isinstance(ffn, EchoFeedForward):
                        ffn.inner.down.weight.mul_(scale)
                        nn.init.zeros_(ffn.lora_b.weight)
                    elif ffn is not None and hasattr(ffn, "down"):
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
        cos_slow = sin_slow = None
        if self.rope_slow is not None:
            cs, ss = self.rope_slow(q_offset + seq_len, device=x.device, dtype=torch.float32)
            cos_slow = cs[:, :, q_offset : q_offset + seq_len]
            sin_slow = ss[:, :, q_offset : q_offset + seq_len]

        v_prev: Tensor | None = None
        history: list[Tensor] = []
        for block in self.blocks:
            if self.variant == "dual_rope":
                assert cos_slow is not None and sin_slow is not None
                x, v_prev = block(
                    x,
                    cos,
                    sin,
                    cos_slow,
                    sin_slow,
                    v_prev=v_prev,
                    cache=cache,
                    q_offset=q_offset,
                )
            elif self.variant == "kv_inherit":
                x, v_prev = block(x, cos, sin, history=history, cache=cache, q_offset=q_offset)
                history.append(v_prev.detach())
            elif self.variant == "resimix":
                x, v_prev = block(x, cos, sin, cache=cache, q_offset=q_offset)
            else:
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
    def from_config(cls, config: Mapping[str, Any]) -> ExperimentalTransformer:
        """Build a model from a config mapping, ignoring bookkeeping keys."""
        payload = {k: v for k, v in dict(config).items() if k in ExperimentalTransformerConfig}
        return cls(payload)
