"""Selective state-space (Mamba-style) language models in pure PyTorch.

Mamba's selective scan normally needs a custom CUDA kernel; at our ~1M scale a
straightforward sequential/parallel scan in float32 is fast enough for training
and keeps the repository free of compiled extensions. The five ``variant``
values below are original small-model compositions on top of that scan — they
are not clones of published Mamba configs.
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
    RMSNorm,
    RotaryEmbedding,
    SwiGLUFeedForward,
    TransformerBlock,
)

__all__ = ["MambaBlock", "MambaLM", "MambaLMConfig", "SelectiveSSM"]

#: Defaults for every key the Mamba family understands.
MambaLMConfig: dict[str, Any] = {
    "vocab_size": 4096,
    "dim": 112,
    "n_layers": 6,
    "ffn_hidden": 256,
    "state_dim": 16,
    "expand": 2,
    "conv_kernel": 4,
    "n_heads": 7,
    "head_dim": 16,
    "n_kv_heads": 1,
    "norm_eps": 1e-6,
    "bias": False,
    "window": 512,
    "qk_norm": True,
    "rope_base": 10000.0,
    "max_seq_len": 1024,
    "tie_embeddings": True,
    "value_residual": True,
    "init_std": 0.02,
    "scale_residual_init": True,
    # pure | attn_tail | multihead | conv_gate | braid
    "variant": "pure",
    "attn_tail_layers": 2,
    "layer_pattern": ["ssm", "ssm", "attention"],
}


class SelectiveSSM(nn.Module):
    """Real-valued selective state space with an optional depthwise short conv.

    The recurrence is the standard Mamba discretisation on a diagonal ``A``:

    ``h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t ⊙ x_t``
    ``y_t = C_t · h_t + D ⊙ x_t``

    ``Δ``, ``B`` and ``C`` are input-dependent (the "selection"); ``A`` and ``D``
    are learned parameters. When ``n_heads > 1`` the inner width is partitioned
    into independent heads that share the same state size — a cheap multi-head
    SSM that is *not* the published Mamba-2 SSD formulation.
    """

    def __init__(
        self,
        dim: int,
        *,
        state_dim: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        n_heads: int = 1,
        bias: bool = False,
        use_conv: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.state_dim = int(state_dim)
        self.expand = int(expand)
        self.inner = self.dim * self.expand
        self.n_heads = int(n_heads)
        if self.inner % self.n_heads != 0:
            raise ValueError(
                f"expanded dim {self.inner} must be divisible by n_heads {self.n_heads}"
            )
        self.head_dim = self.inner // self.n_heads
        self.use_conv = bool(use_conv)
        self.conv_kernel = int(conv_kernel)

        self.in_proj = nn.Linear(self.dim, 2 * self.inner, bias=bias)
        self.dt_proj = nn.Linear(self.inner, self.inner, bias=True)
        self.bc_proj = nn.Linear(self.inner, 2 * self.n_heads * self.state_dim, bias=False)
        # A_log kept negative via -exp so the continuous A is stable.
        self.A_log = nn.Parameter(
            torch.log(torch.linspace(1.0, 16.0, self.n_heads * self.state_dim)).view(
                self.n_heads, self.state_dim
            )
        )
        self.D = nn.Parameter(torch.ones(self.inner))
        if self.use_conv and self.conv_kernel > 0:
            self.conv_weight = nn.Parameter(
                torch.empty(self.inner, 1, self.conv_kernel)
            )
            self.conv_bias = nn.Parameter(torch.zeros(self.inner))
            nn.init.normal_(self.conv_weight, std=0.02)
        else:
            self.conv_weight = None
            self.conv_bias = None
        self.out_proj = nn.Linear(self.inner, self.dim, bias=bias)

    def _causal_conv(self, u: Tensor, conv_state: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """Depthwise causal conv with an explicit left-pad state for streaming."""
        if self.conv_weight is None:
            return u, None
        # u: [B, T, C] ; conv_state: [B, C, K-1]
        u_t = u.transpose(1, 2)
        kernel = self.conv_kernel
        if conv_state is None:
            u_t = F.pad(u_t, (kernel - 1, 0))
        else:
            u_t = torch.cat([conv_state, u_t], dim=2)
        out = F.conv1d(u_t, self.conv_weight, self.conv_bias, groups=self.inner)
        new_state = u_t[:, :, -(kernel - 1) :].detach() if kernel > 1 else None
        return out.transpose(1, 2), new_state

    def _scan(
        self, x: Tensor, delta: Tensor, B: Tensor, C: Tensor, state: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Run the selective recurrence over ``x`` shaped ``[B, T, H, Dh]``.

        State layout is ``[B, H, N, Dh]`` so each channel keeps an independent
        ``N``-dimensional SSM. ``A`` broadcasts as ``[H, N, 1]``.
        """
        batch, seq_len, n_heads, head_dim = x.shape
        A = -torch.exp(self.A_log.float())  # [H, N]
        dA = torch.exp(delta.unsqueeze(3).float() * A.unsqueeze(-1))  # [B,T,H,N,Dh]
        dB = delta.unsqueeze(3).float() * B.unsqueeze(-1).float()  # [B,T,H,N,Dh]

        h = (
            torch.zeros(
                batch, n_heads, self.state_dim, head_dim, device=x.device, dtype=torch.float32
            )
            if state is None
            else state.float()
        )
        ys: list[Tensor] = []
        x_f = x.float()
        C_f = C.float()
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x_f[:, t].unsqueeze(2)
            y_t = (C_f[:, t].unsqueeze(-1) * h).sum(dim=2)  # [B,H,Dh]
            ys.append(y_t)
        y = torch.stack(ys, dim=1).to(x.dtype)
        return y, h.to(x.dtype)

    def forward(
        self, x: Tensor, state: tuple[Tensor | None, Tensor | None] | Tensor | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor | None]]:
        """Return ``(output, (ssm_state, conv_state))`` for ``x`` shaped ``[B, T, dim]``."""
        if isinstance(state, tuple):
            ssm_state, conv_state = state
        else:
            ssm_state, conv_state = state, None

        gate_x, u = self.in_proj(x).chunk(2, dim=-1)
        gate = F.silu(gate_x)
        u, new_conv = self._causal_conv(u, conv_state)
        u = F.silu(u)

        delta = F.softplus(self.dt_proj(u))
        bc = self.bc_proj(u)
        B_par, C_par = bc.chunk(2, dim=-1)
        batch, seq_len, _ = u.shape
        u_h = u.view(batch, seq_len, self.n_heads, self.head_dim)
        delta_h = delta.view(batch, seq_len, self.n_heads, self.head_dim)
        B_h = B_par.view(batch, seq_len, self.n_heads, self.state_dim)
        C_h = C_par.view(batch, seq_len, self.n_heads, self.state_dim)

        y, new_ssm = self._scan(u_h, delta_h, B_h, C_h, ssm_state)
        y = y.reshape(batch, seq_len, self.inner)
        y = y + u * self.D
        return self.out_proj(y * gate), (new_ssm, new_conv)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, inner={self.inner}, state_dim={self.state_dim}, "
            f"n_heads={self.n_heads}, conv={self.use_conv}"
        )


class MambaBlock(nn.Module):
    """Pre-norm selective-SSM + SwiGLU block."""

    def __init__(
        self,
        dim: int,
        ffn_hidden: int,
        *,
        state_dim: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        n_heads: int = 1,
        norm_eps: float = 1e-6,
        bias: bool = False,
        use_conv: bool = True,
    ):
        super().__init__()
        self.mixer_norm = RMSNorm(dim, eps=norm_eps)
        self.mixer = SelectiveSSM(
            dim,
            state_dim=state_dim,
            expand=expand,
            conv_kernel=conv_kernel,
            n_heads=n_heads,
            bias=bias,
            use_conv=use_conv,
        )
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = SwiGLUFeedForward(dim, ffn_hidden, bias=bias)

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return ``(hidden_states, ssm_state)``."""
        mixed, new_state = self.mixer(self.mixer_norm(x), state)
        x = x + mixed
        return x + self.ffn(self.ffn_norm(x)), new_state


class MambaLM(BaseLanguageModel):
    """Decoder-only language model built from :class:`MambaBlock` (+ optional attention).

    Variants
    --------
    ``pure``
        All layers are selective SSM blocks.
    ``attn_tail``
        SSM trunk followed by ``attn_tail_layers`` of local attention (retrieval
        coda — original small-model composition).
    ``multihead``
        Same as ``pure`` but each SSM uses ``n_heads`` independent state heads.
    ``conv_gate``
        Explicit short-conv gating (``use_conv=True``); the default pure variant
        can disable conv, this one forces it and uses expand=2.
    ``braid``
        Alternating SSM / attention following ``layer_pattern``.
    """

    architecture_name = "mamba-lm"

    def __init__(self, config: Mapping[str, Any] | None = None):
        merged = {**MambaLMConfig, **dict(config or {})}
        super().__init__(merged)
        cfg = self.config

        dim = int(cfg["dim"])
        self.dim = dim
        self.n_layers = int(cfg["n_layers"])
        self.vocab_size = int(cfg["vocab_size"])
        self.max_seq_len = int(cfg["max_seq_len"])
        self.tie_embeddings = bool(cfg["tie_embeddings"])
        self.variant = str(cfg["variant"]).strip().lower()
        valid = {"pure", "attn_tail", "multihead", "conv_gate", "braid"}
        if self.variant not in valid:
            raise ValueError(f"unknown mamba variant {self.variant!r}; valid: {sorted(valid)}")

        n_heads = int(cfg["n_heads"])
        head_dim = int(cfg["head_dim"])
        if n_heads * head_dim != dim:
            raise ValueError(
                f"n_heads * head_dim must equal dim (got {n_heads} * {head_dim} != {dim})"
            )
        self.n_kv_heads = int(cfg["n_kv_heads"]) if cfg["n_kv_heads"] else n_heads
        self.window = int(cfg["window"]) if cfg["window"] else None

        ssm_heads = n_heads if self.variant == "multihead" else 1
        use_conv = self.variant != "pure" or int(cfg["conv_kernel"]) > 0
        if self.variant == "pure":
            # Pure keeps conv (classic Mamba block) unless conv_kernel is 0.
            use_conv = int(cfg["conv_kernel"]) > 0
        if self.variant == "conv_gate":
            use_conv = True

        self.embedding = nn.Embedding(self.vocab_size, dim)
        self.rope = RotaryEmbedding(
            head_dim, base=float(cfg["rope_base"]), max_seq_len=self.max_seq_len
        )

        self.layer_types = self._plan_layers(cfg)
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
                    MambaBlock(
                        dim=dim,
                        ffn_hidden=int(cfg["ffn_hidden"]),
                        state_dim=int(cfg["state_dim"]),
                        expand=int(cfg["expand"]),
                        conv_kernel=int(cfg["conv_kernel"]),
                        n_heads=ssm_heads,
                        norm_eps=float(cfg["norm_eps"]),
                        bias=bool(cfg["bias"]),
                        use_conv=use_conv,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(dim, eps=float(cfg["norm_eps"]))
        self.lm_head = None if self.tie_embeddings else nn.Linear(dim, self.vocab_size, bias=False)
        self.init_weights()

    def _plan_layers(self, cfg: Mapping[str, Any]) -> list[str]:
        """Decide the per-layer mixer type from ``variant``."""
        n = self.n_layers
        if self.variant in {"pure", "multihead", "conv_gate"}:
            return ["ssm"] * n
        if self.variant == "attn_tail":
            tail = min(int(cfg["attn_tail_layers"]), n - 1)
            return ["ssm"] * (n - tail) + ["attention"] * tail
        # braid
        return self._expand_pattern(cfg["layer_pattern"], n)

    @staticmethod
    def _expand_pattern(pattern: Sequence[str] | str, n_layers: int) -> list[str]:
        if isinstance(pattern, str):
            pattern = [pattern]
        pattern = [str(p).strip().lower() for p in pattern]
        if not pattern:
            raise ValueError("layer_pattern must not be empty")
        unknown = set(pattern) - {"attention", "ssm", "recurrent"}
        if unknown:
            raise ValueError(f"unknown layer types in layer_pattern: {sorted(unknown)}")
        # Treat "recurrent" as an alias for ssm in this family.
        normalised = ["ssm" if p == "recurrent" else p for p in pattern]
        return [normalised[i % len(normalised)] for i in range(n_layers)]

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
                    elif isinstance(block, MambaBlock):
                        block.mixer.out_proj.weight.mul_(scale)
                    block.ffn.down.weight.mul_(scale)
            for module in self.modules():
                if isinstance(module, RMSNorm):
                    module.weight.fill_(1.0)
                v_logit = getattr(module, "v_lambda_logit", None)
                if isinstance(v_logit, nn.Parameter):
                    v_logit.zero_()
            # Re-assert SSM A_log / D after the generic pass.
            for block in self.blocks:
                if isinstance(block, MambaBlock):
                    n_heads = block.mixer.n_heads
                    state_dim = block.mixer.state_dim
                    block.mixer.A_log.copy_(
                        torch.log(torch.linspace(1.0, 16.0, n_heads * state_dim)).view(
                            n_heads, state_dim
                        )
                    )
                    block.mixer.D.fill_(1.0)

    def forward(
        self,
        tokens: Tensor,
        *,
        return_hidden: bool = False,
        cache: KVCache | None = None,
        states: list[Tensor | None] | None = None,
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
        """Fresh (empty) SSM state slots, one per layer."""
        return [None] * len(self.blocks)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> MambaLM:
        """Build a model from a config mapping, ignoring bookkeeping keys."""
        payload = {k: v for k, v in dict(config).items() if k in MambaLMConfig}
        return cls(payload)
