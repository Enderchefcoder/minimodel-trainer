"""Faithful reimplementation of the *released* Glint-2 checkpoint.

Important: the public `generate.py` on the Glint-2 repo describes a "pure-loop,
1.06M, zero unique layers" model, but the released checkpoint
(`glint-2.pt`) actually carries `coda_layers: 1` and a full `coda.0.*` block,
totalling 1,710,049 parameters. `generate.py` as published cannot load the
checkpoint with `strict=True` (the `coda.*` keys are unexpected). This module
matches the *weights that ship*, so we can evaluate the real artifact.

The forward pass is: embed -> [shared block looped `loops` times, each with a
per-loop embedding and a rank-4 LoRA on the QKV delta] -> coda block -> final
RMSNorm -> tied LM head. Architecture details (RoPE even/odd interleave, causal
window-256 attention, SwiGLU FFN, RMSNorm eps = torch default 1e-5) are copied
verbatim from the repo's `generate.py`.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["Glint2", "load_glint2"]

ATTENTION_WINDOW = 256


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Even/odd interleaved RoPE, exactly as in the Glint-2 release."""
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1
    ).flatten(-2)


class SwiGlu(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin, qkv_delta):
        b, t, dim = x.shape
        qkv = self.qkv(x) + qkv_delta
        q, k, v = qkv.split(dim, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        pos = torch.arange(t, device=x.device)
        mask = (pos[None, :] <= pos[:, None]) & (pos[:, None] - pos[None, :] < ATTENTION_WINDOW)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=1.0 / math.sqrt(self.head_dim)
        )
        return self.out(attended.transpose(1, 2).reshape(b, t, dim))


class Block(nn.Module):
    def __init__(self, dim, n_heads, ffn_hidden):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = SwiGlu(dim, ffn_hidden)

    def forward(self, x, cos, sin, qkv_delta):
        x = x + self.attn(self.attn_norm(x), cos, sin, qkv_delta)
        return x + self.ffn(self.ffn_norm(x))


class LoopLora(nn.Module):
    def __init__(self, dim, rank, max_loops):
        super().__init__()
        self.down = nn.ModuleList(nn.Linear(dim, rank, bias=False) for _ in range(max_loops))
        self.up = nn.ModuleList(nn.Linear(rank, 3 * dim, bias=False) for _ in range(max_loops))

    def forward(self, x, loop_index):
        i = min(loop_index, len(self.down) - 1)
        return self.up[i](self.down[i](x))


class Indexer(nn.Module):
    """Vestigial in the release: a single gate parameter, unused in the forward."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor([0.1]))


class Glint2(nn.Module):
    """The released Glint-2: looped shared block + a unique coda block."""

    def __init__(self, cfg: dict, max_loops: int):
        super().__init__()
        self.cfg = cfg
        self.max_loops = max_loops
        self.loops = int(cfg.get("shared_loops_runtime", 8))
        dim, n_heads, ffn = cfg["dim"], cfg["n_heads"], cfg["ffn_hidden"]
        self.embed = nn.Embedding(cfg["vocab_size"], dim)
        self.indexer = Indexer()
        self.shared = Block(dim, n_heads, ffn)
        self.loop_lora = LoopLora(dim, cfg["lora_rank"], max_loops)
        self.loop_embed = nn.Embedding(max_loops, dim)
        self.coda = nn.ModuleList(
            Block(dim, n_heads, ffn) for _ in range(int(cfg.get("coda_layers", 1)))
        )
        self.final_norm = nn.RMSNorm(dim)

    def rope(self, seq_len: int):
        head_dim = self.cfg["dim"] // self.cfg["n_heads"]
        positions = torch.arange(seq_len, dtype=torch.float32)
        inv_freq = 1.0 / (
            self.cfg["rope_base"] ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        angles = torch.outer(positions, inv_freq)
        return torch.cos(angles), torch.sin(angles)

    def forward(self, tokens, loops: int | None = None, return_hidden: bool = False):
        loops = self.loops if loops is None else loops
        cos, sin = self.rope(tokens.shape[1])
        x = self.embed(tokens)
        for loop_index in range(loops):
            clamped = min(loop_index, self.max_loops - 1)
            gated = x + self.loop_embed.weight[clamped]
            delta = self.loop_lora(gated, loop_index)
            x = self.shared(gated, cos, sin, delta)
        zero_delta = torch.zeros(
            tokens.shape[0], tokens.shape[1], 3 * self.cfg["dim"], device=x.device, dtype=x.dtype
        )
        for block in self.coda:
            x = block(x, cos, sin, zero_delta)
        x = self.final_norm(x)
        if return_hidden:
            return x
        return F.linear(x, self.embed.weight)

    def num_parameters(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total


def load_glint2(path: str | Path = "research/baselines/glint2/glint-2.pt", loops: int = 8) -> Glint2:
    """Load the released checkpoint into the faithful module (strict=True)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["model_config"])
    cfg["shared_loops_runtime"] = loops
    max_loops = ck["model"]["loop_embed.weight"].shape[0]
    model = Glint2(cfg, max_loops=max_loops)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model
