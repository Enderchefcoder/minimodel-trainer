"""Reusable transformer building blocks.

Every layer here is deliberately small and explicit so that the YAML
architecture templates in ``src/architectures/templates`` can be read as a
faithful description of the code:

* :class:`RMSNorm` - root-mean-square normalisation with a configurable epsilon.
* :class:`RotaryEmbedding` / :func:`apply_rope` - even/odd interleaved RoPE.
* :class:`SwiGLUFeedForward` - gated FFN with a fused gate/up projection.
* :class:`FactorizedEmbedding` - low-rank ``[V, r] @ [r, d]`` embedding whose
  factors are reused by the LM head, which is where most of the parameter
  savings in a very small model come from.
* :class:`CausalLocalAttention` - sliding-window causal attention with grouped
  key/value heads, optional QK normalisation, optional value residuals and a
  per-call QKV delta (used by the looped architecture).
* :class:`MoEFeedForward` - sparse mixture-of-experts FFN with a shared expert
  and bias-based, auxiliary-loss-free load balancing.
* :class:`GatedRecurrentUnit` - a real-diagonal gated linear recurrence (the
  Hawk/Griffin RG-LRU) evaluated with a parallel log-space scan.
* :class:`KVCache` - incremental decoding cache shared by all architectures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "CausalLocalAttention",
    "FactorizedEmbedding",
    "GatedRecurrentUnit",
    "KVCache",
    "MoEFeedForward",
    "RMSNorm",
    "RecurrentBlock",
    "RotaryEmbedding",
    "SwiGLUFeedForward",
    "TransformerBlock",
    "apply_rope",
    "build_attention_mask",
    "repeat_kv",
]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm.

    Parameters
    ----------
    dim:
        Feature dimension.
    eps:
        Added to the mean square before the reciprocal square root. The bundled
        templates use ``1e-6`` rather than PyTorch's ``1e-5`` default, so the
        value is always passed explicitly.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: Tensor) -> Tensor:
        """Normalise the last dimension of ``x`` and apply the learned gain."""
        dtype = x.dtype
        x32 = x.float()
        norm = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.to(dtype)) * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings with even/odd interleaved pairing.

    The cosine/sine tables have shape ``[1, 1, T, head_dim // 2]`` so they
    broadcast over ``[B, n_heads, T, head_dim]`` activations. Tables are cached
    and only rebuilt when a longer sequence, a new device or a new dtype is
    requested.
    """

    def __init__(self, head_dim: int, base: float = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = int(head_dim)
        self.base = float(base)
        self.max_seq_len = int(max_seq_len)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len = 0
        self._cos: Tensor | None = None
        self._sin: Tensor | None = None

    def _build(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq.to(device=device, dtype=torch.float32))
        self._cos = angles.cos().to(dtype)[None, None, :, :]
        self._sin = angles.sin().to(dtype)[None, None, :, :]
        self._cached_len = seq_len

    def forward(
        self, seq_len: int, *, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> tuple[Tensor, Tensor]:
        """Return ``(cos, sin)`` tables covering ``seq_len`` positions."""
        needs_rebuild = (
            self._cos is None
            or self._cached_len < seq_len
            or self._cos.device != device
            or self._cos.dtype != dtype
        )
        if needs_rebuild:
            self._build(max(seq_len, self._cached_len), device, dtype)
        assert self._cos is not None and self._sin is not None  # for type checkers
        return self._cos[:, :, :seq_len], self._sin[:, :, :seq_len]

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, base={self.base}, max_seq_len={self.max_seq_len}"


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``x`` in-plane using even/odd channel pairs.

    Parameters
    ----------
    x:
        Activations shaped ``[B, H, T, head_dim]``.
    cos, sin:
        Tables shaped ``[1, 1, T, head_dim // 2]``.

    Notes
    -----
    The pairing convention is ``(x[..., 0::2], x[..., 1::2])``. It must match the
    convention used when the model was trained, which is why it lives in one
    shared function rather than being re-derived per architecture.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


def build_attention_mask(
    q_len: int,
    k_len: int,
    *,
    window: int | None,
    device: torch.device,
    q_offset: int = 0,
) -> Tensor:
    """Build a boolean ``[q_len, k_len]`` mask where ``True`` means "attend".

    ``q_offset`` is the absolute position of the first query, which is non-zero
    during incremental decoding.

    The rule is ``(k_pos <= q_pos) & (q_pos - k_pos < window)``: causal, plus an
    optional sliding window. ``window=None`` disables the window.
    """
    q_pos = torch.arange(q_len, device=device).unsqueeze(1) + q_offset
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)
    mask = k_pos <= q_pos
    if window is not None and window > 0:
        mask = mask & ((q_pos - k_pos) < window)
    return mask


@dataclass
class KVCache:
    """Append-only key/value cache for incremental decoding.

    A cache holds one slot per attention *call site*. Because the looped
    architecture calls its shared blocks several times per forward pass, slots
    are addressed by call order rather than by module identity; call
    :meth:`begin_forward` before each forward pass to rewind the slot cursor.
    """

    keys: list[Tensor] = field(default_factory=list)
    values: list[Tensor] = field(default_factory=list)
    length: int = 0
    max_length: int | None = None
    #: Side-channel for architectures with non-attention state (hybrid models
    #: store their recurrent hidden state here, keyed by layer index).
    recurrent_states: dict[int, Tensor] = field(default_factory=dict)
    _cursor: int = 0

    def begin_forward(self) -> None:
        """Rewind the per-call-site cursor before a new forward pass."""
        self._cursor = 0

    def reset(self) -> None:
        """Drop all cached tensors."""
        self.keys.clear()
        self.values.clear()
        self.recurrent_states.clear()
        self.length = 0
        self._cursor = 0

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Append ``k``/``v`` for the next call site and return the full history."""
        idx = self._cursor
        self._cursor += 1
        if idx < len(self.keys):
            k = torch.cat([self.keys[idx], k], dim=2)
            v = torch.cat([self.values[idx], v], dim=2)
            if self.max_length is not None and k.shape[2] > self.max_length:
                k = k[:, :, -self.max_length :]
                v = v[:, :, -self.max_length :]
            self.keys[idx] = k
            self.values[idx] = v
        else:
            if self.max_length is not None and k.shape[2] > self.max_length:
                k = k[:, :, -self.max_length :]
                v = v[:, :, -self.max_length :]
            self.keys.append(k)
            self.values.append(v)
        return k, v

    @property
    def n_slots(self) -> int:
        """How many attention call sites have been recorded."""
        return len(self.keys)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network with a single fused gate/up projection.

    ``down(silu(gate) * up)`` where ``gate, up = chunk(gate_up(x), 2, dim=-1)``.
    """

    def __init__(self, dim: int, hidden: int, *, bias: bool = False):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.gate_up = nn.Linear(self.dim, 2 * self.hidden, bias=bias)
        self.down = nn.Linear(self.hidden, self.dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the gated projection."""
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, hidden={self.hidden}"


class FactorizedEmbedding(nn.Module):
    """Low-rank token embedding whose factors are shared with the LM head.

    The embedding is ``E[V, r]`` followed by ``proj[r, d]``. Instead of tying a
    full ``[V, d]`` matrix, both factors are reused, which keeps the LM head free
    while cutting the embedding cost from ``V*d`` to ``V*r + r*d``.

    For ``V=4096``, ``d=128``, ``r=64`` that is 270K parameters instead of 524K.
    """

    def __init__(self, vocab_size: int, rank: int, dim: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)
        self.dim = int(dim)
        self.E = nn.Embedding(self.vocab_size, self.rank)
        self.proj = nn.Linear(self.rank, self.dim, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        """Embed ``[B, T]`` token ids into ``[B, T, dim]``."""
        return self.proj(self.E(tokens))

    def logits(self, hidden: Tensor, *, materialize: bool = False) -> Tensor:
        """Project hidden states back to vocabulary logits.

        Two mathematically equivalent paths are available:

        ``materialize=False`` (default, best for training)
            ``x @ proj.weight.T @ E.weight.T`` computed as two small matmuls,
            which never allocates the full ``[V, d]`` matrix.
        ``materialize=True`` (best for single-token decoding)
            builds ``E.weight @ proj.weight.T`` once and does a single matmul.
            The result is a temporary tensor and must never be registered as a
            parameter, or the tying and the parameter budget would break.
        """
        if materialize:
            full = self.E.weight @ self.proj.weight.T
            return F.linear(hidden, full)
        z = F.linear(hidden, self.proj.weight.T)
        return F.linear(z, self.E.weight)

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, rank={self.rank}, dim={self.dim}"


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand grouped key/value heads to match the number of query heads.

    ``x`` is ``[B, n_kv_heads, T, head_dim]``; the output repeats each KV head
    ``n_rep`` times along the head axis.
    """
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class CausalLocalAttention(nn.Module):
    """Causal attention with grouped KV heads and an optional sliding window.

    Beyond textbook attention this supports four options that all pay for
    themselves at small scale:

    ``n_kv_heads`` (grouped-query attention)
        Fewer key/value heads than query heads. Shrinks both the QKV projection
        and the decoding cache; at these model sizes the quality cost is
        negligible while decode memory drops by the grouping factor.
    ``qk_norm``
        RMSNorm on queries and keys before the dot product. This is the cheapest
        known fix for attention-logit blow-up, which is the most common cause of
        loss spikes when training small models at high learning rates.
    ``qkv_delta``
        A per-call additive term on the fused QKV projection. The looped model
        uses it to inject a rank-4 LoRA that differs per loop iteration, giving
        each iteration its own effective attention weights at negligible cost.
    ``value residual``
        A learned per-head blend ``v = lam * v + (1 - lam) * v_prev`` with the
        values of the previous attention layer. This shortens the gradient path
        to early layers and measurably stabilises deep weight-shared stacks.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        *,
        bias: bool = False,
        window: int | None = None,
        value_residual: bool = True,
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
        self.window = window
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.W_qkv = nn.Linear(self.dim, self.dim + 2 * self.kv_dim, bias=bias)
        self.W_out = nn.Linear(self.dim, self.dim, bias=bias)
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.value_residual = bool(value_residual)
        if self.value_residual:
            self.v_lambda_logit = nn.Parameter(torch.zeros(self.n_kv_heads))
        else:
            self.register_parameter("v_lambda_logit", None)

    def _split_heads(self, x: Tensor, n_heads: int) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        qkv_delta: Tensor | None = None,
        window: int | None = None,
        v_prev: Tensor | None = None,
        cache: KVCache | None = None,
        q_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Run attention.

        Returns
        -------
        tuple
            ``(output, v)`` where ``v`` is the *mixed*, pre-expansion value
            tensor, which the caller threads into the next layer as ``v_prev``.
        """
        raw_qkv = self.W_qkv(x)
        if qkv_delta is not None:
            raw_qkv = raw_qkv + qkv_delta
        q, k, v = raw_qkv.split([self.dim, self.kv_dim, self.kv_dim], dim=-1)
        q = self._split_heads(q, self.n_heads)
        k = self._split_heads(k, self.n_kv_heads)
        v = self._split_heads(v, self.n_kv_heads)

        if self.value_residual and v_prev is not None and self.v_lambda_logit is not None:
            lam = torch.sigmoid(self.v_lambda_logit).view(1, self.n_kv_heads, 1, 1).to(v.dtype)
            v = lam * v + (1.0 - lam) * v_prev

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Values are never rotated; only queries and keys carry position.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        mixed_v = v
        if cache is not None:
            k, v = cache.update(k, v)

        eff_window = self.window if window is None else window
        mask = build_attention_mask(
            q.shape[2], k.shape[2], window=eff_window, device=q.device, q_offset=q_offset
        )
        attn = F.scaled_dot_product_attention(
            q, repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep), attn_mask=mask, scale=self.scale
        )
        out = self.W_out(self._merge_heads(attn))
        return out, mixed_v

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, "
            f"head_dim={self.head_dim}, window={self.window}, qk_norm={self.qk_norm}, "
            f"value_residual={self.value_residual}"
        )


class MoEFeedForward(nn.Module):
    """Sparse mixture-of-experts FFN with a shared expert.

    Design notes
    ------------
    * ``n_shared`` experts are always active. A shared expert captures the
      "everyone needs this" computation so the routed experts can specialise,
      and it removes the failure mode where a token routed to a bad expert gets
      no useful transformation at all.
    * Routing is top-``k`` over ``n_routed`` experts with softmax gates.
    * Load balancing is **auxiliary-loss free**: a per-expert bias is added to
      the routing logits *for selection only* and nudged up or down based on
      observed load. Because the bias never enters the gate value, it balances
      the experts without injecting gradient noise into the language-modelling
      objective the way an auxiliary loss does.

    The forward pass loops over experts rather than over tokens, so cost scales
    with the number of experts, not the batch size.
    """

    def __init__(
        self,
        dim: int,
        hidden: int,
        *,
        n_routed: int = 8,
        n_shared: int = 1,
        top_k: int = 2,
        bias: bool = False,
        balance_rate: float = 1e-3,
        normalize_gates: bool = True,
    ):
        super().__init__()
        if top_k < 1 or top_k > n_routed:
            raise ValueError(f"top_k must be in [1, n_routed], got {top_k} with {n_routed} experts")
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.n_routed = int(n_routed)
        self.n_shared = int(n_shared)
        self.top_k = int(top_k)
        self.balance_rate = float(balance_rate)
        self.normalize_gates = bool(normalize_gates)

        self.router = nn.Linear(self.dim, self.n_routed, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUFeedForward(self.dim, self.hidden, bias=bias) for _ in range(self.n_routed)]
        )
        self.shared_experts = nn.ModuleList(
            [SwiGLUFeedForward(self.dim, self.hidden, bias=bias) for _ in range(self.n_shared)]
        )
        # Buffers, not parameters: updated by a rule, never by the optimizer.
        self.register_buffer("expert_bias", torch.zeros(self.n_routed), persistent=True)
        self.register_buffer("expert_load", torch.zeros(self.n_routed), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Route each token to ``top_k`` experts and add the shared experts."""
        b, t, d = x.shape
        flat = x.reshape(-1, d)

        logits = self.router(flat.float())
        gates = torch.softmax(logits, dim=-1)
        scores = gates + self.expert_bias  # bias affects selection only
        _, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = gates.gather(-1, indices)
        if self.normalize_gates:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        weights = weights.to(x.dtype)

        out = torch.zeros_like(flat)
        counts = torch.zeros(self.n_routed, device=x.device, dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts):
            hit = indices == expert_id
            token_idx, slot_idx = hit.nonzero(as_tuple=True)
            counts[expert_id] = token_idx.numel()
            if token_idx.numel() == 0:
                continue
            selected = flat.index_select(0, token_idx)
            contribution = expert(selected) * weights[token_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, contribution.to(out.dtype))

        for shared in self.shared_experts:
            out = out + shared(flat)

        self.expert_load = counts
        if self.training and self.balance_rate > 0:
            self._update_balancing_bias(counts)
        return out.view(b, t, d)

    @torch.no_grad()
    def _update_balancing_bias(self, counts: Tensor) -> None:
        """Nudge the selection bias toward uniform expert load."""
        target = counts.mean()
        error = target - counts
        self.expert_bias += self.balance_rate * torch.sign(error)

    def load_balance_stats(self) -> dict[str, float]:
        """Report how evenly the last batch was routed.

        ``max_over_mean`` of 1.0 is perfectly balanced; the auxiliary-loss-free
        controller typically settles between 1.0 and 1.3.
        """
        load = self.expert_load.float()
        total = load.sum()
        if total <= 0:
            return {"max_over_mean": 1.0, "entropy": 0.0, "dead_experts": 0.0}
        fraction = load / total
        entropy = float(-(fraction * (fraction + 1e-9).log()).sum())
        return {
            "max_over_mean": float(load.max() / load.mean().clamp(min=1e-9)),
            "entropy": entropy,
            "dead_experts": float((load == 0).sum()),
        }

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, hidden={self.hidden}, n_routed={self.n_routed}, "
            f"n_shared={self.n_shared}, top_k={self.top_k}"
        )


class GatedRecurrentUnit(nn.Module):
    """Real-diagonal gated linear recurrence (RG-LRU), as used by Hawk/Griffin.

    The recurrence is ``h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (i_t * x_t)``
    with ``a_t = exp(-8 * softplus(Lambda) * r_t)`` where ``r_t`` and ``i_t`` are
    input-dependent sigmoid gates. Because ``a`` is diagonal, positive and
    strictly less than one, the whole sequence can be solved in parallel with a
    log-space cumulative sum instead of a sequential loop:

    ``h_t = A_t * sum_{s<=t} (b_s / A_s)`` where ``A_t = prod_{s<=t} a_s``.

    That formulation is numerically safe here because ``a`` is bounded away from
    zero over practical sequence lengths, and it means the block trains at
    attention-like speed on GPU without a custom kernel. Cost is O(T) in
    sequence length rather than attention's O(T^2), which is what makes hybrid
    stacks attractive for long-context small models.
    """

    #: Matches the Griffin paper's `c` constant in `a = exp(-c * softplus(L) * r)`.
    DECAY_SCALE = 8.0

    def __init__(self, dim: int, *, expansion: float = 1.0, bias: bool = False):
        super().__init__()
        self.dim = int(dim)
        self.inner = int(dim * expansion)
        self.input_proj = nn.Linear(self.dim, self.inner, bias=bias)
        self.gate_proj = nn.Linear(self.dim, self.inner, bias=bias)
        self.recurrent_gate = nn.Linear(self.inner, self.inner, bias=True)
        self.input_gate = nn.Linear(self.inner, self.inner, bias=True)
        self.out_proj = nn.Linear(self.inner, self.dim, bias=bias)
        # log-space parameterisation keeps Lambda positive without clamping
        self.log_lambda = nn.Parameter(torch.linspace(-4.0, 0.0, self.inner))

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return ``(output, final_state)`` for ``x`` shaped ``[B, T, dim]``."""
        gate = F.gelu(self.gate_proj(x))
        u = self.input_proj(x)

        r = torch.sigmoid(self.recurrent_gate(u))
        i = torch.sigmoid(self.input_gate(u))
        decay = torch.exp(-self.DECAY_SCALE * F.softplus(self.log_lambda) * r)
        driven = torch.sqrt((1.0 - decay.pow(2)).clamp(min=1e-8)) * (i * u)

        log_decay = torch.log(decay.clamp(min=1e-8).float())
        cumulative = torch.cumsum(log_decay, dim=1)
        # h_t = exp(C_t) * (h_0 + sum_{s<=t} exp(-C_s) * b_s)
        scaled = driven.float() * torch.exp(-cumulative)
        running = torch.cumsum(scaled, dim=1)
        if state is not None:
            running = running + state.unsqueeze(1).float()
        h = (torch.exp(cumulative) * running).to(x.dtype)

        return self.out_proj(h * gate), h[:, -1]

    def extra_repr(self) -> str:
        return f"dim={self.dim}, inner={self.inner}"


class RecurrentBlock(nn.Module):
    """Pre-norm block that uses :class:`GatedRecurrentUnit` instead of attention."""

    def __init__(
        self,
        dim: int,
        ffn_hidden: int,
        *,
        expansion: float = 1.0,
        norm_eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.mixer_norm = RMSNorm(dim, eps=norm_eps)
        self.mixer = GatedRecurrentUnit(dim, expansion=expansion, bias=bias)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = SwiGLUFeedForward(dim, ffn_hidden, bias=bias)

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return ``(hidden_states, recurrent_state)``."""
        mixed, new_state = self.mixer(self.mixer_norm(x), state)
        x = x + mixed
        return x + self.ffn(self.ffn_norm(x)), new_state


class TransformerBlock(nn.Module):
    """Pre-norm attention + feed-forward block with residual connections.

    ``timestep_scale`` multiplies the FFN output before the residual add. The
    looped model uses it to give every loop iteration its own output gain, which
    lets one set of shared weights behave differently at different depths.

    Passing ``moe`` swaps the dense FFN for a :class:`MoEFeedForward`.
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
        value_residual: bool = True,
        n_kv_heads: int | None = None,
        qk_norm: bool = False,
        moe: dict | None = None,
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
        if moe:
            self.ffn: nn.Module = MoEFeedForward(dim, ffn_hidden, bias=bias, **moe)
        else:
            self.ffn = SwiGLUFeedForward(dim, ffn_hidden, bias=bias)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        qkv_delta: Tensor | None = None,
        window: int | None = None,
        v_prev: Tensor | None = None,
        timestep_scale: Tensor | None = None,
        cache: KVCache | None = None,
        q_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(hidden_states, mixed_values)``."""
        attn_out, v = self.attention(
            self.attn_norm(x),
            cos,
            sin,
            qkv_delta=qkv_delta,
            window=window,
            v_prev=v_prev,
            cache=cache,
            q_offset=q_offset,
        )
        x = x + attn_out
        ffn_out = self.ffn(self.ffn_norm(x))
        if timestep_scale is not None:
            ffn_out = ffn_out * timestep_scale
        return x + ffn_out, v
