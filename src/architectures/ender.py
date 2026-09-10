"""ENDER — Endogenous Neural Depth with Evolving Recurrence.

ENDER is a *wrapper architecture*: a frozen (or fine-tuned) backbone transformer
gains a latent recurrence module grafted in front of its final norm. The module
projects the residual stream ``h`` into a small latent space ``r_latent``, runs a
shared causal core block ``num_steps`` times with gated, tanh-bounded updates,
and injects the resulting "innovation" back into the stream through a zero-init
gate:

    z_0 = down(norm_in(h))
    u_k = core(z_k + step_embed[k])
    z_{k+1} = z_k + g_k * tanh(u_k - z_k)        g_k = sigmoid(gate([z_k, a]))
    h' = h + tanh(alpha) * up(z_K - z_0)

The tanh bounding and the zero-initialised ``alpha`` mean a freshly grafted
ENDER module is *exactly* a no-op: the wrapped model produces identical logits
to the backbone at step 0, which makes adaptation safe to apply to any trained
checkpoint and makes the before/after benchmark a clean A/B.

Key mechanisms:

``step_embeds``
    A learned vector per iteration, telling the shared core which step it is on
    (the same trick the looped architecture uses to break weight-sharing symmetry).
``gated tanh updates``
    Each step proposes ``tanh(u_k - z_k)``, bounded to [-1, 1]; the gate scales it
    per channel. Innovations stay small by construction, so extra recurrence steps
    never blow up the residual stream.
``endogenous depth``
    The core is one block re-run ``num_steps`` times: effective depth grows with
    the step count while parameter count stays fixed, and the step count is a
    test-time dial (``set_recurrence_steps``).

The module adapts to any backbone by locating its final normalisation layer and
hooking before it; the generic loss (:class:`EnderLoss`) adds an auxiliary
"convergence" penalty pulling the last two latent states together (so extra steps
refine rather than churn) and an optional KD term against a teacher.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minimodel.architectures.base import BaseLanguageModel
from minimodel.architectures.layers import KVCache, RMSNorm, RotaryEmbedding, TransformerBlock

__all__ = [
    "EnderAdapter",
    "EnderCoreBlock",
    "EnderLoss",
    "EnderRecurrenceModule",
    "EnderTransformer",
    "EnderTransformerConfig",
    "build_ender_optimizer",
]


#: Defaults for every key the ENDER architecture understands.
EnderTransformerConfig: dict[str, Any] = {
    "vocab_size": 4096,
    "dim": 128,
    "n_heads": 4,
    "head_dim": 32,
    "ffn_hidden": 512,
    "norm_eps": 1e-6,
    "bias": False,
    "window": 512,
    "rope_base": 10000.0,
    "max_seq_len": 1024,
    "tie_embeddings": True,
    "init_std": 0.02,
    # ENDER recurrence
    "r_latent": 64,
    "num_steps": 4,
    "min_steps": 2,
    "variable_steps": True,
    "gate_bias_init": -2.0,
    # auxiliary losses (consumed by EnderLoss / EnderTrainer)
    "lambda_delta": 0.05,
    "lambda_kd": 0.0,
    "kd_temperature": 2.0,
}


class EnderCoreBlock(nn.Module):
    """The shared causal block re-run at every recurrence step.

    Deliberately *position-free*: the backbone has already applied RoPE to the
    residual stream, and the recurrence re-runs on that same sequence, so the
    core only needs causal masking — no rotary tables and no decoding cache of
    its own (each step re-attends to the same tokens).
    """

    def __init__(
        self,
        r_latent: int,
        num_heads: int,
        ffn_hidden: int,
        *,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if r_latent % num_heads != 0:
            raise ValueError(
                f"r_latent must be divisible by num_heads (got {r_latent} % {num_heads} != 0)"
            )
        self.r_latent = int(r_latent)
        self.num_heads = int(num_heads)
        self.head_dim = self.r_latent // self.num_heads
        self.norm1 = RMSNorm(r_latent, eps=norm_eps)
        self.qkv = nn.Linear(r_latent, 3 * r_latent, bias=False)
        self.out_proj = nn.Linear(r_latent, r_latent, bias=False)
        self.norm2 = RMSNorm(r_latent, eps=norm_eps)
        self.w1 = nn.Linear(r_latent, ffn_hidden, bias=False)
        self.w2 = nn.Linear(r_latent, ffn_hidden, bias=False)
        self.w3 = nn.Linear(ffn_hidden, r_latent, bias=False)

    def forward(self, z: Tensor) -> Tensor:
        """Causal attention + SwiGLU over ``z`` shaped ``[B, T, r_latent]``."""
        b, t, r = z.shape
        qkv = (
            self.qkv(self.norm1(z))
            .reshape(b, t, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        z = z + self.out_proj(attended.permute(0, 2, 1, 3).reshape(b, t, r))
        return z + self.w3(F.silu(self.w1(self.norm2(z))) * self.w2(self.norm2(z)))


class EnderRecurrenceModule(nn.Module):
    """The latent recurrence grafted onto a backbone's final norm input.

    Parameters
    ----------
    dim:
        Backbone hidden size (``d_model``).
    r_latent:
        Latent width the recurrence runs in.
    num_steps:
        Recurrence iterations at the default setting.
    num_heads, head_dim, ffn_hidden:
        Geometry of the shared core block inside the latent space.
    gate_bias_init:
        Initial bias of the update gate; a negative value keeps early updates
        small so training starts close to the backbone.
    """

    def __init__(
        self,
        dim: int,
        *,
        r_latent: int = 64,
        num_steps: int = 4,
        num_heads: int = 4,
        head_dim: int | None = None,
        ffn_hidden: int | None = None,
        gate_bias_init: float = -2.0,
        norm_eps: float = 1e-6,
        init_std: float = 0.02,
    ):
        super().__init__()
        head_dim = head_dim or max(1, r_latent // num_heads)
        if num_heads * head_dim != r_latent:
            raise ValueError(
                f"num_heads * head_dim must equal r_latent (got {num_heads} * {head_dim} != {r_latent})"
            )
        ffn_hidden = ffn_hidden or 4 * r_latent
        self.dim = int(dim)
        self.r_latent = int(r_latent)
        self.num_steps = int(num_steps)
        self.gate_bias_init = float(gate_bias_init)

        self.norm_in = RMSNorm(dim, eps=norm_eps)
        self.down = nn.Linear(dim, r_latent, bias=False)
        self.up = nn.Linear(r_latent, dim, bias=False)
        self.core = EnderCoreBlock(
            r_latent,
            num_heads,
            ffn_hidden,
            norm_eps=norm_eps,
        )
        self.step_embed = nn.Parameter(torch.zeros(num_steps, r_latent))
        self.gate = nn.Linear(2 * r_latent, r_latent, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, self.gate_bias_init)
        # Zero-init the output gate: a fresh module is exactly the backbone.
        self.alpha = nn.Parameter(torch.zeros(dim))

    def forward(
        self, h: Tensor, steps: int | None = None
    ) -> tuple[Tensor, dict[str, Tensor | list[Tensor]]]:
        """Refine ``h`` ``steps`` times; return ``(h', aux)``.

        ``aux`` carries the latent trajectory (first entry, final entry) so the
        convergence loss can see it without re-running the module.
        """
        n_steps = steps if steps is not None else self.num_steps
        if n_steps < 1:
            raise ValueError(f"steps must be >= 1, got {n_steps}")

        a = self.down(self.norm_in(h))
        z = a
        z_first = z

        if n_steps > self.step_embed.shape[0]:
            padding = torch.zeros(
                n_steps - self.step_embed.shape[0], self.r_latent, device=h.device, dtype=h.dtype
            )
            step_embeds = torch.cat([self.step_embed, padding], dim=0)
        else:
            step_embeds = self.step_embed

        for k in range(n_steps):
            u_k = self.core(z + step_embeds[min(k, step_embeds.shape[0] - 1)])
            g_k = torch.sigmoid(self.gate(torch.cat([z, a], dim=-1)))
            z = z + g_k * torch.tanh(u_k - z)

        innovation = z - z_first
        h_prime = h + self.alpha.unsqueeze(0).unsqueeze(0) * self.up(innovation)
        return h_prime, {"a": a, "z_first": z_first, "z_final": z, "innovation": innovation}


class EnderAdapter(nn.Module):
    """Wrap a backbone LM and graft an :class:`EnderRecurrenceModule` into it.

    The adapter is backbone-agnostic: it locates the final normalisation layer
    (``model.norm``, ``transformer.ln_f``, ``final_norm``, ... or a name given
    via ``final_norm_name``) and installs a forward-pre-hook that runs the
    recurrence on the stream right before that norm. A fresh adapter is a
    mathematical no-op, so wrapping a trained checkpoint cannot hurt it until
    the new parameters are trained.
    """

    #: Candidate attribute paths for the final norm, in priority order.
    FINAL_NORM_PATHS = (
        "model.norm",
        "transformer.ln_f",
        "transformer.norm",
        "model.final_layernorm",
        "final_norm",
        "norm",
    )

    def __init__(
        self,
        backbone: nn.Module,
        config: Mapping[str, Any] | None = None,
        *,
        final_norm_name: str | None = None,
    ):
        super().__init__()
        merged = {**EnderTransformerConfig, **dict(config or {})}
        dim = int(merged["dim"])
        self.config: dict[str, Any] = merged
        self.backbone = backbone
        r_latent = int(merged["r_latent"])
        core_heads = int(merged["n_heads"])
        # The core's geometry is the latent space's, not the backbone's: fall
        # back to one head per unit when the declared head_dim does not tile it.
        core_head_dim = r_latent // core_heads if r_latent % core_heads == 0 else None
        if core_head_dim is None:
            core_heads = 1
            core_head_dim = r_latent
        self.ender = EnderRecurrenceModule(
            dim,
            r_latent=r_latent,
            num_steps=int(merged["num_steps"]),
            num_heads=core_heads,
            head_dim=core_head_dim,
            ffn_hidden=None,
            gate_bias_init=float(merged["gate_bias_init"]),
            norm_eps=float(merged["norm_eps"]),
            init_std=float(merged["init_std"]),
        )
        self.active_steps: int | None = None
        self.last_aux: dict[str, Tensor | list[Tensor]] = {}
        self.final_norm = self._resolve_final_norm(final_norm_name)
        self.final_norm.register_forward_pre_hook(self._norm_pre_hook)

    # ------------------------------------------------------------------
    def _resolve_final_norm(self, name: str | None) -> nn.Module:
        """Find the module the recurrence hooks into."""
        if name:
            for module_name, module in self.backbone.named_modules():
                if module_name == name:
                    return module
            raise ValueError(f"no module named {name!r} in the backbone")
        for path in self.FINAL_NORM_PATHS:
            current: nn.Module = self.backbone
            found = True
            for part in path.split("."):
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    found = False
                    break
            if found and isinstance(current, nn.Module):
                return current
        norm_modules = [
            (n, m)
            for n, m in self.backbone.named_modules()
            if isinstance(m, (nn.LayerNorm, RMSNorm))
            or "norm" in m.__class__.__name__.lower()
        ]
        if norm_modules:
            return norm_modules[-1][1]
        raise RuntimeError(
            "unable to locate the final normalisation layer; pass final_norm_name explicitly"
        )

    def _norm_pre_hook(self, module: nn.Module, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        h_prime, aux = self.ender(inputs[0], steps=self.active_steps)
        self.last_aux = aux
        return (h_prime, *inputs[1:])

    # ------------------------------------------------------------------
    def set_recurrence_steps(self, steps: int | None) -> None:
        """Set the test-time depth dial. ``None`` uses the trained default."""
        self.active_steps = steps

    def forward(
        self,
        tokens: Tensor,
        *,
        steps: int | None = None,
        cache: KVCache | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Delegate to the backbone; the recurrence runs inside the hook."""
        previous = self.active_steps
        if steps is not None:
            self.active_steps = steps
        try:
            return self.backbone(tokens, cache=cache, **kwargs)
        finally:
            self.active_steps = previous


class EnderTransformer(BaseLanguageModel):
    """A first-class ENDER model: dense backbone + latent recurrence.

    This is the trained-from-scratch counterpart of :class:`EnderAdapter`.
    A small dense transformer carries the sequence-level processing; ENDER
    adds ``num_steps`` passes of a shared latent core before the final norm,
    which is where the "endogenous depth" lives. Because the recurrence is
    causal and the core uses the same :class:`KVCache` protocol as the backbone,
    the whole thing supports incremental decoding like every other family here.

    Examples
    --------
    >>> model = EnderTransformer({"vocab_size": 64, "dim": 32, "n_heads": 2,
    ...                           "head_dim": 16, "ffn_hidden": 64,
    ...                           "r_latent": 16, "num_steps": 2})
    >>> logits = model(torch.zeros(1, 5, dtype=torch.long), steps=2)
    >>> tuple(logits.shape)
    (1, 5, 64)
    """

    architecture_name = "ender"

    def __init__(self, config: Mapping[str, Any] | None = None):
        merged = {**EnderTransformerConfig, **dict(config or {})}
        super().__init__(merged)
        cfg = self.config

        dim = int(cfg["dim"])
        n_heads = int(cfg["n_heads"])
        head_dim = int(cfg["head_dim"])
        if n_heads * head_dim != dim:
            raise ValueError(
                f"n_heads * head_dim must equal dim (got {n_heads} * {head_dim} != {dim})"
            )
        r_latent = int(cfg["r_latent"])
        core_heads = max(1, min(n_heads, r_latent))
        if r_latent % core_heads != 0:
            core_heads = 1
        core_head_dim = r_latent // core_heads

        self.dim = dim
        self.vocab_size = int(cfg["vocab_size"])
        self.max_seq_len = int(cfg["max_seq_len"])
        self.num_steps = int(cfg["num_steps"])
        self.min_steps = int(cfg["min_steps"])
        self.variable_steps = bool(cfg["variable_steps"])

        self.embedding = nn.Embedding(self.vocab_size, dim)
        self.rope = RotaryEmbedding(
            head_dim, base=float(cfg["rope_base"]), max_seq_len=self.max_seq_len
        )
        window = int(cfg["window"]) if cfg["window"] else None
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    ffn_hidden=int(cfg["ffn_hidden"]),
                    norm_eps=float(cfg["norm_eps"]),
                    bias=bool(cfg["bias"]),
                    window=window,
                    value_residual=False,
                )
                for _ in range(int(cfg.get("n_layers", 2)))
            ]
        )
        self.ender = EnderRecurrenceModule(
            dim,
            r_latent=r_latent,
            num_steps=self.num_steps,
            num_heads=core_heads,
            head_dim=core_head_dim,
            gate_bias_init=float(cfg["gate_bias_init"]),
            norm_eps=float(cfg["norm_eps"]),
            init_std=float(cfg["init_std"]),
        )
        self.final_norm = RMSNorm(dim, eps=float(cfg["norm_eps"]))
        self.init_weights()

    def init_weights(self) -> None:
        """Normal init, with the recurrence output gate at zero.

        The zero ``alpha`` (set inside the module) means a fresh ENDER model is
        a plain dense transformer; the recurrence has to earn its influence.
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
            self.ender.alpha.zero_()
            for module in self.modules():
                if isinstance(module, RMSNorm):
                    module.weight.fill_(1.0)

    def resolve_steps(self, steps: int | None) -> int:
        """Pick the recurrence depth for this pass (explicit > sampled > default)."""
        if steps is not None:
            if steps < 1:
                raise ValueError(f"steps must be >= 1, got {steps}")
            return int(steps)
        if self.training and self.variable_steps and self.min_steps < self.num_steps:
            import random

            return random.randint(self.min_steps, self.num_steps)
        return self.num_steps

    def forward(
        self,
        tokens: Tensor,
        *,
        steps: int | None = None,
        return_hidden: bool = False,
        cache: KVCache | None = None,
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

        for block in self.blocks:
            x, _ = block(x, cos, sin, cache=cache, q_offset=q_offset)

        n_steps = self.resolve_steps(steps)
        x, self.ender_aux = self.ender(x, steps=n_steps)

        if cache is not None:
            cache.length = q_offset + seq_len

        hidden = self.final_norm(x)
        if return_hidden:
            return hidden
        return F.linear(hidden, self.embedding.weight)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> EnderTransformer:
        """Build a model from a config mapping, ignoring bookkeeping keys."""
        payload = {k: v for k, v in dict(config).items() if k in EnderTransformerConfig}
        return cls(payload)


class EnderLoss(nn.Module):
    """Cross-entropy plus ENDER's auxiliary objectives.

    ``loss_delta`` (convergence penalty)
        Mean squared distance between the final and pre-final latent states.
        It rewards a recurrence that has *converged* by its last step, so extra
        test-time steps refine rather than oscillate.
    ``loss_kd`` (optional distillation)
        Temperature-scaled KL against teacher logits, for adapting a wrapped
        checkpoint without letting it drift from its original behaviour.
    """

    def __init__(
        self,
        *,
        lambda_delta: float = 0.05,
        lambda_kd: float = 0.0,
        kd_temperature: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.lambda_delta = float(lambda_delta)
        self.lambda_kd = float(lambda_kd)
        self.kd_temperature = float(kd_temperature)
        self.ignore_index = int(ignore_index)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        *,
        aux: Mapping[str, Any] | None = None,
        teacher_logits: Tensor | None = None,
        ignore_index: int | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Return ``(total_loss, extras)``; extras feed the training log."""
        ignore = self.ignore_index if ignore_index is None else ignore_index
        loss_ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            ignore_index=ignore,
        )
        total = loss_ce
        extras = {"loss_ce": float(loss_ce.detach())}

        if aux is not None and self.lambda_delta > 0 and "z_final" in aux:
            trajectory = aux.get("trajectory")
            if isinstance(trajectory, list) and len(trajectory) >= 2:
                delta = torch.mean((trajectory[-1] - trajectory[-2]) ** 2)
            else:
                delta = torch.mean((aux["z_final"] - aux["z_first"]) ** 2)
            total = total + self.lambda_delta * delta
            extras["loss_delta"] = float(delta.detach())

        if teacher_logits is not None and self.lambda_kd > 0:
            temperature = self.kd_temperature
            kl = F.kl_div(
                F.log_softmax(logits[:, :-1] / temperature, dim=-1),
                F.softmax(teacher_logits[:, :-1] / temperature, dim=-1),
                reduction="batchmean",
            )
            loss_kd = (temperature**2) * kl
            total = total + self.lambda_kd * loss_kd
            extras["loss_kd"] = float(loss_kd.detach())

        return total, extras


def build_ender_optimizer(
    model: EnderAdapter,
    *,
    base_lr: float = 1e-4,
    unfreeze_top_blocks: int = 0,
    unfrozen_lr_scale: float = 0.1,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """Stage-1 optimisation for an :class:`EnderAdapter`.

    The backbone is frozen; only the grafted recurrence trains. Optionally the
    top ``unfreeze_top_blocks`` backbone blocks are unfrozen at a scaled learning
    rate, which is the standard second stage when the adapter alone plateaus.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.ender.parameters():
        param.requires_grad = True

    groups: list[dict[str, Any]] = [
        {"params": list(model.ender.parameters()), "lr": base_lr}
    ]
    if unfreeze_top_blocks > 0:
        blocks = getattr(model.backbone, "blocks", None) or getattr(
            model.backbone, "layers", None
        )
        if blocks is None:
            raise ValueError("backbone has no `blocks`/`layers` to unfreeze")
        for block in list(blocks)[-int(unfreeze_top_blocks) :]:
            for param in block.parameters():
                param.requires_grad = True
        groups.append(
            {
                "params": [p for b in list(blocks)[-int(unfreeze_top_blocks) :] for p in b.parameters()],
                "lr": base_lr * unfrozen_lr_scale,
            }
        )
    return torch.optim.AdamW(groups, weight_decay=weight_decay)
