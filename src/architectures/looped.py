"""The looped (weight-shared recurrent-depth) transformer.

This is the flagship architecture of the repository. Instead of stacking N
distinct blocks, it runs a small set of *shared* blocks several times, which
buys effective depth without paying for it in parameters. A 1.4M-parameter
configuration runs 8 loops over 2 shared blocks, giving 18 effective layers from
4 blocks' worth of weights.

Naive weight sharing collapses: every iteration computes the same function, so
the extra depth adds little. Four mechanisms break that symmetry, and all four
are cheap:

``loop_embed``
    A learned vector added to the residual stream before each iteration, telling
    the shared block which iteration it is in.
``loop_lora``
    A rank-4 LoRA on the fused QKV projection, with its own table per iteration.
    Each iteration therefore has slightly different attention weights.
``timestep_scale``
    A per-iteration, per-channel gain on the FFN output, letting the model damp
    or amplify updates as depth increases.
``outer_residual``
    A learned channel-wise re-injection of the post-embedding stream after every
    iteration, which keeps the input reachable no matter how many loops run.

Additionally, the loop count is sampled during training (``U{min..max}``), so a
single model can trade compute for quality at inference time by asking for more
or fewer loops.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from minimodel.architectures.base import BaseLanguageModel
from minimodel.architectures.layers import (
    FactorizedEmbedding,
    KVCache,
    RMSNorm,
    RotaryEmbedding,
    TiedEmbedding,
    TransformerBlock,
)

__all__ = ["LoopedTransformer", "LoopedTransformerConfig"]

#: Default values for every key the architecture understands. The YAML templates
#: only need to specify what they change.
LoopedTransformerConfig: dict[str, Any] = {
    "vocab_size": 4096,
    "dim": 128,
    "n_heads": 8,
    "head_dim": 16,
    "ffn_hidden": 544,
    "norm_eps": 1e-6,
    "bias": False,
    "window": 256,
    "rope_base": 10000.0,
    "max_seq_len": 4096,
    "embedding_rank": 64,
    #: ``factorized`` (E @ proj, shared with the head) or ``tied`` (plain
    #: [vocab, dim] embedding whose transpose is the head, as Glint-2 uses).
    "embedding_type": "factorized",
    #: Unique blocks before / after the recurrent loop. Glint-2 is (0, 1); our
    #: supra2 is (1, 1). Setting both to 0 gives a pure-loop model.
    "prelude_layers": 1,
    "coda_layers": 1,
    "n_shared_blocks": 2,
    "train_loops": 8,
    "min_loops": 4,
    "variable_loops": True,
    "max_loops_table": 16,
    "loop_lora_rank": 4,
    "outer_gate_init": 0.1,
    #: Per-iteration conditioning knobs. All on for supra2; Glint-2 keeps only
    #: loop_embed + loop_lora, so these let us ablate each mechanism.
    "use_timestep_scale": True,
    "use_outer_residual": True,
    "value_residual": True,
    "init_std": 0.02,
}


class LoopedTransformer(BaseLanguageModel):
    """Recurrent-depth transformer with per-iteration conditioning.

    Parameters
    ----------
    config:
        Mapping merged over :data:`LoopedTransformerConfig`. See
        ``src/architectures/templates/supra2_1406240.yaml`` for a fully
        annotated example.

    Examples
    --------
    >>> model = LoopedTransformer({"vocab_size": 64, "dim": 32, "n_heads": 2,
    ...                            "head_dim": 16, "ffn_hidden": 64,
    ...                            "embedding_rank": 16, "max_loops_table": 4,
    ...                            "train_loops": 2, "min_loops": 1})
    >>> logits = model(torch.zeros(1, 5, dtype=torch.long), loops=2)
    >>> tuple(logits.shape)
    (1, 5, 64)
    """

    architecture_name = "looped-transformer"

    def __init__(self, config: Mapping[str, Any] | None = None):
        merged = {**LoopedTransformerConfig, **dict(config or {})}
        super().__init__(merged)
        cfg = self.config

        dim = int(cfg["dim"])
        n_heads = int(cfg["n_heads"])
        head_dim = int(cfg["head_dim"])
        if n_heads * head_dim != dim:
            raise ValueError(
                f"n_heads * head_dim must equal dim (got {n_heads} * {head_dim} != {dim})"
            )
        n_tables = int(cfg["max_loops_table"])
        if int(cfg["train_loops"]) > n_tables:
            raise ValueError(
                f"train_loops ({cfg['train_loops']}) exceeds max_loops_table ({n_tables})"
            )
        if int(cfg["min_loops"]) > int(cfg["train_loops"]):
            raise ValueError(
                f"min_loops ({cfg['min_loops']}) must be <= train_loops ({cfg['train_loops']})"
            )
        if int(cfg["n_shared_blocks"]) < 1:
            raise ValueError("n_shared_blocks must be at least 1")

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window = int(cfg["window"]) if cfg["window"] else None
        self.n_shared_blocks = int(cfg["n_shared_blocks"])
        self.n_prelude = int(cfg["prelude_layers"])
        self.n_coda = int(cfg["coda_layers"])
        self.train_loops = int(cfg["train_loops"])
        self.min_loops = int(cfg["min_loops"])
        self.variable_loops = bool(cfg["variable_loops"])
        self.max_loops_table = n_tables
        self.loop_lora_rank = int(cfg["loop_lora_rank"])
        self.vocab_size = int(cfg["vocab_size"])
        self.max_seq_len = int(cfg["max_seq_len"])
        self.embedding_type = str(cfg["embedding_type"]).lower()
        self.use_timestep_scale = bool(cfg["use_timestep_scale"])
        self.use_outer_residual = bool(cfg["use_outer_residual"])

        if self.embedding_type == "factorized":
            self.embedding = FactorizedEmbedding(self.vocab_size, int(cfg["embedding_rank"]), dim)
        elif self.embedding_type == "tied":
            self.embedding = TiedEmbedding(self.vocab_size, dim)
        else:
            raise ValueError(f"unknown embedding_type {self.embedding_type!r}")
        self.rope = RotaryEmbedding(
            head_dim, base=float(cfg["rope_base"]), max_seq_len=self.max_seq_len
        )

        block_kwargs = {
            "dim": dim,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "ffn_hidden": int(cfg["ffn_hidden"]),
            "norm_eps": float(cfg["norm_eps"]),
            "bias": bool(cfg["bias"]),
            "window": self.window,
            "value_residual": bool(cfg["value_residual"]),
        }
        self.prelude = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(self.n_prelude)]
        )
        self.shared = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(self.n_shared_blocks)]
        )
        self.coda = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(self.n_coda)]
        )

        # Per-iteration conditioning tables. Iterations beyond `max_loops_table`
        # reuse the last entry, so asking for more loops at inference is safe.
        self.loop_embed = nn.Embedding(n_tables, dim)
        self.loop_lora_down = nn.Parameter(torch.zeros(n_tables, self.loop_lora_rank, dim))
        self.loop_lora_up = nn.Parameter(torch.zeros(n_tables, 3 * dim, self.loop_lora_rank))
        # Only allocate the optional conditioning params when enabled, so
        # disabling them removes params (fair budget comparison against Glint-2).
        if self.use_timestep_scale:
            self.timestep_scale = nn.Parameter(torch.ones(n_tables, dim))
        else:
            self.register_parameter("timestep_scale", None)
        if self.use_outer_residual:
            self.outer_gate = nn.Parameter(torch.full((dim,), float(cfg["outer_gate_init"])))
        else:
            self.register_parameter("outer_gate", None)

        self.final_norm = RMSNorm(dim, eps=float(cfg["norm_eps"]))

        self.init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def init_weights(self) -> None:
        """Apply the initialisation scheme in the order the template requires.

        Generic ``normal(0, std)`` is applied to every ``Linear`` and
        ``Embedding`` first, then the special-cased tensors are re-asserted so
        they survive that pass. RMSNorm gains and the value-residual logits are
        neither Linear nor Embedding and must not be touched by the generic pass.
        """
        std = float(self.config["init_std"])

        def _init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Embedding, TiedEmbedding)):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        self.apply(_init)

        with torch.no_grad():
            nn.init.normal_(self.loop_lora_down, mean=0.0, std=std)
            # A zero up-projection makes every LoRA a no-op at step 0, so the
            # loop conditioning starts from the plain shared-block function.
            self.loop_lora_up.zero_()
            if self.timestep_scale is not None:
                self.timestep_scale.fill_(1.0)
            if self.outer_gate is not None:
                self.outer_gate.fill_(float(self.config["outer_gate_init"]))
            for module in self.modules():
                if isinstance(module, RMSNorm):
                    module.weight.fill_(1.0)
                v_logit = getattr(module, "v_lambda_logit", None)
                if isinstance(v_logit, nn.Parameter):
                    v_logit.zero_()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def resolve_loops(self, loops: int | None) -> int:
        """Decide how many loop iterations this forward pass should run.

        An explicit ``loops`` always wins. Otherwise training samples
        ``U{min_loops..train_loops}`` when variable loops are enabled, and
        evaluation uses ``train_loops``.
        """
        if loops is not None:
            if loops < 1:
                raise ValueError(f"loops must be >= 1, got {loops}")
            return int(loops)
        if self.training and self.variable_loops and self.min_loops < self.train_loops:
            return random.randint(self.min_loops, self.train_loops)
        return self.train_loops

    def _loop_delta(self, gated: Tensor, table_index: int) -> Tensor:
        """Compute the per-iteration LoRA delta on the fused QKV projection."""
        down = self.loop_lora_down[table_index]  # [rank, dim]
        up = self.loop_lora_up[table_index]  # [3 * dim, rank]
        z = torch.nn.functional.linear(gated, down)
        return torch.nn.functional.linear(z, up)

    def forward(
        self,
        tokens: Tensor,
        *,
        loops: int | None = None,
        return_hidden: bool = False,
        cache: KVCache | None = None,
    ) -> Tensor:
        """Map ``[B, T]`` token ids to ``[B, T, vocab_size]`` logits.

        Parameters
        ----------
        tokens:
            Input token ids.
        loops:
            Explicit iteration count. When decoding with a ``cache`` this must
            stay constant across steps, because the cache addresses attention
            layers by call order.
        return_hidden:
            Return the normalised hidden states instead of logits.
        cache:
            Optional :class:`~minimodel.architectures.layers.KVCache` for
            incremental decoding.
        """
        if tokens.dim() != 2:
            raise ValueError(f"expected tokens of shape [B, T], got {tuple(tokens.shape)}")
        seq_len = tokens.shape[1]
        q_offset = cache.length if cache is not None else 0
        if cache is not None:
            cache.begin_forward()

        x = self.embedding(tokens)
        x0 = x
        v_prev: Tensor | None = None

        cos_full, sin_full = self.rope(q_offset + seq_len, device=x.device, dtype=torch.float32)
        cos = cos_full[:, :, q_offset : q_offset + seq_len]
        sin = sin_full[:, :, q_offset : q_offset + seq_len]

        n_loops = self.resolve_loops(loops)

        for block in self.prelude:
            x, v_prev = block(x, cos, sin, v_prev=v_prev, cache=cache, q_offset=q_offset)

        for step in range(n_loops):
            table_index = min(step, self.max_loops_table - 1)
            gated = x + self.loop_embed.weight[table_index]
            delta = self._loop_delta(gated, table_index)
            block = self.shared[step % self.n_shared_blocks]
            tau = self.timestep_scale[table_index].view(1, 1, -1) if self.use_timestep_scale else None
            x_new, v_prev = block(
                gated,
                cos,
                sin,
                qkv_delta=delta,
                v_prev=v_prev,
                timestep_scale=tau,
                cache=cache,
                q_offset=q_offset,
            )
            x = x_new + self.outer_gate * x0 if self.use_outer_residual else x_new

        for block in self.coda:
            x, v_prev = block(x, cos, sin, v_prev=v_prev, cache=cache, q_offset=q_offset)

        if cache is not None:
            cache.length = q_offset + seq_len

        hidden = self.final_norm(x)
        if return_hidden:
            return hidden
        materialize = not self.training and hidden.shape[1] <= 8
        return self.embedding.logits(hidden, materialize=materialize)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LoopedTransformer:
        """Build a model from a config mapping, ignoring bookkeeping keys."""
        payload = {k: v for k, v in dict(config).items() if k in LoopedTransformerConfig}
        return cls(payload)

    def expected_parameter_count(self) -> int:
        """Analytic parameter count, used to validate against a template budget."""
        cfg = self.config
        dim = int(cfg["dim"])
        rank = int(cfg["embedding_rank"])
        vocab = int(cfg["vocab_size"])
        hidden = int(cfg["ffn_hidden"])
        heads = int(cfg["n_heads"])
        tables = int(cfg["max_loops_table"])
        lora_rank = int(cfg["loop_lora_rank"])

        if str(cfg["embedding_type"]).lower() == "tied":
            embed = vocab * dim
        else:
            embed = vocab * rank + rank * dim
        per_block = (
            dim * 3 * dim  # W_qkv
            + dim * dim  # W_out
            + (heads if cfg["value_residual"] else 0)  # v_lambda_logit
            + 2 * dim  # two RMSNorm gains
            + dim * 2 * hidden  # gate_up
            + hidden * dim  # down
        )
        n_blocks = int(cfg["prelude_layers"]) + int(cfg["n_shared_blocks"]) + int(cfg["coda_layers"])
        loop_tables = tables * (lora_rank * dim + 3 * dim * lora_rank)
        return (
            embed
            + n_blocks * per_block
            + loop_tables
            + tables * dim  # loop_embed
            + (tables * dim if cfg["use_timestep_scale"] else 0)  # timestep_scale
            + (dim if cfg["use_outer_residual"] else 0)  # outer_gate
            + dim  # final_norm
        )
