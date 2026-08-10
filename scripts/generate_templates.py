#!/usr/bin/env python3
"""Regenerate the bundled architecture templates.

Every template in ``src/architectures/templates`` except the hand-annotated
``supra2_1406240.yaml`` is produced by this script. Keeping the size ladder in
one place means the parameter counts written into the templates are always the
counts the code actually produces, and adding a new size is a three-line edit
rather than a hand-computed YAML file.

Usage::

    python scripts/generate_templates.py            # write templates
    python scripts/generate_templates.py --check    # verify they are current
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / ".."))

import yaml  # noqa: E402

from minimodel.architectures.builder import TEMPLATE_DIR  # noqa: E402
from minimodel.architectures.registry import ARCHITECTURES  # noqa: E402


@dataclass
class TemplateSpec:
    """One entry in the size ladder."""

    name: str
    family: str
    description: str
    arch: dict[str, Any]
    #: Rough token budget that gets the most out of this size. Small models are
    #: deliberately over-trained relative to Chinchilla-optimal because
    #: inference cost, not training cost, is what matters for them.
    recommended_tokens: str = ""
    training_defaults: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: Rank among the ~1M Glint-2 candidates (1 = strongest prior). None for
    #: unrelated size-ladder templates.
    glint2_rank: int | None = None
    #: researched | novel-transformer | novel-mamba
    candidate_class: str = ""


def ffn_hidden(dim: int, ratio: float = 8 / 3, multiple: int = 32) -> int:
    """SwiGLU hidden size: ``ratio * dim`` rounded up to a multiple of 32.

    The 8/3 ratio keeps a SwiGLU FFN at the same parameter count as a plain
    4x GELU FFN, which is what the ratio is chosen to match.
    """
    raw = int(dim * ratio)
    return ((raw + multiple - 1) // multiple) * multiple


_MM1M_TRAIN = {"lr": 3.0e-3, "batch_tokens": 65536, "seq_len": 512}
_MM1M_BASE = {
    "vocab_size": 4096,
    "dim": 112,
    "n_heads": 7,
    "head_dim": 16,
    "n_kv_heads": 1,
    "window": 512,
    "max_seq_len": 1024,
    "qk_norm": True,
    "tie_embeddings": True,
}


# ---------------------------------------------------------------------------
# The size ladder
# ---------------------------------------------------------------------------
SPECS: list[TemplateSpec] = [
    # ==================================================================
    # ~1M Glint-2 candidates (ordered by prior; mm1m_rXX sorts by rank)
    # ==================================================================
    TemplateSpec(
        name="mm1m_r01_dense_gqa_vr",
        family="dense-transformer",
        description=(
            "Rank-1 ~1.03M dense GQA + value residual + QK-norm — researched "
            "winner shape from the crush-Glint-2 bake-off, retargeted to ~1M."
        ),
        arch={**_MM1M_BASE, "n_layers": 5, "ffn_hidden": 256, "value_residual": True},
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        notes=["Prior from reports 03/11: dense GQA≈MHA, VR+QK-norm mandatory."],
        glint2_rank=1,
        candidate_class="researched",
    ),
    TemplateSpec(
        name="mm1m_r02_dense_mha",
        family="dense-transformer",
        description="Rank-2 ~1.05M full MHA dense (researched; sandbox GQA≈MHA).",
        arch={
            **_MM1M_BASE,
            "n_layers": 4,
            "n_kv_heads": 7,
            "ffn_hidden": 288,
            "value_residual": True,
        },
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=2,
        candidate_class="researched",
    ),
    TemplateSpec(
        name="mm1m_r03_dense_window",
        family="dense-transformer",
        description=(
            "Rank-3 ~1.03M dense with Mistral-style local/global window pattern "
            "(researched)."
        ),
        arch={
            **_MM1M_BASE,
            "n_layers": 5,
            "ffn_hidden": 256,
            "window": 256,
            "window_pattern": 4,
            "value_residual": True,
        },
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=3,
        candidate_class="researched",
    ),
    TemplateSpec(
        name="mm1m_r04_hybrid_griffin",
        family="hybrid-recurrent",
        description="Rank-4 ~0.95M Griffin-style RG-LRU hybrid (researched).",
        arch={
            "vocab_size": 4096,
            "dim": 96,
            "n_layers": 6,
            "n_heads": 6,
            "head_dim": 16,
            "n_kv_heads": 1,
            "ffn_hidden": 192,
            "window": 512,
            "max_seq_len": 2048,
            "layer_pattern": ["recurrent", "recurrent", "attention"],
            "qk_norm": True,
            "tie_embeddings": True,
            "value_residual": True,
        },
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=4,
        candidate_class="researched",
    ),
    TemplateSpec(
        name="mm1m_r05_exp_resimix",
        family="experimental-transformer",
        description=(
            "Rank-5 ~1.04M novel ResiMix Transformer — prenorm/postnorm residual "
            "mix with per-channel gates."
        ),
        arch={**_MM1M_BASE, "n_layers": 5, "ffn_hidden": 256, "variant": "resimix"},
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        notes=["Original improvement; still a Transformer."],
        glint2_rank=5,
        candidate_class="novel-transformer",
    ),
    TemplateSpec(
        name="mm1m_r06_exp_kv_inherit",
        family="experimental-transformer",
        description=(
            "Rank-6 ~1.03M novel KV-inherit Transformer — soft mix over all prior "
            "layer values (generalised value residual)."
        ),
        arch={**_MM1M_BASE, "n_layers": 5, "ffn_hidden": 256, "variant": "kv_inherit"},
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=6,
        candidate_class="novel-transformer",
    ),
    TemplateSpec(
        name="mm1m_r07_dense_deep",
        family="dense-transformer",
        description="Rank-7 ~1.11M thin-deep dense L=8 (researched; usually trails wider).",
        arch={
            "vocab_size": 4096,
            "dim": 96,
            "n_layers": 8,
            "n_heads": 6,
            "head_dim": 16,
            "n_kv_heads": 2,
            "ffn_hidden": 224,
            "window": 512,
            "max_seq_len": 1024,
            "qk_norm": True,
            "tie_embeddings": True,
            "value_residual": True,
        },
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=7,
        candidate_class="researched",
    ),
    TemplateSpec(
        name="mm1m_r08_exp_braid",
        family="experimental-transformer",
        description=(
            "Rank-8 ~1.03M novel braid attention — odd heads local, even heads global "
            "inside one layer."
        ),
        arch={
            **_MM1M_BASE,
            "n_layers": 5,
            "ffn_hidden": 256,
            "variant": "braid",
            "local_window": 128,
        },
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=8,
        candidate_class="novel-transformer",
    ),
    TemplateSpec(
        name="mm1m_r09_exp_dual_rope",
        family="experimental-transformer",
        description=(
            "Rank-9 ~1.03M novel dual-RoPE Transformer — per-head mix of fast and "
            "slow rotary bases."
        ),
        arch={
            **_MM1M_BASE,
            "n_layers": 5,
            "ffn_hidden": 256,
            "variant": "dual_rope",
            "rope_base": 10000.0,
            "rope_base_slow": 500.0,
            "value_residual": True,
        },
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=9,
        candidate_class="novel-transformer",
    ),
    TemplateSpec(
        name="mm1m_r10_mamba_attn_tail",
        family="mamba-lm",
        description=(
            "Rank-10 ~1.06M novel Mamba trunk + attention coda (retrieval tail)."
        ),
        arch={
            **_MM1M_BASE,
            "n_layers": 5,
            "ffn_hidden": 224,
            "variant": "attn_tail",
            "attn_tail_layers": 2,
            "state_dim": 16,
            "expand": 1,
            "conv_kernel": 4,
            "value_residual": True,
        },
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=10,
        candidate_class="novel-mamba",
    ),
    TemplateSpec(
        name="mm1m_r11_exp_echo_ffn",
        family="experimental-transformer",
        description=(
            "Rank-11 ~1.04M novel echo-FFN Transformer — tied double SwiGLU with "
            "LoRA bridge."
        ),
        arch={
            **_MM1M_BASE,
            "n_layers": 5,
            "ffn_hidden": 256,
            "variant": "echo_ffn",
            "echo_lora_rank": 8,
            "value_residual": True,
        },
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=11,
        candidate_class="novel-transformer",
    ),
    TemplateSpec(
        name="mm1m_r12_mamba_braid",
        family="mamba-lm",
        description="Rank-12 ~1.10M novel alternating SSM/attention braid.",
        arch={
            **_MM1M_BASE,
            "n_layers": 6,
            "ffn_hidden": 192,
            "variant": "braid",
            "layer_pattern": ["ssm", "attention"],
            "state_dim": 16,
            "expand": 1,
            "conv_kernel": 4,
            "value_residual": True,
        },
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=12,
        candidate_class="novel-mamba",
    ),
    TemplateSpec(
        name="mm1m_r13_loop_poisson",
        family="looped-transformer",
        description=(
            "Rank-13 ~1.16M researched looped model with stabilisers + Poisson "
            "loop sampling (Huginn-style)."
        ),
        arch={
            "vocab_size": 4096,
            "dim": 112,
            "n_heads": 7,
            "head_dim": 16,
            "ffn_hidden": 768,
            "embedding_rank": 48,
            "window": 512,
            "max_seq_len": 1024,
            "n_shared_blocks": 1,
            "train_loops": 8,
            "min_loops": 4,
            "max_loops_table": 16,
            "loop_lora_rank": 4,
            "value_residual": True,
            "variable_loops": True,
            "loop_sampling": "poisson",
        },
        recommended_tokens="0.5-2B",
        training_defaults={"lr": 2.0e-2, "batch_tokens": 65536, "seq_len": 512},
        notes=["Use Muon for looped (report 04); loops are the test-time dial."],
        glint2_rank=13,
        candidate_class="researched",
    ),
    TemplateSpec(
        name="mm1m_r14_dense_wide",
        family="dense-transformer",
        description="Rank-14 ~1.12M wide-shallow dense L=2 (researched).",
        arch={
            "vocab_size": 4096,
            "dim": 160,
            "n_layers": 2,
            "n_heads": 5,
            "head_dim": 32,
            "n_kv_heads": 1,
            "ffn_hidden": 352,
            "window": 512,
            "max_seq_len": 1024,
            "qk_norm": True,
            "tie_embeddings": True,
            "value_residual": True,
        },
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=14,
        candidate_class="researched",
    ),
    TemplateSpec(
        name="mm1m_r15_moe_micro",
        family="moe-transformer",
        description="Rank-15 ~1.07M micro-MoE (researched; active≈0.85M).",
        arch={
            "vocab_size": 4096,
            "dim": 96,
            "n_layers": 4,
            "n_heads": 6,
            "head_dim": 16,
            "n_kv_heads": 1,
            "ffn_hidden": 128,
            "window": 512,
            "max_seq_len": 1024,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "top_k": 2,
            "first_moe_layer": 1,
            "qk_norm": True,
            "tie_embeddings": True,
        },
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=15,
        candidate_class="researched",
    ),
    TemplateSpec(
        name="mm1m_r16_mamba_multihead",
        family="mamba-lm",
        description="Rank-16 ~1.16M novel multi-head selective SSM.",
        arch={
            **_MM1M_BASE,
            "n_layers": 5,
            "ffn_hidden": 192,
            "variant": "multihead",
            "state_dim": 16,
            "expand": 1,
            "conv_kernel": 4,
            "value_residual": True,
        },
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=16,
        candidate_class="novel-mamba",
    ),
    TemplateSpec(
        name="mm1m_r17_mamba_conv_gate",
        family="mamba-lm",
        description="Rank-17 ~1.11M novel conv-gated selective SSM (forced short conv).",
        arch={
            **_MM1M_BASE,
            "n_layers": 5,
            "ffn_hidden": 224,
            "variant": "conv_gate",
            "state_dim": 16,
            "expand": 1,
            "conv_kernel": 4,
            "value_residual": True,
        },
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=17,
        candidate_class="novel-mamba",
    ),
    TemplateSpec(
        name="mm1m_r18_dense_novr",
        family="dense-transformer",
        description="Rank-18 ~1.03M dense without value residual (researched ablation).",
        arch={**_MM1M_BASE, "n_layers": 5, "ffn_hidden": 256, "value_residual": False},
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=18,
        candidate_class="researched",
    ),
    TemplateSpec(
        name="mm1m_r19_mamba_pure",
        family="mamba-lm",
        description="Rank-19 ~1.11M novel pure selective-SSM stack (no attention).",
        arch={
            **_MM1M_BASE,
            "n_layers": 5,
            "ffn_hidden": 224,
            "variant": "pure",
            "state_dim": 16,
            "expand": 1,
            "conv_kernel": 4,
            "value_residual": True,
        },
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=19,
        candidate_class="novel-mamba",
    ),
    TemplateSpec(
        name="mm1m_r20_dense_ffn4x",
        family="dense-transformer",
        description="Rank-20 ~1.18M dense with ~4x FFN (researched; report 05 width sweep).",
        arch={**_MM1M_BASE, "n_layers": 4, "ffn_hidden": 448, "value_residual": True},
        recommended_tokens="0.3-1B",
        training_defaults=dict(_MM1M_TRAIN),
        glint2_rank=20,
        candidate_class="researched",
    ),
    # ---------------- looped (recurrent depth) ----------------
    TemplateSpec(
        name="looped_4m",
        family="looped-transformer",
        description="Looped transformer, ~4M params. Character-to-word level modelling on a laptop.",
        arch={
            "vocab_size": 8192,
            "dim": 256,
            "n_heads": 8,
            "head_dim": 32,
            "ffn_hidden": ffn_hidden(256),
            "embedding_rank": 96,
            "window": 512,
            "max_seq_len": 2048,
            "n_shared_blocks": 2,
            "train_loops": 8,
            "min_loops": 4,
            "max_loops_table": 16,
            "loop_lora_rank": 8,
        },
        recommended_tokens="1-3B",
        training_defaults={"lr": 2.0e-3, "batch_tokens": 65536, "seq_len": 512},
        notes=[
            "18 effective layers from 4 blocks of weights.",
            "Raise `loops` at inference for a free quality/latency trade.",
        ],
    ),
    TemplateSpec(
        name="looped_16m",
        family="looped-transformer",
        description="Looped transformer, ~16M params. The sweet spot for the looped design.",
        arch={
            "vocab_size": 16384,
            "dim": 448,
            "n_heads": 8,
            "head_dim": 56,
            "ffn_hidden": ffn_hidden(448),
            "embedding_rank": 192,
            "window": 1024,
            "max_seq_len": 4096,
            "n_shared_blocks": 3,
            "train_loops": 8,
            "min_loops": 4,
            "max_loops_table": 16,
            "loop_lora_rank": 16,
        },
        recommended_tokens="5-15B",
        training_defaults={"lr": 1.2e-3, "batch_tokens": 262144, "seq_len": 1024},
    ),
    # ---------------- dense ----------------
    TemplateSpec(
        name="dense_1_4m",
        family="dense-transformer",
        description=(
            "Dense GQA transformer at ~1.406M params — the Glint-2 crusher. "
            "Matched to the 1.4M budget with QK-norm, tied embeddings and value "
            "residuals; preferred over looped at fixed training tokens."
        ),
        arch={
            "vocab_size": 4096,
            "dim": 128,
            "n_layers": 5,
            "n_heads": 8,
            "head_dim": 16,
            "n_kv_heads": 2,
            "ffn_hidden": 352,
            "window": 512,
            "max_seq_len": 1024,
            "qk_norm": True,
            "tie_embeddings": True,
            "value_residual": True,
        },
        recommended_tokens="0.4-1B",
        training_defaults={"lr": 3.0e-3, "batch_tokens": 65536, "seq_len": 1024},
        notes=[
            "Sandbox bake-off winner near 1.4M (tied with MHA/deep variants; "
            "GQA chosen for T4 throughput). Crush-glint2 v2 mix: FineWeb-Edu "
            "55% + DCLM-100BT 28% + TinyStories 12% + soft-label QA 5%.",
        ],
    ),
    TemplateSpec(
        name="dense_3m",
        family="dense-transformer",
        description="Tiny dense transformer, ~3M params. Fast to train end-to-end for debugging.",
        arch={
            "vocab_size": 4096,
            "dim": 192,
            "n_layers": 6,
            "n_heads": 6,
            "head_dim": 32,
            "n_kv_heads": 2,
            "ffn_hidden": ffn_hidden(192),
            "window": 512,
            "max_seq_len": 1024,
            "qk_norm": True,
            "tie_embeddings": True,
        },
        recommended_tokens="0.5-1B",
        training_defaults={"lr": 3.0e-3, "batch_tokens": 32768, "seq_len": 256},
    ),
    TemplateSpec(
        name="dense_12m",
        family="dense-transformer",
        description="Dense transformer, ~12M params. Coherent short-form text on a single GPU.",
        arch={
            "vocab_size": 8192,
            "dim": 384,
            "n_layers": 6,
            "n_heads": 6,
            "head_dim": 64,
            "n_kv_heads": 2,
            "ffn_hidden": ffn_hidden(384),
            "window": 1024,
            "max_seq_len": 2048,
            "qk_norm": True,
            "tie_embeddings": True,
        },
        recommended_tokens="2-6B",
        training_defaults={"lr": 2.0e-3, "batch_tokens": 131072, "seq_len": 512},
    ),
    TemplateSpec(
        name="dense_30m",
        family="dense-transformer",
        description="Dense transformer, ~30M params. Usable instruction following after SFT.",
        arch={
            "vocab_size": 16384,
            "dim": 512,
            "n_layers": 8,
            "n_heads": 8,
            "head_dim": 64,
            "n_kv_heads": 2,
            "ffn_hidden": ffn_hidden(512),
            "window": 1024,
            "window_pattern": 4,
            "max_seq_len": 4096,
            "qk_norm": True,
            "tie_embeddings": True,
        },
        recommended_tokens="6-20B",
        training_defaults={"lr": 1.2e-3, "batch_tokens": 262144, "seq_len": 1024},
        notes=["Every 4th layer uses full attention; the rest are windowed."],
    ),
    TemplateSpec(
        name="dense_60m",
        family="dense-transformer",
        description="Dense transformer, ~60M params. The largest size that still trains overnight on one consumer GPU.",
        arch={
            "vocab_size": 32000,
            "dim": 640,
            "n_layers": 10,
            "n_heads": 10,
            "head_dim": 64,
            "n_kv_heads": 2,
            "ffn_hidden": ffn_hidden(640),
            "window": 2048,
            "window_pattern": 4,
            "max_seq_len": 4096,
            "qk_norm": True,
            "tie_embeddings": True,
        },
        recommended_tokens="15-40B",
        training_defaults={"lr": 8.0e-4, "batch_tokens": 524288, "seq_len": 2048},
    ),
    TemplateSpec(
        name="dense_125m",
        family="dense-transformer",
        description="Dense transformer, ~125M params. GPT-2 small class, modernised.",
        arch={
            "vocab_size": 32000,
            "dim": 768,
            "n_layers": 16,
            "n_heads": 12,
            "head_dim": 64,
            "n_kv_heads": 4,
            "ffn_hidden": ffn_hidden(768),
            "window": 2048,
            "window_pattern": 4,
            "max_seq_len": 8192,
            "qk_norm": True,
            "tie_embeddings": True,
        },
        recommended_tokens="30-100B",
        training_defaults={"lr": 6.0e-4, "batch_tokens": 1048576, "seq_len": 2048},
    ),
    TemplateSpec(
        name="dense_350m",
        family="dense-transformer",
        description="Dense transformer, ~350M params. Multi-GPU territory.",
        arch={
            "vocab_size": 32000,
            "dim": 1024,
            "n_layers": 28,
            "n_heads": 16,
            "head_dim": 64,
            "n_kv_heads": 4,
            "ffn_hidden": ffn_hidden(1024),
            "window": 4096,
            "window_pattern": 4,
            "max_seq_len": 8192,
            "qk_norm": True,
            "tie_embeddings": True,
        },
        recommended_tokens="100-300B",
        training_defaults={"lr": 4.0e-4, "batch_tokens": 2097152, "seq_len": 4096},
    ),
    # ---------------- mixture of experts ----------------
    TemplateSpec(
        name="moe_28m_a16m",
        family="moe-transformer",
        description="Sparse MoE, ~28M total / ~16M active. Dense-16M compute, noticeably better quality.",
        arch={
            "vocab_size": 16384,
            "dim": 384,
            "n_layers": 8,
            "n_heads": 6,
            "head_dim": 64,
            "n_kv_heads": 2,
            "ffn_hidden": 256,
            "window": 1024,
            "max_seq_len": 2048,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "top_k": 2,
            "first_moe_layer": 1,
            "qk_norm": True,
        },
        recommended_tokens="8-25B",
        training_defaults={"lr": 1.5e-3, "batch_tokens": 262144, "seq_len": 1024},
        notes=[
            "Layer 0 stays dense: early representations route poorly.",
            "Balance is maintained by a routing bias, not an auxiliary loss.",
        ],
    ),
    TemplateSpec(
        name="moe_135m_a44m",
        family="moe-transformer",
        description="Sparse MoE, ~135M total / ~44M active.",
        arch={
            "vocab_size": 32000,
            "dim": 512,
            "n_layers": 12,
            "n_heads": 8,
            "head_dim": 64,
            "n_kv_heads": 2,
            "ffn_hidden": 384,
            "window": 2048,
            "max_seq_len": 4096,
            "n_routed_experts": 16,
            "n_shared_experts": 1,
            "top_k": 2,
            "first_moe_layer": 1,
            "qk_norm": True,
        },
        recommended_tokens="30-80B",
        training_defaults={"lr": 1.0e-3, "batch_tokens": 524288, "seq_len": 2048},
    ),
    # ---------------- hybrid recurrent ----------------
    TemplateSpec(
        name="hybrid_35m",
        family="hybrid-recurrent",
        description="Griffin-style hybrid, ~35M params. Constant-size decode state, long context.",
        arch={
            "vocab_size": 16384,
            "dim": 512,
            "n_layers": 9,
            "n_heads": 8,
            "head_dim": 64,
            "n_kv_heads": 2,
            "ffn_hidden": ffn_hidden(512),
            "window": 512,
            "max_seq_len": 8192,
            "layer_pattern": ["recurrent", "recurrent", "attention"],
            "qk_norm": True,
        },
        recommended_tokens="8-25B",
        training_defaults={"lr": 1.2e-3, "batch_tokens": 262144, "seq_len": 1024},
        notes=[
            "Only 1 layer in 3 keeps a KV cache, so long-context decoding stays cheap.",
        ],
    ),
    TemplateSpec(
        name="hybrid_150m",
        family="hybrid-recurrent",
        description="Griffin-style hybrid, ~150M params, 16K context.",
        arch={
            "vocab_size": 32000,
            "dim": 768,
            "n_layers": 18,
            "n_heads": 12,
            "head_dim": 64,
            "n_kv_heads": 4,
            "ffn_hidden": ffn_hidden(768),
            "window": 1024,
            "max_seq_len": 16384,
            "layer_pattern": ["recurrent", "recurrent", "attention"],
            "qk_norm": True,
        },
        recommended_tokens="40-120B",
        training_defaults={"lr": 6.0e-4, "batch_tokens": 1048576, "seq_len": 2048},
    ),
]


def build_template_document(spec: TemplateSpec) -> dict[str, Any]:
    """Instantiate the model to get exact counts, then assemble the YAML body."""
    model_cls = ARCHITECTURES.get(spec.family)
    model = model_cls.from_config(spec.arch)
    document: dict[str, Any] = {
        "name": spec.name,
        "family": spec.family,
        "description": spec.description,
        "params": model.num_parameters(),
    }
    active = getattr(model, "active_parameters", None)
    if callable(active):
        document["active_params"] = active()
    if spec.glint2_rank is not None:
        document["glint2_rank"] = int(spec.glint2_rank)
    if spec.candidate_class:
        document["candidate_class"] = spec.candidate_class
    if spec.recommended_tokens:
        document["recommended_tokens"] = spec.recommended_tokens
    document["arch"] = dict(spec.arch)
    if spec.training_defaults:
        document["training_defaults"] = dict(spec.training_defaults)
    if spec.notes:
        document["notes"] = list(spec.notes)
    return document


def render(spec: TemplateSpec) -> str:
    """Render one template to YAML text."""
    document = build_template_document(spec)
    header = (
        "# Generated by scripts/generate_templates.py - edit that file, not this one.\n"
        f"# {spec.description}\n"
    )
    return header + yaml.safe_dump(document, sort_keys=False, default_flow_style=False)


def main() -> int:
    """Write or verify every generated template."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="exit non-zero if any template is out of date"
    )
    args = parser.parse_args()

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    for spec in SPECS:
        path = TEMPLATE_DIR / f"{spec.name}.yaml"
        text = render(spec)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(spec.name)
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(REPO_ROOT)}")

    if args.check and stale:
        print("out-of-date templates: " + ", ".join(stale), file=sys.stderr)
        return 1
    if args.check:
        print("all templates up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
