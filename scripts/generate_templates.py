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


def ffn_hidden(dim: int, ratio: float = 8 / 3, multiple: int = 32) -> int:
    """SwiGLU hidden size: ``ratio * dim`` rounded up to a multiple of 32.

    The 8/3 ratio keeps a SwiGLU FFN at the same parameter count as a plain
    4x GELU FFN, which is what the ratio is chosen to match.
    """
    raw = int(dim * ratio)
    return ((raw + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# The size ladder
# ---------------------------------------------------------------------------
SPECS: list[TemplateSpec] = [
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
