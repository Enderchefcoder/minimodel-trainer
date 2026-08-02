#!/usr/bin/env python3
"""Regenerate the bundled image-model templates.

Same contract as ``scripts/generate_templates.py``: the parameter counts written
into the templates are always the counts the code actually produces.

Usage::

    python scripts/generate_vision_templates.py
    python scripts/generate_vision_templates.py --check
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from minimodel.vision.architectures.registry import (  # noqa: E402
    IMAGE_ARCHITECTURES,
    VISION_TEMPLATE_DIR,
)


@dataclass
class VisionTemplateSpec:
    """One entry in the image-model ladder."""

    name: str
    family: str
    description: str
    arch: dict[str, Any]
    recommended_data: str = ""
    training_defaults: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


SPECS: list[VisionTemplateSpec] = [
    # ---------------- pixel art ----------------
    VisionTemplateSpec(
        name="pixelgpt_24x24_10m",
        family="pixelgpt",
        description="Autoregressive 24x24 pixel-art model, ~10M params. The flagship sprite model.",
        arch={
            "image_size": 24,
            "palette_size": 64,
            "dim": 320,
            "n_layers": 8,
            "n_heads": 8,
            "head_dim": 40,
            "n_kv_heads": 4,
            "ffn_hidden": 960,
            "qk_norm": True,
        },
        recommended_data="unstonio/pixelgpt-24x24-20k (20K sprites)",
        training_defaults={"lr": 6.0e-4, "batch_size": 64, "max_steps": 20000},
        notes=[
            "576 tokens per image; sampling one sprite is ~576 forward passes with a KV cache.",
            "Palette output is exact by construction - no quantization artifacts.",
            "Horizontal-flip augmentation roughly doubles a sprite corpus for free.",
        ],
    ),
    VisionTemplateSpec(
        name="pixelgpt_16x16_3m",
        family="pixelgpt",
        description="Autoregressive 16x16 pixel-art model, ~3M params. Trains in minutes.",
        arch={
            "image_size": 16,
            "palette_size": 32,
            "dim": 192,
            "n_layers": 8,
            "n_heads": 6,
            "head_dim": 32,
            "n_kv_heads": 2,
            "ffn_hidden": 512,
            "qk_norm": True,
        },
        recommended_data="16x16 sprite sets, or `--synthetic`",
        training_defaults={"lr": 1.0e-3, "batch_size": 64, "max_steps": 5000},
    ),
    VisionTemplateSpec(
        name="pixelgpt_32x32_30m",
        family="pixelgpt",
        description="Autoregressive 32x32 pixel-art model, ~30M params.",
        arch={
            "image_size": 32,
            "palette_size": 128,
            "dim": 512,
            "n_layers": 10,
            "n_heads": 8,
            "head_dim": 64,
            "n_kv_heads": 4,
            "ffn_hidden": 1376,
            "qk_norm": True,
        },
        recommended_data="Norod78/Pixel-Art-Sprites (90K sprites)",
        training_defaults={"lr": 4.0e-4, "batch_size": 32, "max_steps": 40000},
    ),
    # ---------------- diffusion transformers ----------------
    VisionTemplateSpec(
        name="dit_tiny_32",
        family="dit",
        description="Diffusion transformer for 32x32 images, ~10M params.",
        arch={
            "image_size": 32,
            "patch_size": 2,
            "dim": 256,
            "depth": 8,
            "n_heads": 4,
            "condition": "class",
            "num_classes": 10,
        },
        recommended_data="CIFAR-10",
        training_defaults={"lr": 3.0e-4, "batch_size": 128, "max_steps": 50000},
        notes=["Flow-matching objective; 25-50 Euler steps at sampling time."],
    ),
    VisionTemplateSpec(
        name="dit_small_64",
        family="dit",
        description="Diffusion transformer for 64x64 images, ~33M params.",
        arch={
            "image_size": 64,
            "patch_size": 4,
            "dim": 384,
            "depth": 12,
            "n_heads": 6,
            "condition": "class",
            "num_classes": 10,
        },
        recommended_data="CelebA-HQ 64, anime-faces",
        training_defaults={"lr": 2.0e-4, "batch_size": 64, "max_steps": 100000},
    ),
    VisionTemplateSpec(
        name="dit_text_64",
        family="dit",
        description="Text-conditioned diffusion transformer for 64x64 images, ~38M params.",
        arch={
            "image_size": 64,
            "patch_size": 4,
            "dim": 384,
            "depth": 12,
            "n_heads": 6,
            "condition": "text",
            "text_vocab_size": 4096,
            "text_max_len": 32,
        },
        recommended_data="pokemon-blip, diffusiondb-pixelart",
        training_defaults={"lr": 2.0e-4, "batch_size": 64, "max_steps": 100000},
        notes=["Uses the same BPE tokenizer as the language models."],
    ),
    # ---------------- UNets ----------------
    VisionTemplateSpec(
        name="unet_small_32",
        family="unet",
        description="Convolutional UNet for 32x32 images, ~16M params. Best below ~50K images.",
        arch={
            "image_size": 32,
            "base_channels": 96,
            "channel_multipliers": [1, 2, 2],
            "blocks_per_level": 2,
            "attention_resolutions": [16, 8],
            "condition": "class",
            "num_classes": 10,
        },
        recommended_data="CIFAR-10, flowers102, butterflies",
        training_defaults={"lr": 2.0e-4, "batch_size": 128, "max_steps": 50000},
    ),
    VisionTemplateSpec(
        name="unet_base_64",
        family="unet",
        description="Convolutional UNet for 64x64 images, ~35M params.",
        arch={
            "image_size": 64,
            "base_channels": 80,
            "channel_multipliers": [1, 2, 3, 4],
            "blocks_per_level": 2,
            "attention_resolutions": [16, 8],
            "condition": "none",
        },
        recommended_data="CelebA-HQ 64, anime-faces",
        training_defaults={"lr": 1.5e-4, "batch_size": 32, "max_steps": 120000},
    ),
    # ---------------- editing ----------------
    VisionTemplateSpec(
        name="edit_small_64",
        family="image-edit",
        description="Instruction-based image editor for 64x64 images, ~38M params.",
        arch={
            "image_size": 64,
            "patch_size": 4,
            "dim": 384,
            "depth": 12,
            "n_heads": 6,
            "condition": "text",
            "text_vocab_size": 4096,
            "text_max_len": 32,
            "extra_in_channels": 3,
        },
        recommended_data="timbrooks/instructpix2pix-clip-filtered, then osunlp/MagicBrush",
        training_defaults={"lr": 1.0e-4, "batch_size": 32, "max_steps": 100000},
        notes=[
            "Sample with dual guidance: image_guidance ~1.5, text guidance ~7.",
            "Pretrain on synthetic pairs, then fine-tune on human-annotated edits.",
        ],
    ),
    # ---------------- autoencoder ----------------
    VisionTemplateSpec(
        name="vae_f8_64",
        family="vae",
        description="8x-downsampling autoencoder for latent diffusion, ~16M params.",
        arch={
            "image_size": 64,
            "latent_channels": 4,
            "base_channels": 64,
            "channel_multipliers": [1, 2, 4, 4],
            "blocks_per_level": 2,
        },
        recommended_data="Any image corpus; train this before the latent diffusion model.",
        training_defaults={"lr": 1.0e-4, "batch_size": 32, "max_steps": 50000},
        notes=["Train to convergence first, then freeze it and train diffusion in its latents."],
    ),
]


def build_document(spec: VisionTemplateSpec) -> dict[str, Any]:
    """Instantiate the model to get an exact count, then assemble the YAML body."""
    model_cls = IMAGE_ARCHITECTURES.get(spec.family)
    model = model_cls.from_config(spec.arch)
    document: dict[str, Any] = {
        "name": spec.name,
        "family": spec.family,
        "description": spec.description,
        "params": model.num_parameters(),
    }
    if spec.recommended_data:
        document["recommended_data"] = spec.recommended_data
    document["arch"] = dict(spec.arch)
    if spec.training_defaults:
        document["training_defaults"] = dict(spec.training_defaults)
    if spec.notes:
        document["notes"] = list(spec.notes)
    return document


def render(spec: VisionTemplateSpec) -> str:
    """Render one template to YAML text."""
    header = (
        "# Generated by scripts/generate_vision_templates.py - edit that file, not this one.\n"
        f"# {spec.description}\n"
    )
    return header + yaml.safe_dump(build_document(spec), sort_keys=False, default_flow_style=False)


def main() -> int:
    """Write or verify every generated image template."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero if out of date")
    args = parser.parse_args()

    VISION_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    for spec in SPECS:
        path = VISION_TEMPLATE_DIR / f"{spec.name}.yaml"
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
        print("all image templates up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
