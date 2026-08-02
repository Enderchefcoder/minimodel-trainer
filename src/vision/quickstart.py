"""The image pipeline, end to end, on synthetic sprites.

``minimodel vision quickstart`` prepares a small corpus of procedurally
generated sprites, trains both a PixelGPT and a diffusion transformer on it, and
samples from each. Like the text quickstart, the resulting models are toys; the
point is to prove the pipeline works before spending real compute.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from minimodel.core.io_utils import ensure_dir, human_duration
from minimodel.core.logging_utils import get_logger
from minimodel.vision.architectures.dit import DiT
from minimodel.vision.architectures.pixelgpt import PixelGPT
from minimodel.vision.data.datasets import (
    ImageDataset,
    PixelSequenceDataset,
    prepare_image_corpus,
    prepare_pixel_corpus,
    synthetic_sprites,
)
from minimodel.vision.data.palette import Palette
from minimodel.vision.sampling.generate import sample_and_save, sample_pixel_art
from minimodel.vision.training.diffusion import DiffusionConfig, DiffusionTrainer
from minimodel.vision.training.pixel_trainer import PixelGPTConfig, PixelGPTTrainer

__all__ = ["run_vision_quickstart"]

logger = get_logger(__name__)


def run_vision_quickstart(
    output_dir: str | Path = "runs/vision-quickstart",
    *,
    steps: int = 200,
    n_sprites: int = 256,
    size: int = 16,
    palette_size: int = 16,
    batch_size: int = 16,
    seed: int = 1337,
) -> dict[str, Any]:
    """Prepare data, train PixelGPT and a DiT, and sample from both."""
    started = time.perf_counter()
    root = ensure_dir(output_dir)
    summary: dict[str, Any] = {"output_dir": str(root)}

    logger.info("[1/5] generating %d synthetic sprites", n_sprites)
    sprites = synthetic_sprites(n_sprites, size=size, n_colors=palette_size, seed=seed)

    logger.info("[2/5] building palette and RGB corpora")
    palette_stats = prepare_pixel_corpus(
        sprites,
        root / "data" / "palette",
        size=size,
        palette_size=palette_size,
        source="synthetic",
    )
    rgb_stats = prepare_image_corpus(
        sprites, root / "data" / "rgb", size=size, source="synthetic"
    )
    summary["corpora"] = {
        "palette": {"images": palette_stats["n_images"], "colors": palette_stats["palette_size"]},
        "rgb": {"images": rgb_stats["n_images"]},
    }

    logger.info("[3/5] training PixelGPT for %d steps", steps)
    pixel_model = PixelGPT(
        {
            "image_size": size,
            "palette_size": palette_stats["palette_size"],
            "dim": 128,
            "n_layers": 4,
            "n_heads": 4,
            "head_dim": 32,
            "n_kv_heads": 2,
            "ffn_hidden": 320,
            "num_classes": palette_stats["n_classes"],
        }
    )
    pixel_config = PixelGPTConfig(
        run_name="pixelgpt",
        output_dir=str(root),
        max_steps=steps,
        batch_size=batch_size,
        seq_len=size * size,
        lr=1e-3,
        log_every=max(1, steps // 5),
        eval_every=0,
        save_every=0,
        seed=seed,
        resume=False,
    )
    pixel_result = PixelGPTTrainer(
        pixel_model,
        pixel_config,
        train_dataset=PixelSequenceDataset(root / "data" / "palette"),
    ).fit()
    summary["pixelgpt"] = {
        "params": pixel_model.num_parameters(),
        "steps": pixel_result.steps,
        "final_loss": round(pixel_result.final_loss, 4),
    }

    logger.info("[4/5] training a diffusion transformer for %d steps", steps)
    dit = DiT(
        {
            "image_size": size,
            "patch_size": 2,
            "dim": 128,
            "depth": 4,
            "n_heads": 4,
            "condition": "class" if rgb_stats["n_classes"] else "none",
            "num_classes": rgb_stats["n_classes"],
        }
    )
    diffusion_config = DiffusionConfig(
        run_name="dit",
        output_dir=str(root),
        max_steps=steps,
        batch_size=batch_size,
        lr=3e-4,
        log_every=max(1, steps // 5),
        save_every=0,
        seed=seed,
        resume=False,
    )
    diffusion_trainer = DiffusionTrainer(
        dit, diffusion_config, dataset=ImageDataset(root / "data" / "rgb")
    )
    diffusion_result = diffusion_trainer.fit()
    summary["dit"] = {
        "params": dit.num_parameters(),
        "steps": diffusion_result["steps"],
        "final_loss": round(diffusion_result["final_loss"], 4),
    }

    logger.info("[5/5] sampling from both models")
    palette = Palette.load(root / "data" / "palette" / "palette.json")
    pixel_samples = sample_pixel_art(
        pixel_model,
        n_samples=9,
        palette=palette,
        temperature=0.8,
        seed=seed,
        output=root / "samples_pixelgpt.png",
        scale=8,
    )
    diffusion_samples = sample_and_save(
        dit,
        root / "samples_dit.png",
        n_samples=9,
        n_steps=25,
        labels=[0] * 9 if rgb_stats["n_classes"] else None,
        seed=seed,
        scale=8,
    )
    summary["samples"] = {
        "pixelgpt": str(pixel_samples),
        "dit": str(diffusion_samples),
    }

    pixel_model.save_pretrained(root / "model_pixelgpt")
    palette.save(root / "model_pixelgpt" / "palette.json")
    diffusion_trainer.export(root / "model_dit")
    summary["models"] = {
        "pixelgpt": str(root / "model_pixelgpt"),
        "dit": str(root / "model_dit"),
    }
    summary["elapsed"] = human_duration(time.perf_counter() - started)
    logger.info("vision quickstart finished in %s", summary["elapsed"])
    return summary
