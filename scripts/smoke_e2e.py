#!/usr/bin/env python3
"""End-to-end smoke test of every pipeline, offline.

Runs the text pipeline (tokenizer, corpus, pretrain, SFT, CoT, DPO, RLVR, SPIN,
merge, benchmark, model card) and the image pipeline (palette corpus, PixelGPT,
diffusion, sampling) on the bundled data, with tiny models and step counts.

This is what CI runs to catch integration breakage that unit tests miss - a
config key renamed in one place, a trainer that no longer accepts the batch
shape its dataset produces, an export path that writes the wrong file.

Usage::

    python scripts/smoke_e2e.py [--output DIR] [--keep]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from minimodel.architectures.builder import build_model  # noqa: E402
from minimodel.benchmarking.bench import run_suite  # noqa: E402
from minimodel.cardgen.modelcard_autogen import generate_model_card  # noqa: E402
from minimodel.core.logging_utils import get_logger, setup_logging  # noqa: E402
from minimodel.datasets.builtin import builtin_records  # noqa: E402
from minimodel.datasets.loader import PackedTextDataset, SupervisedDataset  # noqa: E402
from minimodel.datasets.tokenize_datasets import (  # noqa: E402
    tokenize_chat_records,
    tokenize_preference_records,
    tokenize_text_records,
)
from minimodel.merging.slerp import merge_models  # noqa: E402
from minimodel.tokenization.tokenize import BPETokenizer  # noqa: E402
from minimodel.training.instruct_cot_posttrainer import CoTTrainer, CoTTrainerConfig  # noqa: E402
from minimodel.training.instruct_posttrainer import (  # noqa: E402
    InstructTrainer,
    InstructTrainerConfig,
)
from minimodel.training.rl.dpo import DPOConfig, DPOTrainer  # noqa: E402
from minimodel.training.rl.rlvr import RLVRConfig, RLVRTrainer  # noqa: E402
from minimodel.training.rl.spin import SPINConfig, SPINTrainer  # noqa: E402
from minimodel.training.trainer import Trainer, TrainerConfig  # noqa: E402

logger = get_logger("smoke")

SMALL_MODEL = {
    "dim": 64,
    "n_layers": 2,
    "n_heads": 2,
    "head_dim": 32,
    "n_kv_heads": 1,
    "ffn_hidden": 128,
    "max_seq_len": 256,
    "window": 128,
}


def _stage(name: str, index: int, total: int) -> None:
    logger.info("[%d/%d] %s", index, total, name)


def run_text_pipeline(root: Path) -> dict[str, Any]:
    """Every text stage, end to end."""
    results: dict[str, Any] = {}
    total = 9

    _stage("tokenizer", 1, total)
    texts = [record["text"] for record in builtin_records("pretrain", repeat=6)]
    tokenizer = BPETokenizer.train(texts, vocab_size=512)
    tokenizer_path = tokenizer.save(root / "tokenizer.json")
    results["tokenizer"] = {"vocab_size": tokenizer.vocab_size, "path": str(tokenizer_path)}

    _stage("corpora", 2, total)
    tokenize_text_records(({"text": t} for t in texts), tokenizer, root / "data" / "pretrain")
    tokenize_chat_records(builtin_records("sft", repeat=8), tokenizer, root / "data" / "sft")
    tokenize_chat_records(builtin_records("cot", repeat=10), tokenizer, root / "data" / "cot")
    tokenize_preference_records(
        builtin_records("preference", repeat=6), tokenizer, root / "data" / "pairs.jsonl"
    )
    (root / "data" / "tasks.jsonl").write_text(
        "\n".join(json.dumps(task) for task in builtin_records("rlvr")), encoding="utf-8"
    )
    (root / "data" / "sft.jsonl").write_text(
        "\n".join(json.dumps(record) for record in builtin_records("sft")), encoding="utf-8"
    )

    overrides = {**SMALL_MODEL, "vocab_size": tokenizer.vocab_size}

    _stage("pretraining", 3, total)
    model = build_model("dense_3m", overrides=overrides, verify_budget=False)
    pretrain = Trainer(
        model,
        TrainerConfig(
            run_name="pretrain",
            output_dir=str(root / "runs"),
            max_steps=30,
            batch_size=4,
            seq_len=64,
            lr=3e-3,
            log_every=15,
            eval_every=15,
            eval_batches=2,
            save_every=15,
            resume=False,
        ),
        train_dataset=PackedTextDataset(root / "data" / "pretrain", seq_len=64),
        eval_dataset=PackedTextDataset(root / "data" / "pretrain", seq_len=64, seed=7),
        tokenizer=tokenizer,
    )
    pretrain_result = pretrain.fit()
    pretrain.export(root / "runs" / "pretrain" / "model")
    tokenizer.save(root / "runs" / "pretrain" / "model" / "tokenizer.json")
    results["pretrain"] = {"loss": round(pretrain_result.final_loss, 4)}

    _stage("instruction tuning", 4, total)
    sft = InstructTrainer(
        model,
        InstructTrainerConfig(
            run_name="sft",
            output_dir=str(root / "runs"),
            max_steps=15,
            batch_size=4,
            seq_len=64,
            lr=5e-4,
            log_every=15,
            eval_every=0,
            save_every=0,
            resume=False,
        ),
        train_dataset=SupervisedDataset(root / "data" / "sft", seq_len=64),
        tokenizer=tokenizer,
    )
    results["sft"] = {"loss": round(sft.fit().final_loss, 4)}
    sft.export(root / "runs" / "sft" / "model")

    _stage("chain-of-thought", 5, total)
    cot = CoTTrainer(
        build_model("dense_3m", overrides=overrides, verify_budget=False),
        CoTTrainerConfig(
            run_name="cot",
            output_dir=str(root / "runs"),
            max_steps=10,
            batch_size=4,
            seq_len=64,
            lr=5e-4,
            log_every=10,
            eval_every=0,
            save_every=0,
            reasoning_loss_weight=0.5,
            enforce_think_close=0.1,
            resume=False,
        ),
        train_dataset=SupervisedDataset(root / "data" / "cot", seq_len=64),
        tokenizer=tokenizer,
    )
    results["cot"] = {"loss": round(cot.fit().final_loss, 4)}

    _stage("DPO", 6, total)
    dpo = DPOTrainer(
        build_model("dense_3m", overrides=overrides, verify_budget=False),
        DPOConfig(
            run_name="dpo",
            output_dir=str(root / "runs"),
            max_steps=6,
            batch_size=2,
            lr=1e-5,
            log_every=6,
            eval_every=0,
            save_every=0,
            resume=False,
        ),
        pairs_path=root / "data" / "pairs.jsonl",
        tokenizer=tokenizer,
    )
    results["dpo"] = {"loss": round(dpo.fit().final_loss, 4)}

    _stage("RLVR", 7, total)
    rlvr = RLVRTrainer(
        build_model("dense_3m", overrides=overrides, verify_budget=False),
        RLVRConfig(
            run_name="rlvr",
            output_dir=str(root / "runs"),
            max_steps=2,
            batch_size=2,
            group_size=4,
            max_new_tokens=8,
            log_every=2,
            eval_every=0,
            save_every=0,
            resume=False,
        ),
        tasks_path=root / "data" / "tasks.jsonl",
        tokenizer=tokenizer,
    )
    results["rlvr"] = {"loss": round(rlvr.fit().final_loss, 4)}

    _stage("SPIN", 8, total)
    spin = SPINTrainer(
        build_model("dense_3m", overrides=overrides, verify_budget=False),
        SPINConfig(
            run_name="spin",
            output_dir=str(root / "runs"),
            max_steps=4,
            batch_size=2,
            iterations=2,
            max_new_tokens=8,
            lr=1e-5,
            log_every=4,
            eval_every=0,
            save_every=0,
            resume=False,
        ),
        dataset_path=root / "data" / "sft.jsonl",
        tokenizer=tokenizer,
    )
    results["spin"] = {"loss": round(spin.fit().final_loss, 4)}

    _stage("merge, benchmark and card", 9, total)
    merge_models(
        [root / "runs" / "pretrain" / "model", root / "runs" / "sft" / "model"],
        method="slerp",
        output=root / "runs" / "merged",
        t=0.5,
    )
    benchmark = run_suite(
        model,
        tokenizer,
        perplexity_corpus=root / "data" / "pretrain",
        model_name="smoke",
        include_throughput=True,
    )
    benchmark_path = benchmark.save(root / "benchmark.json")
    card = generate_model_card(
        root / "README.md",
        run_dir=root / "runs" / "pretrain",
        model_dir=root / "runs" / "pretrain" / "model",
        benchmark_paths=[benchmark_path],
        name="smoke-model",
        organization="local",
    )
    results["benchmark"] = benchmark.headline()
    results["card"] = str(card)
    results["merged"] = str(root / "runs" / "merged")
    return results


def run_vision_pipeline(root: Path) -> dict[str, Any]:
    """Every image stage, end to end."""
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

    results: dict[str, Any] = {}
    sprites = synthetic_sprites(48, size=16, n_colors=8, seed=0)

    _stage("image corpora", 1, 4)
    palette_stats = prepare_pixel_corpus(
        sprites, root / "images" / "palette", size=16, palette_size=16
    )
    prepare_image_corpus(sprites, root / "images" / "rgb", size=16)
    results["corpus"] = {
        "images": palette_stats["n_images"],
        "colors": palette_stats["palette_size"],
    }

    _stage("PixelGPT", 2, 4)
    pixel_model = build_pixel_model(palette_stats)
    pixel_trainer = PixelGPTTrainer(
        pixel_model,
        PixelGPTConfig(
            run_name="pixelgpt",
            output_dir=str(root / "runs"),
            max_steps=10,
            batch_size=8,
            seq_len=256,
            lr=1e-3,
            log_every=10,
            eval_every=0,
            save_every=0,
            resume=False,
        ),
        train_dataset=PixelSequenceDataset(root / "images" / "palette"),
    )
    results["pixelgpt"] = {"loss": round(pixel_trainer.fit().final_loss, 4)}

    _stage("diffusion", 3, 4)
    from minimodel.vision.architectures.dit import DiT

    dit = DiT({"image_size": 16, "patch_size": 2, "dim": 64, "depth": 2, "n_heads": 2})
    diffusion = DiffusionTrainer(
        dit,
        DiffusionConfig(
            run_name="dit",
            output_dir=str(root / "runs"),
            max_steps=10,
            batch_size=8,
            lr=3e-4,
            log_every=10,
            save_every=0,
            resume=False,
        ),
        dataset=ImageDataset(root / "images" / "rgb"),
    )
    results["diffusion"] = {"loss": round(diffusion.fit()["final_loss"], 4)}

    _stage("sampling", 4, 4)
    palette = Palette.load(root / "images" / "palette" / "palette.json")
    sample_pixel_art(
        pixel_model,
        n_samples=4,
        palette=palette,
        seed=0,
        output=root / "sprites.png",
        scale=4,
    )
    sample_and_save(dit, root / "diffusion.png", n_samples=4, n_steps=5, seed=0, scale=4)
    results["samples"] = [str(root / "sprites.png"), str(root / "diffusion.png")]
    return results


def build_pixel_model(palette_stats: dict[str, Any]):
    """A tiny PixelGPT sized for the smoke corpus."""
    from minimodel.vision.architectures.pixelgpt import PixelGPT

    return PixelGPT(
        {
            "image_size": 16,
            "palette_size": palette_stats["palette_size"],
            "dim": 64,
            "n_layers": 2,
            "n_heads": 2,
            "head_dim": 32,
            "n_kv_heads": 1,
            "ffn_hidden": 128,
            "num_classes": palette_stats["n_classes"],
        }
    )


def main() -> int:
    """Run both pipelines and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", help="where to write artifacts (default: a temp dir)")
    parser.add_argument("--keep", action="store_true", help="keep the output directory")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--vision-only", action="store_true")
    args = parser.parse_args()

    setup_logging(force=True)
    torch.manual_seed(0)

    root = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="minimodel-smoke-"))
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary: dict[str, Any] = {"output": str(root)}

    try:
        if not args.vision_only:
            logger.info("=== text pipeline ===")
            summary["text"] = run_text_pipeline(root / "text")
        if not args.text_only:
            logger.info("=== image pipeline ===")
            summary["vision"] = run_vision_pipeline(root / "vision")
    except Exception:
        logger.exception("smoke test failed")
        return 1
    finally:
        if not args.keep and not args.output:
            shutil.rmtree(root, ignore_errors=True)

    summary["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    print(json.dumps(summary, indent=2, default=str))
    logger.info("smoke test passed in %.1fs", summary["elapsed_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
