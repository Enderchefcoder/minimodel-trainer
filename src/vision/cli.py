"""The ``minimodel vision`` command group.

Kept separate from :mod:`minimodel.cli` so that the text commands stay readable
and so that importing the text CLI does not pull in the vision stack until it is
needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from minimodel.core.logging_utils import get_logger

__all__ = ["add_vision_parser"]

logger = get_logger(__name__)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def cmd_vision_models(args: argparse.Namespace) -> int:
    """List image architectures and templates."""
    from minimodel.vision.architectures.registry import (
        build_image_model,
        list_image_architectures,
        list_image_templates,
        load_vision_template,
    )

    rows = []
    for name in list_image_templates():
        template = load_vision_template(name)
        row = {
            "template": name,
            "family": template.get("family"),
            "params": template.get("params"),
            "description": template.get("description", ""),
        }
        if args.verify:
            row["built_params"] = build_image_model(name, verify_budget=False).num_parameters()
        rows.append(row)

    if args.json:
        _print_json({"architectures": list_image_architectures(), "templates": rows})
        return 0
    print("Image architectures: " + ", ".join(list_image_architectures()))
    print(f"\n{'TEMPLATE':<24} {'FAMILY':<14} {'PARAMS':>13}  DESCRIPTION")
    for row in rows:
        params = f"{row['params']:,}" if row["params"] else "-"
        print(f"{row['template']:<24} {str(row['family']):<14} {params:>13}  {row['description'][:52]}")
    return 0


def cmd_vision_datasets(args: argparse.Namespace) -> int:
    """List registered image datasets."""
    from minimodel.vision.registry import list_image_datasets

    datasets = list_image_datasets(kind=args.kind)
    if args.json:
        _print_json([d.to_dict() for d in datasets])
        return 0
    print(f"{'NAME':<26} {'KIND':<12} {'SIZE':<8} {'IMAGES':<10} SOURCE")
    for spec in datasets:
        print(
            f"{spec.name:<26} {spec.kind:<12} {str(spec.image_size):<8} "
            f"{str(spec.images or '-'):<10} {spec.display}"
        )
    return 0


def cmd_vision_prepare(args: argparse.Namespace) -> int:
    """Build an image corpus from a directory, a registered dataset or synthetic data."""
    from minimodel.vision.data.datasets import (
        load_images_from_directory,
        prepare_image_corpus,
        prepare_pixel_corpus,
        synthetic_sprites,
    )
    from minimodel.vision.registry import iter_image_records

    if args.synthetic:
        images = synthetic_sprites(args.limit or 512, size=args.size, seed=args.seed)
    elif args.input:
        images = list(load_images_from_directory(args.input, args.size, limit=args.limit))
    elif args.dataset:
        images = list(iter_image_records(args.dataset, size=args.size, limit=args.limit))
    else:
        logger.error("give --input, --dataset or --synthetic")
        return 1

    if args.mode == "palette":
        stats = prepare_pixel_corpus(
            images,
            args.output,
            size=args.size,
            palette_size=args.palette_size,
            palette_method=args.palette_method,
            source=args.dataset or args.input or "synthetic",
        )
    else:
        stats = prepare_image_corpus(
            images, args.output, size=args.size, source=args.dataset or args.input or "synthetic"
        )
    if args.json:
        _print_json(stats)
    else:
        logger.info("prepared %d images -> %s", stats.get("n_images"), args.output)
    return 0


def cmd_vision_info(args: argparse.Namespace) -> int:
    """Print statistics for an image corpus."""
    from minimodel.vision.data.shards import ImageCorpus

    _print_json(ImageCorpus(args.path).stats())
    return 0


def cmd_vision_train(args: argparse.Namespace) -> int:
    """Train an image model from a recipe."""
    from minimodel.vision.recipe import run_vision_recipe

    summary = run_vision_recipe(args.config, cli_overrides=args.overrides)
    if args.json:
        _print_json(summary)
    return 0


def cmd_vision_sample(args: argparse.Namespace) -> int:
    """Sample images from a trained model."""
    from minimodel.vision.architectures.registry import load_image_model
    from minimodel.vision.data.palette import Palette
    from minimodel.vision.sampling.generate import sample_and_save, sample_pixel_art

    model = load_image_model(args.model, device=args.device)
    if model.architecture_name == "pixelgpt":
        palette_path = args.palette or Path(args.model) / "palette.json"
        output = sample_pixel_art(
            model,
            n_samples=args.n,
            palette=Palette.load(palette_path) if Path(palette_path).exists() else None,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            output=args.output,
            scale=args.scale,
            device=args.device,
        )
    else:
        text_tokens = None
        if args.prompt:
            from minimodel.tokenization.tokenize import BPETokenizer

            if not args.tokenizer:
                logger.error("--prompt needs --tokenizer")
                return 1
            import torch

            tokenizer = BPETokenizer.load(args.tokenizer)
            ids = tokenizer.encode(args.prompt, allow_special=False)[:32]
            ids = ids + [0] * (32 - len(ids))
            text_tokens = torch.tensor([ids] * args.n, dtype=torch.long)

        output = sample_and_save(
            model,
            args.output,
            n_samples=args.n,
            sampler=args.sampler,
            n_steps=args.steps,
            guidance_scale=args.guidance,
            labels=[args.label] * args.n if args.label is not None else None,
            text_tokens=text_tokens,
            seed=args.seed,
            scale=args.scale,
            device=args.device,
        )
    if args.json:
        _print_json({"output": str(output)})
    else:
        print(output)
    return 0


def cmd_vision_edit(args: argparse.Namespace) -> int:
    """Apply an instruction edit to an image."""
    import numpy as np
    import torch

    from minimodel.tokenization.tokenize import BPETokenizer
    from minimodel.vision.architectures.registry import load_image_model
    from minimodel.vision.data.datasets import _load_pil
    from minimodel.vision.sampling.generate import save_image_grid
    from minimodel.vision.sampling.samplers import sample_images

    model = load_image_model(args.model, device=args.device)
    size = int(model.config.get("image_size", 64))
    source = _load_pil(Path(args.image), size)
    source_tensor = torch.from_numpy(
        np.asarray(source, dtype=np.float32) / 127.5 - 1.0
    ).permute(2, 0, 1).unsqueeze(0)

    text_tokens = None
    if args.tokenizer:
        tokenizer = BPETokenizer.load(args.tokenizer)
        ids = tokenizer.encode(args.instruction, allow_special=False)[:32]
        text_tokens = torch.tensor([ids + [0] * (32 - len(ids))], dtype=torch.long)

    images = sample_images(
        model,
        n_samples=1,
        sampler=args.sampler,
        n_steps=args.steps,
        guidance_scale=args.guidance,
        text_tokens=text_tokens,
        reference=source_tensor,
        seed=args.seed,
        device=args.device,
    )
    output = save_image_grid(images, args.output, scale=args.scale)
    if args.json:
        _print_json({"output": str(output), "instruction": args.instruction})
    else:
        print(output)
    return 0


def cmd_vision_quickstart(args: argparse.Namespace) -> int:
    """Run the whole image pipeline on synthetic sprites."""
    from minimodel.vision.quickstart import run_vision_quickstart

    summary = run_vision_quickstart(
        output_dir=args.output, steps=args.steps, n_sprites=args.sprites, size=args.size
    )
    if args.json:
        _print_json(summary)
    return 0


def add_vision_parser(subparsers: Any, add_common: Any) -> argparse.ArgumentParser:
    """Attach the ``vision`` command group to the main parser."""
    vision = subparsers.add_parser("vision", help="image model commands")
    vision_sub = vision.add_subparsers(dest="subcommand", metavar="<subcommand>")

    models = vision_sub.add_parser("models", help="list image architectures and templates")
    models.add_argument("--verify", action="store_true")
    add_common(models)
    models.set_defaults(func=cmd_vision_models)

    datasets = vision_sub.add_parser("datasets", help="list registered image datasets")
    datasets.add_argument("--kind", choices=["generation", "pixelart", "edit", "classification"])
    add_common(datasets)
    datasets.set_defaults(func=cmd_vision_datasets)

    data = vision_sub.add_parser("data", help="image corpus commands")
    data_sub = data.add_subparsers(dest="datacommand", metavar="<subcommand>")

    prepare = data_sub.add_parser("prepare", help="build an image corpus")
    prepare.add_argument("-o", "--output", required=True)
    prepare.add_argument("--input", help="directory of images")
    prepare.add_argument("--dataset", help="registered image dataset name")
    prepare.add_argument("--synthetic", action="store_true", help="generate sprites offline")
    prepare.add_argument("--size", type=int, default=24)
    prepare.add_argument("--mode", default="rgb", choices=["rgb", "palette"])
    prepare.add_argument("--palette-size", type=int, default=64)
    prepare.add_argument(
        "--palette-method", default="auto", choices=["auto", "exact", "median_cut", "kmeans"]
    )
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--seed", type=int, default=0)
    add_common(prepare)
    prepare.set_defaults(func=cmd_vision_prepare)

    info = data_sub.add_parser("info", help="show image corpus statistics")
    info.add_argument("path")
    add_common(info)
    info.set_defaults(func=cmd_vision_info)

    train = vision_sub.add_parser("train", help="train an image model from a recipe")
    train.add_argument("-c", "--config", required=True)
    add_common(train)
    train.set_defaults(func=cmd_vision_train)

    sample = vision_sub.add_parser("sample", help="sample images from a model")
    sample.add_argument("-m", "--model", required=True)
    sample.add_argument("-o", "--output", default="samples.png")
    sample.add_argument("-n", type=int, default=16, help="number of samples")
    sample.add_argument("--sampler", default="euler", choices=["euler", "heun", "ddim", "ddpm"])
    sample.add_argument("--steps", type=int, default=50)
    sample.add_argument("--guidance", type=float, default=1.0)
    sample.add_argument("--label", type=int)
    sample.add_argument("--prompt", help="text prompt (text-conditioned models)")
    sample.add_argument("--tokenizer", help="tokenizer for --prompt")
    sample.add_argument("--palette", help="palette.json for pixel-art models")
    sample.add_argument("--temperature", type=float, default=0.9)
    sample.add_argument("--top-k", type=int, default=0)
    sample.add_argument("--top-p", type=float, default=0.9)
    sample.add_argument("--scale", type=int, default=4, help="upscale factor when saving")
    sample.add_argument("--seed", type=int)
    sample.add_argument("--device", default="auto")
    add_common(sample)
    sample.set_defaults(func=cmd_vision_sample)

    edit = vision_sub.add_parser("edit", help="edit an image with an instruction")
    edit.add_argument("-m", "--model", required=True)
    edit.add_argument("-i", "--image", required=True)
    edit.add_argument("--instruction", required=True)
    edit.add_argument("-t", "--tokenizer")
    edit.add_argument("-o", "--output", default="edited.png")
    edit.add_argument("--sampler", default="euler", choices=["euler", "heun", "ddim", "ddpm"])
    edit.add_argument("--steps", type=int, default=50)
    edit.add_argument("--guidance", type=float, default=3.0)
    edit.add_argument("--scale", type=int, default=1)
    edit.add_argument("--seed", type=int)
    edit.add_argument("--device", default="auto")
    add_common(edit)
    edit.set_defaults(func=cmd_vision_edit)

    quickstart = vision_sub.add_parser(
        "quickstart", help="run the image pipeline on synthetic sprites"
    )
    quickstart.add_argument("-o", "--output", default="runs/vision-quickstart")
    quickstart.add_argument("--steps", type=int, default=200)
    quickstart.add_argument("--sprites", type=int, default=256)
    quickstart.add_argument("--size", type=int, default=16)
    add_common(quickstart)
    quickstart.set_defaults(func=cmd_vision_quickstart)

    return vision
