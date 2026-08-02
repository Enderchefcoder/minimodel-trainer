"""Run image-model training from a YAML recipe.

Mirrors :mod:`minimodel.training.recipe` for the vision side::

    kind: diffusion            # diffusion | pixelgpt | edit | vae
    run_name: sprites-dit
    model:
      template: dit_small_32
      overrides:
        num_classes: 4
    data:
      train: data/images/sprites
      eval: data/images/sprites-val
    training:
      max_steps: 20000
      batch_size: 64
      lr: 1.0e-4
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from minimodel.core.config import Config, ConfigError, load_config
from minimodel.core.logging_utils import get_logger
from minimodel.training.post_train import build_trainer_config
from minimodel.vision.architectures.registry import build_image_model, load_image_model
from minimodel.vision.data.datasets import ImageDataset, PairedImageDataset, PixelSequenceDataset
from minimodel.vision.data.shards import ImageCorpus
from minimodel.vision.training.diffusion import DiffusionConfig, DiffusionTrainer
from minimodel.vision.training.pixel_trainer import (
    PixelGPTConfig,
    PixelGPTTrainer,
    VAETrainer,
    VAETrainerConfig,
)

__all__ = ["VISION_RECIPE_KINDS", "run_vision_recipe"]

logger = get_logger(__name__)

#: Recipe kinds :func:`run_vision_recipe` understands.
VISION_RECIPE_KINDS = ("diffusion", "pixelgpt", "edit", "vae")


def _load_tokenizer(path: str | Path | None):
    """Load a text tokenizer for caption/instruction conditioning."""
    if not path:
        return None
    from minimodel.tokenization.tokenize import BPETokenizer

    return BPETokenizer.load(path)


def run_vision_recipe(
    config: str | Path | Config | Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
    cli_overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load a vision recipe and train the model it describes."""
    if isinstance(config, (str, Path)):
        cfg = load_config(config, overrides=overrides, cli_overrides=cli_overrides)
    else:
        cfg = Config(dict(config))
        if overrides:
            cfg = cfg.merged(overrides)
        if cli_overrides:
            cfg = cfg.update_from_overrides(cli_overrides)

    kind = str(cfg.get("kind", "diffusion")).strip().lower()
    if kind not in VISION_RECIPE_KINDS:
        raise ConfigError(
            f"unknown vision recipe kind {kind!r}; expected one of {', '.join(VISION_RECIPE_KINDS)}"
        )

    data_section = cfg.section("data")
    train_path = data_section.get("train")
    if not train_path:
        raise ConfigError("vision recipe needs data.train pointing at an image corpus")
    train_corpus = ImageCorpus(train_path)
    eval_corpus = ImageCorpus(data_section["eval"]) if data_section.get("eval") else None

    tokenizer = _load_tokenizer(cfg.get("tokenizer"))

    model_section = cfg.section("model")
    model_overrides = dict(model_section.get("overrides") or {})
    if kind == "pixelgpt":
        model_overrides.setdefault("image_size", train_corpus.index.height)
        model_overrides.setdefault("palette_size", train_corpus.index.palette_size)
        if train_corpus.index.n_classes:
            model_overrides.setdefault("num_classes", train_corpus.index.n_classes)
    else:
        model_overrides.setdefault("image_size", train_corpus.index.height)
        if train_corpus.index.n_classes and model_overrides.get("condition") == "class":
            model_overrides.setdefault("num_classes", train_corpus.index.n_classes)
        if tokenizer is not None:
            model_overrides.setdefault("text_vocab_size", tokenizer.vocab_size)

    checkpoint = model_section.get("checkpoint")
    if checkpoint:
        model = load_image_model(checkpoint)
    else:
        template = model_section.get("template") or model_section.get("architecture")
        if not template:
            raise ConfigError("vision recipe model section needs `template` or `checkpoint`")
        model = build_image_model(template, overrides=model_overrides, verify_budget=False)

    logger.info(
        "training %s (%s params) on %d images",
        model.architecture_name,
        f"{model.num_parameters():,}",
        len(train_corpus),
    )

    training_section = dict(cfg.section("training"))
    training_section.setdefault("run_name", cfg.get("run_name", kind))

    if kind == "pixelgpt":
        trainer_config = build_trainer_config(PixelGPTConfig, training_section)
        train_dataset = PixelSequenceDataset(
            train_corpus, horizontal_flip=trainer_config.horizontal_flip
        )
        eval_dataset = PixelSequenceDataset(eval_corpus) if eval_corpus else None
        trainer = PixelGPTTrainer(
            model, trainer_config, train_dataset=train_dataset, eval_dataset=eval_dataset
        )
        result = trainer.fit()
        trainer.export(cfg.get("export_dir") or trainer.run_dir / "model")
        return result.to_dict()

    if kind == "vae":
        trainer_config = build_trainer_config(VAETrainerConfig, training_section)
        trainer = VAETrainer(
            model,
            trainer_config,
            train_dataset=ImageDataset(train_corpus),
            eval_dataset=ImageDataset(eval_corpus) if eval_corpus else None,
        )
        result = trainer.fit()
        trainer.export(cfg.get("export_dir") or trainer.run_dir / "model")
        return result.to_dict()

    diffusion_config = build_trainer_config(DiffusionConfig, training_section)
    dataset_class = PairedImageDataset if kind == "edit" else ImageDataset
    train_dataset = dataset_class(
        train_corpus, horizontal_flip=diffusion_config.horizontal_flip, tokenizer=tokenizer
    )
    eval_dataset = dataset_class(eval_corpus, tokenizer=tokenizer) if eval_corpus else None

    trainer = DiffusionTrainer(
        model, diffusion_config, dataset=train_dataset, eval_dataset=eval_dataset
    )
    summary = trainer.fit()
    trainer.export(cfg.get("export_dir") or trainer.run_dir / "model")
    return summary
