"""Run a pretraining recipe from YAML.

A recipe is the single artifact that describes a training run, so a run can be
reproduced from the file alone::

    run_name: supra2-demo
    model:
      template: supra2_1406240
    tokenizer: artifacts/tokenizer.json
    data:
      train: data/tokenized/fineweb-edu
      eval: data/tokenized/fineweb-edu-val
      # or a weighted blend:
      # mixture:
      #   - { path: data/tokenized/fineweb-edu, weight: 0.7 }
      #   - { path: data/tokenized/cosmopedia, weight: 0.3 }
    training:
      max_steps: 20000
      batch_size: 32
      seq_len: 512
      lr: 3.0e-3
      optimizer: muon

Recipes may ``extends:`` another file, so a size-specific recipe can inherit a
shared base and change three lines.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from minimodel.architectures.builder import build_model, load_model
from minimodel.core.config import Config, ConfigError, load_config
from minimodel.core.logging_utils import get_logger
from minimodel.datasets.loader import MixtureDataset, PackedTextDataset, SupervisedDataset
from minimodel.datasets.shards import TokenizedCorpus
from minimodel.tokenization.tokenize import BPETokenizer
from minimodel.training.callbacks import Callback, SampleGenerator
from minimodel.training.post_train import build_trainer_config
from minimodel.training.trainer import Trainer, TrainerConfig, TrainingResult

__all__ = ["build_dataset_from_recipe", "run_pretrain_recipe"]

logger = get_logger(__name__)


def build_dataset_from_recipe(
    section: Mapping[str, Any],
    key: str,
    seq_len: int,
    *,
    seed: int = 0,
) -> Any:
    """Build one dataset (or mixture) described by a recipe's ``data`` section.

    Accepts either ``<key>: path`` or ``<key>_mixture: [{path, weight}, ...]``;
    the plain ``mixture`` key is treated as the training mixture.
    """
    mixture_key = "mixture" if key == "train" else f"{key}_mixture"
    mixture = section.get(mixture_key)
    if mixture:
        datasets = []
        weights = []
        names = []
        for component in mixture:
            path = component["path"] if isinstance(component, Mapping) else component
            weight = float(component.get("weight", 1.0)) if isinstance(component, Mapping) else 1.0
            corpus = TokenizedCorpus(path)
            dataset = (
                SupervisedDataset(corpus, seq_len, seed=seed)
                if corpus.supervised
                else PackedTextDataset(corpus, seq_len, seed=seed)
            )
            datasets.append(dataset)
            weights.append(weight)
            names.append(Path(str(path)).name)
        logger.info(
            "%s mixture: %s",
            key,
            ", ".join(f"{n} {w:.0%}" for n, w in zip(names, weights, strict=True)),
        )
        return MixtureDataset(datasets, weights, names=names, seed=seed)

    path = section.get(key)
    if not path:
        return None
    corpus = TokenizedCorpus(path)
    if corpus.supervised:
        return SupervisedDataset(corpus, seq_len, seed=seed)
    return PackedTextDataset(corpus, seq_len, seed=seed)


def run_pretrain_recipe(
    config: str | Path | Config | Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
    cli_overrides: list[str] | None = None,
    extra_callbacks: list[Callback] | None = None,
) -> TrainingResult:
    """Load a recipe and run pretraining."""
    if isinstance(config, (str, Path)):
        cfg = load_config(config, overrides=overrides, cli_overrides=cli_overrides)
    else:
        cfg = Config(dict(config))
        if overrides:
            cfg = cfg.merged(overrides)
        if cli_overrides:
            cfg = cfg.update_from_overrides(cli_overrides)

    tokenizer = None
    tokenizer_path = cfg.get("tokenizer")
    if tokenizer_path:
        tokenizer = BPETokenizer.load(tokenizer_path)
        logger.info("loaded tokenizer with %d tokens", tokenizer.vocab_size)

    training_section = dict(cfg.section("training"))
    if "run_name" not in training_section and cfg.get("run_name"):
        training_section["run_name"] = cfg.get("run_name")
    trainer_config: TrainerConfig = build_trainer_config(TrainerConfig, training_section)

    model_section = cfg.section("model")
    checkpoint = model_section.get("checkpoint")
    if checkpoint:
        model = load_model(checkpoint, device=trainer_config.device)
    else:
        template = model_section.get("template") or model_section.get("architecture")
        if not template:
            raise ConfigError("recipe model section needs `template` or `checkpoint`")
        model_overrides = dict(model_section.get("overrides") or {})
        if tokenizer is not None:
            model_overrides.setdefault("vocab_size", tokenizer.vocab_size)
        max_seq = model_overrides.get("max_seq_len")
        if max_seq is not None and trainer_config.seq_len > int(max_seq):
            raise ConfigError(
                f"training.seq_len ({trainer_config.seq_len}) exceeds the model's "
                f"max_seq_len ({max_seq})"
            )
        model = build_model(template, overrides=model_overrides)

    data_section = cfg.section("data")
    train_dataset = build_dataset_from_recipe(
        data_section, "train", trainer_config.seq_len, seed=trainer_config.seed
    )
    if train_dataset is None:
        raise ConfigError("recipe data section needs `train` or `mixture`")
    eval_dataset = build_dataset_from_recipe(
        data_section, "eval", trainer_config.seq_len, seed=trainer_config.seed + 1
    )

    callbacks: list[Callback] = list(extra_callbacks or [])
    sample_prompt = cfg.get("sample_prompt")
    if sample_prompt and tokenizer is not None:
        callbacks.append(SampleGenerator(str(sample_prompt), max_new_tokens=48))

    trainer = Trainer(
        model,
        trainer_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
        model_forward_kwargs=dict(cfg.get("model_forward_kwargs") or {}),
    )
    result = trainer.fit()

    if cfg.get("export", True):
        export_dir = cfg.get("export_dir") or (trainer.run_dir / "model")
        trainer.export(export_dir)
        if tokenizer is not None:
            tokenizer.save(Path(export_dir) / "tokenizer.json")
    return result
