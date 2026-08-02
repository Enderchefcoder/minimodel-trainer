"""One entry point for every post-training stage.

Post-training always starts from an existing checkpoint and always follows the
same skeleton: load a model, load a tokenized dataset, pick an objective, train.
:func:`post_train` implements that skeleton once, dispatching on the ``stage``
field of a recipe so that ``minimodel posttrain --config recipe.yaml`` works for
SFT, chain-of-thought, DPO, SPIN and RLVR alike.

Recipe shape::

    stage: sft                 # sft | cot | dpo | spin | rlvr
    model:
      checkpoint: runs/pretrain/checkpoints/step_010000
    tokenizer: artifacts/tokenizer.json
    data:
      train: data/tokenized/smoltalk
      eval: data/tokenized/smoltalk-val
    training:
      max_steps: 2000
      lr: 2.0e-5
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

from minimodel.architectures.builder import build_model, load_model
from minimodel.core.config import Config, ConfigError, load_config
from minimodel.core.logging_utils import get_logger
from minimodel.datasets.loader import PackedTextDataset, SupervisedDataset
from minimodel.datasets.shards import TokenizedCorpus
from minimodel.tokenization.tokenize import BPETokenizer
from minimodel.training.instruct_cot_posttrainer import CoTTrainer, CoTTrainerConfig
from minimodel.training.instruct_posttrainer import InstructTrainer, InstructTrainerConfig
from minimodel.training.trainer import Trainer, TrainerConfig, TrainingResult

__all__ = ["POST_TRAIN_STAGES", "build_trainer_config", "load_pretrained", "post_train"]

logger = get_logger(__name__)

#: Stages :func:`post_train` knows how to run.
POST_TRAIN_STAGES = ("sft", "instruct", "cot", "reasoning", "dpo", "spin", "rlvr", "continue")


def build_trainer_config(config_class: type, section: Mapping[str, Any]) -> Any:
    """Instantiate a trainer config dataclass from a YAML section.

    Unknown keys are dropped with a warning rather than raising, so a recipe
    written for a newer version still runs.
    """
    valid = {f.name for f in fields(config_class)}
    payload = {k: v for k, v in dict(section).items() if k in valid}
    unknown = set(dict(section)) - valid
    if unknown:
        logger.warning("ignoring unknown training keys: %s", ", ".join(sorted(unknown)))
    return config_class(**payload)


def load_pretrained(
    spec: Mapping[str, Any],
    *,
    vocab_size: int | None = None,
    device: str | None = None,
):
    """Load the starting model for a post-training run.

    Accepts either ``checkpoint:`` (a directory written by the checkpoint
    manager or by ``save_pretrained``) or ``template:`` for training from
    scratch.
    """
    checkpoint = spec.get("checkpoint") or spec.get("path")
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            raise ConfigError(f"model checkpoint not found: {path}")
        logger.info("loading model from %s", path)
        return load_model(path, device=device)

    template = spec.get("template") or spec.get("architecture")
    if not template:
        raise ConfigError("model section needs either `checkpoint` or `template`")
    overrides = dict(spec.get("overrides") or {})
    if vocab_size is not None:
        overrides.setdefault("vocab_size", vocab_size)
    logger.info("building a fresh model from template %s", template)
    return build_model(template, overrides=overrides, device=device)


def _load_dataset(path: str | Path, seq_len: int, *, seed: int, supervised_hint: bool):
    """Open a tokenized corpus as the right dataset class."""
    corpus = TokenizedCorpus(path)
    if corpus.supervised and supervised_hint:
        return SupervisedDataset(corpus, seq_len, seed=seed)
    return PackedTextDataset(corpus, seq_len, seed=seed)


def post_train(
    config: str | Path | Config | Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
    cli_overrides: list[str] | None = None,
) -> TrainingResult | dict[str, Any]:
    """Run a post-training stage described by ``config``.

    Returns the trainer's :class:`~minimodel.training.trainer.TrainingResult`,
    or a plain dictionary for stages (like RLVR) that report differently.
    """
    if isinstance(config, (str, Path)):
        cfg = load_config(config, overrides=overrides, cli_overrides=cli_overrides)
    else:
        cfg = Config(dict(config))
        if overrides:
            cfg = cfg.merged(overrides)
        if cli_overrides:
            cfg = cfg.update_from_overrides(cli_overrides)

    stage = str(cfg.get("stage", "sft")).strip().lower()
    if stage not in POST_TRAIN_STAGES:
        raise ConfigError(
            f"unknown post-training stage {stage!r}; expected one of {', '.join(POST_TRAIN_STAGES)}"
        )

    tokenizer = None
    tokenizer_path = cfg.get("tokenizer")
    if tokenizer_path:
        tokenizer = BPETokenizer.load(tokenizer_path)

    model = load_pretrained(
        cfg.section("model"),
        vocab_size=tokenizer.vocab_size if tokenizer else None,
        device=cfg.get("training.device"),
    )

    training_section = dict(cfg.section("training"))
    data_section = cfg.section("data")

    if stage in {"dpo", "spin"}:
        return _run_preference_stage(stage, cfg, model, tokenizer, training_section)
    if stage == "rlvr":
        return _run_rlvr_stage(cfg, model, tokenizer, training_section)

    if stage in {"cot", "reasoning"}:
        trainer_config = build_trainer_config(CoTTrainerConfig, training_section)
        trainer_class: type[Trainer] = CoTTrainer
    elif stage in {"sft", "instruct"}:
        trainer_config = build_trainer_config(InstructTrainerConfig, training_section)
        trainer_class = InstructTrainer
    else:  # "continue" - more pretraining from an existing checkpoint
        trainer_config = build_trainer_config(TrainerConfig, training_section)
        trainer_class = Trainer

    supervised = stage in {"sft", "instruct", "cot", "reasoning"}
    train_path = data_section.get("train")
    if not train_path:
        raise ConfigError("data.train must point at a tokenized corpus directory")
    train_dataset = _load_dataset(
        train_path, trainer_config.seq_len, seed=trainer_config.seed, supervised_hint=supervised
    )
    eval_dataset = None
    if data_section.get("eval"):
        eval_dataset = _load_dataset(
            data_section["eval"],
            trainer_config.seq_len,
            seed=trainer_config.seed + 1,
            supervised_hint=supervised,
        )
    replay_dataset = None
    if data_section.get("replay"):
        replay_dataset = _load_dataset(
            data_section["replay"],
            trainer_config.seq_len,
            seed=trainer_config.seed + 2,
            supervised_hint=False,
        )

    kwargs: dict[str, Any] = {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
    }
    if trainer_class is not Trainer:
        kwargs["replay_dataset"] = replay_dataset

    trainer = trainer_class(model, trainer_config, **kwargs)
    return trainer.fit()


def _run_preference_stage(
    stage: str,
    cfg: Config,
    model: Any,
    tokenizer: Any,
    training_section: Mapping[str, Any],
) -> TrainingResult:
    """Dispatch to the DPO or SPIN trainer."""
    from minimodel.training.rl.dpo import DPOConfig, DPOTrainer
    from minimodel.training.rl.spin import SPINConfig, SPINTrainer

    pairs = cfg.get("data.pairs") or cfg.get("data.train")
    if not pairs:
        raise ConfigError(f"{stage} needs data.pairs pointing at a preference JSONL file")

    if stage == "dpo":
        dpo_config = build_trainer_config(DPOConfig, training_section)
        trainer = DPOTrainer(model, dpo_config, pairs_path=pairs, tokenizer=tokenizer)
    else:
        spin_config = build_trainer_config(SPINConfig, training_section)
        trainer = SPINTrainer(model, spin_config, dataset_path=pairs, tokenizer=tokenizer)
    return trainer.fit()


def _run_rlvr_stage(
    cfg: Config, model: Any, tokenizer: Any, training_section: Mapping[str, Any]
) -> dict[str, Any]:
    """Dispatch to the verifiable-reward RL trainer."""
    from minimodel.training.rl.rlvr import RLVRConfig, RLVRTrainer

    tasks_path = cfg.get("data.tasks") or cfg.get("data.train")
    if not tasks_path:
        raise ConfigError("rlvr needs data.tasks pointing at a prompt/answer JSONL file")
    if tokenizer is None:
        raise ConfigError("rlvr needs a tokenizer to decode samples for verification")
    rlvr_config = build_trainer_config(RLVRConfig, training_section)
    trainer = RLVRTrainer(model, rlvr_config, tasks_path=tasks_path, tokenizer=tokenizer)
    return trainer.fit().to_dict()
