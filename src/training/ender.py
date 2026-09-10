"""The ENDER trainer: cross-entropy plus the recurrence's auxiliary objectives.

:class:`EnderTrainer` subclasses :class:`~minimodel.training.trainer.Trainer` and
only changes the loss, following the package contract: everything else (AMP,
accumulation, checkpointing, resume, scheduling) is inherited. It also exposes a
staged mode for *adapting* a pre-trained backbone: stage 1 trains only the
grafted recurrence with the backbone frozen, stage 2 unfreezes the top backbone
blocks at a reduced learning rate (the :func:`configure_optimization_stages`
pattern from the reference ENDER implementation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor, nn

from minimodel.architectures.ender import EnderAdapter, EnderTransformer
from minimodel.training.trainer import Trainer, TrainerConfig

__all__ = ["EnderTrainer", "EnderTrainerConfig"]


@dataclass
class EnderTrainerConfig(TrainerConfig):
    """Trainer config plus the ENDER auxiliary-loss weights."""

    lambda_delta: float = 0.05
    lambda_kd: float = 0.0
    kd_temperature: float = 2.0
    #: Staged adaptation: freeze the backbone (0) or unfreeze the top N blocks.
    unfreeze_top_blocks: int = 0
    unfrozen_lr_scale: float = 0.1
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)


class EnderTrainer(Trainer):
    """Trainer for :class:`EnderTransformer` and :class:`EnderAdapter` models.

    The extra logging scalars (``loss_delta``, ``loss_kd``) come back from
    :meth:`compute_loss` and land in ``metrics.jsonl`` like any other trainer
    extra.
    """

    def __init__(self, model: nn.Module, config: EnderTrainerConfig, **kwargs: Any):
        super().__init__(model, config, **kwargs)
        cfg = self.config
        assert isinstance(cfg, EnderTrainerConfig)
        self.ender_loss = self._build_loss(cfg)
        self._apply_staged_optimizer(cfg)

    # ------------------------------------------------------------------
    def _build_loss(self, cfg: EnderTrainerConfig):
        from minimodel.architectures.ender import EnderLoss

        return EnderLoss(
            lambda_delta=cfg.lambda_delta,
            lambda_kd=cfg.lambda_kd,
            kd_temperature=cfg.kd_temperature,
            ignore_index=cfg.ignore_index,
        )

    def _apply_staged_optimizer(self, cfg: EnderTrainerConfig) -> None:
        """Rebuild the optimizer when staged freezing is requested."""
        if cfg.unfreeze_top_blocks <= 0 and all(
            p.requires_grad for p in self.raw_model.parameters()
        ):
            return
        model = self.raw_model
        if isinstance(model, EnderAdapter):
            from minimodel.architectures.ender import build_ender_optimizer

            self.optimizer = build_ender_optimizer(
                model,
                base_lr=cfg.lr,
                unfreeze_top_blocks=cfg.unfreeze_top_blocks,
                unfrozen_lr_scale=cfg.unfrozen_lr_scale,
                weight_decay=cfg.weight_decay,
            )
        elif isinstance(model, EnderTransformer):
            for param in model.ender.parameters():
                param.requires_grad = True
        # The scheduler wraps the optimizer by reference, so rebuild it too.
        from minimodel.training.schedules import build_scheduler, resolve_warmup

        warmup_steps = resolve_warmup(cfg.warmup, cfg.max_steps)
        self.scheduler = build_scheduler(
            self.optimizer,
            cfg.lr_schedule,
            total_steps=cfg.max_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=cfg.min_lr_ratio,
            **cfg.schedule_kwargs,
        )

    # ------------------------------------------------------------------
    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """CE loss plus the ENDER convergence / distillation terms."""
        model = self.raw_model
        tokens = batch["input_ids"]
        labels = batch["labels"]
        logits = model(tokens, **self.model_forward_kwargs)
        aux = getattr(model, "ender_aux", None) or (
            model.last_aux if isinstance(model, EnderAdapter) else None
        )
        return self.ender_loss(logits, labels, aux=aux)
