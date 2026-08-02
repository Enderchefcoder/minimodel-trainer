"""Training loops for every stage of the model lifecycle.

===================================  ====================================
Stage                                Class
===================================  ====================================
Pretraining                          :class:`Trainer`
Instruction tuning                   :class:`InstructTrainer`
Chain-of-thought distillation        :class:`CoTTrainer`
Preference optimization              :class:`~minimodel.training.rl.DPOTrainer`
Self-play fine-tuning                :class:`~minimodel.training.rl.SPINTrainer`
Verifiable-reward RL                 :class:`~minimodel.training.rl.RLVRTrainer`
===================================  ====================================

Every class in that table derives from :class:`Trainer` and overrides only
``compute_loss``, so any improvement to the loop applies to all of them.
"""

from __future__ import annotations

from minimodel.training.callbacks import (
    Callback,
    CallbackList,
    ConsoleLogger,
    EarlyStopping,
    GradientMonitor,
    SampleGenerator,
)
from minimodel.training.instruct_cot_posttrainer import CoTTrainer, CoTTrainerConfig
from minimodel.training.instruct_posttrainer import InstructTrainer, InstructTrainerConfig
from minimodel.training.optim import OPTIMIZERS, Lion, Muon, build_optimizer, param_groups
from minimodel.training.post_train import post_train
from minimodel.training.schedules import SCHEDULES, build_scheduler
from minimodel.training.trainer import Trainer, TrainerConfig, TrainingResult

__all__ = [
    "OPTIMIZERS",
    "SCHEDULES",
    "Callback",
    "CallbackList",
    "CoTTrainer",
    "CoTTrainerConfig",
    "ConsoleLogger",
    "EarlyStopping",
    "GradientMonitor",
    "InstructTrainer",
    "InstructTrainerConfig",
    "Lion",
    "Muon",
    "SampleGenerator",
    "Trainer",
    "TrainerConfig",
    "TrainingResult",
    "build_optimizer",
    "build_scheduler",
    "param_groups",
    "post_train",
]
