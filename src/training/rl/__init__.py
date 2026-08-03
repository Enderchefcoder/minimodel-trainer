"""Preference optimization and reinforcement learning.

============  ==========================================================
Method        When to use it
============  ==========================================================
DPO           You have preference pairs (chosen vs rejected).
SPIN          You have SFT data but no preferences; generates its own.
RLVR (GRPO)   The task has a checkable answer: maths, code, puzzles.
============  ==========================================================

All three subclass :class:`~minimodel.training.trainer.Trainer`, so they share
checkpointing, logging, scheduling and resume with the pretraining loop.
"""

from __future__ import annotations

from minimodel.training.rl.dpo import DPOConfig, DPOTrainer, dpo_loss
from minimodel.training.rl.rlvr import (
    VERIFIERS,
    RLVRConfig,
    RLVRTrainer,
    extract_final_number,
    group_advantages,
)
from minimodel.training.rl.spin import SPINConfig, SPINTrainer, generate_self_play_pairs

__all__ = [
    "VERIFIERS",
    "DPOConfig",
    "DPOTrainer",
    "RLVRConfig",
    "RLVRTrainer",
    "SPINConfig",
    "SPINTrainer",
    "dpo_loss",
    "extract_final_number",
    "generate_self_play_pairs",
    "group_advantages",
]
