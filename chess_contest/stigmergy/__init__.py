"""Stigmergy — Diffusive Pheromone Field Chess Engine (DPFE).

A deliberately non-generic architecture: multi-channel pheromone diffusion,
bilinear cross-color interactions, ternary ant-trail memory, and deep IDAS.
"""

from __future__ import annotations

from chess_contest.stigmergy.engine import StigmergyEngine
from chess_contest.stigmergy.weights import (
    StigmergyWeights,
    default_weights,
    load_weights,
    save_weights,
)

__all__ = [
    "StigmergyEngine",
    "StigmergyWeights",
    "default_weights",
    "load_weights",
    "save_weights",
]
