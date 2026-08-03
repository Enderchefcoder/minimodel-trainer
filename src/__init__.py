"""minimodel-trainer: an end-to-end toolkit for training small language models.

The package is laid out as one subpackage per stage of the model lifecycle:

============================  ==========================================================
Subpackage                    Responsibility
============================  ==========================================================
:mod:`minimodel.core`         Config loading, registries, logging, seeding, device setup
:mod:`minimodel.tokenization` Byte-level BPE training, encoding and decoding
:mod:`minimodel.datasets`     Pulling corpora, tokenizing to shards, batching
:mod:`minimodel.architectures` Model definitions and YAML-driven model construction
:mod:`minimodel.training`     Pretraining, post-training and reinforcement learning
:mod:`minimodel.checkpointing` Checkpoint IO, resume, ETA estimation, loss plots
:mod:`minimodel.inference`    Sampling, generation and chat
:mod:`minimodel.benchmarking` Evaluation harness, comparison tables and charts
:mod:`minimodel.merging`      Weight-space model merging
:mod:`minimodel.cardgen`      Automatic model card generation
============================  ==========================================================

The top level only re-exports a handful of names so that ``import minimodel`` stays
cheap; everything else is imported lazily from its subpackage.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "Config",
    "__version__",
    "build_model",
    "get_logger",
    "set_seed",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy-import shim
    """Lazily resolve the small set of re-exported top-level names.

    Importing :mod:`torch` costs a noticeable amount of time, so the heavier
    subpackages are only imported when one of the re-exported names is actually
    touched.
    """
    if name == "Config":
        from minimodel.core.config import Config

        return Config
    if name == "get_logger":
        from minimodel.core.logging_utils import get_logger

        return get_logger
    if name == "set_seed":
        from minimodel.core.seeding import set_seed

        return set_seed
    if name == "build_model":
        from minimodel.architectures.builder import build_model

        return build_model
    raise AttributeError(f"module 'minimodel' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - introspection helper
    return sorted(__all__)
