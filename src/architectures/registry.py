"""Architecture registry.

Templates refer to architectures by their ``family`` string. Registering here is
the only step needed to make a new architecture available to every CLI command,
config file and test.
"""

from __future__ import annotations

from minimodel.architectures.base import BaseLanguageModel
from minimodel.architectures.dense import DenseTransformer
from minimodel.architectures.experimental import ExperimentalTransformer
from minimodel.architectures.hybrid import HybridRecurrentTransformer
from minimodel.architectures.looped import LoopedTransformer
from minimodel.architectures.moe import MoETransformer
from minimodel.architectures.ssm import MambaLM
from minimodel.core.registry import Registry

__all__ = ["ARCHITECTURES", "list_architectures", "register_architecture"]

#: Maps a ``family`` string to a :class:`BaseLanguageModel` subclass.
ARCHITECTURES: Registry[type[BaseLanguageModel]] = Registry("architecture")

ARCHITECTURES.add(
    LoopedTransformer.architecture_name,
    LoopedTransformer,
    aliases=("looped", "supra", "supra2", "recurrent_depth"),
)
ARCHITECTURES.add(
    DenseTransformer.architecture_name,
    DenseTransformer,
    aliases=("dense", "gpt", "transformer", "llama"),
)
ARCHITECTURES.add(
    MoETransformer.architecture_name,
    MoETransformer,
    aliases=("moe", "mixture_of_experts", "sparse"),
)
ARCHITECTURES.add(
    HybridRecurrentTransformer.architecture_name,
    HybridRecurrentTransformer,
    aliases=("hybrid", "griffin", "recurrent"),
)
ARCHITECTURES.add(
    ExperimentalTransformer.architecture_name,
    ExperimentalTransformer,
    aliases=("experimental", "novel", "variant"),
)
ARCHITECTURES.add(
    MambaLM.architecture_name,
    MambaLM,
    aliases=("mamba", "ssm", "selective_ssm"),
)


def register_architecture(
    name: str, cls: type[BaseLanguageModel], *, aliases: tuple[str, ...] = ()
) -> None:
    """Register a custom architecture at runtime.

    >>> from minimodel.architectures.dense import DenseTransformer
    >>> register_architecture("my-gpt", DenseTransformer)
    >>> "my-gpt" in ARCHITECTURES
    True
    """
    ARCHITECTURES.add(name, cls, aliases=aliases, overwrite=True)


def list_architectures() -> list[str]:
    """All registered architecture family names."""
    return ARCHITECTURES.names()
