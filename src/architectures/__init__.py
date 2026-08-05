"""Model definitions and YAML-driven model construction.

Available families
------------------
``looped-transformer``
    Weight-shared recurrent-depth transformer. Best parameters-per-quality at
    very small scale; the depth/compute trade is adjustable at inference time.
``dense-transformer``
    Conventional decoder-only stack with RMSNorm, RoPE, SwiGLU, grouped-query
    attention and QK normalisation. The familiar, predictable baseline.
``moe-transformer``
    Sparse mixture-of-experts. Highest quality per training FLOP once the model
    is large enough for routing to pay off.
``hybrid-recurrent``
    Griffin-style interleaving of gated linear recurrences and local attention.
    Constant-size decoding state, so it is the choice for long context.
``experimental-transformer``
    Novel Transformer wirings (ResiMix, KV-inherit, braid, echo-FFN, dual-RoPE)
    for ~1M Glint-2 bake-offs.
``mamba-lm``
    Pure-PyTorch selective SSM (Mamba-style) with hybrid attention variants.

See ``docs/architecture.md`` for how to pick between them.
"""

from __future__ import annotations

from minimodel.architectures.base import BaseLanguageModel, ModelOutput
from minimodel.architectures.builder import (
    TEMPLATE_DIR,
    build_model,
    describe_model,
    list_glint2_candidates,
    list_templates,
    load_model,
    load_template,
    template_to_model_config,
)
from minimodel.architectures.dense import DenseTransformer, DenseTransformerConfig
from minimodel.architectures.experimental import (
    ExperimentalTransformer,
    ExperimentalTransformerConfig,
)
from minimodel.architectures.hybrid import HybridRecurrentConfig, HybridRecurrentTransformer
from minimodel.architectures.layers import (
    CausalLocalAttention,
    FactorizedEmbedding,
    GatedRecurrentUnit,
    KVCache,
    MoEFeedForward,
    RecurrentBlock,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUFeedForward,
    TransformerBlock,
)
from minimodel.architectures.looped import LoopedTransformer, LoopedTransformerConfig
from minimodel.architectures.moe import MoETransformer, MoETransformerConfig
from minimodel.architectures.registry import (
    ARCHITECTURES,
    list_architectures,
    register_architecture,
)
from minimodel.architectures.ssm import MambaLM, MambaLMConfig

__all__ = [
    "ARCHITECTURES",
    "TEMPLATE_DIR",
    "BaseLanguageModel",
    "CausalLocalAttention",
    "DenseTransformer",
    "DenseTransformerConfig",
    "ExperimentalTransformer",
    "ExperimentalTransformerConfig",
    "FactorizedEmbedding",
    "GatedRecurrentUnit",
    "HybridRecurrentConfig",
    "HybridRecurrentTransformer",
    "KVCache",
    "LoopedTransformer",
    "LoopedTransformerConfig",
    "MambaLM",
    "MambaLMConfig",
    "MoEFeedForward",
    "MoETransformer",
    "MoETransformerConfig",
    "ModelOutput",
    "RMSNorm",
    "RecurrentBlock",
    "RotaryEmbedding",
    "SwiGLUFeedForward",
    "TransformerBlock",
    "build_model",
    "describe_model",
    "list_architectures",
    "list_glint2_candidates",
    "list_templates",
    "load_model",
    "load_template",
    "register_architecture",
    "template_to_model_config",
]
