"""Image model architectures.

=================  =========================================================
Family             Use it for
=================  =========================================================
``dit``            General image generation. Scales best with data.
``unet``           General image generation below ~50K training images.
``pixelgpt``       Pixel art. Exact palette output, autoregressive.
``image-edit``     Instruction-based editing of an existing image.
``vae``            Compressing images to latents for latent diffusion.
=================  =========================================================

See ``docs/vision.md`` for how to choose and how to train each one.
"""

from __future__ import annotations

from minimodel.vision.architectures.base import BaseImageModel
from minimodel.vision.architectures.dit import DiT, DiTConfig
from minimodel.vision.architectures.edit import ImageEditConfig, ImageEditModel
from minimodel.vision.architectures.layers import (
    AttentionBlock2d,
    DiTBlock,
    Downsample2d,
    FinalLayer,
    LabelEmbedding,
    PatchEmbed,
    ResBlock2d,
    SelfAttention,
    TextConditioner,
    TimestepEmbedding,
    Upsample2d,
)
from minimodel.vision.architectures.pixelgpt import PixelGPT, PixelGPTConfig
from minimodel.vision.architectures.registry import (
    IMAGE_ARCHITECTURES,
    VISION_TEMPLATE_DIR,
    build_image_model,
    describe_image_model,
    list_image_architectures,
    list_image_templates,
    load_image_model,
    load_vision_template,
)
from minimodel.vision.architectures.unet import UNet, UNetConfig
from minimodel.vision.architectures.vae import VAE, VAEConfig, VAEOutput

__all__ = [
    "IMAGE_ARCHITECTURES",
    "VAE",
    "VISION_TEMPLATE_DIR",
    "AttentionBlock2d",
    "BaseImageModel",
    "DiT",
    "DiTBlock",
    "DiTConfig",
    "Downsample2d",
    "FinalLayer",
    "ImageEditConfig",
    "ImageEditModel",
    "LabelEmbedding",
    "PatchEmbed",
    "PixelGPT",
    "PixelGPTConfig",
    "ResBlock2d",
    "SelfAttention",
    "TextConditioner",
    "TimestepEmbedding",
    "UNet",
    "UNetConfig",
    "Upsample2d",
    "VAEConfig",
    "VAEOutput",
    "build_image_model",
    "describe_image_model",
    "list_image_architectures",
    "list_image_templates",
    "load_image_model",
    "load_vision_template",
]
