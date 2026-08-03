"""Image models: architectures, data, training and sampling.

The image side of the toolkit mirrors the text side stage for stage::

    minimodel vision data prepare   # images       -> image corpus
    minimodel vision train          # corpus       -> trained model
    minimodel vision sample         # model        -> images
    minimodel vision edit           # image + text -> edited image

Four model families are available - a diffusion transformer, a convolutional
UNet, an autoregressive pixel-art model and an instruction-based editor - plus a
latent autoencoder for higher resolutions. See ``docs/vision.md``.
"""

from __future__ import annotations

from minimodel.vision.architectures import (
    IMAGE_ARCHITECTURES,
    VAE,
    BaseImageModel,
    DiT,
    ImageEditModel,
    PixelGPT,
    UNet,
    build_image_model,
    list_image_architectures,
    list_image_templates,
    load_image_model,
)
from minimodel.vision.data import (
    ImageCorpus,
    ImageDataset,
    PairedImageDataset,
    Palette,
    PixelSequenceDataset,
    prepare_image_corpus,
    prepare_pixel_corpus,
    synthetic_sprites,
)
from minimodel.vision.sampling import sample_images, sample_pixel_art, save_image_grid
from minimodel.vision.training import (
    DiffusionConfig,
    DiffusionTrainer,
    PixelGPTTrainer,
    VAETrainer,
)

__all__ = [
    "IMAGE_ARCHITECTURES",
    "VAE",
    "BaseImageModel",
    "DiT",
    "DiffusionConfig",
    "DiffusionTrainer",
    "ImageCorpus",
    "ImageDataset",
    "ImageEditModel",
    "PairedImageDataset",
    "Palette",
    "PixelGPT",
    "PixelGPTTrainer",
    "PixelSequenceDataset",
    "UNet",
    "VAETrainer",
    "build_image_model",
    "list_image_architectures",
    "list_image_templates",
    "load_image_model",
    "prepare_image_corpus",
    "prepare_pixel_corpus",
    "sample_images",
    "sample_pixel_art",
    "save_image_grid",
    "synthetic_sprites",
]
