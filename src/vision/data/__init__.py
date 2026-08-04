"""Image corpora: preparation, storage and batching."""

from __future__ import annotations

from minimodel.vision.data.datasets import (
    ImageDataset,
    PairedImageDataset,
    PixelSequenceDataset,
    build_image_dataloader,
    collate_images,
    load_images_from_directory,
    prepare_image_corpus,
    prepare_pixel_corpus,
    synthetic_sprites,
)
from minimodel.vision.data.palette import (
    Palette,
    build_palette,
    exact_palette,
    kmeans_palette,
    median_cut_palette,
    quantize_to_palette,
)
from minimodel.vision.data.shards import ImageCorpus, ImageShardIndex, ImageShardWriter

__all__ = [
    "ImageCorpus",
    "ImageDataset",
    "ImageShardIndex",
    "ImageShardWriter",
    "PairedImageDataset",
    "Palette",
    "PixelSequenceDataset",
    "build_image_dataloader",
    "build_palette",
    "collate_images",
    "exact_palette",
    "kmeans_palette",
    "load_images_from_directory",
    "median_cut_palette",
    "prepare_image_corpus",
    "prepare_pixel_corpus",
    "quantize_to_palette",
    "synthetic_sprites",
]
