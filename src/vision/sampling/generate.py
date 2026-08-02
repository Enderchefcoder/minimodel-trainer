"""Turning model outputs into image files.

Saving a grid of samples requires Pillow; when it is missing the tensors are
written as ``.npy`` instead so a headless run never fails at the last step.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from minimodel.core.io_utils import ensure_dir
from minimodel.core.logging_utils import get_logger
from minimodel.vision.data.palette import Palette
from minimodel.vision.sampling.samplers import sample_images

__all__ = [
    "make_grid",
    "save_image_grid",
    "sample_pixel_art",
    "tensor_to_uint8",
]

logger = get_logger(__name__)


def tensor_to_uint8(images: Tensor) -> np.ndarray:
    """``[B, C, H, W]`` in ``[-1, 1]`` -> ``[B, H, W, C]`` uint8."""
    array = images.detach().float().clamp(-1.0, 1.0).cpu().numpy()
    array = (array + 1.0) * 127.5
    array = np.clip(np.round(array), 0, 255).astype(np.uint8)
    return np.transpose(array, (0, 2, 3, 1))


def make_grid(images: np.ndarray, *, columns: int | None = None, padding: int = 2) -> np.ndarray:
    """Tile ``[N, H, W, C]`` images into a single padded grid."""
    if images.ndim == 3:
        images = images[..., None]
    n, height, width, channels = images.shape
    columns = columns or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / columns))

    grid_height = rows * height + (rows + 1) * padding
    grid_width = columns * width + (columns + 1) * padding
    grid = np.zeros((grid_height, grid_width, channels), dtype=np.uint8)

    for index in range(n):
        row, column = divmod(index, columns)
        y = padding + row * (height + padding)
        x = padding + column * (width + padding)
        grid[y : y + height, x : x + width] = images[index]
    return grid


def save_image_grid(
    images: Tensor | np.ndarray,
    path: str | Path,
    *,
    columns: int | None = None,
    scale: int = 1,
    palette: Palette | None = None,
) -> Path:
    """Save a batch of images as one grid file.

    Parameters
    ----------
    images:
        Either float tensors in ``[-1, 1]`` or integer palette indices.
    palette:
        Required when ``images`` holds palette indices.
    scale:
        Nearest-neighbour upscaling factor, so a 24px sprite is actually
        visible in a file browser.
    """
    if isinstance(images, Tensor):
        if images.dtype in (torch.int64, torch.int32, torch.uint8) and palette is not None:
            array = palette.dequantize(images.detach().cpu().numpy())
        else:
            array = tensor_to_uint8(images)
    else:
        array = np.asarray(images)
        if array.ndim == 3 and palette is not None:
            array = palette.dequantize(array)

    grid = make_grid(array, columns=columns)
    if scale > 1:
        grid = np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)

    path = Path(path)
    ensure_dir(path.parent)
    try:
        from PIL import Image

        image = Image.fromarray(grid.squeeze() if grid.shape[-1] == 1 else grid)
        image.save(path)
    except ImportError:
        fallback = path.with_suffix(".npy")
        np.save(fallback, grid)
        logger.warning("Pillow is not installed; wrote %s instead of %s", fallback, path)
        return fallback
    logger.info("wrote %s", path)
    return path


def sample_pixel_art(
    model: nn.Module,
    *,
    n_samples: int = 16,
    palette: Palette | str | Path | None = None,
    temperature: float = 0.9,
    top_k: int = 0,
    top_p: float = 0.9,
    labels: Tensor | None = None,
    seed: int | None = None,
    output: str | Path | None = None,
    scale: int = 8,
    device: torch.device | str | None = None,
) -> Any:
    """Sample sprites from a :class:`~minimodel.vision.architectures.PixelGPT`.

    Returns the palette-index tensor, or the written path when ``output`` is
    given.
    """
    indices = model.generate(
        n_samples,
        labels=labels,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        device=device,
    )
    if output is None:
        return indices
    if palette is None:
        raise ValueError("saving pixel art needs a palette")
    if not isinstance(palette, Palette):
        palette = Palette.load(palette)
    return save_image_grid(indices, output, palette=palette, scale=scale)


def sample_and_save(
    model: nn.Module,
    output: str | Path,
    *,
    n_samples: int = 16,
    sampler: str = "euler",
    n_steps: int = 50,
    guidance_scale: float = 1.0,
    labels: Sequence[int] | None = None,
    text_tokens: Tensor | None = None,
    reference: Tensor | None = None,
    seed: int | None = None,
    scale: int = 1,
    device: torch.device | str | None = None,
) -> Path:
    """Sample from a diffusion model and write a grid."""
    label_tensor = (
        torch.tensor(list(labels), dtype=torch.long) if labels is not None else None
    )
    images = sample_images(
        model,
        n_samples=n_samples,
        sampler=sampler,
        n_steps=n_steps,
        guidance_scale=guidance_scale,
        labels=label_tensor,
        text_tokens=text_tokens,
        reference=reference,
        seed=seed,
        device=device,
    )
    return save_image_grid(images, output, scale=scale)
