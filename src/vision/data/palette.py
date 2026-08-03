"""Palette extraction and quantization for pixel art.

:class:`PixelGPT` works over palette indices, so a corpus of RGB sprites has to
be reduced to a shared palette first. Two algorithms are provided:

``median_cut``
    Recursively split the colour box along its longest axis. Deterministic,
    fast, and it never drops a colour region entirely - which matters because
    losing a sprite's single-pixel highlight colour is very visible.
``kmeans``
    Lloyd's algorithm from a median-cut initialisation. Slightly better average
    error, at the cost of being iterative.

Pixel art usually already has a small exact palette. :func:`exact_palette`
detects that case, and when the corpus uses fewer distinct colours than the
budget, it is used verbatim and quantization is lossless.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from minimodel.core.io_utils import read_json, write_json
from minimodel.core.logging_utils import get_logger

__all__ = [
    "Palette",
    "build_palette",
    "exact_palette",
    "kmeans_palette",
    "median_cut_palette",
    "quantize_to_palette",
]

logger = get_logger(__name__)


@dataclass
class Palette:
    """An ordered list of RGB colours plus lookup helpers."""

    colors: np.ndarray  # [K, 3] uint8

    def __post_init__(self) -> None:
        self.colors = np.asarray(self.colors, dtype=np.uint8).reshape(-1, 3)

    def __len__(self) -> int:
        return int(self.colors.shape[0])

    def __repr__(self) -> str:
        return f"Palette({len(self)} colors)"

    def quantize(self, image: np.ndarray) -> np.ndarray:
        """Map an ``[H, W, 3]`` image to ``[H, W]`` palette indices."""
        return quantize_to_palette(image, self.colors)

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """Map palette indices back to an ``[..., 3]`` uint8 image."""
        clipped = np.clip(np.asarray(indices), 0, len(self) - 1)
        return self.colors[clipped]

    def save(self, path: str | Path) -> Path:
        """Write the palette as JSON."""
        return write_json(path, {"colors": self.colors.tolist(), "size": len(self)})

    @classmethod
    def load(cls, path: str | Path) -> Palette:
        """Read a palette written by :meth:`save`."""
        payload = read_json(path)
        return cls(np.asarray(payload["colors"], dtype=np.uint8))

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view."""
        return {"colors": self.colors.tolist(), "size": len(self)}


def quantize_to_palette(image: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Nearest-colour quantization in RGB space.

    Distances are computed in chunks so that a large image against a large
    palette does not allocate an ``H*W*K*3`` intermediate.
    """
    pixels = np.asarray(image, dtype=np.int32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.int32).reshape(-1, 3)
    out = np.empty(pixels.shape[0], dtype=np.int64)
    chunk = max(1, 1_000_000 // max(1, colors.shape[0]))
    for start in range(0, pixels.shape[0], chunk):
        block = pixels[start : start + chunk]
        distances = ((block[:, None, :] - colors[None, :, :]) ** 2).sum(axis=2)
        out[start : start + chunk] = distances.argmin(axis=1)
    return out.reshape(np.asarray(image).shape[:-1])


def exact_palette(images: Iterable[np.ndarray], max_colors: int = 256) -> Palette | None:
    """Return the exact palette if the corpus uses at most ``max_colors``.

    Returns ``None`` as soon as the distinct-colour count exceeds the budget,
    without scanning the rest of the corpus.
    """
    seen: set[tuple[int, int, int]] = set()
    for image in images:
        pixels = np.asarray(image, dtype=np.uint8).reshape(-1, 3)
        unique = np.unique(pixels, axis=0)
        for color in unique:
            seen.add((int(color[0]), int(color[1]), int(color[2])))
            if len(seen) > max_colors:
                return None
    if not seen:
        return None
    ordered = sorted(seen)
    return Palette(np.asarray(ordered, dtype=np.uint8))


def median_cut_palette(pixels: np.ndarray, n_colors: int = 64) -> Palette:
    """Median-cut colour quantization.

    Repeatedly splits the colour bucket with the largest range along its widest
    channel, then averages each final bucket.
    """
    pixels = np.asarray(pixels, dtype=np.int32).reshape(-1, 3)
    if pixels.shape[0] == 0:
        raise ValueError("median_cut_palette needs at least one pixel")
    buckets: list[np.ndarray] = [pixels]

    while len(buckets) < n_colors:
        # Pick the bucket whose colours span the widest range.
        splittable = [(i, b) for i, b in enumerate(buckets) if b.shape[0] > 1]
        if not splittable:
            break
        index, bucket = max(
            splittable, key=lambda pair: (pair[1].max(axis=0) - pair[1].min(axis=0)).max()
        )
        channel = int((bucket.max(axis=0) - bucket.min(axis=0)).argmax())
        ordered = bucket[bucket[:, channel].argsort()]
        middle = ordered.shape[0] // 2
        buckets.pop(index)
        buckets.extend([ordered[:middle], ordered[middle:]])

    colors = np.stack([bucket.mean(axis=0) for bucket in buckets if bucket.shape[0]])
    return Palette(np.clip(np.round(colors), 0, 255).astype(np.uint8))


def kmeans_palette(
    pixels: np.ndarray, n_colors: int = 64, *, iterations: int = 12, seed: int = 0
) -> Palette:
    """K-means colour quantization initialised from median cut."""
    pixels = np.asarray(pixels, dtype=np.float32).reshape(-1, 3)
    if pixels.shape[0] == 0:
        raise ValueError("kmeans_palette needs at least one pixel")

    rng = np.random.default_rng(seed)
    if pixels.shape[0] > 200_000:
        pixels = pixels[rng.choice(pixels.shape[0], 200_000, replace=False)]

    centroids = median_cut_palette(pixels.astype(np.int32), n_colors).colors.astype(np.float32)
    for _ in range(iterations):
        distances = ((pixels[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assignments = distances.argmin(axis=1)
        for index in range(centroids.shape[0]):
            members = pixels[assignments == index]
            if members.shape[0]:
                centroids[index] = members.mean(axis=0)
            else:
                # Re-seed a dead centroid on the worst-fit pixel rather than
                # leaving a wasted palette entry.
                centroids[index] = pixels[distances.min(axis=1).argmax()]
    return Palette(np.clip(np.round(centroids), 0, 255).astype(np.uint8))


def build_palette(
    images: Sequence[np.ndarray],
    n_colors: int = 64,
    *,
    method: str = "auto",
    seed: int = 0,
    max_sample_pixels: int = 500_000,
) -> Palette:
    """Build a palette for a corpus.

    ``method="auto"`` first tries :func:`exact_palette` and falls back to
    median cut, which is the right default for pixel art: most sprite sets
    already use fewer colours than the budget, and quantizing them at all would
    only introduce error.
    """
    if not len(images):
        raise ValueError("build_palette needs at least one image")

    normalized = method.strip().lower()
    if normalized in {"auto", "exact"}:
        palette = exact_palette(images, max_colors=n_colors)
        if palette is not None:
            logger.info("corpus uses %d distinct colours; using them exactly", len(palette))
            return palette
        if normalized == "exact":
            raise ValueError(
                f"corpus uses more than {n_colors} distinct colours; "
                "use method='median_cut' or 'kmeans'"
            )
        normalized = "median_cut"

    rng = np.random.default_rng(seed)
    stacked = np.concatenate([np.asarray(i, dtype=np.uint8).reshape(-1, 3) for i in images])
    if stacked.shape[0] > max_sample_pixels:
        stacked = stacked[rng.choice(stacked.shape[0], max_sample_pixels, replace=False)]

    if normalized == "median_cut":
        return median_cut_palette(stacked, n_colors)
    if normalized == "kmeans":
        return kmeans_palette(stacked, n_colors, seed=seed)
    raise ValueError(f"unknown palette method {method!r}")
