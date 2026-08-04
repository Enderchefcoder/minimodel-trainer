"""PyTorch datasets over image corpora, plus corpus preparation.

Images are stored as ``uint8`` and converted to the ``[-1, 1]`` float range on
the way out, which is the convention every diffusion model in this package
expects. Palette corpora skip that conversion entirely and hand PixelGPT the
integer indices directly.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from minimodel.core.logging_utils import get_logger
from minimodel.core.seeding import seed_worker
from minimodel.vision.data.palette import Palette, build_palette
from minimodel.vision.data.shards import ImageCorpus, ImageShardWriter

__all__ = [
    "ImageDataset",
    "PairedImageDataset",
    "PixelSequenceDataset",
    "build_image_dataloader",
    "load_images_from_directory",
    "prepare_image_corpus",
    "prepare_pixel_corpus",
    "synthetic_sprites",
]

logger = get_logger(__name__)


def _to_float(array: np.ndarray) -> torch.Tensor:
    """``[H, W, C]`` uint8 -> ``[C, H, W]`` float in ``[-1, 1]``."""
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32) / 127.5 - 1.0)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(-1)
    return tensor.permute(2, 0, 1).contiguous()


class ImageDataset(Dataset):
    """Images (and optional labels/captions) from an :class:`ImageCorpus`.

    Returns a dict with ``image`` in ``[-1, 1]`` and, when present, ``label``
    and ``caption``.
    """

    def __init__(
        self,
        corpus: ImageCorpus | str | Path,
        *,
        horizontal_flip: bool = False,
        tokenizer: Any = None,
        caption_max_len: int = 32,
        seed: int = 0,
    ):
        self.corpus = corpus if isinstance(corpus, ImageCorpus) else ImageCorpus(corpus)
        self.horizontal_flip = bool(horizontal_flip)
        self.tokenizer = tokenizer
        self.caption_max_len = int(caption_max_len)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.corpus)

    def __repr__(self) -> str:
        return f"ImageDataset({len(self)} images, {self.corpus.index.height}px)"

    def _tokenize(self, caption: str) -> torch.Tensor:
        ids = self.tokenizer.encode(caption, allow_special=False)[: self.caption_max_len]
        padded = ids + [0] * (self.caption_max_len - len(ids))
        return torch.tensor(padded, dtype=torch.long)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one training example."""
        array = self.corpus.image(int(index))
        if self.horizontal_flip:
            rng = np.random.default_rng(self.seed * 7919 + int(index))
            if rng.random() < 0.5:
                array = array[:, ::-1].copy()
        item: dict[str, Any] = {"image": _to_float(array)}
        label = self.corpus.label(int(index))
        if label >= 0:
            item["label"] = torch.tensor(label, dtype=torch.long)
        if self.tokenizer is not None:
            item["text_tokens"] = self._tokenize(self.corpus.caption(int(index)))
        return item


class PairedImageDataset(ImageDataset):
    """Source/target image pairs with an instruction, for edit training."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return ``image`` (target), ``source`` and ``text_tokens``."""
        item = super().__getitem__(index)
        source = self.corpus.source_image(int(index))
        if source is None:
            raise ValueError(
                f"{self.corpus.directory} has no paired source images; "
                "prepare it with with_pairs=True"
            )
        item["source"] = _to_float(source)
        return item


class PixelSequenceDataset(Dataset):
    """Flattened palette indices for :class:`~minimodel.vision.architectures.PixelGPT`."""

    def __init__(
        self,
        corpus: ImageCorpus | str | Path,
        *,
        horizontal_flip: bool = False,
        seed: int = 0,
    ):
        self.corpus = corpus if isinstance(corpus, ImageCorpus) else ImageCorpus(corpus)
        if self.corpus.index.mode != "palette":
            raise ValueError(
                f"{self.corpus.directory} is not a palette corpus; "
                "prepare it with prepare_pixel_corpus()"
            )
        self.horizontal_flip = bool(horizontal_flip)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.corpus)

    def __repr__(self) -> str:
        return f"PixelSequenceDataset({len(self)} images, palette={self.corpus.index.palette_size})"

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return ``pixels`` as a flat ``[H*W]`` long tensor."""
        array = self.corpus.image(int(index))
        if self.horizontal_flip:
            rng = np.random.default_rng(self.seed * 7919 + int(index))
            if rng.random() < 0.5:
                array = array[:, ::-1].copy()
        item: dict[str, torch.Tensor] = {
            "pixels": torch.from_numpy(np.asarray(array, dtype=np.int64).reshape(-1))
        }
        label = self.corpus.label(int(index))
        if label >= 0:
            item["label"] = torch.tensor(label, dtype=torch.long)
        return item


def collate_images(items: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of example dicts into a batch."""
    keys = items[0].keys()
    return {key: torch.stack([item[key] for item in items]) for key in keys}


def build_image_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 0,
    drop_last: bool = True,
    pin_memory: bool = False,
) -> DataLoader:
    """Wrap an image dataset in a ``DataLoader``."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_images,
        drop_last=drop_last,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Corpus preparation
# ---------------------------------------------------------------------------
def _load_pil(data: bytes | str | Path, size: int) -> np.ndarray:
    """Decode and resize one image to ``[size, size, 3]`` uint8."""
    from PIL import Image

    source = io.BytesIO(data) if isinstance(data, bytes) else data
    with Image.open(source) as image:
        image = image.convert("RGB")
        if image.size != (size, size):
            # NEAREST preserves hard pixel-art edges; anything smoother turns a
            # sprite into mush.
            resample = Image.NEAREST if size <= 64 else Image.BICUBIC
            image = image.resize((size, size), resample)
        return np.asarray(image, dtype=np.uint8)


def load_images_from_directory(
    directory: str | Path, size: int = 32, *, limit: int | None = None
) -> Iterator[tuple[np.ndarray, str]]:
    """Yield ``(image, class_name)`` for every image file under ``directory``.

    Sub-directory names become class labels, matching the usual ImageFolder
    convention.
    """
    directory = Path(directory)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
    count = 0
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in extensions:
            continue
        class_name = path.parent.name if path.parent != directory else ""
        try:
            yield _load_pil(path, size), class_name
        except (OSError, ValueError) as exc:
            logger.warning("skipping %s: %s", path, exc)
            continue
        count += 1
        if limit is not None and count >= limit:
            return


def prepare_image_corpus(
    images: Iterable[tuple[np.ndarray, str] | np.ndarray],
    output_dir: str | Path,
    *,
    size: int = 32,
    captions: Sequence[str] | None = None,
    source_images: Sequence[np.ndarray] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Write RGB images into an image corpus directory."""
    materialised: list[tuple[np.ndarray, str]] = []
    for entry in images:
        if isinstance(entry, tuple):
            materialised.append((np.asarray(entry[0], dtype=np.uint8), str(entry[1])))
        else:
            materialised.append((np.asarray(entry, dtype=np.uint8), ""))
    if not materialised:
        raise ValueError("no images to write")

    class_names = sorted({name for _, name in materialised if name})
    class_index = {name: i for i, name in enumerate(class_names)}

    with ImageShardWriter(
        output_dir,
        height=size,
        width=size,
        channels=3,
        mode="rgb",
        class_names=class_names,
        source=source,
        with_pairs=source_images is not None,
    ) as writer:
        for index, (image, class_name) in enumerate(materialised):
            writer.write(
                image,
                label=class_index.get(class_name) if class_name else None,
                caption=captions[index] if captions and index < len(captions) else None,
                source_image=source_images[index] if source_images is not None else None,
            )
        index_data = writer.index
    return index_data.to_dict()


def prepare_pixel_corpus(
    images: Iterable[tuple[np.ndarray, str] | np.ndarray],
    output_dir: str | Path,
    *,
    size: int = 24,
    palette_size: int = 64,
    palette: Palette | None = None,
    palette_method: str = "auto",
    source: str | None = None,
) -> dict[str, Any]:
    """Quantize images to a shared palette and write a palette corpus.

    The palette is saved next to the shards as ``palette.json`` so that samples
    can be converted back to RGB later.
    """
    materialised: list[tuple[np.ndarray, str]] = []
    for entry in images:
        if isinstance(entry, tuple):
            materialised.append((np.asarray(entry[0], dtype=np.uint8), str(entry[1])))
        else:
            materialised.append((np.asarray(entry, dtype=np.uint8), ""))
    if not materialised:
        raise ValueError("no images to write")

    if palette is None:
        palette = build_palette(
            [image for image, _ in materialised], palette_size, method=palette_method
        )
    logger.info("palette has %d colours", len(palette))

    class_names = sorted({name for _, name in materialised if name})
    class_index = {name: i for i, name in enumerate(class_names)}

    with ImageShardWriter(
        output_dir,
        height=size,
        width=size,
        mode="palette",
        palette_size=len(palette),
        class_names=class_names,
        source=source,
    ) as writer:
        for image, class_name in materialised:
            indices = palette.quantize(image).astype(np.uint8)
            writer.write(indices, label=class_index.get(class_name) if class_name else None)
        index_data = writer.index

    palette.save(Path(output_dir) / "palette.json")
    payload = index_data.to_dict()
    payload["palette"] = str(Path(output_dir) / "palette.json")
    return payload


def synthetic_sprites(
    n: int = 256, size: int = 24, *, n_colors: int = 8, seed: int = 0
) -> list[tuple[np.ndarray, str]]:
    """Generate simple symmetric sprites for offline testing.

    Each sprite is a vertically mirrored blob on a background, which is enough
    structure for a small PixelGPT to visibly learn (symmetry and a bounded
    palette) within a few hundred steps - and it needs no downloads.
    """
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(n_colors, 3), dtype=np.uint8)
    palette[0] = np.array([16, 16, 24], dtype=np.uint8)  # background

    sprites: list[tuple[np.ndarray, str]] = []
    half = size // 2
    for index in range(n):
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:, :] = palette[0]
        body_color = palette[rng.integers(1, n_colors)]
        accent_color = palette[rng.integers(1, n_colors)]

        top = int(rng.integers(2, max(3, size // 3)))
        bottom = int(rng.integers(size - size // 3, size - 1))
        for row in range(top, bottom):
            width = int(rng.integers(2, half))
            canvas[row, half - width : half] = body_color
            canvas[row, half : half + width] = body_color[::-1] if index % 3 == 0 else body_color

        eye_row = top + max(1, (bottom - top) // 4)
        offset = max(1, half // 3)
        canvas[eye_row, half - offset] = accent_color
        canvas[eye_row, half + offset - 1] = accent_color

        # Mirror the left half onto the right so every sprite is symmetric.
        canvas[:, half:] = canvas[:, :half][:, ::-1]
        sprites.append((canvas, f"class_{index % 4}"))
    return sprites
