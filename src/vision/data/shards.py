"""On-disk format for image corpora.

Images are stored as one flat ``uint8`` array plus an ``index.json``::

    data/sprites/
      index.json
      images_0000.bin      # [N, H, W, C] uint8, or [N, H, W] palette indices
      labels_0000.bin      # optional int32 class labels
      captions.jsonl       # optional captions/instructions, one per image

Fixed-size ``uint8`` records mean the whole corpus is one memmap and a random
batch is a single fancy-index - no per-image decode, no JPEG library on the
training hot path. A 20K sprite set at 24x24 RGB is 34 MiB, so it stays in page
cache permanently.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from minimodel.core.io_utils import ensure_dir, human_bytes, write_json
from minimodel.core.logging_utils import get_logger

__all__ = ["ImageCorpus", "ImageShardIndex", "ImageShardWriter"]

logger = get_logger(__name__)


@dataclass
class ImageShardIndex:
    """Metadata describing an image corpus directory."""

    height: int = 0
    width: int = 0
    channels: int = 3
    n_images: int = 0
    mode: str = "rgb"  # rgb | palette
    palette_size: int = 0
    n_classes: int = 0
    class_names: list[str] = field(default_factory=list)
    has_captions: bool = False
    has_pairs: bool = False
    shards: list[dict[str, Any]] = field(default_factory=list)
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "format": "minimodel-image-shards/1",
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
            "n_images": self.n_images,
            "mode": self.mode,
            "palette_size": self.palette_size,
            "n_classes": self.n_classes,
            "class_names": self.class_names,
            "has_captions": self.has_captions,
            "has_pairs": self.has_pairs,
            "shards": self.shards,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageShardIndex:
        """Parse an ``index.json`` payload."""
        return cls(
            height=int(data.get("height", 0)),
            width=int(data.get("width", 0)),
            channels=int(data.get("channels", 3)),
            n_images=int(data.get("n_images", 0)),
            mode=str(data.get("mode", "rgb")),
            palette_size=int(data.get("palette_size", 0)),
            n_classes=int(data.get("n_classes", 0)),
            class_names=list(data.get("class_names", [])),
            has_captions=bool(data.get("has_captions", False)),
            has_pairs=bool(data.get("has_pairs", False)),
            shards=list(data.get("shards", [])),
            source=data.get("source"),
        )

    @property
    def item_shape(self) -> tuple[int, ...]:
        """Shape of one stored image."""
        if self.mode == "palette":
            return (self.height, self.width)
        return (self.height, self.width, self.channels)

    @property
    def item_size(self) -> int:
        """Number of bytes per stored image."""
        return int(np.prod(self.item_shape))


class ImageShardWriter:
    """Streaming writer for an image corpus.

    Parameters
    ----------
    directory:
        Output directory.
    height, width:
        Image size. Every image must match.
    mode:
        ``rgb`` for ``[H, W, 3]`` uint8, or ``palette`` for ``[H, W]`` indices.
    shard_images:
        Images per shard file.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        height: int,
        width: int,
        channels: int = 3,
        mode: str = "rgb",
        palette_size: int = 0,
        class_names: Sequence[str] = (),
        shard_images: int = 50_000,
        source: str | None = None,
        with_pairs: bool = False,
    ):
        self.directory = ensure_dir(directory)
        self.shard_images = int(shard_images)
        self.index = ImageShardIndex(
            height=int(height),
            width=int(width),
            channels=int(channels) if mode == "rgb" else 1,
            mode=mode,
            palette_size=int(palette_size),
            n_classes=len(class_names),
            class_names=list(class_names),
            has_pairs=bool(with_pairs),
            source=source,
        )
        self._buffer: list[np.ndarray] = []
        self._source_buffer: list[np.ndarray] = []
        self._labels: list[int] = []
        self._captions: list[str] = []
        self._shard_id = 0
        self._with_pairs = bool(with_pairs)

    def write(
        self,
        image: np.ndarray,
        *,
        label: int | None = None,
        caption: str | None = None,
        source_image: np.ndarray | None = None,
    ) -> None:
        """Append one image (with optional label, caption and source image)."""
        array = np.asarray(image, dtype=np.uint8)
        if array.shape != self.index.item_shape:
            raise ValueError(
                f"image shape {array.shape} does not match corpus shape {self.index.item_shape}"
            )
        self._buffer.append(array)
        self._labels.append(-1 if label is None else int(label))
        if caption is not None:
            self.index.has_captions = True
        self._captions.append(caption or "")
        if self._with_pairs:
            if source_image is None:
                raise ValueError("this corpus stores pairs, so source_image is required")
            self._source_buffer.append(np.asarray(source_image, dtype=np.uint8))
        self.index.n_images += 1
        if len(self._buffer) >= self.shard_images:
            self.flush()

    def write_many(self, images: Iterable[np.ndarray], **kwargs: Any) -> None:
        """Append many images with the same options."""
        for image in images:
            self.write(image, **kwargs)

    def flush(self) -> None:
        """Write the buffered images out as a shard."""
        if not self._buffer:
            return
        name = f"images_{self._shard_id:04d}.bin"
        np.stack(self._buffer).tofile(self.directory / name)
        entry: dict[str, Any] = {"path": name, "n_images": len(self._buffer)}
        if any(label >= 0 for label in self._labels):
            label_name = f"labels_{self._shard_id:04d}.bin"
            np.asarray(self._labels, dtype=np.int32).tofile(self.directory / label_name)
            entry["labels_path"] = label_name
        if self._with_pairs:
            source_name = f"sources_{self._shard_id:04d}.bin"
            np.stack(self._source_buffer).tofile(self.directory / source_name)
            entry["sources_path"] = source_name
        self.index.shards.append(entry)
        self._buffer.clear()
        self._source_buffer.clear()
        self._labels.clear()
        self._shard_id += 1

    def close(self) -> ImageShardIndex:
        """Flush, write captions and ``index.json``."""
        captions = list(self._captions)
        self.flush()
        if self.index.has_captions:
            with (self.directory / "captions.jsonl").open("w", encoding="utf-8") as handle:
                for caption in captions:
                    handle.write(json.dumps({"caption": caption}) + "\n")
        write_json(self.directory / "index.json", self.index.to_dict())
        size = self.index.n_images * self.index.item_size
        logger.info(
            "wrote %d images (%s) to %s",
            self.index.n_images,
            human_bytes(size),
            self.directory,
        )
        return self.index

    def __enter__(self) -> ImageShardWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class ImageCorpus:
    """Read-only memmapped view over an image corpus directory."""

    def __init__(self, directory: str | Path, *, mmap: bool = True):
        self.directory = Path(directory)
        index_path = self.directory / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"{self.directory} is not an image corpus (no index.json). "
                "Run `minimodel vision data prepare` first."
            )
        self.index = ImageShardIndex.from_dict(
            json.loads(index_path.read_text(encoding="utf-8"))
        )
        shape = self.index.item_shape
        self._arrays: list[np.ndarray] = []
        self._sources: list[np.ndarray | None] = []
        self._labels: list[np.ndarray | None] = []
        self._offsets: list[int] = []

        offset = 0
        for shard in self.index.shards:
            count = int(shard["n_images"])
            array = np.memmap(
                self.directory / shard["path"], dtype=np.uint8, mode="r", shape=(count, *shape)
            ) if mmap else np.fromfile(self.directory / shard["path"], dtype=np.uint8).reshape(count, *shape)
            self._arrays.append(array)
            self._offsets.append(offset)
            offset += count

            labels_path = shard.get("labels_path")
            self._labels.append(
                np.fromfile(self.directory / labels_path, dtype=np.int32) if labels_path else None
            )
            sources_path = shard.get("sources_path")
            self._sources.append(
                np.memmap(
                    self.directory / sources_path, dtype=np.uint8, mode="r", shape=(count, *shape)
                )
                if sources_path
                else None
            )
        self.n_images = offset

        self.captions: list[str] = []
        captions_path = self.directory / "captions.jsonl"
        if captions_path.exists():
            with captions_path.open("r", encoding="utf-8") as handle:
                self.captions = [json.loads(line)["caption"] for line in handle if line.strip()]

    def __len__(self) -> int:
        return self.n_images

    def __repr__(self) -> str:
        return (
            f"ImageCorpus({self.directory.name!r}, n={self.n_images}, "
            f"{self.index.height}x{self.index.width}, mode={self.index.mode})"
        )

    def _locate(self, index: int) -> tuple[int, int]:
        for shard_index in range(len(self._offsets) - 1, -1, -1):
            if self._offsets[shard_index] <= index:
                return shard_index, index - self._offsets[shard_index]
        raise IndexError(index)

    def image(self, index: int) -> np.ndarray:
        """Return image ``index`` as a uint8 array."""
        if not 0 <= index < self.n_images:
            raise IndexError(f"image index {index} out of range (n={self.n_images})")
        shard_index, offset = self._locate(index)
        return np.asarray(self._arrays[shard_index][offset])

    def source_image(self, index: int) -> np.ndarray | None:
        """Return the paired source image, for edit corpora."""
        shard_index, offset = self._locate(index)
        sources = self._sources[shard_index]
        return None if sources is None else np.asarray(sources[offset])

    def label(self, index: int) -> int:
        """Return the class label, or ``-1`` when unlabelled."""
        shard_index, offset = self._locate(index)
        labels = self._labels[shard_index]
        return -1 if labels is None else int(labels[offset])

    def caption(self, index: int) -> str:
        """Return the caption/instruction, or an empty string."""
        return self.captions[index] if index < len(self.captions) else ""

    def stats(self) -> dict[str, Any]:
        """Summary for ``minimodel vision data info``."""
        return {
            "directory": str(self.directory),
            "n_images": self.n_images,
            "size": f"{self.index.height}x{self.index.width}",
            "mode": self.index.mode,
            "channels": self.index.channels,
            "palette_size": self.index.palette_size,
            "n_classes": self.index.n_classes,
            "has_captions": bool(self.captions),
            "has_pairs": self.index.has_pairs,
            "bytes": self.n_images * self.index.item_size,
        }
