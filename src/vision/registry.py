"""Image dataset registry.

Same idea as :mod:`minimodel.datasets.registry`, for image corpora. Entries live
in ``config/image_datasets.yaml`` next to the text registry so both catalogues
are edited the same way.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from minimodel.core.config import ConfigError
from minimodel.core.logging_utils import get_logger

__all__ = [
    "IMAGE_CONFIG_PATH",
    "ImageDatasetSpec",
    "get_image_dataset",
    "iter_image_records",
    "list_image_datasets",
    "load_image_registry",
]

logger = get_logger(__name__)

#: Path of the bundled image dataset registry.
IMAGE_CONFIG_PATH = Path(__file__).parent.parent / "datasets" / "config" / "image_datasets.yaml"


@dataclass
class ImageDatasetSpec:
    """One image dataset recipe."""

    name: str
    source: str = "huggingface"
    repo: str | None = None
    config: str | None = None
    split: str = "train"
    image_field: str = "image"
    caption_field: str | None = None
    source_image_field: str | None = None
    label_field: str | None = None
    kind: str = "generation"
    image_size: int = 64
    images: str | None = None
    license: str | None = None
    description: str = ""
    path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any]) -> ImageDatasetSpec:
        """Build a spec from a registry entry."""
        known = {f for f in cls.__dataclass_fields__ if f not in {"name", "extra"}}
        payload = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(name=name, extra=extra, **payload)

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping for tables and metadata."""
        return {
            "name": self.name,
            "source": self.source,
            "repo": self.repo,
            "kind": self.kind,
            "image_size": self.image_size,
            "images": self.images,
            "license": self.license,
        }

    @property
    def display(self) -> str:
        """Human-readable identifier."""
        if self.source == "huggingface" and self.repo:
            return f"{self.repo}:{self.config}" if self.config else self.repo
        if self.source == "local" and self.path:
            return self.path
        return f"{self.source}:{self.name}"


@lru_cache(maxsize=2)
def load_image_registry(path: str | Path | None = None) -> dict[str, Any]:
    """Load the image dataset registry YAML."""
    registry_path = Path(path) if path else IMAGE_CONFIG_PATH
    if not registry_path.exists():
        raise ConfigError(f"image dataset registry not found: {registry_path}")
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    return {"datasets": data.get("datasets") or {}, "path": str(registry_path)}


def list_image_datasets(
    *, kind: str | None = None, path: str | Path | None = None
) -> list[ImageDatasetSpec]:
    """All registered image datasets, optionally filtered by kind."""
    registry = load_image_registry(path)
    specs = [ImageDatasetSpec.from_dict(n, d) for n, d in registry["datasets"].items()]
    if kind:
        specs = [s for s in specs if s.kind == kind]
    return sorted(specs, key=lambda s: (s.kind, s.name))


def get_image_dataset(name: str, *, path: str | Path | None = None) -> ImageDatasetSpec:
    """Look up one image dataset by name."""
    registry = load_image_registry(path)
    data = registry["datasets"].get(name)
    if data is None:
        available = ", ".join(sorted(registry["datasets"]))
        raise ConfigError(f"unknown image dataset {name!r}; available: {available}")
    return ImageDatasetSpec.from_dict(name, data)


def _to_array(value: Any, size: int) -> np.ndarray | None:
    """Coerce a dataset field into an ``[size, size, 3]`` uint8 array."""
    from minimodel.vision.data.datasets import _load_pil

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.shape[:2] != (size, size):
            from PIL import Image

            image = Image.fromarray(array.astype(np.uint8)).resize(
                (size, size), Image.NEAREST if size <= 64 else Image.BICUBIC
            )
            array = np.asarray(image)
        return array.astype(np.uint8)
    if isinstance(value, (bytes, str, Path)):
        return _load_pil(value, size)
    if isinstance(value, Mapping) and "bytes" in value:
        return _load_pil(value["bytes"], size)
    # datasets returns PIL images for image columns.
    if hasattr(value, "convert"):
        from PIL import Image

        image = value.convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.NEAREST if size <= 64 else Image.BICUBIC)
        return np.asarray(image, dtype=np.uint8)
    return None


def iter_image_records(
    name: str,
    *,
    size: int = 64,
    limit: int | None = None,
    registry_path: str | Path | None = None,
) -> Iterator[tuple[np.ndarray, str]]:
    """Yield ``(image, class_name)`` pairs for a registered image dataset."""
    spec = get_image_dataset(name, path=registry_path)

    if spec.source == "local":
        from minimodel.vision.data.datasets import load_images_from_directory

        if not spec.path:
            raise ConfigError(f"image dataset {name!r} has source 'local' but no path")
        yield from load_images_from_directory(spec.path, size, limit=limit)
        return

    if spec.source != "huggingface":
        raise ConfigError(f"unknown image dataset source {spec.source!r}")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "pulling Hugging Face image datasets requires the `datasets` package.\n"
            "Install it with:  pip install 'minimodel-trainer[hf]'"
        ) from exc

    kwargs: dict[str, Any] = {"split": spec.split, "streaming": True}
    if spec.config:
        kwargs["name"] = spec.config
    dataset = load_dataset(spec.repo, **kwargs)  # pragma: no cover - network

    for index, record in enumerate(dataset):  # pragma: no cover - network
        if limit is not None and index >= limit:
            return
        array = _to_array(record.get(spec.image_field), size)
        if array is None:
            continue
        label = ""
        if spec.label_field and record.get(spec.label_field) is not None:
            label = str(record[spec.label_field])
        yield array, label
