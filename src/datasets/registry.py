"""Access to the dataset and mixture registries.

The registries are plain YAML (``config/datasets.yaml`` for text,
``config/image_datasets.yaml`` for images) so that adding a corpus never
requires touching Python. This module loads them, validates the references
between mixtures and datasets, and exposes lookup helpers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from minimodel.core.config import ConfigError

__all__ = [
    "CONFIG_DIR",
    "DatasetSpec",
    "MixtureSpec",
    "get_dataset",
    "get_mixture",
    "list_datasets",
    "list_mixtures",
    "load_registry",
    "resolve_mixture",
]

#: Directory holding the bundled registry YAML files.
CONFIG_DIR = Path(__file__).parent / "config"


@dataclass
class DatasetSpec:
    """One dataset recipe."""

    name: str
    source: str = "huggingface"
    repo: str | None = None
    config: str | None = None
    split: str = "train"
    text_field: str = "text"
    stage: str = "pretrain"
    format: str = "text"
    fields: dict[str, str] = field(default_factory=dict)
    tokens: str | None = None
    license: str | None = None
    description: str = ""
    path: str | None = None
    url: str | None = None
    verifier: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any]) -> DatasetSpec:
        """Build a spec from a registry entry."""
        known = {f for f in cls.__dataclass_fields__ if f not in {"name", "extra"}}
        payload = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(name=name, extra=extra, **payload)

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping, used in run metadata and model cards."""
        return {
            "name": self.name,
            "source": self.source,
            "repo": self.repo,
            "config": self.config,
            "split": self.split,
            "stage": self.stage,
            "format": self.format,
            "tokens": self.tokens,
            "license": self.license,
        }

    @property
    def display(self) -> str:
        """Human-readable identifier, e.g. ``HuggingFaceTB/smollm-corpus:cosmopedia-v2``."""
        if self.source == "huggingface" and self.repo:
            return f"{self.repo}:{self.config}" if self.config else self.repo
        if self.source == "local" and self.path:
            return self.path
        if self.source == "url" and self.url:
            return self.url
        return f"{self.source}:{self.name}"


@dataclass
class MixtureSpec:
    """A weighted blend of datasets."""

    name: str
    description: str = ""
    stage: str = "pretrain"
    components: list[dict[str, Any]] = field(default_factory=list)

    def normalized_weights(self) -> list[tuple[str, float]]:
        """Return ``[(dataset_name, weight)]`` with weights summing to 1."""
        pairs = [(str(c["dataset"]), float(c.get("weight", 1.0))) for c in self.components]
        total = sum(w for _, w in pairs)
        if total <= 0:
            raise ConfigError(f"mixture {self.name!r} has non-positive total weight")
        return [(name, weight / total) for name, weight in pairs]


@lru_cache(maxsize=4)
def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a registry YAML file.

    Defaults to the bundled text registry. Results are cached, so callers can
    treat this as cheap.
    """
    registry_path = Path(path) if path else CONFIG_DIR / "datasets.yaml"
    if not registry_path.exists():
        raise ConfigError(f"dataset registry not found: {registry_path}")
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    datasets = data.get("datasets") or {}
    mixtures = data.get("mixtures") or {}

    for mixture_name, mixture in mixtures.items():
        for component in mixture.get("components", []):
            referenced = component.get("dataset")
            if referenced not in datasets:
                raise ConfigError(
                    f"mixture {mixture_name!r} references unknown dataset {referenced!r} "
                    f"in {registry_path}"
                )
    return {"datasets": datasets, "mixtures": mixtures, "path": str(registry_path)}


def list_datasets(
    *, stage: str | None = None, path: str | Path | None = None
) -> list[DatasetSpec]:
    """All registered datasets, optionally filtered by training stage."""
    registry = load_registry(path)
    specs = [DatasetSpec.from_dict(name, data) for name, data in registry["datasets"].items()]
    if stage:
        specs = [s for s in specs if s.stage == stage]
    return sorted(specs, key=lambda s: (s.stage, s.name))


def get_dataset(name: str, *, path: str | Path | None = None) -> DatasetSpec:
    """Look up one dataset by registry name."""
    registry = load_registry(path)
    data = registry["datasets"].get(name)
    if data is None:
        available = ", ".join(sorted(registry["datasets"]))
        raise ConfigError(f"unknown dataset {name!r}; available: {available}")
    return DatasetSpec.from_dict(name, data)


def list_mixtures(*, path: str | Path | None = None) -> list[MixtureSpec]:
    """All registered mixtures."""
    registry = load_registry(path)
    return [
        MixtureSpec(
            name=name,
            description=data.get("description", ""),
            stage=data.get("stage", "pretrain"),
            components=list(data.get("components", [])),
        )
        for name, data in sorted(registry["mixtures"].items())
    ]


def get_mixture(name: str, *, path: str | Path | None = None) -> MixtureSpec:
    """Look up one mixture by registry name."""
    registry = load_registry(path)
    data = registry["mixtures"].get(name)
    if data is None:
        available = ", ".join(sorted(registry["mixtures"]))
        raise ConfigError(f"unknown mixture {name!r}; available: {available}")
    return MixtureSpec(
        name=name,
        description=data.get("description", ""),
        stage=data.get("stage", "pretrain"),
        components=list(data.get("components", [])),
    )


def resolve_mixture(
    name: str, *, path: str | Path | None = None
) -> list[tuple[DatasetSpec, float]]:
    """Expand a mixture into ``[(spec, normalised_weight)]``."""
    mixture = get_mixture(name, path=path)
    return [
        (get_dataset(dataset_name, path=path), weight)
        for dataset_name, weight in mixture.normalized_weights()
    ]
