"""Image architecture registry and template builder.

Mirrors :mod:`minimodel.architectures.builder` so that the CLI, configs and
tests treat image models exactly like language models.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from minimodel.core.config import Config, ConfigError, load_config, merge_configs
from minimodel.core.devices import resolve_device, resolve_dtype
from minimodel.core.logging_utils import get_logger
from minimodel.core.registry import Registry
from minimodel.vision.architectures.base import BaseImageModel
from minimodel.vision.architectures.dit import DiT
from minimodel.vision.architectures.edit import ImageEditModel
from minimodel.vision.architectures.pixelgpt import PixelGPT
from minimodel.vision.architectures.unet import UNet
from minimodel.vision.architectures.vae import VAE

__all__ = [
    "IMAGE_ARCHITECTURES",
    "VISION_TEMPLATE_DIR",
    "build_image_model",
    "describe_image_model",
    "list_image_architectures",
    "list_image_templates",
    "load_image_model",
    "load_vision_template",
]

logger = get_logger(__name__)

#: Directory holding the bundled image-model templates.
VISION_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"

#: Maps a ``family`` string to a :class:`BaseImageModel` subclass.
IMAGE_ARCHITECTURES: Registry[type[BaseImageModel]] = Registry("image architecture")
IMAGE_ARCHITECTURES.add(DiT.architecture_name, DiT, aliases=("diffusion_transformer", "dit_image"))
IMAGE_ARCHITECTURES.add(UNet.architecture_name, UNet, aliases=("unet2d", "conv_unet"))
IMAGE_ARCHITECTURES.add(PixelGPT.architecture_name, PixelGPT, aliases=("pixel_gpt", "pixelart"))
IMAGE_ARCHITECTURES.add(
    ImageEditModel.architecture_name, ImageEditModel, aliases=("edit", "instructpix2pix")
)
IMAGE_ARCHITECTURES.add(VAE.architecture_name, VAE, aliases=("autoencoder", "kl_vae"))


def list_image_architectures() -> list[str]:
    """All registered image architecture names."""
    return IMAGE_ARCHITECTURES.names()


def list_image_templates() -> list[str]:
    """Names of the bundled image templates."""
    if not VISION_TEMPLATE_DIR.exists():
        return []
    return sorted(p.stem for p in VISION_TEMPLATE_DIR.glob("*.yaml"))


def resolve_vision_template_path(spec: str | Path) -> Path:
    """Resolve a template name or path to a file."""
    path = Path(spec)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        return path
    for candidate in (
        VISION_TEMPLATE_DIR / f"{spec}.yaml",
        VISION_TEMPLATE_DIR / f"{spec}.yml",
        VISION_TEMPLATE_DIR / str(spec),
    ):
        if candidate.exists():
            return candidate
    available = ", ".join(list_image_templates()) or "<none>"
    raise ConfigError(f"image template {spec!r} not found; bundled templates: {available}")


def load_vision_template(spec: str | Path) -> Config:
    """Load an image template by name or path."""
    return load_config(resolve_vision_template_path(spec))


def build_image_model(
    spec: str | Path | Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    verify_budget: bool = True,
) -> BaseImageModel:
    """Instantiate an image model from a template name, path or mapping.

    >>> model = build_image_model("pixelgpt_24x24_10m")
    >>> model.architecture_name
    'pixelgpt'
    """
    if isinstance(spec, (str, Path)):
        template: Mapping[str, Any] = load_vision_template(spec).to_dict()
    else:
        template = dict(spec)

    if "family" not in template and "architecture" in template:
        family = str(template["architecture"])
        flat = {k: v for k, v in template.items() if k != "architecture"}
    else:
        family = str(template.get("family") or "")
        if not family:
            raise ConfigError("image template must declare `family` (or `architecture`)")
        arch_block = template.get("arch") or {}
        if not isinstance(arch_block, Mapping):
            raise ConfigError("`arch` section of an image template must be a mapping")
        flat = dict(arch_block)

    if overrides:
        flat = merge_configs(flat, dict(overrides))

    model_cls = IMAGE_ARCHITECTURES.get(family)
    model = model_cls.from_config(flat)

    if verify_budget and template.get("params") is not None:
        declared = int(template["params"])
        actual = model.num_parameters()
        if declared != actual:
            logger.warning(
                "template %s declares %s parameters but the built model has %s",
                template.get("name", family),
                f"{declared:,}",
                f"{actual:,}",
            )

    if device is not None or dtype is not None:
        target_device = resolve_device(device) if device is not None else torch.device("cpu")
        target_dtype = resolve_dtype(dtype) if dtype is not None else torch.float32
        model = model.to(device=target_device, dtype=target_dtype)
    return model


def load_image_model(
    directory: str | Path,
    *,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> BaseImageModel:
    """Load a model written by :meth:`BaseImageModel.save_pretrained`."""
    import json

    directory = Path(directory)
    config_path = directory / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"no config.json in {directory}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = build_image_model(config, device=device, verify_budget=False)
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=strict)
    if device is not None:
        model = model.to(resolve_device(device))
    return model


def describe_image_model(model: BaseImageModel) -> dict[str, Any]:
    """Summarise an image model for logging and cards."""
    return {
        "architecture": model.architecture_name,
        "parameters": model.num_parameters(),
        "breakdown": model.parameter_breakdown(),
        "config": dict(model.config),
    }
