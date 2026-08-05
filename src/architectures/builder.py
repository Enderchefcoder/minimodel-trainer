"""Turn architecture templates into instantiated models.

A *template* is a YAML file describing one model. Two layouts are supported:

**Canonical (recommended)** - a flat ``arch:`` block whose keys map 1:1 onto the
architecture's config::

    name: mm-30m
    family: dense-transformer
    arch:
      vocab_size: 16384
      dim: 384
      n_layers: 8

**Annotated** - the fully documented nested layout used by
``supra2_1406240.yaml``, where the model is described section by section
(``model.attention.window``, ``model.topology.recurrent.train_loops``, ...).
This layout doubles as the architecture's written specification, so the builder
knows how to read it rather than requiring the values to be duplicated.

Both forms are resolved by :func:`build_model`, which also verifies the declared
``params`` / ``param_budget.total`` against the model it actually built.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from minimodel.architectures.base import BaseLanguageModel
from minimodel.architectures.registry import ARCHITECTURES
from minimodel.core.config import Config, ConfigError, load_config, merge_configs
from minimodel.core.devices import resolve_device, resolve_dtype
from minimodel.core.logging_utils import get_logger

__all__ = [
    "TEMPLATE_DIR",
    "build_model",
    "describe_model",
    "list_glint2_candidates",
    "list_templates",
    "load_model",
    "load_template",
    "template_to_model_config",
]

logger = get_logger(__name__)

#: Directory holding the bundled architecture templates.
TEMPLATE_DIR = Path(__file__).parent / "templates"


def list_templates() -> list[str]:
    """Names of the bundled templates (file stems), sorted."""
    if not TEMPLATE_DIR.exists():
        return []
    return sorted(p.stem for p in TEMPLATE_DIR.glob("*.yaml"))


def list_glint2_candidates() -> list[dict[str, Any]]:
    """Return the ~1M Glint-2 candidate templates ordered by ``glint2_rank``.

    Each entry carries ``name``, ``rank``, ``candidate_class``, ``params`` and
    ``family`` so callers can print a leaderboard without re-parsing YAML.
    """
    rows: list[dict[str, Any]] = []
    for name in list_templates():
        template = load_template(name)
        rank = template.get("glint2_rank")
        if rank is None:
            continue
        rows.append(
            {
                "name": name,
                "rank": int(rank),
                "candidate_class": str(template.get("candidate_class") or ""),
                "params": int(template.get("params") or 0),
                "family": str(template.get("family") or ""),
                "description": str(template.get("description") or ""),
            }
        )
    return sorted(rows, key=lambda row: row["rank"])


def resolve_template_path(spec: str | Path) -> Path:
    """Resolve a template name or path to a file.

    Accepts a bare template name (``"supra2_1406240"``), a name with extension,
    or any filesystem path.
    """
    path = Path(spec)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        return path
    candidates = [
        TEMPLATE_DIR / f"{spec}.yaml",
        TEMPLATE_DIR / f"{spec}.yml",
        TEMPLATE_DIR / str(spec),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    available = ", ".join(list_templates()) or "<none>"
    raise ConfigError(f"architecture template {spec!r} not found; bundled templates: {available}")


def load_template(spec: str | Path) -> Config:
    """Load a template by name or path."""
    return load_config(resolve_template_path(spec))


def _extract_annotated_looped(template: Mapping[str, Any]) -> dict[str, Any]:
    """Read the flat looped-transformer config out of the annotated layout."""
    cfg = Config(template)
    model = cfg.section("model")
    recurrent = model.section("topology").section("recurrent")
    variable = recurrent.section("variable_loops")
    per_loop = recurrent.section("per_loop")

    out: dict[str, Any] = {}

    def _put(key: str, value: Any) -> None:
        if value is not None:
            out[key] = value

    _put("vocab_size", cfg.get("tokenizer.vocab_size", model.get("embedding.vocab_size")))
    _put("dim", model.get("dim"))
    _put("n_heads", model.get("n_heads"))
    _put("head_dim", model.get("head_dim"))
    _put("ffn_hidden", model.get("ffn_hidden"))
    _put("norm_eps", model.get("norm_eps"))
    _put("bias", model.get("bias"))
    _put("max_seq_len", model.get("max_position_embeddings"))
    _put("embedding_rank", model.get("embedding.rank"))
    _put("window", model.get("attention.window"))
    _put("value_residual", model.get("attention.value_residual.enabled"))
    _put("rope_base", model.get("position_encoding.base"))
    _put("n_shared_blocks", recurrent.get("n_shared_blocks"))
    _put("train_loops", recurrent.get("train_loops"))
    _put("min_loops", variable.get("min"))
    _put("variable_loops", variable.get("enabled"))
    _put("max_loops_table", recurrent.get("max_loops_table"))
    _put("loop_lora_rank", per_loop.get("loop_lora.rank"))

    gate_init = recurrent.get("outer_residual.gate_init")
    if isinstance(gate_init, str) and "," in gate_init:
        # e.g. "full(128, 0.1)" -> 0.1
        with contextlib.suppress(ValueError):
            out["outer_gate_init"] = float(gate_init.rsplit(",", 1)[1].strip(" )"))
    elif isinstance(gate_init, (int, float)):
        out["outer_gate_init"] = float(gate_init)

    init = cfg.section("weight_init")
    linear_init = init.get("linear")
    if isinstance(linear_init, str) and "std=" in linear_init:
        with contextlib.suppress(ValueError):
            out["init_std"] = float(linear_init.split("std=")[1].strip(" )"))
    return out


def template_to_model_config(template: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return ``(family, flat_config)`` for a template mapping.

    The canonical ``arch:`` block wins; anything it does not define is filled in
    from the annotated layout, so a template can mix the two.
    """
    cfg = Config(template)
    family = cfg.get("family") or cfg.get("architecture")
    if not family:
        raise ConfigError("architecture template must declare `family` (or `architecture`)")

    extracted: dict[str, Any] = {}
    if ARCHITECTURES.normalize(str(family)) in {"looped_transformer"} or "topology" in cfg.get(
        "model", {}
    ):
        extracted = _extract_annotated_looped(template)

    arch_block = cfg.get("arch") or {}
    if not isinstance(arch_block, Mapping):
        raise ConfigError("`arch` section of an architecture template must be a mapping")

    flat = merge_configs(extracted, dict(arch_block))
    # A tokenizer section is a convenient place to declare vocab size once.
    vocab = cfg.get("tokenizer.vocab_size")
    if vocab is not None:
        flat.setdefault("vocab_size", vocab)
    return str(family), flat


def build_model(
    spec: str | Path | Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    verify_budget: bool = True,
) -> BaseLanguageModel:
    """Instantiate a model from a template name, path or mapping.

    Parameters
    ----------
    spec:
        Bundled template name, path to a YAML file, or an already-loaded
        template mapping. A mapping containing an ``architecture`` key and flat
        config values (i.e. a saved ``config.json``) is also accepted.
    overrides:
        Flat config keys merged over the template, e.g. ``{"vocab_size": 8192}``.
        This is how the vocabulary size from a trained tokenizer gets injected.
    device, dtype:
        Where and in what precision to place the model. ``None`` leaves the
        model on CPU in float32.
    verify_budget:
        Warn when the built model's parameter count differs from the ``params``
        or ``param_budget.total`` declared by the template.

    Examples
    --------
    >>> model = build_model("supra2_1406240")
    >>> model.num_parameters()
    1406240
    """
    if isinstance(spec, (str, Path)):
        template: Mapping[str, Any] = load_template(spec).to_dict()
    else:
        template = dict(spec)

    if "family" not in template and "architecture" in template:
        # A saved model config: flat keys plus an `architecture` field.
        family = str(template["architecture"])
        flat = {k: v for k, v in template.items() if k != "architecture"}
    else:
        family, flat = template_to_model_config(template)

    if overrides:
        flat = merge_configs(flat, dict(overrides))

    model_cls = ARCHITECTURES.get(family)
    model = model_cls.from_config(flat)

    if verify_budget:
        declared = template.get("params") or Config(template).get("param_budget.total")
        if declared is not None:
            actual = model.num_parameters()
            if int(declared) != actual:
                logger.warning(
                    "template %s declares %s parameters but the built model has %s",
                    template.get("name", family),
                    f"{int(declared):,}",
                    f"{actual:,}",
                )

    if device is not None or dtype is not None:
        target_device = resolve_device(device) if device is not None else torch.device("cpu")
        target_dtype = resolve_dtype(dtype) if dtype is not None else torch.float32
        model = model.to(device=target_device, dtype=target_dtype)
    return model


def load_model(
    directory: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    strict: bool = True,
) -> BaseLanguageModel:
    """Load a model previously written by :meth:`BaseLanguageModel.save_pretrained`."""
    directory = Path(directory)
    config_path = directory / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"no config.json in {directory}")
    import json

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = build_model(config, device=device, dtype=dtype, verify_budget=False)
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=strict)
    if device is not None:
        model = model.to(resolve_device(device))
    return model


def describe_model(model: BaseLanguageModel) -> dict[str, Any]:
    """Summarise a model for logging, run metadata and model cards."""
    info: dict[str, Any] = {
        "architecture": model.architecture_name,
        "parameters": model.num_parameters(),
        "trainable_parameters": model.num_parameters(trainable_only=True),
        "vocab_size": model.vocab_size,
        "max_seq_len": model.max_seq_len,
        "breakdown": model.parameter_breakdown(),
        "config": dict(model.config),
    }
    active = getattr(model, "active_parameters", None)
    if callable(active):
        info["active_parameters"] = active()
    return info
