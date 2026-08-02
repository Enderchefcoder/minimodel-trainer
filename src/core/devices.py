"""Device and dtype resolution.

Recipes say ``device: auto`` and ``dtype: auto``; this module turns that into a
concrete :class:`torch.device` and :class:`torch.dtype` for whatever hardware is
actually present, and provides the matching autocast context manager.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch

__all__ = [
    "autocast_context",
    "count_parameters",
    "describe_device",
    "device_memory_stats",
    "resolve_device",
    "resolve_dtype",
    "supports_bf16",
]

_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "f32": torch.float32,
    "tf32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "f16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "fp64": torch.float64,
}


def resolve_device(spec: str | torch.device | None = "auto") -> torch.device:
    """Return a concrete device for ``spec``.

    ``"auto"`` (or ``None``) prefers CUDA, then Apple MPS, then CPU.

    >>> resolve_device("cpu")
    device(type='cpu')
    """
    if isinstance(spec, torch.device):
        return spec
    text = (spec or "auto").strip().lower()
    if text in {"auto", "", "default"}:
        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():  # pragma: no cover - hardware dependent
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(text)


def supports_bf16(device: torch.device | None = None) -> bool:
    """Whether bfloat16 autocast is usable on ``device``."""
    device = device or resolve_device("auto")
    if device.type == "cuda":  # pragma: no cover - hardware dependent
        try:
            return bool(torch.cuda.is_bf16_supported())
        except (RuntimeError, AttributeError):
            return False
    if device.type == "cpu":
        # CPU bf16 autocast works everywhere on modern PyTorch, it is just slow
        # without AMX/AVX512-BF16. It is still numerically valid.
        return True
    return False  # pragma: no cover - MPS has no bf16 autocast


def resolve_dtype(
    spec: str | torch.dtype | None = "auto", device: torch.device | None = None
) -> torch.dtype:
    """Return the compute dtype for ``spec`` on ``device``.

    ``"auto"`` selects bfloat16 on CUDA hardware that supports it, otherwise
    float32 (fp16 without a GradScaler is a common source of NaNs, so it is
    never chosen automatically).
    """
    if isinstance(spec, torch.dtype):
        return spec
    text = (spec or "auto").strip().lower()
    if text in {"auto", "", "default"}:
        device = device or resolve_device("auto")
        if device.type == "cuda" and supports_bf16(device):  # pragma: no cover - hardware
            return torch.bfloat16
        return torch.float32
    if text not in _DTYPE_ALIASES:
        options = ", ".join(sorted(_DTYPE_ALIASES))
        raise ValueError(f"unknown dtype {spec!r}; available: {options}")
    return _DTYPE_ALIASES[text]


@contextlib.contextmanager
def autocast_context(
    device: torch.device, dtype: torch.dtype, *, enabled: bool = True
) -> Iterator[None]:
    """Autocast for ``device``/``dtype``, or a no-op when it does not apply.

    Autocast is skipped for float32 and on devices without autocast support so
    that callers can always wrap their forward pass in this context.
    """
    use_amp = enabled and dtype in (torch.float16, torch.bfloat16) and device.type in ("cuda", "cpu")
    if not use_amp:
        yield
        return
    with torch.autocast(device_type=device.type, dtype=dtype):
        yield


def describe_device(device: torch.device | None = None) -> dict[str, Any]:
    """Return a JSON-friendly description of the compute device.

    Used by run metadata and by the model-card generator to fill in the
    "Hardware & Infrastructure" section automatically.
    """
    device = device or resolve_device("auto")
    info: dict[str, Any] = {
        "type": device.type,
        "index": device.index,
        "torch_version": torch.__version__,
    }
    if device.type == "cuda" and torch.cuda.is_available():  # pragma: no cover - hardware
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "capability": f"{props.major}.{props.minor}",
                "device_count": torch.cuda.device_count(),
                "bf16": supports_bf16(device),
            }
        )
    elif device.type == "cpu":
        import platform

        info.update(
            {
                "name": platform.processor() or platform.machine(),
                "threads": torch.get_num_threads(),
                "platform": platform.platform(),
            }
        )
    else:  # pragma: no cover - MPS / XPU
        info["name"] = device.type
    return info


def device_memory_stats(device: torch.device | None = None) -> dict[str, float]:
    """Return allocated/reserved/peak memory in MiB (zeros on non-CUDA)."""
    device = device or resolve_device("auto")
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"allocated_mib": 0.0, "reserved_mib": 0.0, "max_allocated_mib": 0.0}
    scale = 1024.0**2  # pragma: no cover - hardware dependent
    return {  # pragma: no cover - hardware dependent
        "allocated_mib": torch.cuda.memory_allocated(device) / scale,
        "reserved_mib": torch.cuda.memory_reserved(device) / scale,
        "max_allocated_mib": torch.cuda.max_memory_allocated(device) / scale,
    }


def count_parameters(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    """Count parameters, de-duplicating tied weights by storage identity."""
    seen: set[int] = set()
    total = 0
    for param in module.parameters():
        if trainable_only and not param.requires_grad:
            continue
        key = id(param)
        if key in seen:
            continue
        seen.add(key)
        total += param.numel()
    return total
