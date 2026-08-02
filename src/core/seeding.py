"""Deterministic seeding and RNG state capture.

Resuming a run is only truly seamless if the data order and dropout masks pick
up where they left off, so :func:`get_rng_state` / :func:`set_rng_state` are
stored inside checkpoints alongside the model and optimizer state.
"""

from __future__ import annotations

import contextlib
import os
import random
from typing import Any

import numpy as np
import torch

__all__ = [
    "get_rng_state",
    "seed_everything",
    "seed_worker",
    "set_deterministic",
    "set_rng_state",
    "set_seed",
]


def set_seed(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch (CPU + all CUDA devices).

    Parameters
    ----------
    seed:
        Base seed. Values are taken modulo ``2**32`` where a 32-bit seed is
        required.
    deterministic:
        Also request deterministic cuDNN/cuBLAS kernels. This is slower, so it
        is opt-in.

    Returns
    -------
    int
        The seed that was applied, for logging.
    """
    seed = int(seed) % (2**32)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        set_deterministic(True)
    return seed


#: Alias matching the naming used by several other training frameworks.
seed_everything = set_seed


def set_deterministic(enabled: bool = True, *, warn_only: bool = True) -> None:
    """Toggle deterministic algorithms in PyTorch and cuDNN."""
    torch.backends.cudnn.deterministic = bool(enabled)
    torch.backends.cudnn.benchmark = not enabled
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with contextlib.suppress(RuntimeError, TypeError):  # older torch builds
        torch.use_deterministic_algorithms(bool(enabled), warn_only=warn_only)


def seed_worker(worker_id: int) -> None:
    """``DataLoader`` worker init function that gives each worker its own seed."""
    base = torch.initial_seed() % (2**32)
    worker_seed = (base + int(worker_id)) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_rng_state() -> dict[str, Any]:
    """Snapshot every RNG the toolkit touches."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any] | None) -> None:
    """Restore a snapshot produced by :func:`get_rng_state`.

    Missing or partially-missing snapshots are tolerated: an older checkpoint
    should still resume, just without bit-exact data ordering.
    """
    if not state:
        return
    if "python" in state:
        python_state = state["python"]
        # JSON round-trips turn the inner tuple into a list.
        if isinstance(python_state, list):
            python_state = (python_state[0], tuple(python_state[1]), python_state[2])
        random.setstate(python_state)
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        tensor = state["torch"]
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.uint8)
        torch.set_rng_state(tensor.cpu().to(torch.uint8))
    if "cuda" in state and torch.cuda.is_available():  # pragma: no cover - hardware
        with contextlib.suppress(RuntimeError, ValueError):
            torch.cuda.set_rng_state_all(state["cuda"])
