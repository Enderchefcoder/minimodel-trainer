"""Minimal helpers for optional single-node / multi-node DDP training.

Everything here degrades gracefully to a single process: when ``torchrun`` did
not set the usual environment variables, :func:`setup_distributed` returns a
world size of one and the rest of the helpers become no-ops. That means the
trainer can call them unconditionally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

__all__ = [
    "DistInfo",
    "all_reduce_mean",
    "barrier",
    "gather_object",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_main_process",
    "setup_distributed",
    "teardown_distributed",
]


@dataclass(frozen=True)
class DistInfo:
    """Resolved distributed topology for the current process."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None
    initialized: bool = False

    @property
    def is_main(self) -> bool:
        """Whether this process should own logging, checkpointing and IO."""
        return self.rank == 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view for run metadata."""
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "backend": self.backend,
            "initialized": self.initialized,
        }


def is_distributed() -> bool:
    """True when a process group is initialized and larger than one process."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    """Global rank, or 0 outside distributed runs."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    """Node-local rank, used to pick the CUDA device."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Number of participating processes (1 when not distributed)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    """Whether this process is rank 0."""
    return get_rank() == 0


def setup_distributed(backend: str | None = None, *, timeout_minutes: int = 30) -> DistInfo:
    """Initialise the default process group when launched under ``torchrun``.

    Returns a single-process :class:`DistInfo` when the launcher environment
    variables are absent, so callers never need to branch.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1 or not dist.is_available():
        return DistInfo(world_size=1)
    if dist.is_initialized():  # pragma: no cover - requires a live process group
        return DistInfo(
            rank=dist.get_rank(),
            local_rank=get_local_rank(),
            world_size=dist.get_world_size(),
            backend=dist.get_backend(),
            initialized=True,
        )

    # pragma: no cover below - exercised only under a real multi-process launch
    import datetime

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend, timeout=datetime.timedelta(minutes=timeout_minutes)
    )
    local_rank = get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistInfo(
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
        backend=backend,
        initialized=True,
    )


def teardown_distributed() -> None:
    """Destroy the process group if one exists."""
    if dist.is_available() and dist.is_initialized():  # pragma: no cover - needs a group
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronise all ranks (no-op in single-process runs)."""
    if is_distributed():  # pragma: no cover - needs a group
        dist.barrier()


def all_reduce_mean(value: float | torch.Tensor, device: torch.device | None = None) -> float:
    """Average a scalar across ranks; returns it unchanged when not distributed."""
    if not is_distributed():
        return float(value.item() if isinstance(value, torch.Tensor) else value)
    # pragma: no cover below - needs a live process group
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(float(value))
    tensor = tensor.detach().float().to(device or tensor.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / dist.get_world_size())


def gather_object(obj: Any) -> list[Any]:
    """Gather a picklable object from every rank onto every rank."""
    if not is_distributed():
        return [obj]
    output: list[Any] = [None] * dist.get_world_size()  # pragma: no cover - needs a group
    dist.all_gather_object(output, obj)  # pragma: no cover - needs a group
    return output  # pragma: no cover - needs a group
