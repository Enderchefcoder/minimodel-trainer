"""Checkpoint saving, loading, resuming and retention.

A checkpoint here is a directory, not a single file::

    runs/my-run/checkpoints/step_001000/
      model.pt            model weights
      config.json         architecture config (so it reloads without the recipe)
      trainer.pt          optimizer, scheduler, scaler, step counter, RNG state
      metrics.json        the metrics recorded at this step

Separating weights from trainer state means a finished model can be shipped by
copying two small files, while a resume still restores the optimizer moments and
the exact data order.

Retention keeps the last ``keep_last`` checkpoints plus the best ``keep_best``
by a monitored metric, so a long run does not silently fill the disk.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from minimodel.core.io_utils import ensure_dir, human_bytes, read_json, write_json
from minimodel.core.logging_utils import get_logger
from minimodel.core.seeding import get_rng_state, set_rng_state

__all__ = ["Checkpoint", "CheckpointManager", "find_latest_checkpoint"]

logger = get_logger(__name__)

_STEP_PATTERN = re.compile(r"step_(\d+)$")


@dataclass
class Checkpoint:
    """A checkpoint directory and its metadata."""

    path: Path
    step: int
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def model_path(self) -> Path:
        """Path to the weights file."""
        return self.path / "model.pt"

    @property
    def trainer_path(self) -> Path:
        """Path to the optimizer/scheduler state file."""
        return self.path / "trainer.pt"

    def size_bytes(self) -> int:
        """Total on-disk size of the checkpoint."""
        return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())

    def __repr__(self) -> str:
        return f"Checkpoint(step={self.step}, path={self.path})"


def find_latest_checkpoint(directory: str | Path) -> Checkpoint | None:
    """Return the highest-step checkpoint in ``directory``, or ``None``."""
    directory = Path(directory)
    if not directory.exists():
        return None
    candidates: list[Checkpoint] = []
    for child in directory.iterdir():
        if not child.is_dir():
            continue
        match = _STEP_PATTERN.match(child.name)
        if not match or not (child / "model.pt").exists():
            continue
        metrics_path = child / "metrics.json"
        metrics = read_json(metrics_path) if metrics_path.exists() else {}
        candidates.append(Checkpoint(child, int(match.group(1)), metrics))
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.step)


class CheckpointManager:
    """Saves, restores and prunes checkpoints for one run.

    Parameters
    ----------
    directory:
        Where checkpoints are written.
    keep_last:
        Number of most-recent checkpoints to retain. ``0`` keeps all of them.
    keep_best:
        Number of best-scoring checkpoints to retain in addition to the last
        ones.
    monitor:
        Metric key used to rank checkpoints, e.g. ``"val_loss"``.
    mode:
        ``"min"`` (default) or ``"max"``.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        keep_last: int = 3,
        keep_best: int = 1,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.directory = ensure_dir(directory)
        self.keep_last = int(keep_last)
        self.keep_best = int(keep_best)
        self.monitor = monitor
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.mode = mode
        self.history: list[Checkpoint] = []
        self._rescan()

    def _rescan(self) -> None:
        """Rebuild the in-memory history from what is on disk."""
        found: list[Checkpoint] = []
        for child in sorted(self.directory.iterdir()) if self.directory.exists() else []:
            if not child.is_dir():
                continue
            match = _STEP_PATTERN.match(child.name)
            if not match:
                continue
            metrics_path = child / "metrics.json"
            metrics = read_json(metrics_path) if metrics_path.exists() else {}
            found.append(Checkpoint(child, int(match.group(1)), metrics))
        self.history = sorted(found, key=lambda c: c.step)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save(
        self,
        step: int,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        metrics: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        save_rng: bool = True,
    ) -> Checkpoint:
        """Write a checkpoint for ``step`` and prune old ones."""
        path = ensure_dir(self.directory / f"step_{step:06d}")

        module = model.module if hasattr(model, "module") else model
        torch.save(module.state_dict(), path / "model.pt")

        config = dict(getattr(module, "config", {}))
        config["architecture"] = getattr(module, "architecture_name", "unknown")
        write_json(path / "config.json", config)

        trainer_state: dict[str, Any] = {"step": int(step)}
        if optimizer is not None:
            trainer_state["optimizer"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            trainer_state["scheduler"] = scheduler.state_dict()
        if scaler is not None and hasattr(scaler, "state_dict"):
            trainer_state["scaler"] = scaler.state_dict()
        if save_rng:
            trainer_state["rng"] = get_rng_state()
        if extra:
            trainer_state["extra"] = dict(extra)
        torch.save(trainer_state, path / "trainer.pt")

        recorded = {"step": int(step), **dict((metrics or {}).items())}
        write_json(path / "metrics.json", recorded)

        checkpoint = Checkpoint(path, int(step), recorded)
        self.history = [c for c in self.history if c.step != step] + [checkpoint]
        self.history.sort(key=lambda c: c.step)
        logger.info("saved checkpoint %s (%s)", path.name, human_bytes(checkpoint.size_bytes()))
        self.prune()
        return checkpoint

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def latest(self) -> Checkpoint | None:
        """Most recent checkpoint, or ``None`` when the run is fresh."""
        return self.history[-1] if self.history else None

    def best(self) -> Checkpoint | None:
        """Best checkpoint by the monitored metric, or ``None``."""
        scored = [c for c in self.history if self.monitor in c.metrics]
        if not scored:
            return None
        picker = min if self.mode == "min" else max
        return picker(scored, key=lambda c: float(c.metrics[self.monitor]))

    def load(
        self,
        checkpoint: Checkpoint | str | Path | None = None,
        *,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Restore state from a checkpoint.

        Returns the trainer-state dictionary, whose ``step`` key tells the
        caller where to resume from.
        """
        if checkpoint is None:
            checkpoint = self.latest()
        if checkpoint is None:
            raise FileNotFoundError(f"no checkpoints found in {self.directory}")
        path = checkpoint.path if isinstance(checkpoint, Checkpoint) else Path(checkpoint)

        if model is not None:
            state = torch.load(path / "model.pt", map_location=map_location, weights_only=True)
            module = model.module if hasattr(model, "module") else model
            missing, unexpected = module.load_state_dict(state, strict=strict)
            if missing or unexpected:
                logger.warning(
                    "state dict mismatch: %d missing, %d unexpected", len(missing), len(unexpected)
                )

        trainer_path = path / "trainer.pt"
        if not trainer_path.exists():
            return {"step": 0}
        # `weights_only=False` is required: the payload holds RNG state and
        # optimizer objects, not just tensors. Only load checkpoints you trust.
        trainer_state = torch.load(trainer_path, map_location=map_location, weights_only=False)

        if optimizer is not None and "optimizer" in trainer_state:
            optimizer.load_state_dict(trainer_state["optimizer"])
        if (
            scheduler is not None
            and "scheduler" in trainer_state
            and hasattr(scheduler, "load_state_dict")
        ):
            scheduler.load_state_dict(trainer_state["scheduler"])
        if scaler is not None and "scaler" in trainer_state and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(trainer_state["scaler"])
        if restore_rng:
            set_rng_state(trainer_state.get("rng"))

        logger.info("resumed from %s at step %s", path.name, trainer_state.get("step", 0))
        return trainer_state

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------
    def prune(self) -> list[Path]:
        """Delete checkpoints outside the retention policy; return what was removed."""
        if self.keep_last <= 0:
            return []
        keep: set[Path] = {c.path for c in self.history[-self.keep_last :]}
        if self.keep_best > 0:
            scored = [c for c in self.history if self.monitor in c.metrics]
            reverse = self.mode == "max"
            ranked = sorted(scored, key=lambda c: float(c.metrics[self.monitor]), reverse=reverse)
            keep.update(c.path for c in ranked[: self.keep_best])

        removed: list[Path] = []
        for checkpoint in list(self.history):
            if checkpoint.path in keep:
                continue
            shutil.rmtree(checkpoint.path, ignore_errors=True)
            self.history.remove(checkpoint)
            removed.append(checkpoint.path)
        if removed:
            logger.debug("pruned %d checkpoint(s)", len(removed))
        return removed

    def export_model(self, destination: str | Path, checkpoint: Checkpoint | None = None) -> Path:
        """Copy just the weights and config of a checkpoint to ``destination``.

        This is what you publish: no optimizer moments, no RNG state.
        """
        source = checkpoint or self.best() or self.latest()
        if source is None:
            raise FileNotFoundError(f"no checkpoints found in {self.directory}")
        destination = ensure_dir(destination)
        shutil.copy2(source.model_path, destination / "model.pt")
        shutil.copy2(source.path / "config.json", destination / "config.json")
        metrics_path = source.path / "metrics.json"
        if metrics_path.exists():
            shutil.copy2(metrics_path, destination / "metrics.json")
        logger.info("exported %s to %s", source.path.name, destination)
        return destination

    def summary(self) -> list[dict[str, Any]]:
        """Table of retained checkpoints for ``minimodel ckpt list``."""
        return [
            {
                "step": c.step,
                "path": str(c.path),
                "size": human_bytes(c.size_bytes()),
                **{k: v for k, v in c.metrics.items() if k != "step"},
            }
            for c in self.history
        ]


def save_model_bundle(
    model: torch.nn.Module,
    destination: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a self-contained, shippable model directory.

    The bundle contains the weights, the architecture config, optionally the
    tokenizer, and a metadata file describing how the model was trained.
    """
    destination = ensure_dir(destination)
    module = model.module if hasattr(model, "module") else model
    torch.save(module.state_dict(), destination / "model.pt")
    config = dict(getattr(module, "config", {}))
    config["architecture"] = getattr(module, "architecture_name", "unknown")
    write_json(destination / "config.json", config)
    if tokenizer_path is not None:
        source = Path(tokenizer_path)
        if source.is_dir():
            source = source / "tokenizer.json"
        if source.exists():
            shutil.copy2(source, destination / "tokenizer.json")
    if metadata:
        write_json(destination / "training_metadata.json", dict(metadata))
    return destination


def load_checkpoint_metrics(directory: str | Path) -> list[dict[str, Any]]:
    """Read the metrics recorded by every checkpoint under ``directory``."""
    directory = Path(directory)
    rows: list[dict[str, Any]] = []
    for child in sorted(directory.glob("step_*")):
        metrics_path = child / "metrics.json"
        if metrics_path.exists():
            rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    return rows
