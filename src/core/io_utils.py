"""Filesystem and formatting helpers shared across the toolkit.

The JSON helpers are deliberately tolerant: metric dictionaries frequently
contain ``torch.Tensor`` scalars or ``numpy`` floats, and losing a whole run log
because one value was not JSON-serialisable is a bad trade.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "append_jsonl",
    "atomic_write_bytes",
    "atomic_write_text",
    "ensure_dir",
    "human_bytes",
    "human_count",
    "human_duration",
    "jsonable",
    "read_json",
    "read_jsonl",
    "write_json",
    "write_jsonl",
]


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write ``text`` to ``path`` via a temporary file plus rename.

    An interrupted run then leaves either the old file or the new one, never a
    half-written checkpoint index or metrics file.
    """
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Byte-oriented counterpart of :func:`atomic_write_text`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return p


def jsonable(value: Any) -> Any:
    """Best-effort conversion of ``value`` into JSON-serialisable data.

    Handles tensors and numpy scalars/arrays without importing either library
    eagerly, and falls back to ``str`` for anything else.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    tolist = getattr(value, "tolist", None)
    if callable(item):
        try:
            return jsonable(item())
        except (ValueError, TypeError, RuntimeError):
            pass
    if callable(tolist):
        try:
            return jsonable(tolist())
        except (ValueError, TypeError, RuntimeError):
            pass
    return str(value)


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    """Atomically write ``data`` as pretty-printed JSON."""
    return atomic_write_text(path, json.dumps(jsonable(data), indent=indent) + "\n")


def read_json(path: str | Path) -> Any:
    """Read a JSON document."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Atomically write an iterable of mappings as JSON Lines."""
    payload = "".join(json.dumps(jsonable(row)) + "\n" for row in rows)
    return atomic_write_text(path, payload)


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> Path:
    """Append a single record to a JSON Lines file, creating it if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(row)) + "\n")
    return p


def read_jsonl(path: str | Path, *, skip_bad_lines: bool = False) -> Iterator[dict[str, Any]]:
    """Yield records from a JSON Lines file.

    Blank lines are always skipped. Malformed lines raise unless
    ``skip_bad_lines`` is set, which is handy for partially-flushed logs from a
    killed run.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                if skip_bad_lines:
                    continue
                raise ValueError(f"{path}:{lineno} is not valid JSON") from None


def human_bytes(num_bytes: float) -> str:
    """Format a byte count with binary units.

    >>> human_bytes(1536)
    '1.5 KiB'
    """
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(value) < 1024.0 or unit == "PiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PiB"  # pragma: no cover - unreachable, loop always returns


def human_count(count: float) -> str:
    """Format a large count compactly.

    >>> human_count(1_406_240)
    '1.41M'
    """
    value = float(count)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:.2f}{suffix}"
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}"


def human_duration(seconds: float) -> str:
    """Format a duration as ``1d 2h 3m`` / ``4m 5s`` / ``6.7s``.

    >>> human_duration(3661)
    '1h 1m 1s'
    """
    if seconds != seconds or seconds in (float("inf"), float("-inf")):  # NaN / inf
        return "unknown"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = round(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)
