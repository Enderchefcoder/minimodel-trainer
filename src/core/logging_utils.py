"""Console and structured logging.

Two logging surfaces are used throughout the toolkit:

* :func:`get_logger` returns a normal :mod:`logging` logger whose records are
  suppressed on non-zero ranks by default, so distributed runs stay readable.
* :class:`JsonlLogger` appends one JSON object per step to ``metrics.jsonl``,
  which is what the plotting, comparison and model-card tools read back.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from minimodel.core.io_utils import append_jsonl, ensure_dir, jsonable

__all__ = [
    "JsonlLogger",
    "MetricFormatter",
    "get_logger",
    "setup_logging",
]

_ROOT_LOGGER_NAME = "minimodel"
_CONFIGURED = False


class _RankFilter(logging.Filter):
    """Drop records from non-zero ranks unless they are warnings or worse."""

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        if self.rank == 0:
            return True
        return record.levelno >= logging.WARNING


class _ConsoleFormatter(logging.Formatter):
    """Compact ``HH:MM:SS | LEVEL | name | message`` console format."""

    _LEVEL_ABBREV = {
        logging.DEBUG: "DBG",
        logging.INFO: "INF",
        logging.WARNING: "WRN",
        logging.ERROR: "ERR",
        logging.CRITICAL: "CRT",
    }

    def format(self, record: logging.LogRecord) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = self._LEVEL_ABBREV.get(record.levelno, record.levelname[:3])
        name = record.name
        if name.startswith(_ROOT_LOGGER_NAME + "."):
            name = name[len(_ROOT_LOGGER_NAME) + 1 :]
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        return f"{stamp} | {level} | {name} | {message}"


def setup_logging(
    level: int | str = logging.INFO,
    *,
    rank: int = 0,
    log_file: str | Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``minimodel`` logger hierarchy exactly once.

    Parameters
    ----------
    level:
        Level name or numeric level for console output.
    rank:
        Process rank; ranks other than 0 only emit warnings and errors.
    log_file:
        Optional path that receives a full-detail copy of every record.
    force:
        Re-configure even if logging was already set up (used by tests).
    """
    global _CONFIGURED
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if _CONFIGURED and not force:
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.INFO

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(_ConsoleFormatter())
    console.addFilter(_RankFilter(rank))
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        ensure_dir(path.parent)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        file_handler.addFilter(_RankFilter(rank))
        logger.addHandler(file_handler)

    _CONFIGURED = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger, configuring defaults on first use."""
    if not _CONFIGURED:
        setup_logging()
    if not name or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    # Module paths like "minimodel.training.trainer" arrive with the prefix; bare
    # names such as "trainer" get it added.
    short = name.split(".")[-1] if name.startswith("minimodel") else name
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{short}")


class MetricFormatter:
    """Render metric dictionaries as a stable, human-readable one-liner.

    >>> MetricFormatter().format({"step": 10, "loss": 3.14159, "lr": 0.0003})
    'step 10 | loss 3.1416 | lr 3.00e-04'
    """

    #: Keys rendered first, in this order, when present.
    PRIORITY = ("step", "epoch", "loss", "lr")

    def __init__(self, float_digits: int = 4):
        self.float_digits = float_digits

    def _format_value(self, key: str, value: Any) -> str:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if value != value:
                return "nan"
            if key == "lr" or (value != 0.0 and abs(value) < 1e-3):
                return f"{value:.2e}"
            return f"{value:.{self.float_digits}f}"
        return str(value)

    def format(self, metrics: Mapping[str, Any]) -> str:
        """Format ``metrics`` into ``key value | key value`` text."""
        flat = {k: jsonable(v) for k, v in metrics.items()}
        ordered: list[str] = []
        for key in self.PRIORITY:
            if key in flat:
                ordered.append(f"{key} {self._format_value(key, flat.pop(key))}")
        for key in sorted(flat):
            ordered.append(f"{key} {self._format_value(key, flat[key])}")
        return " | ".join(ordered)


class JsonlLogger:
    """Append-only JSON Lines metric sink.

    Every record is stamped with wall-clock time and the elapsed seconds since
    the logger was created, so downstream tools can compute throughput without
    the trainer having to log it explicitly.
    """

    def __init__(self, path: str | Path, *, enabled: bool = True, flush_every: int = 1):
        self.path = Path(path)
        self.enabled = enabled
        self.flush_every = max(1, int(flush_every))
        self._buffer: list[dict[str, Any]] = []
        self._start = time.time()
        if self.enabled:
            ensure_dir(self.path.parent)

    def log(self, metrics: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
        """Record one row and return the row that was (or would be) written."""
        row: dict[str, Any] = {
            "wall_time": round(time.time(), 3),
            "elapsed_s": round(time.time() - self._start, 3),
        }
        row.update({str(k): jsonable(v) for k, v in metrics.items()})
        row.update({str(k): jsonable(v) for k, v in extra.items()})
        if not self.enabled:
            return row
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every:
            self.flush()
        return row

    def flush(self) -> None:
        """Write any buffered rows to disk."""
        if not self.enabled or not self._buffer:
            self._buffer.clear()
            return
        for row in self._buffer:
            append_jsonl(self.path, row)
        self._buffer.clear()

    def close(self) -> None:
        """Flush and stop accepting new rows."""
        self.flush()
        self.enabled = False

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
