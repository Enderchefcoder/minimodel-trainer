"""The on-disk format for tokenized corpora.

A tokenized corpus is a directory containing one or more flat binary shards of
token ids plus an ``index.json`` describing them::

    data/fineweb-edu/
      index.json
      shard_0000.bin
      shard_0001.bin
      labels_0000.bin      # only for supervised corpora

Why a flat memmap rather than a row-oriented format: pretraining reads random
fixed-length windows across document boundaries, so any per-record framing is
pure overhead. A memmapped ``uint16`` array gives zero-copy random access, costs
2 bytes per token, and lets the OS page cache do the buffering. A 1B-token
corpus is 2 GiB on disk and needs no RAM to iterate.

Supervised corpora additionally store a parallel ``labels`` array holding the
loss mask (``-100`` where the position is not supervised), which keeps the
prompt-masking decision with the data instead of re-deriving it at train time.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from minimodel.core.io_utils import ensure_dir, human_count, write_json
from minimodel.core.logging_utils import get_logger

__all__ = [
    "IGNORE_INDEX",
    "ShardIndex",
    "ShardWriter",
    "TokenizedCorpus",
    "choose_dtype",
]

logger = get_logger(__name__)

#: Label value that the loss ignores.
IGNORE_INDEX = -100


def choose_dtype(vocab_size: int) -> str:
    """Smallest unsigned integer type that can hold ``vocab_size`` ids.

    >>> choose_dtype(4096)
    'uint16'
    >>> choose_dtype(100000)
    'uint32'
    """
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if vocab_size <= 2**16:
        return "uint16"
    return "uint32"


@dataclass
class ShardIndex:
    """Metadata describing a tokenized corpus directory."""

    dtype: str = "uint16"
    vocab_size: int = 0
    n_tokens: int = 0
    n_documents: int = 0
    shards: list[dict[str, Any]] = field(default_factory=list)
    supervised: bool = False
    tokenizer: str | None = None
    source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "format": "minimodel-token-shards/1",
            "dtype": self.dtype,
            "vocab_size": self.vocab_size,
            "n_tokens": self.n_tokens,
            "n_documents": self.n_documents,
            "supervised": self.supervised,
            "tokenizer": self.tokenizer,
            "source": self.source,
            "shards": self.shards,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShardIndex:
        """Parse an ``index.json`` payload."""
        known = {
            "format",
            "dtype",
            "vocab_size",
            "n_tokens",
            "n_documents",
            "supervised",
            "tokenizer",
            "source",
            "shards",
        }
        return cls(
            dtype=data.get("dtype", "uint16"),
            vocab_size=int(data.get("vocab_size", 0)),
            n_tokens=int(data.get("n_tokens", 0)),
            n_documents=int(data.get("n_documents", 0)),
            shards=list(data.get("shards", [])),
            supervised=bool(data.get("supervised", False)),
            tokenizer=data.get("tokenizer"),
            source=data.get("source"),
            extra={k: v for k, v in data.items() if k not in known},
        )


class ShardWriter:
    """Streaming writer that splits a token stream into fixed-size shards.

    Parameters
    ----------
    directory:
        Output directory; created if missing.
    dtype:
        ``uint16`` or ``uint32``.
    shard_tokens:
        Maximum tokens per shard. Smaller shards parallelise better across
        dataloader workers; larger shards mean fewer file handles.
    supervised:
        Also write a parallel label array.

    Examples
    --------
    >>> import tempfile
    >>> with ShardWriter(tempfile.mkdtemp(), dtype="uint16") as writer:
    ...     writer.write_document([1, 2, 3])
    ...     index = writer.index
    >>> index.n_tokens
    3
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        dtype: str = "uint16",
        shard_tokens: int = 100_000_000,
        supervised: bool = False,
        vocab_size: int = 0,
        tokenizer: str | None = None,
        source: str | None = None,
    ):
        self.directory = ensure_dir(directory)
        self.dtype = dtype
        self.np_dtype = np.dtype(dtype)
        self.shard_tokens = int(shard_tokens)
        self.supervised = bool(supervised)
        self.index = ShardIndex(
            dtype=dtype,
            vocab_size=vocab_size,
            supervised=self.supervised,
            tokenizer=tokenizer,
            source=source,
        )
        self._buffer: list[np.ndarray] = []
        self._label_buffer: list[np.ndarray] = []
        self._buffered = 0
        self._shard_id = 0

    def write_document(
        self, tokens: Sequence[int], labels: Sequence[int] | None = None
    ) -> None:
        """Append one document (and its labels for supervised corpora)."""
        if len(tokens) == 0:
            return
        array = np.asarray(tokens, dtype=self.np_dtype)
        self._buffer.append(array)
        if self.supervised:
            if labels is None:
                labels = list(tokens)
            if len(labels) != len(tokens):
                raise ValueError(
                    f"labels length {len(labels)} does not match tokens length {len(tokens)}"
                )
            # int32 keeps room for the negative IGNORE_INDEX sentinel.
            self._label_buffer.append(np.asarray(labels, dtype=np.int32))
        self._buffered += len(array)
        self.index.n_documents += 1
        if self._buffered >= self.shard_tokens:
            self.flush()

    def write_documents(
        self, documents: Iterable[Sequence[int]], labels: Iterable[Sequence[int]] | None = None
    ) -> None:
        """Append many documents."""
        if labels is None:
            for doc in documents:
                self.write_document(doc)
        else:
            for doc, label in zip(documents, labels, strict=True):
                self.write_document(doc, label)

    def flush(self) -> None:
        """Write the buffered tokens out as a shard."""
        if not self._buffer:
            return
        tokens = np.concatenate(self._buffer)
        name = f"shard_{self._shard_id:04d}.bin"
        tokens.tofile(self.directory / name)
        entry: dict[str, Any] = {"path": name, "n_tokens": int(tokens.size)}
        if self.supervised:
            label_name = f"labels_{self._shard_id:04d}.bin"
            np.concatenate(self._label_buffer).tofile(self.directory / label_name)
            entry["labels_path"] = label_name
        self.index.shards.append(entry)
        self.index.n_tokens += int(tokens.size)
        self._buffer.clear()
        self._label_buffer.clear()
        self._buffered = 0
        self._shard_id += 1

    def close(self) -> ShardIndex:
        """Flush and write ``index.json``."""
        self.flush()
        write_json(self.directory / "index.json", self.index.to_dict())
        logger.info(
            "wrote %s tokens across %d shard(s) to %s",
            human_count(self.index.n_tokens),
            len(self.index.shards),
            self.directory,
        )
        return self.index

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class TokenizedCorpus:
    """Read-only memmapped view over a tokenized corpus directory.

    The shards are presented as one logical array, so callers can index into
    ``[0, n_tokens)`` without caring where the shard boundaries fall.
    """

    def __init__(self, directory: str | Path, *, mmap: bool = True):
        self.directory = Path(directory)
        index_path = self.directory / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"{self.directory} is not a tokenized corpus (no index.json). "
                "Run `minimodel data tokenize` first."
            )
        self.index = ShardIndex.from_dict(json.loads(index_path.read_text(encoding="utf-8")))
        self.dtype = np.dtype(self.index.dtype)
        self._mmap_mode: str | None = "r" if mmap else None
        self._arrays: list[np.ndarray] = []
        self._label_arrays: list[np.ndarray] = []
        self._offsets: list[int] = []

        offset = 0
        for shard in self.index.shards:
            path = self.directory / shard["path"]
            array = np.memmap(path, dtype=self.dtype, mode="r") if mmap else np.fromfile(path, dtype=self.dtype)
            self._arrays.append(array)
            self._offsets.append(offset)
            offset += int(array.size)
            if self.index.supervised and "labels_path" in shard:
                label_path = self.directory / shard["labels_path"]
                labels = (
                    np.memmap(label_path, dtype=np.int32, mode="r")
                    if mmap
                    else np.fromfile(label_path, dtype=np.int32)
                )
                self._label_arrays.append(labels)
        self.n_tokens = offset

    def __len__(self) -> int:
        return self.n_tokens

    def __repr__(self) -> str:
        return (
            f"TokenizedCorpus({self.directory.name!r}, n_tokens={self.n_tokens:,}, "
            f"shards={len(self._arrays)}, supervised={self.index.supervised})"
        )

    @property
    def supervised(self) -> bool:
        """Whether a parallel label array is present."""
        return bool(self._label_arrays)

    def _locate(self, start: int) -> tuple[int, int]:
        """Return ``(shard_index, offset_within_shard)`` for a global position."""
        low, high = 0, len(self._offsets) - 1
        while low < high:
            mid = (low + high + 1) // 2
            if self._offsets[mid] <= start:
                low = mid
            else:
                high = mid - 1
        return low, start - self._offsets[low]

    def read(self, start: int, length: int, *, labels: bool = False) -> np.ndarray:
        """Read ``length`` tokens starting at global position ``start``.

        Reads that straddle a shard boundary are stitched together, so shard
        layout never affects the data a model sees.
        """
        if start < 0 or length < 0:
            raise ValueError("start and length must be non-negative")
        if start + length > self.n_tokens:
            raise IndexError(
                f"read of {length} tokens at {start} exceeds corpus length {self.n_tokens}"
            )
        source = self._label_arrays if labels else self._arrays
        if labels and not self._label_arrays:
            raise ValueError(f"{self.directory} has no label array")

        shard_idx, offset = self._locate(start)
        pieces: list[np.ndarray] = []
        remaining = length
        while remaining > 0:
            array = source[shard_idx]
            take = min(remaining, int(array.size) - offset)
            pieces.append(np.asarray(array[offset : offset + take]))
            remaining -= take
            shard_idx += 1
            offset = 0
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

    def iter_windows(self, seq_len: int, *, stride: int | None = None) -> Iterator[np.ndarray]:
        """Yield consecutive windows of ``seq_len`` tokens."""
        stride = stride or seq_len
        position = 0
        while position + seq_len <= self.n_tokens:
            yield self.read(position, seq_len)
            position += stride

    def stats(self) -> dict[str, Any]:
        """Summary used by ``minimodel data info`` and model cards."""
        return {
            "directory": str(self.directory),
            "n_tokens": self.n_tokens,
            "n_documents": self.index.n_documents,
            "n_shards": len(self._arrays),
            "dtype": str(self.dtype),
            "vocab_size": self.index.vocab_size,
            "supervised": self.supervised,
            "tokenizer": self.index.tokenizer,
            "source": self.index.source,
            "size_bytes": self.n_tokens * self.dtype.itemsize,
        }
