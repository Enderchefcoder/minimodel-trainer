"""Datasets and dataloaders that feed the trainers.

Three dataset classes cover every stage:

:class:`PackedTextDataset`
    Random fixed-length windows over a memmapped token stream. Used for
    pretraining. Windows deliberately cross document boundaries: the model needs
    to learn that documents end, and the end-of-text token teaches it.
:class:`SupervisedDataset`
    Windows over a corpus that also has a label array, so prompt tokens can be
    excluded from the loss. Used for SFT and CoT.
:class:`MixtureDataset`
    Samples from several datasets according to fixed weights, which is how a
    blend like "70% web, 30% synthetic" is realised without physically
    interleaving the files.

All three return ``(input_ids, labels)`` tensors of the same shape, so the
trainer never branches on which one it was given.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from minimodel.core.logging_utils import get_logger
from minimodel.core.seeding import seed_worker
from minimodel.datasets.shards import IGNORE_INDEX, TokenizedCorpus

__all__ = [
    "MixtureDataset",
    "PackedTextDataset",
    "SupervisedDataset",
    "build_dataloader",
    "collate_batch",
    "infinite_loader",
]

logger = get_logger(__name__)


def collate_batch(items: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack ``(input_ids, labels)`` pairs into a batch dictionary."""
    inputs = torch.stack([item[0] for item in items])
    labels = torch.stack([item[1] for item in items])
    return {"input_ids": inputs, "labels": labels}


class PackedTextDataset(Dataset):
    """Fixed-length windows over a tokenized pretraining corpus.

    Parameters
    ----------
    corpus:
        A :class:`~minimodel.datasets.shards.TokenizedCorpus` or a path to one.
    seq_len:
        Window length. The model sees ``seq_len`` inputs and predicts the same
        number of next tokens, so ``seq_len + 1`` raw tokens are read per item.
    stride:
        Distance between successive window starts in sequential mode. Defaults
        to ``seq_len`` (no overlap).
    shuffle:
        Sample window starts uniformly at random instead of walking the corpus
        in order. Random sampling is the right default for pretraining: it
        decorrelates consecutive batches without needing a shuffle buffer.
    """

    def __init__(
        self,
        corpus: TokenizedCorpus | str | Path,
        seq_len: int,
        *,
        stride: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        length: int | None = None,
    ):
        self.corpus = corpus if isinstance(corpus, TokenizedCorpus) else TokenizedCorpus(corpus)
        self.seq_len = int(seq_len)
        self.stride = int(stride or seq_len)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        if self.corpus.n_tokens < self.seq_len + 1:
            raise ValueError(
                f"corpus has {self.corpus.n_tokens} tokens, which is fewer than "
                f"seq_len + 1 = {self.seq_len + 1}"
            )
        self.max_start = self.corpus.n_tokens - self.seq_len - 1
        if length is not None:
            self._length = int(length)
        elif self.shuffle:
            self._length = max(1, self.corpus.n_tokens // self.seq_len)
        else:
            self._length = max(1, self.max_start // self.stride + 1)

    def __len__(self) -> int:
        return self._length

    def __repr__(self) -> str:
        return (
            f"PackedTextDataset(tokens={self.corpus.n_tokens:,}, seq_len={self.seq_len}, "
            f"windows={len(self)})"
        )

    def _start_for(self, index: int) -> int:
        if not self.shuffle:
            return min(index * self.stride, self.max_start)
        # Deterministic per-index randomness: the same index always yields the
        # same window, which keeps epochs reproducible across workers and
        # across a resume.
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        return int(rng.integers(0, self.max_start + 1))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(input_ids, labels)`` for one window."""
        start = self._start_for(int(index))
        window = self.corpus.read(start, self.seq_len + 1)
        tokens = torch.from_numpy(np.asarray(window, dtype=np.int64))
        return tokens[:-1], tokens[1:]

    def token_count(self) -> int:
        """Total tokens in the underlying corpus."""
        return self.corpus.n_tokens


class SupervisedDataset(Dataset):
    """Windows over a corpus with a parallel label mask (SFT / CoT).

    Positions whose label is :data:`IGNORE_INDEX` do not contribute to the loss.
    Windows are aligned so the label at position ``t`` supervises the prediction
    made from input ``t``, i.e. labels are already shifted.
    """

    def __init__(
        self,
        corpus: TokenizedCorpus | str | Path,
        seq_len: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        length: int | None = None,
    ):
        self.corpus = corpus if isinstance(corpus, TokenizedCorpus) else TokenizedCorpus(corpus)
        if not self.corpus.supervised:
            raise ValueError(
                f"{self.corpus.directory} has no label array; tokenize it with "
                "format='chat' to use SupervisedDataset"
            )
        self.seq_len = int(seq_len)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        if self.corpus.n_tokens < self.seq_len + 1:
            raise ValueError(
                f"corpus has {self.corpus.n_tokens} tokens, fewer than seq_len + 1"
            )
        self.max_start = self.corpus.n_tokens - self.seq_len - 1
        self._length = int(length) if length is not None else max(
            1, self.corpus.n_tokens // self.seq_len
        )

    def __len__(self) -> int:
        return self._length

    def __repr__(self) -> str:
        return (
            f"SupervisedDataset(tokens={self.corpus.n_tokens:,}, seq_len={self.seq_len}, "
            f"windows={len(self)})"
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(input_ids, labels)`` with prompt positions masked out."""
        if self.shuffle:
            rng = np.random.default_rng(self.seed * 1_000_003 + int(index))
            start = int(rng.integers(0, self.max_start + 1))
        else:
            start = min(int(index) * self.seq_len, self.max_start)
        tokens = torch.from_numpy(
            np.asarray(self.corpus.read(start, self.seq_len + 1), dtype=np.int64)
        )
        labels = torch.from_numpy(
            np.asarray(self.corpus.read(start, self.seq_len + 1, labels=True), dtype=np.int64)
        )
        return tokens[:-1], labels[1:]


class MixtureDataset(Dataset):
    """Weighted blend of several datasets.

    The weight of a component is the probability that a given item comes from
    it, so a 70/30 blend produces roughly 70% of its tokens from the first
    dataset regardless of how large the underlying corpora are.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        weights: Sequence[float] | None = None,
        *,
        length: int | None = None,
        seed: int = 0,
        names: Sequence[str] | None = None,
    ):
        if not datasets:
            raise ValueError("MixtureDataset needs at least one component")
        self.datasets = list(datasets)
        raw_weights = list(weights) if weights is not None else [1.0] * len(self.datasets)
        if len(raw_weights) != len(self.datasets):
            raise ValueError("weights and datasets must have the same length")
        total = float(sum(raw_weights))
        if total <= 0:
            raise ValueError("mixture weights must sum to a positive number")
        self.weights = [w / total for w in raw_weights]
        self.names = list(names) if names else [f"component_{i}" for i in range(len(self.datasets))]
        self.seed = int(seed)
        self._cumulative = np.cumsum(self.weights)
        self._length = int(length) if length is not None else sum(len(d) for d in self.datasets)

    def __len__(self) -> int:
        return self._length

    def __repr__(self) -> str:
        parts = ", ".join(f"{n}={w:.2f}" for n, w in zip(self.names, self.weights, strict=True))
        return f"MixtureDataset({parts}, length={len(self)})"

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick a component by weight, then an item from it."""
        rng = np.random.default_rng(self.seed * 7_919 + int(index))
        which = int(np.searchsorted(self._cumulative, rng.random(), side="right"))
        which = min(which, len(self.datasets) - 1)
        component = self.datasets[which]
        inner = int(rng.integers(0, len(component)))
        return component[inner]

    def component_stats(self) -> list[dict[str, Any]]:
        """Per-component name, weight and size, for logging."""
        return [
            {"name": name, "weight": weight, "items": len(dataset)}
            for name, weight, dataset in zip(
                self.names, self.weights, self.datasets, strict=True
            )
        ]


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 0,
    drop_last: bool = True,
    pin_memory: bool = False,
) -> DataLoader:
    """Wrap a dataset in a ``DataLoader`` with deterministic worker seeding.

    ``shuffle`` defaults to false because :class:`PackedTextDataset` and
    :class:`MixtureDataset` already randomise internally; turning on both just
    costs a permutation.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        drop_last=drop_last,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def infinite_loader(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    """Cycle a dataloader forever.

    Training is driven by a step count rather than by epochs, so the loop never
    needs to know how long an epoch is.
    """
    while True:
        for batch in loader:
            yield batch


class JsonlPairDataset(IterableDataset):
    """Streams preference pairs written by ``tokenize_preference_records``.

    Kept iterable rather than indexable because DPO batches are small and pair
    files are typically read once per epoch.
    """

    def __init__(self, path: str | Path, *, max_length: int = 1024, repeat: bool = True):
        self.path = Path(path)
        self.max_length = int(max_length)
        self.repeat = bool(repeat)
        if not self.path.exists():
            raise FileNotFoundError(f"preference file not found: {self.path}")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield one padded chosen/rejected pair at a time."""
        import json

        while True:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    yield {
                        key: torch.tensor(value[: self.max_length], dtype=torch.long)
                        for key, value in row.items()
                    }
            if not self.repeat:
                return

    @staticmethod
    def collate(items: Sequence[dict[str, torch.Tensor]], pad_id: int = 0) -> dict[str, torch.Tensor]:
        """Right-pad a list of pairs into batched tensors."""
        keys = ("chosen_ids", "chosen_labels", "rejected_ids", "rejected_labels")
        max_len = max(int(item[key].numel()) for item in items for key in keys)
        batch: dict[str, torch.Tensor] = {}
        for key in keys:
            pad_value = IGNORE_INDEX if key.endswith("labels") else pad_id
            rows = []
            for item in items:
                tensor = item[key]
                padding = torch.full((max_len - tensor.numel(),), pad_value, dtype=torch.long)
                rows.append(torch.cat([tensor, padding]))
            batch[key] = torch.stack(rows)
        return batch
