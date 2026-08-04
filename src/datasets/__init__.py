"""Corpus acquisition, tokenization and batching.

The data path has four steps, each of which is independently runnable::

    minimodel data pull      <name>   # remote corpus  -> data/raw/<name>.jsonl
    minimodel data tokenize  <name>   # JSONL          -> data/tokenized/<name>/
    minimodel data info      <name>   # inspect a tokenized corpus
    # loading happens inside the trainers, via PackedTextDataset & friends

Datasets are described declaratively in ``config/datasets.yaml``; see
:mod:`minimodel.datasets.registry` for the schema and ``docs/data.md`` for the
catalogue.
"""

from __future__ import annotations

from minimodel.datasets.builtin import builtin_records
from minimodel.datasets.loader import (
    JsonlPairDataset,
    MixtureDataset,
    PackedTextDataset,
    SupervisedDataset,
    build_dataloader,
    collate_batch,
    infinite_loader,
)
from minimodel.datasets.pull_datasets import iter_records, pull_dataset, pull_mixture
from minimodel.datasets.registry import (
    DatasetSpec,
    MixtureSpec,
    get_dataset,
    get_mixture,
    list_datasets,
    list_mixtures,
    resolve_mixture,
)
from minimodel.datasets.shards import (
    IGNORE_INDEX,
    ShardIndex,
    ShardWriter,
    TokenizedCorpus,
    choose_dtype,
)
from minimodel.datasets.soft_labels import (
    align_steps_to_tokenizer,
    entries_to_plain_texts,
    iter_plain_texts,
    load_soft_label_dataset,
    soft_kl_loss,
    write_plain_jsonl,
)
from minimodel.datasets.tokenize_datasets import (
    tokenize_chat_records,
    tokenize_jsonl,
    tokenize_preference_records,
    tokenize_registered,
    tokenize_text_records,
)

__all__ = [
    "IGNORE_INDEX",
    "DatasetSpec",
    "JsonlPairDataset",
    "MixtureDataset",
    "MixtureSpec",
    "PackedTextDataset",
    "ShardIndex",
    "ShardWriter",
    "SupervisedDataset",
    "TokenizedCorpus",
    "align_steps_to_tokenizer",
    "build_dataloader",
    "builtin_records",
    "choose_dtype",
    "collate_batch",
    "entries_to_plain_texts",
    "get_dataset",
    "get_mixture",
    "infinite_loader",
    "iter_plain_texts",
    "iter_records",
    "list_datasets",
    "list_mixtures",
    "load_soft_label_dataset",
    "pull_dataset",
    "pull_mixture",
    "resolve_mixture",
    "soft_kl_loss",
    "tokenize_chat_records",
    "tokenize_jsonl",
    "tokenize_preference_records",
    "tokenize_registered",
    "tokenize_text_records",
    "write_plain_jsonl",
]
