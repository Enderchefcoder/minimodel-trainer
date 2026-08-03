"""Shared pytest fixtures.

Fixtures are deliberately tiny: a 300-token vocabulary, a two-layer model and a
20K-token corpus. Every test in the suite should run in well under a second so
that the whole thing stays usable as an inner-loop check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from minimodel.datasets.builtin import builtin_records
from minimodel.datasets.tokenize_datasets import (
    tokenize_chat_records,
    tokenize_preference_records,
    tokenize_text_records,
)
from minimodel.tokenization.tokenize import BPETokenizer

#: Config for a model small enough to train inside a test.
TINY_MODEL = {
    "dim": 32,
    "n_layers": 2,
    "n_heads": 2,
    "head_dim": 16,
    "n_kv_heads": 1,
    "ffn_hidden": 64,
    "max_seq_len": 128,
    "window": 64,
}


@pytest.fixture(scope="session")
def texts() -> list[str]:
    """The bundled demo corpus, repeated enough for BPE to find pairs."""
    return [record["text"] for record in builtin_records("pretrain", repeat=4)]


@pytest.fixture(scope="session")
def tokenizer(texts: list[str]) -> BPETokenizer:
    """A small byte-level BPE tokenizer trained on the demo corpus."""
    return BPETokenizer.train(texts, vocab_size=400, min_frequency=2)


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory, tokenizer: BPETokenizer, texts: list[str]) -> Path:
    """A tokenized pretraining corpus."""
    path = tmp_path_factory.mktemp("corpus")
    tokenize_text_records(({"text": t} for t in texts), tokenizer, path)
    return path


@pytest.fixture(scope="session")
def sft_dir(tmp_path_factory, tokenizer: BPETokenizer) -> Path:
    """A tokenized supervised corpus with a loss mask."""
    path = tmp_path_factory.mktemp("sft")
    tokenize_chat_records(builtin_records("sft", repeat=8), tokenizer, path)
    return path


@pytest.fixture(scope="session")
def cot_dir(tmp_path_factory, tokenizer: BPETokenizer) -> Path:
    """A tokenized chain-of-thought corpus."""
    path = tmp_path_factory.mktemp("cot")
    tokenize_chat_records(builtin_records("cot", repeat=10), tokenizer, path)
    return path


@pytest.fixture(scope="session")
def pairs_path(tmp_path_factory, tokenizer: BPETokenizer) -> Path:
    """A preference-pair JSONL file for DPO."""
    path = tmp_path_factory.mktemp("prefs") / "pairs.jsonl"
    tokenize_preference_records(builtin_records("preference", repeat=6), tokenizer, path)
    return path


@pytest.fixture(scope="session")
def tasks_path(tmp_path_factory) -> Path:
    """A verifiable-task JSONL file for RLVR."""
    path = tmp_path_factory.mktemp("tasks") / "tasks.jsonl"
    path.write_text(
        "\n".join(json.dumps(task) for task in builtin_records("rlvr")), encoding="utf-8"
    )
    return path


@pytest.fixture(scope="session")
def sft_jsonl(tmp_path_factory) -> Path:
    """Raw prompt/answer records for SPIN."""
    path = tmp_path_factory.mktemp("spin") / "sft.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in builtin_records("sft")), encoding="utf-8"
    )
    return path


@pytest.fixture
def tiny_model(tokenizer: BPETokenizer):
    """A freshly initialised two-layer dense transformer."""
    from minimodel.architectures.builder import build_model

    return build_model(
        "dense_3m",
        overrides={**TINY_MODEL, "vocab_size": tokenizer.vocab_size},
        verify_budget=False,
    )


@pytest.fixture
def sprites():
    """A handful of synthetic 16x16 sprites."""
    from minimodel.vision.data.datasets import synthetic_sprites

    return synthetic_sprites(24, size=16, n_colors=8, seed=0)


@pytest.fixture(autouse=True)
def _deterministic():
    """Seed every test so failures reproduce."""
    torch.manual_seed(1234)
