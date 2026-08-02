"""Tests for the dataset registry, shard format, tokenization and loaders."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from minimodel.core.config import ConfigError
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
    get_dataset,
    get_mixture,
    list_datasets,
    list_mixtures,
    load_registry,
    resolve_mixture,
)
from minimodel.datasets.shards import (
    IGNORE_INDEX,
    ShardIndex,
    ShardWriter,
    TokenizedCorpus,
    choose_dtype,
)
from minimodel.datasets.tokenize_datasets import (
    extract_text,
    tokenize_chat_records,
    tokenize_jsonl,
    tokenize_preference_records,
    tokenize_registered,
    tokenize_text_records,
)


class TestRegistry:
    """The YAML dataset catalogue."""

    def test_registry_loads_and_validates(self):
        registry = load_registry()
        assert len(registry["datasets"]) > 30
        assert len(registry["mixtures"]) >= 8

    def test_every_stage_is_represented(self):
        stages = {spec.stage for spec in list_datasets()}
        assert {"pretrain", "sft", "cot", "preference", "rlvr", "eval"} <= stages

    def test_named_datasets_present(self):
        for name in (
            "fineweb-edu-dedup",
            "cosmopedia-v2",
            "alpaca-cleaned",
            "casual-conversation",
            "openthoughts3",
            "ultrafeedback-binarized",
            "gsm8k-rlvr",
            "builtin-demo",
        ):
            assert get_dataset(name).name == name

    def test_mixture_weights_normalise(self):
        mixture = get_mixture("smollm-corpus")
        weights = mixture.normalized_weights()
        assert pytest.approx(sum(w for _, w in weights)) == 1.0
        assert len(resolve_mixture("smollm-corpus")) == len(weights)

    def test_stage_filter(self):
        assert all(spec.stage == "cot" for spec in list_datasets(stage="cot"))
        assert len(list_mixtures()) >= 8

    def test_unknown_names_raise(self):
        with pytest.raises(ConfigError, match="unknown dataset"):
            get_dataset("no-such-dataset")
        with pytest.raises(ConfigError, match="unknown mixture"):
            get_mixture("no-such-mixture")

    def test_spec_display_and_dict(self):
        spec = get_dataset("cosmopedia-v2")
        assert ":" in spec.display
        assert spec.to_dict()["stage"] == "pretrain"

    def test_missing_registry_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_registry(tmp_path / "nope.yaml")

    def test_dangling_mixture_reference_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "datasets: {a: {source: builtin}}\n"
            "mixtures: {m: {components: [{dataset: missing}]}}\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="unknown dataset"):
            load_registry(path)


class TestShards:
    """The binary token shard format."""

    def test_choose_dtype(self):
        assert choose_dtype(4096) == "uint16"
        assert choose_dtype(100_000) == "uint32"
        with pytest.raises(ValueError):
            choose_dtype(0)

    def test_write_and_read_across_shards(self, tmp_path):
        with ShardWriter(tmp_path, dtype="uint16", shard_tokens=10, vocab_size=64) as writer:
            for _ in range(5):
                writer.write_document(list(range(6)))
            writer.write_document([])  # empty documents are ignored
        corpus = TokenizedCorpus(tmp_path)
        assert corpus.n_tokens == 30
        assert len(corpus.index.shards) >= 2
        # A read that straddles a shard boundary must be stitched correctly.
        assert corpus.read(8, 6).tolist() == [2, 3, 4, 5, 0, 1]
        assert len(corpus) == 30
        assert "TokenizedCorpus" in repr(corpus)

    def test_supervised_labels_roundtrip(self, tmp_path):
        with ShardWriter(tmp_path, supervised=True, vocab_size=64) as writer:
            writer.write_document([1, 2, 3], [IGNORE_INDEX, 2, 3])
        corpus = TokenizedCorpus(tmp_path)
        assert corpus.supervised
        assert corpus.read(0, 3, labels=True).tolist() == [IGNORE_INDEX, 2, 3]

    def test_label_length_mismatch_rejected(self, tmp_path):
        writer = ShardWriter(tmp_path, supervised=True)
        with pytest.raises(ValueError, match="labels length"):
            writer.write_document([1, 2], [1])

    def test_out_of_range_reads_raise(self, tmp_path):
        with ShardWriter(tmp_path) as writer:
            writer.write_document([1, 2, 3])
        corpus = TokenizedCorpus(tmp_path)
        with pytest.raises(IndexError):
            corpus.read(0, 99)
        with pytest.raises(ValueError):
            corpus.read(-1, 1)
        with pytest.raises(ValueError, match="no label array"):
            corpus.read(0, 1, labels=True)

    def test_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="index.json"):
            TokenizedCorpus(tmp_path)

    def test_iter_windows_and_stats(self, corpus_dir):
        corpus = TokenizedCorpus(corpus_dir)
        windows = list(corpus.iter_windows(32))
        assert all(len(w) == 32 for w in windows)
        stats = corpus.stats()
        assert stats["n_tokens"] == corpus.n_tokens
        assert stats["size_bytes"] > 0

    def test_shard_index_roundtrip(self):
        index = ShardIndex(dtype="uint16", n_tokens=5, extra={"custom": 1})
        restored = ShardIndex.from_dict(index.to_dict())
        assert restored.n_tokens == 5
        assert restored.extra["custom"] == 1

    def test_write_documents_with_labels(self, tmp_path):
        with ShardWriter(tmp_path, supervised=True) as writer:
            writer.write_documents([[1, 2], [3, 4]], [[1, 2], [3, 4]])
        assert TokenizedCorpus(tmp_path).n_tokens == 4


class TestPullAndTokenize:
    """Pulling raw records and turning them into shards."""

    def test_builtin_records_available(self):
        for stage in ("pretrain", "sft", "cot", "preference", "rlvr"):
            assert len(builtin_records(stage)) > 0
        assert len(builtin_records("pretrain", repeat=3)) == 3 * len(builtin_records("pretrain"))
        with pytest.raises(ValueError, match="unknown builtin stage"):
            builtin_records("nope")

    def test_pull_builtin(self, tmp_path):
        path = pull_dataset("builtin-demo", tmp_path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) > 10
        assert path.with_suffix(".meta.json").exists()
        # A second pull is a no-op unless overwrite is requested.
        assert pull_dataset("builtin-demo", tmp_path) == path

    def test_pull_local_files(self, tmp_path):
        source = tmp_path / "docs"
        source.mkdir()
        (source / "a.txt").write_text("one\n\ntwo\n", encoding="utf-8")
        (source / "b.jsonl").write_text('{"text": "three"}\n', encoding="utf-8")
        (source / "c.json").write_text('[{"text": "four"}]', encoding="utf-8")

        from minimodel.datasets.registry import DatasetSpec

        spec = DatasetSpec(name="local", source="local", path=str(source))
        records = list(iter_records(spec))
        assert {r["text"] for r in records} == {"one", "two", "three", "four"}
        assert len(list(iter_records(spec, limit=2))) == 2

    def test_local_without_path_raises(self):
        from minimodel.datasets.registry import DatasetSpec

        with pytest.raises(ValueError, match="no `path`"):
            list(iter_records(DatasetSpec(name="x", source="local")))

    def test_unknown_source_raises(self):
        from minimodel.datasets.registry import DatasetSpec

        with pytest.raises(ValueError, match="unknown dataset source"):
            list(iter_records(DatasetSpec(name="x", source="carrier-pigeon")))

    def test_pull_mixture_splits_limit(self, tmp_path, monkeypatch):
        registry_path = tmp_path / "registry.yaml"
        registry_path.write_text(
            "datasets:\n"
            "  a: {source: builtin, stage: pretrain}\n"
            "  b: {source: builtin, stage: pretrain}\n"
            "mixtures:\n"
            "  m: {components: [{dataset: a, weight: 0.5}, {dataset: b, weight: 0.5}]}\n",
            encoding="utf-8",
        )
        outputs = pull_mixture("m", tmp_path / "raw", total_records=10, registry_path=registry_path)
        assert set(outputs) == {"a", "b"}

    def test_extract_text_field_fallbacks(self):
        assert extract_text({"content": "hi"}) == "hi"
        assert extract_text({"body": "hi"}) == "hi"
        assert extract_text({"nothing": 1}) is None

    def test_tokenize_text_records(self, tokenizer, tmp_path):
        stats = tokenize_text_records(
            [{"text": "hello world"}, {"nothing": 1}], tokenizer, tmp_path
        )
        assert stats["n_documents"] == 1
        assert stats["skipped"] == 1
        assert stats["n_tokens"] > 0

    def test_tokenize_chat_records_masks_prompt(self, tokenizer, tmp_path):
        tokenize_chat_records(builtin_records("sft"), tokenizer, tmp_path)
        corpus = TokenizedCorpus(tmp_path)
        labels = corpus.read(0, corpus.n_tokens, labels=True)
        assert (labels == IGNORE_INDEX).any()
        assert (labels != IGNORE_INDEX).any()

    def test_tokenize_chat_truncates(self, tokenizer, tmp_path):
        stats = tokenize_chat_records(
            builtin_records("sft"), tokenizer, tmp_path, max_length=4
        )
        assert stats["truncated"] > 0

    def test_tokenize_chat_skips_bad_records(self, tokenizer, tmp_path):
        stats = tokenize_chat_records([{"garbage": 1}], tokenizer, tmp_path)
        assert stats["skipped"] == 1

    def test_tokenize_preference_records(self, tokenizer, tmp_path):
        stats = tokenize_preference_records(
            [*builtin_records("preference"), {"chosen": "a"}], tokenizer, tmp_path / "p.jsonl"
        )
        assert stats["pairs"] == len(builtin_records("preference"))
        assert stats["skipped"] == 1

    def test_tokenize_preference_from_message_lists(self, tokenizer, tmp_path):
        record = {
            "chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "good"}],
            "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "bad"}],
        }
        stats = tokenize_preference_records([record], tokenizer, tmp_path / "p.jsonl")
        assert stats["pairs"] == 1

    def test_tokenize_jsonl_dispatch(self, tokenizer, tmp_path):
        source = tmp_path / "raw.jsonl"
        source.write_text('{"text": "hello world"}\n', encoding="utf-8")
        assert tokenize_jsonl(source, tokenizer, tmp_path / "t", format="text")["n_tokens"] > 0
        with pytest.raises(ValueError, match="unknown tokenization format"):
            tokenize_jsonl(source, tokenizer, tmp_path / "u", format="hieroglyphs")

    def test_tokenize_registered_requires_pull(self, tokenizer, tmp_path):
        with pytest.raises(FileNotFoundError, match="data pull"):
            tokenize_registered("builtin-demo", tokenizer, raw_dir=tmp_path)

    def test_tokenize_registered_after_pull(self, tokenizer, tmp_path):
        pull_dataset("builtin-demo", tmp_path / "raw")
        stats = tokenize_registered(
            "builtin-demo", tokenizer, raw_dir=tmp_path / "raw", output_root=tmp_path / "tok"
        )
        assert stats["n_tokens"] > 0


class TestLoaders:
    """Datasets and dataloaders."""

    def test_packed_dataset_shapes_and_shift(self, corpus_dir):
        dataset = PackedTextDataset(corpus_dir, seq_len=16, shuffle=False)
        inputs, labels = dataset[0]
        assert inputs.shape == (16,) and labels.shape == (16,)
        # Labels are the inputs shifted by one.
        assert torch.equal(inputs[1:], labels[:-1])
        assert "PackedTextDataset" in repr(dataset)
        assert dataset.token_count() > 0

    def test_packed_dataset_is_deterministic(self, corpus_dir):
        a = PackedTextDataset(corpus_dir, seq_len=16, seed=3)
        b = PackedTextDataset(corpus_dir, seq_len=16, seed=3)
        assert torch.equal(a[5][0], b[5][0])
        c = PackedTextDataset(corpus_dir, seq_len=16, seed=4)
        assert not torch.equal(a[5][0], c[5][0])

    def test_packed_dataset_rejects_short_corpus(self, corpus_dir):
        with pytest.raises(ValueError, match="fewer than"):
            PackedTextDataset(corpus_dir, seq_len=10**7)

    def test_supervised_dataset_masks(self, sft_dir):
        dataset = SupervisedDataset(sft_dir, seq_len=16)
        _, labels = dataset[0]
        assert labels.shape == (16,)
        assert len(dataset) > 0

    def test_supervised_dataset_requires_labels(self, corpus_dir):
        with pytest.raises(ValueError, match="no label array"):
            SupervisedDataset(corpus_dir, seq_len=8)

    def test_mixture_respects_weights(self, corpus_dir, sft_dir):
        first = PackedTextDataset(corpus_dir, seq_len=8)
        second = SupervisedDataset(sft_dir, seq_len=8)
        mixture = MixtureDataset([first, second], [0.9, 0.1], names=["a", "b"], length=200)
        assert len(mixture) == 200
        assert [c["name"] for c in mixture.component_stats()] == ["a", "b"]
        assert "a=0.90" in repr(mixture)
        for index in range(20):
            inputs, _ = mixture[index]
            assert inputs.shape == (8,)

    def test_mixture_validation(self, corpus_dir):
        dataset = PackedTextDataset(corpus_dir, seq_len=8)
        with pytest.raises(ValueError, match="at least one"):
            MixtureDataset([])
        with pytest.raises(ValueError, match="same length"):
            MixtureDataset([dataset], [1.0, 2.0])
        with pytest.raises(ValueError, match="positive"):
            MixtureDataset([dataset], [0.0])

    def test_collate_and_dataloader(self, corpus_dir):
        dataset = PackedTextDataset(corpus_dir, seq_len=8)
        batch = collate_batch([dataset[0], dataset[1]])
        assert batch["input_ids"].shape == (2, 8)
        loader = build_dataloader(dataset, batch_size=2)
        first = next(iter(loader))
        assert first["labels"].shape == (2, 8)

    def test_infinite_loader_cycles(self, corpus_dir):
        loader = build_dataloader(PackedTextDataset(corpus_dir, seq_len=8, length=2), batch_size=1)
        iterator = infinite_loader(loader)
        assert len([next(iterator) for _ in range(6)]) == 6

    def test_pair_dataset_iterates_and_pads(self, pairs_path):
        dataset = JsonlPairDataset(pairs_path, max_length=32, repeat=False)
        rows = list(dataset)
        assert len(rows) > 0
        batch = JsonlPairDataset.collate(rows[:2])
        assert batch["chosen_ids"].shape == batch["rejected_ids"].shape
        assert batch["chosen_ids"].shape[0] == 2

    def test_pair_dataset_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            JsonlPairDataset(tmp_path / "nope.jsonl")

    def test_corpus_dtype_matches_vocab(self, corpus_dir):
        corpus = TokenizedCorpus(corpus_dir)
        assert corpus.dtype == np.dtype("uint16")
