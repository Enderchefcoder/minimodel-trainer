# datasets/

Declarative corpus handling.

- `config/datasets.yaml` — 48 text dataset recipes + 10 mixtures (this is the
  file to edit to add a corpus). `config/image_datasets.yaml` — the image
  catalogue.
- `registry.py` — loads/validates the YAML; `DatasetSpec`, `MixtureSpec`.
- `pull_datasets.py` — HF/local/URL/builtin → `data/raw/<name>.jsonl`.
- `tokenize_datasets.py` — JSONL → token shards (`text`), token+label shards
  (`chat`/`instruction`), or `pairs.jsonl` (`preference`).
- `shards.py` — the memmapped shard format (`ShardWriter`/`TokenizedCorpus`).
- `loader.py` — `PackedTextDataset`, `SupervisedDataset`, `MixtureDataset`,
  `JsonlPairDataset`, deterministic per (seed, index).
- `builtin.py` — the offline corpus every test and quickstart uses.

Docs: [docs/data.md](../../docs/data.md).
