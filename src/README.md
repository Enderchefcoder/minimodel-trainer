# src/ — the `minimodel` package

Sources live here but import as `minimodel` (`package-dir = {minimodel = "src"}`
in pyproject.toml). One subpackage per lifecycle stage:

| Subpackage | Does | Docs |
| --- | --- | --- |
| `core/` | config, registries, logging, seeding, devices, IO, DDP helpers | [configuration.md](../docs/configuration.md) |
| `tokenization/` | byte-level BPE + chat template | [tokenization.md](../docs/tokenization.md) |
| `datasets/` | registry, pulling, tokenizing, loaders | [data.md](../docs/data.md) |
| `architectures/` | dense / looped / MoE / hybrid LMs + templates | [architecture.md](../docs/architecture.md) |
| `training/` | Trainer, SFT, CoT, `rl/` (DPO, RLVR, SPIN), recipes | [training.md](../docs/training.md) |
| `checkpointing/` | checkpoint manager, ETA, loss plots | [checkpointing.md](../docs/checkpointing.md) |
| `inference/` | sampling, generate, chat | [inference.md](../docs/inference.md) |
| `benchmarking/` | eval harness, compare, charts | [evaluation.md](../docs/evaluation.md) |
| `merging/` | weight-space merges | [merging.md](../docs/merging.md) |
| `cardgen/` | model cards from run artifacts | [model-cards.md](../docs/model-cards.md) |
| `vision/` | the image-model pipeline | [vision.md](../docs/vision.md) |
| `cli.py` | the `minimodel` command | [cli.md](../docs/cli.md) |
| `quickstart.py` | the offline end-to-end demo | — |

Conventions for changing anything here: [AGENTS.md](../AGENTS.md).
