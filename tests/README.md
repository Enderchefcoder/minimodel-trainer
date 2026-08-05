# tests/

~455 tests, fully offline, ~30s on CPU. Session-scoped fixtures in
`conftest.py` build one tiny tokenizer/corpus/model set shared by everything.

| File | Covers |
| --- | --- |
| `test_core.py` | config/extends/interpolation, registry, IO, logging, seeding, devices, dist |
| `test_architectures.py` | layers, LM families (incl. cache-equivalence), builder, templates |
| `test_mm1m_candidates.py` | 20 ordered ~1M Glint-2 candidates + novel/mamba families |
| `test_tokenization.py` | BPE roundtrips, total split pattern, HF interop, chat masking |
| `test_datasets.py` | registry validation, shard format, tokenize paths, loaders |
| `test_checkpointing.py` | save/load/retention/export, ETA, plots |
| `test_training.py` | optimizers (Muon/Lion), schedules, trainer, resume, callbacks, SFT/CoT |
| `test_rl.py` | DPO losses/trainer, verifiers, GRPO advantages/rollouts, SPIN |
| `test_inference.py` | filters/penalties, generate/stream equivalence, runner |
| `test_benchmarking.py` | task normalisation, harness, compare, charts, merging |
| `test_cardgen_and_cli.py` | card generation, recipe runners, CLI command paths |
| `test_posttrain_dispatch.py` | every posttrain stage via recipes, vision CLI |
| `test_vision.py` | vision layers/models/data/training/samplers |
| `test_pipelines.py` | quickstarts, vision recipes, bundled config parsing |

Run: `make test` · coverage: `make coverage` · conventions: [AGENTS.md](../AGENTS.md).
