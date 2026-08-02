# AGENTS.md

Instructions for coding agents (and humans) working in this repository. Follow
them; the test suite and CI enforce most of them.

## What this repository is

`minimodel-trainer` trains small language models (1M-350M params) and image
models (3M-90M) end to end: tokenizer -> data -> pretrain -> post-train (SFT /
CoT / DPO / RLVR / SPIN) -> eval -> merge -> inference -> model card. Text and
vision are parallel pipelines with the same shape.

## Layout in 30 seconds

```
src/                     the `minimodel` package (import minimodel.<subpackage>)
  core/                  config, registries, logging, seeding, devices, io
  tokenization/          byte-level BPE + chat template
  datasets/              registry (config/datasets.yaml), pull, tokenize, loaders
  architectures/         LM families + templates/*.yaml (generated, verified)
  training/              Trainer + SFT/CoT + rl/{dpo,rlvr,spin} + recipe runner
  checkpointing/         checkpoint manager, ETA, loss plots
  inference/             sampling, generate, chat
  benchmarking/          eval harness, compare, charts
  merging/               slerp/linear/ties/task-arithmetic/dare
  cardgen/               model card generation
  vision/                image models: architectures/, data/, training/, sampling/
  cli.py                 `minimodel` entry point (vision subcommands in vision/cli.py)
configs/                 YAML recipes (pretrain/, sft/, cot/, rl/, vision/)
scripts/                 template generators + smoke_e2e.py
tests/                   pytest suite (offline, ~30s)
docs/                    user documentation
research/                reading lists and design notes
```

Note: sources live in `src/` but the package name is `minimodel`
(`package-dir = {minimodel = "src"}` in pyproject.toml). Import
`minimodel.training.trainer`, never `src.training.trainer`.

## Environment

```bash
python3 -m venv venv && source venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU wheel is fine
pip install -r requirements-dev.txt && pip install -e .
```

## Non-negotiable checks before you finish

Run all three; they are fast:

```bash
venv/bin/ruff check src tests            # lint (line length 100, rules in pyproject)
venv/bin/pytest                          # full suite, ~30s, no network needed
venv/bin/python scripts/smoke_e2e.py     # every pipeline end to end, ~2s
```

If you touched architecture code or the template generators, also run:

```bash
venv/bin/python scripts/generate_templates.py --check
venv/bin/python scripts/generate_vision_templates.py --check
```

## Rules that are easy to break by accident

1. **Templates are generated.** Never hand-edit
   `src/architectures/templates/*.yaml` (except `supra2_1406240.yaml`, which is
   the hand-annotated spec) or `src/vision/templates/*.yaml`. Edit
   `scripts/generate_templates.py` / `scripts/generate_vision_templates.py` and
   regenerate. Declared `params:` must equal the built model's count - tests
   verify every template.

2. **Everything must work offline.** Tests, the smoke script and both
   quickstarts run with no network. If you add a feature, exercise it with the
   builtin corpus (`minimodel.datasets.builtin`) or synthetic sprites
   (`minimodel.vision.data.datasets.synthetic_sprites`). Network access happens
   only inside `pull_datasets.py` / `vision/registry.py` behind optional
   imports of `datasets`.

3. **Heavy dependencies stay optional.** Core installs need only torch, numpy,
   pyyaml, tqdm, requests. `datasets`, `tokenizers`, `matplotlib`, `PIL`,
   `safetensors` are imported lazily inside the functions that need them, with
   a helpful error message. Never import them at module top level.

4. **New trainers subclass `Trainer` and override `compute_loss`.** That is
   the whole contract - you inherit checkpoint/resume, AMP, accumulation,
   scheduling, logging and callbacks. Look at `InstructTrainer` (30 lines of
   logic) before writing a loop from scratch.

5. **New architectures implement `BaseLanguageModel` / `BaseImageModel`** and
   register in `architectures/registry.py` (or `vision/architectures/registry.py`).
   They must support the incremental `KVCache` protocol, and there is a test
   that full-sequence and token-by-token decoding agree to ~1e-4 - your model
   has to pass it.

6. **Datasets are registry entries, not code.** To add a corpus, append to
   `src/datasets/config/datasets.yaml` (or `image_datasets.yaml`) with
   `stage`, `format`, `license` and a `description` that says *why* one would
   use it. Mixtures referencing unknown datasets fail a test.

7. **Determinism.** Data order and augmentation draw from
   `np.random.default_rng(seed * K + index)` per item, never from global RNG,
   so a resumed run sees the same data. Preserve this pattern in new datasets.

8. **Config keys are permanent.** Recipes in the wild reference them. Adding a
   key is fine (with a default); renaming or removing one requires updating
   every recipe under `configs/` plus `docs/`.

9. **Docstrings are documentation.** Every public module, class and function
   carries a docstring saying *why*, not just what - the codebase is meant to
   be read as a textbook. Comments that narrate the code ("increment counter")
   are deleted on sight.

10. **No hidden state in tests.** Tests share session-scoped fixtures from
    `tests/conftest.py` (tiny tokenizer, corpora, models). Keep every test
    under ~1s; mark anything slower `@pytest.mark.slow`.

## Style

- Python 3.10+, ruff-formatted (`make format`), line length 100.
- Imports at the top of the module, sorted; lazy imports only for optional
  heavy dependencies (rule 3) and documented circular-dependency breaks.
- Type hints on public signatures; `from __future__ import annotations`
  everywhere.
- f-strings in code, %-style in logging calls.
- Exhaustive `switch`-style dispatch: when matching on a closed set of names
  (stage, format, method), end with an informative `raise ValueError` listing
  the valid options, so a typo fails loudly and helpfully.

## Commit and PR conventions

- One logical change per commit; imperative mood subject lines.
- If a change alters training behaviour (loss, optimizer, schedule, data
  order), say so explicitly in the commit body - reproducibility depends on it.
- PRs should paste the output of `pytest` and `scripts/smoke_e2e.py`.

## Things that look like bugs but are not

- `supra2_1406240.yaml` disagreeing in style with other templates: it is the
  hand-written annotated spec; the others are generated summaries.
- `LOOPED` models re-running the same block: that is the architecture.
- `expert_bias` in MoE never receiving gradients: it is updated by a
  controller rule, not the optimizer, by design (aux-loss-free balancing).
- The tokenizer's `|.` regex fallback branch: required so the split pattern is
  total; removing it silently drops characters from the corpus.
- `weights_only=False` in `CheckpointManager.load` for `trainer.pt`: the file
  holds RNG state and optimizer objects. Model weights (`model.pt`) always
  load with `weights_only=True`.

## Adding a new model size (the most common task)

1. Edit the `SPECS` list in `scripts/generate_templates.py` (or the vision
   one).
2. `venv/bin/python scripts/generate_templates.py` to write the YAML.
3. `venv/bin/pytest tests/test_architectures.py -k template` to verify counts.
4. Optionally add a recipe under `configs/pretrain/` and a line to the README
   table.
