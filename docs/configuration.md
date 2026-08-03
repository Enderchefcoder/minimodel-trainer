# Configuration

One config system drives everything: YAML files -> `Config` objects ->
dataclass trainer configs. No global state, no hidden defaults.

## Recipe anatomy

```yaml
# configs/pretrain/my_run.yaml
extends: _base.yaml            # single file or list; deep-merged, child wins

run_name: my-run
tokenizer: artifacts/tokenizer.json

model:
  template: dense_30m          # bundled name, path to a YAML, or...
  # checkpoint: runs/prev/model  ...a saved model to continue from
  overrides: { window: 2048 }  # any architecture config key

data:
  train: data/tokenized/main   # or mixture: [{path, weight}, ...]
  eval: data/tokenized/heldout

training:                      # maps onto TrainerConfig fields
  max_steps: 100000
  lr: 1.2e-3
```

## Features

**Inheritance.** `extends:` resolves relative to the file, supports chains and
lists, and deep-merges mappings (scalars and lists replace). Every bundled
recipe under `configs/` inherits from a `_base.yaml`, so a size variant is a
handful of lines.

**Environment interpolation.** `${VAR}` and `${VAR:-default}` inside string
values. Missing variables without a default fail loudly at load time, not at
step 40000.

**Command-line overrides.** Any command accepts repeated
`--set dotted.key=value`; values parse as YAML scalars *plus* bare scientific
notation (`--set training.lr=3e-4` is the float, not the string - plain YAML
1.1 gets this wrong).

```bash
minimodel train -c recipe.yaml --set training.max_steps=500 --set model.template=dense_12m
```

Precedence: file < `extends` chain < programmatic overrides < `--set`.

**Dotted access in code.**

```python
from minimodel.core.config import load_config
cfg = load_config("configs/pretrain/dense_30m.yaml")
cfg["training.lr"]           # 0.0012
cfg.get("training.nope", 7)  # default
cfg.section("model")         # sub-Config
cfg.require("data.train")    # raises ConfigError with the file name if absent
```

**Validation philosophy.** Unknown keys inside `training:` produce a warning,
not an error (recipes should survive version skew); *wrong* values fail fast
with the list of valid options (`unknown optimizer 'adamvv'; available:
adamw, lion, muon, sgd`). Structural mistakes - a missing `data.train`, a
`seq_len` longer than the model's context - are checked before any compute is
spent.

## Where configs come from, per stage

| Command | Config consumed by |
| --- | --- |
| `minimodel train` | `training.recipe.run_pretrain_recipe` -> `TrainerConfig` |
| `minimodel posttrain` (stage: sft) | `InstructTrainerConfig` |
| `minimodel posttrain` (stage: cot) | `CoTTrainerConfig` |
| `minimodel posttrain` (stage: dpo/spin) | `DPOConfig` / `SPINConfig` |
| `minimodel posttrain` (stage: rlvr) | `RLVRConfig` |
| `minimodel vision train` (kind: ...) | `DiffusionConfig` / `PixelGPTConfig` / `VAETrainerConfig` |

Each dataclass documents its fields in-source; `docs/training.md` and friends
explain the ones with judgement attached.

## Reproducibility contract

Every run writes `run_metadata.json`: the fully-resolved config, model
config + parameter count, device description, dtype, world size and planned
token budget. A recipe file plus that metadata is sufficient to reproduce a
run; the model card generator reads the same file so published cards match
reality.
