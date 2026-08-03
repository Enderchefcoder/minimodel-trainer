# Contributing

Thanks for looking. The bar for merging is simple: the checks pass, the code
reads like the rest of the codebase, and the docs moved with the code.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements-dev.txt && pip install -e .
```

## Before opening a PR

```bash
make lint      # ruff check src tests
make test      # full suite, offline, ~30s
make smoke     # scripts/smoke_e2e.py — every pipeline end to end
# if you touched architectures or template generators:
venv/bin/python scripts/generate_templates.py --check
venv/bin/python scripts/generate_vision_templates.py --check
```

CI runs exactly these on 3.10 and 3.12.

## Ground rules

The full conventions live in [AGENTS.md](AGENTS.md) (they are written for
coding agents, which makes them unambiguous for humans too). The short
version:

- **Offline-first**: tests and demos must run with no network; exercise new
  features against the builtin corpus or synthetic sprites.
- **Optional deps stay optional**: `datasets`, `tokenizers`, `matplotlib`,
  `PIL` are imported lazily with helpful errors, never at module top level.
- **Templates are generated**: edit `scripts/generate_*.py`, not the YAML.
- **New objectives subclass `Trainer`**; new architectures implement the base
  class + `KVCache` protocol and register themselves.
- **Docstrings explain why.** Narration comments get deleted.
- **Config keys are forever** — adding is fine, renaming means migrating
  every recipe in `configs/` and the docs.

## What contributions are welcome

Datasets for the registry (with licenses), architectures with a template and
budget test, verifiers for RLVR, benchmark task loaders, documentation fixes
grounded in something that confused you, and reproductions/refutations of the
numbers in `src/benchmarking/BENCHMARKS.md` — measured claims beat vibes.

## Reporting bugs

A failing test is the perfect bug report. Second best: the exact command,
the full traceback, and `pip freeze | grep -E 'torch|minimodel'`.
