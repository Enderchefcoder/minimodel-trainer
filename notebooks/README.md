# notebooks/

Interactive walkthroughs of the Python API (the CLI equivalents are in
[docs/recipes.md](../docs/recipes.md)). Both run offline on CPU in a couple of
minutes.

| Notebook | Shows |
| --- | --- |
| [01_train_a_language_model.ipynb](01_train_a_language_model.ipynb) | Tokenizer → corpus → pretrain → SFT → evaluate → generate, all in-process |
| [02_pixel_art_generator.ipynb](02_pixel_art_generator.ipynb) | Sprites → palette corpus → PixelGPT → sampling grids, plus a DiT comparison |

Setup from the repo root:

```bash
pip install -e . matplotlib jupyter
jupyter lab notebooks/
```

Convention: notebooks are teaching artifacts, not pipelines — anything worth
running twice belongs in a recipe YAML or a script. Keep outputs cleared when
committing (`.pre-commit-config.yaml` warns on giant files).
