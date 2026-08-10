# notebooks/

Interactive walkthroughs of the Python API (the CLI equivalents are in
[docs/recipes.md](../docs/recipes.md)). Both run offline on CPU in a couple of
minutes.

| Notebook | Shows |
| --- | --- |
| [01_train_a_language_model.ipynb](01_train_a_language_model.ipynb) | Tokenizer → corpus → pretrain → SFT → evaluate → generate, all in-process |
| [02_pixel_art_generator.ipynb](02_pixel_art_generator.ipynb) | Sprites → palette corpus → PixelGPT → sampling grids, plus a DiT comparison |
| [03_crush_glint2_colab.ipynb](03_crush_glint2_colab.ipynb) | **One-cell Colab T4**: train `dense_1_4m` on FineWeb-Edu/DCLM/TinyStories/soft-QA for ≤4h with Drive backup (recipe report 11) |
| [04_stigmergy_train.ipynb](04_stigmergy_train.ipynb) | **Chess contest**: train Stigmergy-DPFE (pheromone fields) on CPU/CUDA → JSON weights |
| [05_stigmergy_fields.ipynb](05_stigmergy_fields.ipynb) | Visualize the 10-channel diffused pheromone lattice |
| [06_stigmergy_elo.ipynb](06_stigmergy_elo.ipynb) | Elo ladder + uniqueness → contest composite score |

Setup from the repo root:

```bash
pip install -e . matplotlib jupyter
jupyter lab notebooks/
```

Convention: notebooks are teaching artifacts, not pipelines — anything worth
running twice belongs in a recipe YAML or a script. Keep outputs cleared when
committing (`.pre-commit-config.yaml` warns on giant files).
