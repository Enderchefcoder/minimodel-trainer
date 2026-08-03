# research/

Active research program: **make our small models beat Glint-Research's Glint-2
and push past what a 1M-class model is assumed to do.** Everything here is
reproducible from the scripts in `experiments/`, evaluated by one shared harness,
and written up in `reports/`.

## Reports (read in order)

| # | Report | One-line finding |
| --- | --- | --- |
| 00 | [competitor_glint2](reports/00_competitor_glint2.md) | Glint-2 dissected; the target and its two inference tricks |
| 01 | [baseline_validation](reports/01_baseline_validation.md) | Harness validated; the released Glint-2 is 1.71M (not 1.06M); "ppl" is byte-normalised |
| 02 | [slm_techniques](reports/02_slm_techniques.md) | Literature + the techniques we adopt, with citations |
| 03 | [architecture_bakeoff](reports/03_architecture_bakeoff.md) | At a fixed token budget, dense beats looped per-token |
| 04 | [stabilizers_optimizer](reports/04_stabilizers_optimizer.md) | Value residuals + Muon; what helps looped training |
| 07 | [loop_robustness](reports/07_loop_robustness.md) | Glint-2 breaks off 8 loops (35×); ours span 1.04–1.19× and scale up |
| 08 | [inference_wins](reports/08_inference_wins.md) | Effort ladder + quality probe: architecture-agnostic upside |
| 09 | [synthesis](reports/09_synthesis.md) | The recipe to beat Glint-2, and the CPU-budget contender |
| — | [RESULTS](reports/RESULTS.md) | Auto-aggregated tables + figures from all runs |

## Experiments (`experiments/`)

| Script | What it does |
| --- | --- |
| `pull_eval_data.py` | Cache BLiMP / ARC-Easy / WikiText-2 as JSONL |
| `pull_train_data.py` | Cache TinyStories / fineweb-edu text |
| `setup_corpus.py` | Train the 4096-BPE tokenizer, tokenize corpora to shards |
| `eval_harness.py` | Model-agnostic BLiMP / ARC / byte-ppl scorer (shared by all) |
| `eval_glint2.py` | Run the released Glint-2 through the harness |
| `run_experiment.py` | Train one config, evaluate, save a result JSON |
| `ablations.py` | Grouped compute-matched ablations (arch, stabilizers, optimizer, ffn, ...) |
| `loops_scaling.py` | Evaluate a model across loop counts |
| `loop_robustness.py` | Controlled fixed-vs-uniform-vs-Poisson loop-training comparison |
| `contender.py` | Train the best-effort dense contender + measure inference lift |
| `aggregate.py` | Build `reports/RESULTS.md` and figures from result JSONs |

## Baselines (`baselines/`)

`glint2_model.py` — a faithful, strict-loading reimplementation of the *released*
Glint-2 checkpoint (the public `generate.py` cannot load its own weights).

## Reproduce

```bash
pip install datasets tokenizers huggingface_hub matplotlib
python research/experiments/pull_eval_data.py
python research/experiments/pull_train_data.py tinystories tinystories_val
python research/experiments/setup_corpus.py --vocab 4096
python research/experiments/eval_glint2.py            # baseline
python research/experiments/ablations.py arch          # bake-off
python research/experiments/loops_scaling.py --which glint
python research/experiments/loop_robustness.py 800
python research/experiments/contender.py 4000
python research/experiments/aggregate.py
```

Large downloads (data, the Glint-2 checkpoint) are git-ignored; code, reports,
result JSONs and figures are tracked. See [reports/09_synthesis.md](reports/09_synthesis.md)
for the bottom line.

## Design notes and reading lists

The paper-by-paper background for the main `minimodel` package lives beside this
file: [slm-reading-list.md](slm-reading-list.md),
[architecture-notes.md](architecture-notes.md),
[looped-transformers.md](looped-transformers.md),
[post-training-notes.md](post-training-notes.md),
[image-model-notes.md](image-model-notes.md).
