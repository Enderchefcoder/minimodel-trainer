# Getting started

## Install

```bash
git clone https://github.com/Enderchefcoder/minimodel-trainer
cd minimodel-trainer
python3 -m venv venv && source venv/bin/activate

# CPU-only torch (fine for everything up to ~15M params):
pip install --index-url https://download.pytorch.org/whl/cpu torch
# or let pip pick the CUDA build on a GPU machine:
# pip install torch

pip install -e .
```

Check it works - this runs the entire pipeline on a corpus bundled inside the
package, no downloads:

```bash
minimodel quickstart
```

About a minute later you have `runs/quickstart/` containing a tokenizer, a
tokenized corpus, a pretrained + instruction-tuned model, a benchmark JSON and
a generated model card. Talk to the (admittedly very small) result:

```bash
minimodel chat --model runs/quickstart/model
```

The image side has the same smoke test, using synthetic sprites:

```bash
minimodel vision quickstart
# -> runs/vision-quickstart/samples_pixelgpt.png and samples_dit.png
```

## The five commands you will actually use

```bash
minimodel models                 # what architectures/sizes exist
minimodel data list              # what datasets/mixtures exist
minimodel train     -c <recipe>  # pretrain
minimodel posttrain -c <recipe>  # sft / cot / dpo / spin / rlvr
minimodel bench     -m <model>   # measure
```

Every command takes `--json` for machine-readable output and `--set a.b=c` to
override any config key without editing files.

## Your first real model

The demo model is deliberately tiny. Here is the smallest run that produces
something genuinely usable - a ~12M model on TinyStories that writes coherent
short stories. On a single consumer GPU this takes an evening; on CPU, leave
it overnight.

```bash
# 1. Pull ~500K stories (about 300MB of text)
minimodel data pull tinystories --limit 500000

# 2. Train a tokenizer on them. 8192 tokens is plenty for TinyStories' vocabulary.
minimodel tokenizer train --dataset tinystories --limit 50000 \
    --vocab-size 8192 -o artifacts/tokenizer.json

# 3. Tokenize into training shards
minimodel data tokenize tinystories -t artifacts/tokenizer.json

# 4. Train. The recipe file is ~20 lines; open it and look.
minimodel train --config configs/pretrain/demo_tiny.yaml \
    --set data.train=data/tokenized/tinystories \
    --set model.template=dense_12m \
    --set training.max_steps=20000

# 5. Watch it learn (from another terminal)
minimodel plot runs/demo-tiny --output loss.png

# 6. Sample from it
minimodel generate -m runs/demo-tiny/model -p "Once upon a time" --max-new-tokens 200
```

What to expect: loss around 6.9 at step 0 (that is `ln(vocab_size)` - random
guessing), dropping below 2.0 by the end, and samples that hold a simple story
together. If your loss curve looks different, see
[troubleshooting.md](troubleshooting.md).

## Where things land on disk

```
artifacts/tokenizer.json        the tokenizer (one file, self-contained)
data/raw/<name>.jsonl           pulled datasets
data/tokenized/<name>/          binary token shards + index.json
runs/<run_name>/
  train.log                     full log
  metrics.jsonl                 one JSON object per step (plot/compare read this)
  run_metadata.json             config + environment, for reproducibility
  checkpoints/step_XXXXXX/      model.pt + config.json + trainer.pt
  model/                        exported best weights + tokenizer (shippable)
```

## Choosing a size

Rules of thumb, assuming reasonable data (see [data.md](data.md)):

| Params | Trains on | What it can learn |
| --- | --- | --- |
| 1-5M | CPU, minutes-hours | Grammar, TinyStories-level narration |
| 10-30M | 1 GPU, hours | Coherent paragraphs, simple instructions after SFT |
| 60-125M | 1 GPU, days | Usable assistant for narrow domains, basic reasoning |
| 350M | multi-GPU | The ceiling of this repo's ambitions |

Run `minimodel models` to see all bundled templates with exact parameter
counts, and read [architecture.md](architecture.md) for the dense / looped /
MoE / hybrid trade-offs.

## Next steps

- [recipes.md](recipes.md) - complete walkthroughs (assistant, reasoning
  model, pixel-art generator, image editor)
- [training.md](training.md) - what every training knob does and why the
  defaults are what they are
- [data.md](data.md) - the dataset catalogue and how to mix corpora
