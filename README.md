# minimodel-trainer

An end-to-end toolkit for training **small models** - language models from ~1M
to ~350M parameters and image models from ~3M to ~90M - on hardware you
actually have.

Everything a model needs to exist is in this one repository: tokenizer
training, dataset acquisition, pretraining, instruction tuning,
chain-of-thought distillation, preference optimization (DPO), verifiable-reward
RL (GRPO), self-play (SPIN), evaluation, model merging, inference, model-card
generation - plus a parallel pipeline for image models (diffusion, pixel art,
instruction-based editing).

```bash
pip install -e .
minimodel quickstart          # tokenizer -> data -> train -> eval -> chat, offline, ~1 min
minimodel vision quickstart   # the same for image models, on synthetic sprites
```

## Why small models

Small models are not scaled-down afterthoughts. They are the right tool when
you want to *own the whole stack*: train from scratch on data you chose,
iterate on architecture in hours instead of weeks, deploy on CPU, and
understand every part of the system because every part is inspectable. This
repository is built around that workflow - every stage runs on a laptop, and
every stage scales up when you get GPUs.

## What's inside

### Four language-model architectures, one interface

| Family | What it is | When to pick it |
| --- | --- | --- |
| `dense-transformer` | Modern GPT: RMSNorm, RoPE, SwiGLU, GQA, QK-norm, sliding-window patterns | The default. Predictable, well understood |
| `looped-transformer` | Weight-shared recurrent depth with per-iteration LoRA/gain conditioning | Best quality per *parameter*; adjustable depth at inference |
| `moe-transformer` | Sparse mixture-of-experts, shared expert, aux-loss-free balancing | Best quality per training *FLOP* at 30M+ |
| `hybrid-recurrent` | Griffin-style gated linear recurrences + occasional attention | Long context with a constant-size decode state |

13 bundled size templates from **1.4M** (`supra2_1406240`, whose annotated YAML
is also the architecture's written spec) to **343M** (`dense_350m`). Every
template's declared parameter count is verified against the built model in CI.

### Five image-model architectures

| Family | What it is |
| --- | --- |
| `pixelgpt` | Autoregressive pixel-art generator over an exact palette (the 24x24, ~10M config is tuned for `unstonio/pixelgpt-24x24-20k`) |
| `dit` | Diffusion transformer with rectified-flow training, class/text conditioning, CFG |
| `unet` | Convolutional diffusion for small datasets |
| `image-edit` | InstructPix2Pix-style editing: source image + text instruction, dual guidance |
| `vae` | Latent autoencoder for scaling diffusion past 64px |

### A dataset registry, not dataset scripts

48 text datasets and 15+ image datasets are described declaratively in
[`src/datasets/config/datasets.yaml`](src/datasets/config/datasets.yaml) -
SmolLM corpus (FineWeb-Edu + Cosmopedia), TinyStories, math/code corpora, SFT
sets (SmolTalk, Alpaca-cleaned, casual-conversation, ...), reasoning traces
(OpenThoughts3-1.2M, OpenR1-Math), preference data (UltraFeedback), RLVR tasks
(GSM8K, Countdown), and the standard eval suites (BLiMP, ARC, HellaSwag,
WikiText). Weighted mixtures compose them:

```bash
minimodel data list                        # browse the catalogue
minimodel data pull cosmopedia-v2 --limit 100000
minimodel data tokenize cosmopedia-v2 -t artifacts/tokenizer.json
```

### Training that covers the whole lifecycle

```
pretrain -> sft -> cot -> dpo / spin / rlvr -> merge -> bench -> card
```

Every stage is a subclass of one `Trainer` (so checkpoint/resume, mixed
precision, gradient accumulation, LR schedules, ETA estimation, early stopping
and divergence detection are shared), driven by YAML recipes with inheritance:

```bash
minimodel train     --config configs/pretrain/dense_30m.yaml
minimodel posttrain --config configs/sft/instruct.yaml
minimodel posttrain --config configs/rl/dpo.yaml --set training.beta=0.2
```

Optimizers include AdamW, **Muon** (Newton-Schulz orthogonalised momentum) and
Lion; schedules include cosine and **WSD** (warmup-stable-decay, for when you
don't know your token budget up front).

### Everything after training

- `minimodel bench` - perplexity, multiple-choice (ARC/HellaSwag-style),
  minimal pairs (BLiMP-style), verifiable generation, throughput
- `minimodel compare` - Markdown tables with best-per-column highlighting
- `minimodel merge` - linear / SLERP / task-arithmetic / TIES / DARE
- `minimodel chat` - streaming terminal chat with any trained model
- `minimodel card` - model cards generated from run artifacts, not memory

## Installation

```bash
git clone https://github.com/Enderchefcoder/minimodel-trainer
cd minimodel-trainer
python -m venv venv && source venv/bin/activate
pip install -e .                      # core (torch, numpy, pyyaml, tqdm, requests)
pip install -e ".[hf]"                # + HF datasets/tokenizers interop
pip install -e ".[viz]"               # + matplotlib charts
pip install -r requirements-dev.txt   # + pytest, ruff
```

CPU-only PyTorch (laptops, CI): `pip install --index-url
https://download.pytorch.org/whl/cpu torch` first.

## A real workflow, end to end

```bash
# 1. Data
minimodel data pull fineweb-edu-10bt --limit 2000000
minimodel tokenizer train --dataset fineweb-edu-10bt --vocab-size 16384 -o artifacts/tokenizer.json
minimodel data tokenize fineweb-edu-10bt -t artifacts/tokenizer.json

# 2. Pretrain a 30M model (see the recipe for the full hyperparameters)
minimodel train --config configs/pretrain/dense_30m.yaml

# 3. Make it an assistant
minimodel data pull smoltalk --limit 200000
minimodel data tokenize smoltalk -t artifacts/tokenizer.json
minimodel posttrain --config configs/sft/instruct.yaml

# 4. Measure it, then talk to it
minimodel bench --model runs/dense-30m-instruct/model -o bench.json
minimodel chat  --model runs/dense-30m-instruct/model
```

And the pixel-art model:

```bash
minimodel vision data prepare --dataset pixelgpt-24x24 --mode palette --size 24 -o data/images/sprites
minimodel vision train --config configs/vision/pixelgpt_24x24.yaml
minimodel vision sample -m runs/pixelgpt-24x24/model -n 16 -o sprites.png --scale 8
```

## Research: beating Glint-2

The [`research/`](research/) directory holds a full study that uses this toolkit
to reverse-engineer and beat Glint-Research's [Glint-2](https://huggingface.co/Glint-Research/Glint-2)
(a ~1M-class looped model). Headline: on a matched head-to-head — same ~1.7M
parameters, same byte-normalised perplexity, same eval harness, same
distribution — our dense contender reaches **byte-ppl 1.525 vs Glint-2's 2.405
(37% better)**, trained in 32 minutes on 4 CPU cores. We also show Glint-2's
test-time loop scaling is broken (byte-ppl 3.5 → 125 gibberish from 8 → 16
loops) while our stabilised, Poisson-loop-sampled models stay flat or improve.
See [`research/reports/09_synthesis.md`](research/reports/09_synthesis.md) for
the bottom line and [`research/README.md`](research/README.md) for the full
program. Everything is reproducible and every claim is a committed result JSON.

## Documentation

| | |
| --- | --- |
| [Getting started](docs/getting-started.md) | Install, quickstart, first real run |
| [Architectures](docs/architecture.md) | The four LM families, how to choose, how to add one |
| [Data](docs/data.md) | Registry, formats, mixtures, adding your own |
| [Tokenization](docs/tokenization.md) | Byte-level BPE, chat template, special tokens |
| [Training](docs/training.md) | The trainer, optimizers, schedules, recipes |
| [Post-training](docs/post-training.md) | SFT and chain-of-thought distillation |
| [RL](docs/rl.md) | DPO, RLVR/GRPO, SPIN |
| [Evaluation](docs/evaluation.md) | The harness and what small-model numbers mean |
| [Vision](docs/vision.md) | Image models: diffusion, pixel art, editing |
| [Inference](docs/inference.md) | Sampling, chat, reasoning mode |
| [Merging](docs/merging.md) | Five merge methods and when each wins |
| [Model cards](docs/model-cards.md) | Automatic card generation |
| [CLI reference](docs/cli.md) | Every command and flag |
| [Recipes cookbook](docs/recipes.md) | Copy-paste walkthroughs |
| [Troubleshooting](docs/troubleshooting.md) | Loss spikes, NaNs, OOM, bad samples |
| [Research notes](research/README.md) | The reading list behind the design choices |

## Development

```bash
make install-cpu   # venv + CPU torch + dev deps + editable install
make test          # 455 tests, ~30s
make coverage      # ~90% line coverage
make lint          # ruff
make smoke         # every pipeline end to end, offline, ~2s
```

The test suite and the smoke pipeline run entirely offline on a bundled
corpus; no downloads, tokens or GPUs are needed to develop here. See
[AGENTS.md](AGENTS.md) for repository conventions (they apply to humans too).

## License

MIT. Datasets referenced by the registry carry their own licenses - check
`minimodel data list` output before publishing a model trained on them.
