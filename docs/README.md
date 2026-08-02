# Documentation

Start with [getting-started.md](getting-started.md) if you are new, or the
[recipes cookbook](recipes.md) if you want copy-paste workflows.

## The pipeline

```
                 tokenizer train
                       |
        data pull -> data tokenize
                       |
                    pretrain ------------------+
                       |                       |
              sft (instruction tuning)      bench / compare
                       |                       |
              cot (reasoning distill)       merge
                       |                       |
          dpo / spin / rlvr (preferences)   card
                       |
                 chat / generate
```

Each arrow is one CLI command, and every stage reads what the previous one
wrote - there is no hidden state between them.

## Pages

### Using the toolkit

| Page | Contents |
| --- | --- |
| [getting-started.md](getting-started.md) | Install, the two quickstarts, your first real model |
| [cli.md](cli.md) | Every command, flag and output format |
| [recipes.md](recipes.md) | End-to-end walkthroughs for common goals |
| [troubleshooting.md](troubleshooting.md) | Loss spikes, NaNs, OOM, gibberish samples |
| [faq.md](faq.md) | Short answers to recurring questions |

### The stages

| Page | Contents |
| --- | --- |
| [tokenization.md](tokenization.md) | Byte-level BPE, vocab sizing, chat template |
| [data.md](data.md) | The registry, formats, mixtures, adding datasets |
| [architecture.md](architecture.md) | The four LM families and how to choose |
| [configuration.md](configuration.md) | Recipe YAML, `extends:`, `--set` overrides |
| [training.md](training.md) | The trainer, optimizers, schedules, resume |
| [post-training.md](post-training.md) | SFT and chain-of-thought distillation |
| [rl.md](rl.md) | DPO, RLVR (GRPO), SPIN |
| [evaluation.md](evaluation.md) | The harness, and reading small-model numbers |
| [checkpointing.md](checkpointing.md) | Checkpoint layout, retention, export |
| [merging.md](merging.md) | The five merge methods |
| [inference.md](inference.md) | Sampling controls, chat, reasoning mode |
| [model-cards.md](model-cards.md) | Generating cards from run artifacts |
| [vision.md](vision.md) | The whole image-model pipeline |

### Background

| Page | Contents |
| --- | --- |
| [../research/README.md](../research/README.md) | Reading lists and design notes |
| [../AGENTS.md](../AGENTS.md) | Repository conventions (for contributors) |
