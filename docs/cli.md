# CLI reference

```
minimodel <command> [<subcommand>] [flags]
```

Global conventions: every command is non-interactive (except `chat`), exits
non-zero on failure, accepts `--json` for machine-readable output, `--set
key=value` (repeatable) for config overrides, and `--log-level`. `mmt` is an
alias for `minimodel`.

## tokenizer

```
minimodel tokenizer train [--input FILES... | --dataset NAME] [--limit N]
    [--vocab-size 4096] [--min-frequency 2] [--backend auto|python|fast]
    [-o artifacts/tokenizer.json]
minimodel tokenizer inspect TOKENIZER [--text STR | --ids N N ...]
```

`train` with no input uses the bundled demo corpus. `inspect --text` shows
ids, token pieces and a roundtrip check.

## data

```
minimodel data list [--stage pretrain|sft|cot|preference|rlvr|eval]
minimodel data pull NAME [-o data/raw] [--limit N] [--overwrite]     # dataset or mixture
minimodel data tokenize [NAME] [--input FILE] [-t TOKENIZER] [-o DIR]
    [--format auto|text|chat|instruction|preference] [--limit N] [--max-length N]
minimodel data info DIR
```

`tokenize` gets the format from the registry when given a name (`auto`).
`info` prints token counts, dtype, supervision and size for a shard
directory.

## train / posttrain

```
minimodel train -c RECIPE [--set training.lr=3e-4 ...]
minimodel posttrain -c RECIPE [--set ...]        # stage: sft|cot|dpo|spin|rlvr|continue
```

Multi-GPU: `torchrun --nproc_per_node=N -m minimodel.cli train -c RECIPE`.

## generate / chat

```
minimodel generate -m MODEL -p PROMPT [--max-new-tokens 128] [--temperature .8]
    [--top-k 0] [--top-p .9] [--min-p 0] [--repetition-penalty 1.1] [--seed N]
    [--chat] [--completion-only] [--loops N] [--device auto] [-t TOKENIZER]
minimodel chat -m MODEL [--system STR] [--temperature .8] [--max-new-tokens 200]
    [--no-stream]
```

`chat` REPL commands: `/reset`, `/exit`.

## bench / compare / plot

```
minimodel bench -m MODEL [--task FILES...] [--task-kind multiple_choice|minimal_pairs|generation|perplexity]
    [--perplexity-corpus DIR] [--limit N] [--no-throughput] [--name STR] [-o out.json]
minimodel compare PATHS... [--runs] [--title STR] [-o report.md|.json|.csv]
minimodel plot RUN [-o curve.png] [--keys loss val_loss ...] [--smoothing 0.9]
```

`bench` with no `--task` runs the bundled offline sanity tasks. `compare
--runs` reads `metrics.jsonl` from run directories instead of benchmark
JSONs. `plot` without `-o` still writes a PNG when matplotlib exists,
otherwise prints ASCII.

## merge / card / models / quickstart

```
minimodel merge MODELS... --method linear|slerp|task_arithmetic|ties|dare
    [-o DIR] [--weights W...] [--base MODEL] [--t 0.5] [--density 0.2] [--drop-rate 0.9]
minimodel card [-o README.md] [--run DIR] [--model DIR] [--benchmark FILES...]
    [--name STR] [--org STR] [--license SPDX] [--dataset NAMES...]
    [--stage base|instruct|reasoning] [--base-model STR]
minimodel models [--verify]
minimodel quickstart [-o runs/quickstart] [--steps 200] [--template dense_3m]
    [--vocab-size 1024] [--seq-len 128] [--no-sft]
```

## vision

```
minimodel vision models [--verify]
minimodel vision datasets [--kind generation|pixelart|edit|classification]
minimodel vision data prepare (-o DIR) [--input DIR | --dataset NAME | --synthetic]
    [--size 24] [--mode rgb|palette] [--palette-size 64]
    [--palette-method auto|exact|median_cut|kmeans] [--limit N]
minimodel vision data info DIR
minimodel vision train -c RECIPE                  # kind: diffusion|pixelgpt|edit|vae
minimodel vision sample -m MODEL [-n 16] [-o samples.png] [--sampler euler|heun|ddim|ddpm]
    [--steps 50] [--guidance 1.0] [--label N] [--prompt STR --tokenizer TOK]
    [--temperature .9] [--top-p .9] [--palette FILE] [--scale 4] [--seed N]
minimodel vision edit -m MODEL -i IMAGE --instruction STR [-t TOKENIZER]
    [--guidance 3.0] [--steps 50] [-o edited.png]
minimodel vision quickstart [-o DIR] [--steps 200] [--sprites 256] [--size 16]
```

`sample` auto-detects PixelGPT vs diffusion models and switches sampling
accordingly (palette + temperature vs ODE steps + guidance).

## Exit codes

`0` success; `1` any handled error (missing file, bad config, unknown name -
message on stderr); `130` interrupted.
