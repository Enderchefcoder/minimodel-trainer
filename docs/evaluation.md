# Evaluation

## Running it

```bash
# Bundled sanity tasks + throughput (works offline):
minimodel bench -m runs/dense-30m/model -o bench.json

# Real benchmarks: pull them, then point at the JSONL
minimodel data pull hellaswag && minimodel data pull arc-easy && minimodel data pull blimp
minimodel bench -m runs/dense-30m/model \
    --task data/raw/hellaswag.jsonl data/raw/arc-easy.jsonl \
    --task-kind multiple_choice \
    --perplexity-corpus data/tokenized/wikitext \
    -o bench.json

minimodel compare bench_a.json bench_b.json          # markdown table, best bolded
```

## How scoring works (and why)

Everything that *can* be scored by log-likelihood *is*:

- **multiple_choice** (ARC, HellaSwag, PIQA, WinoGrande layouts auto-detected):
  each choice is scored by the log-probability of its tokens given the
  question; highest wins. Both raw `accuracy` and length-normalised
  `accuracy_norm` are reported - quote the normalised one, raw is biased
  toward short options.
- **minimal_pairs** (BLiMP): is the grammatical sentence more likely than the
  ungrammatical one?
- **perplexity**: sequential non-overlapping windows, every token scored once.
  Only comparable between models sharing a tokenizer; `bits_per_token` is the
  tokenizer-robust-ish alternative.
- **generation**: free-form decoding checked by a verifier (the RLVR
  verifiers: numeric, boxed, exact, expression). The only kind that needs
  sampling.

Likelihood scoring matters at small scale: a 20M base model may *know* "the
cat sleeps" beats "the cat sleep" while completely failing to answer the same
thing posed as an instruction. Likelihood measures what the model learned;
generation additionally measures prompt-format compliance. Keep the two
separate in your head.

## Reading small-model numbers

Chance is 0.25 on ARC/HellaSwag, 0.5 on PIQA/WinoGrande/BLiMP (the harness
reports `chance` per task). Expectations by size, trained on good data:

| Benchmark | 5M | 30M | 125M |
| --- | --- | --- | --- |
| BLiMP | 0.60-0.70 | 0.72-0.80 | 0.80+ |
| HellaSwag (norm) | ~0.26 | 0.28-0.31 | 0.33-0.38 |
| ARC-Easy (norm) | ~0.26 | 0.30-0.36 | 0.40-0.48 |
| WinoGrande | ~0.50 | ~0.51 | 0.52-0.55 |

Two practical rules follow. **BLiMP is the benchmark that discriminates under
50M** - it measures syntax, which small models actually acquire, while
knowledge benchmarks sit near chance and mostly measure noise. And **a +-1
point move on a 1000-item benchmark is noise** (binomial std dev ~1.5 points
at p=0.3); rerun with `--limit 0` removed and both seeds before believing a
regression.

## Throughput

`bench` also measures prefill and decode separately (they are bound by
different resources - compute vs memory bandwidth), reporting
tokens/second, ms/token and peak memory. `--no-throughput` skips it.

## Comparing

```bash
minimodel compare a.json b.json c.json -o report.md   # models on shared tasks
minimodel compare runs/run1 runs/run2 --runs          # training runs via metrics.jsonl
```

Tables bold the best value per column (direction-aware: perplexity down,
accuracy up; identity columns like `params` are never highlighted). For a size
ladder, `minimodel.benchmarking.pareto_frontier` answers "which of these sizes
is actually worth training" and `plot_scaling_curve` draws the log-log quality
curve - a knee in it almost always means one point was under-trained, not that
scaling broke.

## In code

```python
from minimodel.benchmarking import run_suite, load_task

tasks = [load_task("hellaswag", "data/raw/hellaswag.jsonl", "multiple_choice", limit=500)]
result = run_suite(model, tokenizer, tasks=tasks, perplexity_corpus="data/tokenized/val")
result.headline()   # {"hellaswag": 0.287, ...}
result.save("bench.json")
```

For looped models, evaluate at multiple depths by passing
`model_kwargs={"loops": k}` - the quality-vs-compute curve across `loops` is
the most interesting single plot that architecture produces.
