# Training

How pretraining works here, and what every knob does.

## The shape of a run

Training is **step-driven**: you specify optimizer steps, not epochs, because
a pretraining corpus is effectively infinite relative to your compute. One
step is:

```
grad_accum_steps x (forward + backward on [batch_size, seq_len])
-> clip -> optimizer step -> scheduler step
```

so tokens/step = `batch_size * seq_len * grad_accum_steps` (the trainer logs
this as `tokens_per_step`, and `planned_tokens = tokens_per_step * max_steps`
is written into the run metadata - check it against your data budget before
launching).

## A recipe, annotated

```yaml
run_name: dense-30m
tokenizer: artifacts/tokenizer.json     # vocab_size flows into the model automatically

model:
  template: dense_30m                   # or checkpoint: runs/prev/model to continue
  overrides: { max_seq_len: 4096 }      # any flat config key

data:
  mixture:                              # or train: <one corpus dir>
    - { path: data/tokenized/fineweb-edu, weight: 0.7 }
    - { path: data/tokenized/cosmopedia,  weight: 0.3 }
  eval: data/tokenized/heldout

training:
  max_steps: 100000
  batch_size: 24
  seq_len: 1024
  grad_accum_steps: 11                  # ~264K tokens/step

  optimizer: muon                       # adamw | muon | lion | sgd
  lr: 1.2e-3
  weight_decay: 0.1                     # matrices only; norms/biases never decay
  grad_clip: 1.0

  lr_schedule: wsd                      # cosine | wsd | linear | constant | inverse_sqrt
  warmup: 0.005                         # fraction of max_steps, or an absolute count
  min_lr_ratio: 0.1

  device: auto                          # cuda > mps > cpu
  dtype: auto                           # bf16 on capable GPUs, else fp32
  compile: true                         # torch.compile; worth it past ~1K steps

  log_every: 20
  eval_every: 2000
  save_every: 5000
  keep_last: 3                          # checkpoint retention
  keep_best: 1                          #   + best by val_loss
  seed: 1337
  resume: true                          # picks up the latest checkpoint automatically

sample_prompt: "The river runs"          # prints a sample at every eval
```

Recipes support `extends:` (see [configuration.md](configuration.md)) and any
key can be overridden at launch: `--set training.lr=8e-4`.

## Optimizers

**AdamW** - the default. `betas=(0.9, 0.95)`, decay on matrices only.

**Muon** - orthogonalises the momentum of 2D parameters with a Newton-Schulz
iteration; typically reaches a target loss in noticeably fewer steps than
AdamW on small transformers. It only applies to matrices, so
`build_optimizer` automatically pairs it with AdamW for embeddings, the LM
head, norms and gates (`optimizer_kwargs.adamw_lr` controls that side). Muon
learning rates look alarming - `0.02` is normal (it is scale-invariant in a
way AdamW is not).

**Lion** - sign-momentum; half the optimizer memory of AdamW. Use ~5-10x lower
lr and ~10x higher weight decay than AdamW.

MoE note: keep the router on AdamW (i.e. use `optimizer: adamw` for MoE
models). Orthogonalised updates on a tiny router matrix destabilise routing.

## Schedules

**cosine** - warmup then cosine to `min_lr_ratio`. Right when you know the
budget.

**wsd** (warmup-stable-decay) - warmup, long flat plateau, short decay
(`decay_ratio: 0.1`, shape `sqrt` slightly beats linear). Two properties make
it the better default for exploratory work: you can extend the plateau after
starting (the schedule does not bake in the end date), and any plateau
checkpoint is a valid branch point for a decayed fine-tune.

**inverse_sqrt** - scale-free; for open-ended runs.

Warmup: 1-2% of steps. Too little warmup at high lr is the classic source of
step-50 divergence.

## Precision

`dtype: auto` selects bf16 autocast on GPUs that support it, else fp32. fp16
is supported (with GradScaler, automatically) but never auto-selected - at
these scales the debugging cost of fp16 overflow exceeds its speedup. The
optimizer states and master weights are always fp32.

## What gets logged

`metrics.jsonl` gains one row per step: `loss`, `lr`, `grad_norm`,
`tokens_per_second`, `total_tokens`, and per-trainer extras (token accuracy
for SFT, reward stats for RL). Watch a run:

```bash
minimodel plot runs/dense-30m               # ASCII in the terminal
minimodel plot runs/dense-30m -o loss.png   # PNG with EMA smoothing
tail -f runs/dense-30m/train.log
```

Grad-norm reading: a healthy run settles into a fairly flat band after
warmup. A slow upward drift late in training usually means the lr floor
(`min_lr_ratio`) is too high; spikes correlated with loss spikes mean data or
lr problems - see [troubleshooting.md](troubleshooting.md).

## Resume semantics

`resume: true` finds the latest checkpoint in the run directory and restores
model, optimizer moments, scheduler position, GradScaler state **and RNG
state** - data order continues exactly where it stopped, so a resumed curve is
indistinguishable from an uninterrupted one. Checkpoints also store the token
count, so throughput accounting survives restarts.

Retention: `keep_last` most recent plus `keep_best` by `monitor` (default
`val_loss`); everything else is pruned automatically.

## Multi-GPU

The trainer wraps the model in DDP when launched under `torchrun`:

```bash
torchrun --nproc_per_node=4 -m minimodel.cli train -c configs/pretrain/dense_125m.yaml
```

Rank 0 owns logging, checkpoints and eval; per-rank seeds are offset so data
differs across ranks. Tokens/step scales by world size - divide
`grad_accum_steps` accordingly if you were saturating a global batch size.

## Callbacks

`ConsoleLogger` and `GradientMonitor` (halts on NaN or a 4x loss explosion -
preserving the last good checkpoint instead of burning the night) are always
on. Optional: `EarlyStopping` (via `early_stopping_patience`),
`SampleGenerator` (via `sample_prompt`). Custom behaviour is a subclass with
any of `on_train_begin/end`, `on_step_begin/end`, `on_evaluate`,
`on_checkpoint` - pass instances to `Trainer(callbacks=[...])`.

## Custom objectives

Subclass and override one method:

```python
class MyTrainer(Trainer):
    def compute_loss(self, batch):
        logits = self.raw_model(batch["input_ids"])
        loss = my_objective(logits, batch["labels"])
        return loss, {"my_metric": float(...)}   # extras land in metrics.jsonl
```

Everything else - AMP, accumulation, clipping, schedules, checkpoints, resume,
logging - is inherited. This is exactly how the SFT, CoT, DPO, RLVR and SPIN
trainers are built; the largest of them overrides ~60 lines.
