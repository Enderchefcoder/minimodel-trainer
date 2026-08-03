# Checkpointing

## Layout

A checkpoint is a directory, not a file:

```
runs/my-run/checkpoints/step_010000/
  model.pt        weights only (loads with weights_only=True)
  config.json     architecture config + family, so it reloads with no recipe
  trainer.pt      optimizer moments, scheduler, GradScaler, step, RNG state
  metrics.json    metrics at save time (used for best-k retention)
```

Weights are separated from trainer state so that *shipping* a model is
copying two small files, while *resuming* restores everything - including RNG
state, so the data order continues bit-exactly and a resumed loss curve is
indistinguishable from an uninterrupted one.

Security note: `model.pt` always loads with `weights_only=True`;
`trainer.pt` cannot (it holds RNG/optimizer objects), so only resume from
checkpoints you produced.

## Retention

`keep_last: 3` most recent + `keep_best: 1` by the monitored metric
(`monitor: val_loss`); the rest are pruned after every save. `keep_last: 0`
disables pruning. The manager rescans its directory on construction, so
retention survives restarts and manual deletion.

## Export

```bash
# from code: trainer.export("release/")     - best (else latest) checkpoint
# result: model.pt + config.json + metrics.json (+ tokenizer.json via recipes)
```

Everything downstream - `bench`, `chat`, `generate`, `merge`, `card` - accepts
these exported directories.

## ETA and throughput

`ETREstimator` predicts completion from a sliding window of recent step times
(the first steps are excluded - compilation and cold caches make them
unrepresentative, and averaging them in keeps the ETA wrong for hours).
`ThroughputMeter` tracks tokens/s the same way. Both feed the console line:

```
step 12000 | loss 2.4173 | lr 8.31e-04 | ... | eta 2h 14m 3s (1.42 it/s)
```

`estimate_training_time(steps, tokens_per_step, tokens_per_sec)` answers "can
I finish this overnight" *before* launching, from a 100-step benchmark run.

## Loss curves

```bash
minimodel plot runs/my-run                      # ASCII plot in the terminal
minimodel plot runs/my-run -o loss.png          # PNG, EMA-smoothed, raw shown faint
minimodel plot runs/my-run --keys lr grad_norm  # any logged series
```

The ASCII path is deliberate, not a fallback: checking a run over SSH should
not require forwarding X or copying files. `summarize_run(path)` produces the
final/best loss, perplexity, token count and duration dictionary that
`compare --runs` and the model-card generator consume.
