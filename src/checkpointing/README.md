# checkpointing/

- `checkpointing.py` — `CheckpointManager`: directory-per-step checkpoints
  (weights separate from optimizer/RNG state), keep-last + keep-best
  retention, bit-exact resume, weights-only export for shipping.
- `etr.py` — sliding-window ETA (`ETREstimator`) and tokens/s
  (`ThroughputMeter`); `estimate_training_time` for pre-launch planning.
- `loss_visualization.py` — `metrics.jsonl` → PNG (matplotlib) or ASCII plots
  (SSH-friendly), plus `summarize_run` for compare/cards.

Docs: [docs/checkpointing.md](../../docs/checkpointing.md).
