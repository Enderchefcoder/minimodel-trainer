# training/

One loop, many objectives. `trainer.py` owns stepping, accumulation, AMP,
clipping, schedules, eval, checkpoints, resume, callbacks and metrics; every
stage subclasses it and overrides `compute_loss`:

- `instruct_posttrainer.py` — SFT (masked loss, replay mixing, token accuracy).
- `instruct_cot_posttrainer.py` — CoT distillation (reasoning-span weighting,
  think-close enforcement).
- `rl/` — DPO, RLVR (GRPO), SPIN. See its README.
- `optim.py` — AdamW / **Muon** (Newton-Schulz) / Lion / SGD + param grouping.
- `schedules.py` — cosine / **WSD** / linear / constant / inverse-sqrt.
- `callbacks.py` — console logging, divergence guard, early stopping, sampling.
- `recipe.py` / `post_train.py` — YAML recipe runners for `minimodel train`
  and `minimodel posttrain`.

Docs: [training.md](../../docs/training.md),
[post-training.md](../../docs/post-training.md).
