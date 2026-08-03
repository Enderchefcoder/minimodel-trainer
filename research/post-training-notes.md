# Post-training notes

What the literature says, and what actually transfers below ~100M params.

## SFT

- **InstructGPT** (arXiv:2203.02155) set the template; **LIMA**
  (arXiv:2305.11206) showed 1K excellent examples suffice at 65B. At small
  scale both hold *more* strongly: capacity is scarce, so noisy SFT data does
  visible damage — prefer `smoltalk`/`no-robots` quality over volume.
- Loss masking on prompts is not optional at small scale (the model will
  otherwise learn to generate user turns — measurable in our
  `supervised_frac`/samples).
- Replay (5–15% pretrain batches) is the cheapest forgetting control
  (rehearsal, classic result); `replay_fraction` implements it.

## Chain of thought

- Distilling traces (**Orca**, arXiv:2306.02707; **OpenThoughts**,
  arXiv:2506.04178) works, but trace *length* must fit the student:
  our `reasoning_loss_weight` / `max_length` / answer-only mode exist because
  a <30M model trained on 1K-token traces learns to ramble, not reason.
- **s1: budget forcing** (arXiv:2501.19393) — cap thinking tokens, force the
  close marker, then answer. Implemented as `generate_with_reasoning` +
  `enforce_think_close`.

## Preferences

- **DPO** (arXiv:2305.18290): the workhorse. lr ≈ 100× below SFT — the single
  most common failure is an SFT-sized lr.
- **IPO** (arXiv:2310.12036): non-saturating; better under ~10K pairs.
- **CPO** (arXiv:2401.08417): reference-free + SFT anchor; halves memory —
  relevant when the reference copy doesn't fit.
- **SPIN** (arXiv:2401.01335): SFT data → preference pairs via self-play;
  2–3 iterations then diminishing returns (matches our experiments with the
  builtin corpus).

## Verifiable rewards

- **GRPO** (DeepSeekMath, arXiv:2402.03300; DeepSeek-R1, arXiv:2501.12948):
  group-mean baseline instead of a critic — at our sizes a value head is pure
  noise, so this isn't just simpler, it's *better*.
- k3 KL estimator (Schulman, 2020, joschu.net/blog/kl-approx) — unbiased,
  non-negative, low variance; what `rlvr.py` uses.
- RLVR sharpens, doesn't create: start from an SFT/CoT checkpoint that solves
  >0% of tasks, or every group reward is 0 and the advantage is 0 everywhere.

## Sequencing (the whole point)

pretrain → (domain continue) → SFT → CoT → {DPO | RLVR} → merge back toward
the SFT parent if generality suffered. Each arrow exists as a recipe under
`configs/`.
