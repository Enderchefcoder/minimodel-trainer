# Preference optimization and RL

Three methods, one decision rule:

| You have | Use |
| --- | --- |
| Preference pairs (chosen vs rejected) | **DPO** |
| Only SFT data, want preference-style gains | **SPIN** |
| Tasks with checkable answers (math, code, puzzles) | **RLVR** |

All three subclass the same `Trainer`, so recipes, logging, checkpoints and
resume work identically to every other stage.

## DPO

```bash
minimodel data pull ultrafeedback-binarized --limit 20000
minimodel data tokenize ultrafeedback-binarized -t artifacts/tokenizer.json \
    --format preference -o data/tokenized/prefs
minimodel posttrain --config configs/rl/dpo.yaml
```

DPO removes RLHF's reward model: the reward is re-expressed through the policy
itself, and fitting preferences becomes classification on pairs:

```
L = -log sigmoid(beta * [(log pi/pi_ref)(chosen) - (log pi/pi_ref)(rejected)])
```

A frozen copy of the starting policy (`pi_ref`) anchors the update - without
it the objective is maximised by degenerate outputs.

**The knobs that matter:**

- `lr: 5e-7`. Not a typo: DPO rates are ~100x below SFT. The most common DPO
  failure is simply an SFT-sized learning rate.
- `beta: 0.1` - the strength of the pull toward the reference. Lower = more
  movement. If `reward_margin` grows but samples get worse, raise it.
- `loss_type`:
  - `sigmoid` - standard DPO.
  - `ipo` - squared loss that does not saturate; overfits less on small pair
    sets (< ~10K pairs). Use `beta: 0.3`.
  - `hinge` - most robust to label noise.
  - `cpo` - reference-free (+ an SFT term via `sft_weight`); halves memory.
- `length_normalize: true` if your chosen responses are systematically shorter
  than rejected ones, or the model just learns "short is good".
- `label_smoothing` (conservative DPO) for noisy preference labels.

**The diagnostic to watch:** `reward_accuracy` - the fraction of pairs the
implicit reward ranks correctly. It should pass 0.6 within a few hundred
steps. Flat at 0.5 means the pairs are inconsistent or the lr is too low;
1.0 almost immediately means the pairs are trivially separable (or leaked).

## RLVR (GRPO)

```bash
minimodel data pull gsm8k-rlvr
minimodel posttrain --config configs/rl/rlvr_gsm8k.yaml
```

RLVR replaces the learned reward with a **program**: extract the model's final
number and compare to ground truth. Nothing to train, nothing for the policy
to hack.

The optimizer is **GRPO**: sample `group_size` completions per prompt and use
the group's own mean as the baseline -

```
A_i = (r_i - mean(r)) / (std(r) + eps)
```

- so there is no value network (at 30M parameters a value head is noise
anyway). The loss is the PPO clipped-ratio objective with a KL penalty toward
the frozen reference (the low-variance k3 estimator).

**Verifiers** (`training.verifier`, or per-task in the JSONL): `numeric` (last
number match - GSM8K), `latex_answer` (`\boxed{...}`), `exact_match`,
`expression` (evaluates arithmetic safely - Countdown), `length`. Add your own
with `@VERIFIERS.register("name")` on a `(completion, reference) -> float`
function.

**Reading the logs:** `solve_rate` on greedy eval is the truth; training
`reward_mean` is inflated by sampling. `kl` creeping up while solve rate is
flat means the model is drifting without learning - raise `kl_coefficient`.
Groups where every sample gets the same reward contribute zero gradient
(correctly), so if `reward_mean` is 0.0 *or* 1.0, there is no signal: pick
easier/harder tasks or start from a stronger SFT checkpoint. `format_bonus:
0.01` gives a model gradient before its first correct answer.

Practicalities: `batch_size x group_size` full generations happen per step -
this is by far the slowest stage per step. `grad_clip: 0.2`, `lr: 1e-6`.

## SPIN

```bash
minimodel posttrain --config configs/rl/spin.yaml
```

Self-Play fIne-tuNing turns an SFT dataset into preference data with no labels
and no reward: at iteration *t*, the previous iterate generates a response to
each prompt; the human answer is "chosen", the model's own is "rejected", and
a DPO loss pushes toward the former. The negatives are exactly the model's
current failure modes, which is why a SPIN pass beats another SFT epoch on the
same data.

- `iterations: 2-3`. Beyond that the model's samples are close enough to the
  data that the gradient vanishes.
- `sample_temperature: 0.9` - negatives sampled too cold are degenerate and
  trivially separable (no signal); too hot and they are noise.
- Negatives are regenerated at every iteration boundary and the reference
  resets to the current policy - both automatic. Pair files are kept under
  `runs/<name>/pairs/iteration_XXX.jsonl` for inspection; reading a few is the
  best way to see what the model is being pushed away from.

## Sequencing

`SFT -> DPO` is the standard assistant pipeline. `SFT -> CoT -> RLVR` is the
reasoning pipeline. SPIN slots in wherever you would have run a second SFT
epoch. Merging (see [merging.md](merging.md)) an RL checkpoint back into its
SFT parent at 0.3-0.5 is a cheap way to trade a little of the RL gain for
recovered generality.
