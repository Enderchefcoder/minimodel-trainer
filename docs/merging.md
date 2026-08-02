# Model merging

Combining fine-tunes in weight space, no training required. It works because
fine-tunes of the same base stay within one loss basin, so interpolations are
also low-loss - and often better than either endpoint, because independent
fine-tunes make independent errors.

Hard requirement: **same architecture, same base**. Merging models pretrained
from different seeds produces noise.

## The methods

```bash
minimodel merge A B -o out --method slerp --t 0.5
minimodel merge A B C -o out --method linear --weights 0.5 0.3 0.2
minimodel merge FT1 FT2 -o out --method ties --base BASE --density 0.2
```

| Method | Needs base? | Use when |
| --- | --- | --- |
| `linear` | no | 3+ models, or a quick check |
| `slerp` | no | exactly 2 models - the default choice |
| `task_arithmetic` | yes | dialling a capability up/down, or *subtracting* one (negative weight) |
| `ties` | yes | 3+ fine-tunes whose updates conflict |
| `dare` | yes | de-noising deltas; composes with the others |

**Why SLERP over linear for pairs:** averaging two weight tensors of similar
norm but different direction *shrinks* the result, and shrinking every matrix
a few percent measurably damages a model. Spherical interpolation preserves
norms. (1-D tensors - gains, biases - are lerped; they have no meaningful
direction.)

**TIES** = trim (keep top `density` of each delta by magnitude) -> elect a
sign per parameter (by total magnitude) -> average only agreeing entries.
This is the fix for the failure mode of naive averaging at 3+ models, where
opposing updates cancel to nothing. `--density 0.2` is the paper value and
holds up.

**DARE** randomly drops `drop_rate` (default 0.9!) of each delta's entries and
rescales survivors by `1/(1-p)` - the delta's expectation is unchanged, and
fine-tune deltas are redundant enough that this is nearly lossless. Useful
alone as regularisation, or before TIES to reduce conflicts.

## Recipes that earn their keep

```bash
# The classic: recover generality an RL stage traded away
minimodel merge runs/sft/model runs/dpo/model -o merged --method slerp --t 0.6

# Combine specialists into a generalist
minimodel merge runs/math/model runs/chat/model runs/code/model \
    -o generalist --method ties --base runs/base/model --weights 1.0 1.0 1.0

# Remove a behaviour: negative task arithmetic
minimodel merge runs/verbose-ft/model -o terser \
    --method task_arithmetic --base runs/base/model --weights -0.5
```

Always `minimodel bench` the merge against both parents; a merge that beats
neither parent on anything is telling you the parents diverged too far
(common after long RL runs - merge earlier checkpoints instead).

The output directory carries the config of the first input, so it loads,
chats and benches like any trained model.
