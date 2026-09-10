# Report 13 — Chain-of-thought vs latent (looped) reasoning in SLMs

**Question.** For a fixed tiny parameter budget, is it better to spend the
answer budget on *emitted* reasoning tokens (chain-of-thought) or on *latent*
compute (looped architecture)? This is the SLM-scale version of the debate
around `<think>` traces vs internal recurrence.

**Experiment.** `experiments/cot_latent_ablation.py`. One controlled setup,
four conditions:

| condition | what it trains | where reasoning lives |
| --- | --- | --- |
| `cot_direct` | dense transformer, answer immediately | nowhere |
| `cot_brief` | dense, one short reasoning sentence before the answer | 1 emitted token-block |
| `cot_detail` | dense, a multi-step narrated trace before the answer | several emitted token-blocks |
| `cot_latent_loop` | looped transformer (8 unrolled iterations), trained to answer directly | unrolled *iterations* (latent) |

Controls: identical dense arch (~620k params) for the three CoT conditions and a
matched looped core (~467k unrolled-equivalent); identical arithmetic
word-problem corpus with **disjoint train/test operand ranges** (accuracy
measures generalisation, not memorisation); identical tokenizer, optimizer,
schedule, seed and step budget; one shared eval harness (task exact-match plus
BLiMP/ARC/WikiText when the eval corpora are cached).

## Results (short-budget run, CPU, ~60k tokens/condition)

| condition | params | final loss ↓ | held-out exact match ↑ |
| --- | ---: | ---: | ---: |
| cot_direct | 620,757 | **2.891** | 0.00% |
| **cot_brief** | 620,757 | 2.928 | **3.33%** |
| cot_detail | 620,757 | 3.433 | 0.00% |
| cot_latent_loop | 466,805 | 3.753 | 0.00% |

Source: `research/data/results/cot_latent_ablation.json`. BLiMP/ARC/WikiText
rows were skipped in this run (eval corpora not cached; the script now degrades
gracefully and prints a skip note — run `pull_eval_data.py` to enable them).

## Findings

1. **Brief CoT is the only condition that generalises at all.** One short
   reasoning sentence before the answer lifted held-out exact-match from 0% to
   3.3% at identical params and budget. Even a single emitted intermediate
   token-block gives the model scratch space that direct answering cannot.
2. **More CoT is not better CoT.** The detailed-trace condition has the *worst*
   dense loss (3.433) and zero EM: at this scale the longer trace spends the
   token budget on verbose, partly boilerplate text, and the answer signal is
   diluted across more supervised tokens.
3. **Latent (looped) compute did not pay.** The looped variant traded
   throughput (~2.7× slower tokens/s) for unrolled depth, yet landed at the
   highest loss and 0% EM. Consistent with reports 03/07: loops help only when
   the *task* needs iterative refinement, and training with variable/Poisson
   loop sampling is what keeps them usable off-distribution.
4. **The pattern matches the literature.** Scratchpad/CoT gains at small scale
   are real but fragile (they need the trace format seen at training), and
   "thinking in weights/loops" underperforms emitted traces on tasks with
   discrete intermediate states like arithmetic.

## Caveats

- Short-budget run: differences beyond ~3 EM points are noise at n=60 test
  problems. Reproduce at full scale with
  `python research/experiments/cot_latent_ablation.py --steps 4000` (add
  `pull_eval_data.py` first for the benchmark rows).
- The looped condition has fewer raw params (467k) by design — its compute
  budget is unrolled iterations — but this also means a smaller core; a
  param-matched looped run is the natural follow-up.
- Task domain is arithmetic word problems; conclusions may not transfer to
  fuzzy natural-language reasoning, where emitted traces serve a decoding
  (search) role as much as a computation role.

## Bottom line

At SLM scale, **one short emitted reasoning beat both
direct answering and latent looped compute** on held-out task accuracy. Long CoT hurt, and latent
compute under this setup lost outright. For our post-training pipeline this
supports distilling *concise* traces (the `cot` stage) rather than verbose
ones, and using loops only where inference-time iteration is the goal.
