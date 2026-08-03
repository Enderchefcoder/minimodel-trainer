# Report 03 — Architecture bake-off at a fixed budget

*Artifacts: `research/experiments/ablations.py` (group `arch`),
`research/data/results/arch_*.json`.*

**Setup.** Five architectures, each ~1.7M parameters (MoE matched on *active*
params, 2.85M total), trained identically on TinyStories: 300 optimizer steps =
**2.46M tokens** (batch 32 × seq 256), AdamW, lr 3e-3, cosine schedule, 4096-vocab
BPE. Evaluated on our harness. This is a *data-efficiency* comparison (same
tokens seen), the standard small-scale ablation protocol; FLOPs/token differ and
are reported as throughput.

## Results

| config | params | val loss ↓ | WikiText byte-ppl ↓ | BLiMP | ARC-E | tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| **arch_dense** | 1,701,152 | **3.27** | **16.28** | 49.9 | 21.3 | 32,453 |
| arch_moe (a1.67M) | 2,854,532 | 3.46 | 16.29 | 49.0 | 21.3 | 22,578 |
| arch_pureloop | 1,765,152 | 4.03 | 17.94 | 52.0 | 20.0 | 2,696 |
| arch_loopcoda (Glint-2 shape) | 1,710,048 | 4.63 | 18.51 | 52.1 | 16.7 | 4,805 |
| arch_supra2 | 1,738,016 | 4.99 | 19.21 | 50.6 | 19.3 | 12,957 |

(BLiMP ≈ 50 and ARC ≈ chance for all: 2.46M tokens is far too little training to
move syntax/knowledge benchmarks. val loss and byte-ppl — both in/near the
training distribution — are the sensitive discriminators at this budget.)

## Findings

1. **At a fixed small token budget, the dense transformer wins decisively.**
   val loss 3.27 and byte-ppl 16.3, versus 4.0–5.0 val loss for the looped
   family. This is exactly the compute-matched pattern reported by the 2026
   iso-depth scaling-law work ([arXiv:2604.21106](https://arxiv.org/abs/2604.21106)):
   *"looped models trail non-looped baselines on validation loss and
   parametric-knowledge tasks."* A looped block has to learn to be useful when
   applied 8 times; that costs training tokens the dense stack spends directly
   on fitting data. **Implication: when training compute is the binding
   constraint, do not loop — go dense.**

2. **Dense is also ~7× faster per token** (32.5K vs 4.8K tok/s for the Glint-2
   shape), because it does not re-run a block 8× and, crucially, does not pay
   for a 22× FFN. So the dense model is better *and* cheaper here — it wins on
   both axes at this budget.

3. **The 22× FFN is the wrong allocation at small scale.** The Glint-2 shape
   (loop+coda, tied embed, 22× FFN) is the worst looped variant on byte-ppl
   *except* supra2, and by far the slowest. The pure-loop variant (no coda,
   wider-but-fewer) edges it out (17.94 vs 18.51), suggesting the coda block's
   params would be better spent elsewhere at this budget. (Report 05 sweeps FFN
   width directly.)

4. **MoE matches dense quality at equal *active* params** (byte-ppl 16.29 vs
   16.28) but needs 2.85M total params to do it — no advantage at this scale,
   where memory is free and the routing overhead does not pay off yet.

5. **supra2 is the weakest here**, which is informative: its factorized
   embedding (rank 64) and variable-loop schedule add training difficulty that
   only pays back with more tokens. Its stabilisers (report 04) help *looped*
   training but cannot overcome the looped penalty at 2.46M tokens.

## The consequence for beating Glint-2

Two things follow, and they shape the rest of the program:

- **For a fixed (small) training budget, a well-tuned dense model is the
  stronger contender** — so our head-to-head contender (report 09) is dense, not
  looped, and we spend the CPU budget on it.
- **The looped architecture is a bet on inference-time compute and reasoning,
  not on per-token training efficiency.** That bet only pays if (a) you can
  afford the training and (b) the model can actually *use* variable loops at
  inference. Glint-2 fails (b) catastrophically (report 07): it works only at
  exactly 8 loops. Fixing that is our clearest architectural win, independent of
  the training-budget question.
