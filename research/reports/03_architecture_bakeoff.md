# Report 03 — Architecture bake-off at a fixed budget

*Artifacts: `research/experiments/ablations.py` (group `arch`),
`research/data/results/arch_*.json`, `research/data/results/mm1m_r*.json`,
`research/data/results/arch_bakeoff_merged.json`,
`research/experiments/bakeoff_1m_candidates.py` (report 12 extension).*

**Setup.** Architectures trained identically on TinyStories: 300 optimizer steps
= **2.46M tokens** (batch 32 × seq 256), AdamW, lr 3e-3, cosine schedule,
4096-vocab BPE. Evaluated on our harness (BLiMP 15/paradigm, ARC-Easy 150,
WikiText 10k tokens). This is a *data-efficiency* comparison (same tokens
seen), the standard small-scale ablation protocol; FLOPs/token differ and are
reported as throughput.

Two waves share this protocol:

1. **Original five** (~1.7M params; MoE 2.85M total / ~1.67M active) — this
   report's first cut.
2. **Twenty `mm1m_r*` candidates** (~1.0–1.2M params; report 12) — same
   tokens, harness, and optimizer; merged below into one 25-run board.

## Results — original five (~1.7M)

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

## Findings (original five)

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
   params would be better spent elsewhere at this budget. (Report 05, the
   FFN-width sweep, tests whether the 22× ratio is the right allocation.)

4. **MoE matches dense quality at equal *active* params** (byte-ppl 16.29 vs
   16.28) but needs 2.85M total params to do it — no advantage at this scale,
   where memory is free and the routing overhead does not pay off yet.

5. **supra2 is the weakest here**, which is informative: its factorized
   embedding (rank 64) and variable-loop schedule add training difficulty that
   only pays back with more tokens. Its stabilisers (report 04) help *looped*
   training but cannot overcome the looped penalty at 2.46M tokens.

## Results — twenty ~1M candidates (same protocol)

Graded with `bakeoff_1m_candidates.py --full`. Classes: researched (10),
novel-transformer (5), novel-mamba (5). Sorted by measured val_loss.

| Meas. | Template | Class | Params | val↓ | byte-ppl↓ | BLiMP | ARC-E | tok/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `mm1m_r17_mamba_conv_gate` | novel-mamba | 1,109,104 | **3.282** | 15.70 | 50.0 | 22.0 | 7,193 |
| 2 | `mm1m_r19_mamba_pure` | novel-mamba | 1,109,104 | **3.282** | 15.70 | 50.0 | 22.0 | 7,204 |
| 3 | `mm1m_r16_mamba_multihead` | novel-mamba | 1,163,344 | 3.300 | 16.46 | 49.4 | 23.3 | 6,856 |
| 4 | `mm1m_r10_mamba_attn_tail` | novel-mamba | 1,057,394 | 3.393 | 15.43 | 50.5 | 22.0 | 10,353 |
| 5 | `mm1m_r04_hybrid_griffin` | researched | 954,786 | 3.421 | **14.51** | **51.7** | **25.3** | 23,192 |
| 6 | `mm1m_r12_mamba_braid` | novel-mamba | 1,097,075 | 3.447 | 15.98 | 48.0 | 20.0 | 10,000 |
| 7 | `mm1m_r08_exp_braid` | novel-transformer | 1,033,584 | 3.549 | 15.76 | 48.0 | 20.7 | 16,434 |
| 8 | `mm1m_r14_dense_wide` | researched | 1,117,090 | 3.553 | 16.53 | 49.1 | 23.3 | **56,726** |
| 9 | `mm1m_r01_dense_gqa_vr` | researched | 1,033,589 | 3.554 | 16.13 | 49.1 | 24.0 | 33,758 |
| 10 | `mm1m_r03_dense_window` | researched | 1,033,589 | 3.554 | 16.13 | 49.1 | 24.0 | 32,834 |
| 11 | `mm1m_r06_exp_kv_inherit` | novel-transformer | 1,033,609 | 3.562 | 16.08 | 48.9 | 20.7 | 33,247 |
| 12 | `mm1m_r18_dense_novr` | researched | 1,033,584 | 3.562 | 16.08 | 48.9 | 20.7 | 32,733 |
| 13 | `mm1m_r20_dense_ffn4x` | researched | 1,176,692 | 3.573 | 16.52 | 50.5 | 22.0 | 34,008 |
| 14 | `mm1m_r07_dense_deep` | researched | 1,107,824 | 3.606 | 14.91 | 51.0 | 22.7 | 25,466 |
| 15 | `mm1m_r15_moe_micro` | researched | 1,071,204 | 3.650 | 15.40 | 51.1 | 21.3 | 35,566 |
| 16 | `mm1m_r05_exp_resimix` | novel-transformer | 1,035,824 | 3.706 | 15.84 | 49.6 | 18.7 | 30,288 |
| 17 | `mm1m_r02_dense_mha` | researched | 1,047,692 | 3.721 | 15.93 | **51.7** | 22.0 | 32,968 |
| 18 | `mm1m_r09_exp_dual_rope` | novel-transformer | 1,033,624 | 3.723 | 15.22 | 49.6 | 21.3 | 32,266 |
| 19 | `mm1m_r11_exp_echo_ffn` | novel-transformer | 1,043,109 | 3.978 | 16.41 | 51.5 | 21.3 | 27,304 |
| 20 | `mm1m_r13_loop_poisson` | researched | 1,159,829 | 5.437 | 20.20 | 51.5 | 20.0 | 10,970 |

### Findings at ~1.05M params

1. **Selective SSM leads val loss** among the ~1M ladder (`mamba_conv_gate` /
   `mamba_pure` 3.282). At this short budget the recurrence fits TinyStories
   CE faster than dense attention — a different comparison than the original
   five's dense-vs-looped contrast.
2. **`mamba_pure` ≡ `mamba_conv_gate` here**: both use `conv_kernel=4`, so the
   templates are architecturally identical (tie on every metric). Distinguish
   later with `pure` at `conv_kernel=0`.
3. **Best WikiText byte-ppl and ARC-E among mm1m**: `hybrid_griffin` (14.51 /
   25.3).
4. **Dense GQA remains solid** (val 3.554, ARC 24.0) and **~5× faster** than
   pure Mamba on CPU — preferred when wall-clock binds.
5. **Windowed dense ≈ full attention** at seq=256 (r01 ties r03).
6. **Looped + Poisson is last** (5.437), matching the original five: looping
   loses at fixed small token budgets.

## Merged board — all 25 runs

Same protocol; parameter budgets differ (~1.0–1.2M mm1m vs ~1.7M original).
Sorted by val_loss. Source: `arch_bakeoff_merged.json`.

| Rank | Name | Wave | Class | Params | val↓ | byte-ppl↓ | BLiMP | ARC-E | tok/s |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `arch_dense` | original | original-arch | 1,701,152 | **3.266** | 16.28 | 49.9 | 21.3 | 32,453 |
| 2 | `mm1m_r17_mamba_conv_gate` | mm1m | novel-mamba | 1,109,104 | 3.282 | 15.70 | 50.0 | 22.0 | 7,193 |
| 3 | `mm1m_r19_mamba_pure` | mm1m | novel-mamba | 1,109,104 | 3.282 | 15.70 | 50.0 | 22.0 | 7,204 |
| 4 | `mm1m_r16_mamba_multihead` | mm1m | novel-mamba | 1,163,344 | 3.300 | 16.46 | 49.4 | 23.3 | 6,856 |
| 5 | `mm1m_r10_mamba_attn_tail` | mm1m | novel-mamba | 1,057,394 | 3.393 | 15.43 | 50.5 | 22.0 | 10,353 |
| 6 | `mm1m_r04_hybrid_griffin` | mm1m | researched | 954,786 | 3.421 | **14.51** | 51.7 | **25.3** | 23,192 |
| 7 | `mm1m_r12_mamba_braid` | mm1m | novel-mamba | 1,097,075 | 3.447 | 15.98 | 48.0 | 20.0 | 10,000 |
| 8 | `arch_moe` | original | original-arch | 2,854,532 | 3.460 | 16.29 | 49.0 | 21.3 | 22,578 |
| 9 | `mm1m_r08_exp_braid` | mm1m | novel-transformer | 1,033,584 | 3.549 | 15.76 | 48.0 | 20.7 | 16,434 |
| 10 | `mm1m_r14_dense_wide` | mm1m | researched | 1,117,090 | 3.553 | 16.53 | 49.1 | 23.3 | 56,726 |
| 11 | `mm1m_r01_dense_gqa_vr` | mm1m | researched | 1,033,589 | 3.554 | 16.13 | 49.1 | 24.0 | 33,758 |
| 12 | `mm1m_r03_dense_window` | mm1m | researched | 1,033,589 | 3.554 | 16.13 | 49.1 | 24.0 | 32,834 |
| 13 | `mm1m_r06_exp_kv_inherit` | mm1m | novel-transformer | 1,033,609 | 3.562 | 16.08 | 48.9 | 20.7 | 33,247 |
| 14 | `mm1m_r18_dense_novr` | mm1m | researched | 1,033,584 | 3.562 | 16.08 | 48.9 | 20.7 | 32,733 |
| 15 | `mm1m_r20_dense_ffn4x` | mm1m | researched | 1,176,692 | 3.573 | 16.52 | 50.5 | 22.0 | 34,008 |
| 16 | `mm1m_r07_dense_deep` | mm1m | researched | 1,107,824 | 3.606 | 14.91 | 51.0 | 22.7 | 25,466 |
| 17 | `mm1m_r15_moe_micro` | mm1m | researched | 1,071,204 | 3.650 | 15.40 | 51.1 | 21.3 | 35,566 |
| 18 | `mm1m_r05_exp_resimix` | mm1m | novel-transformer | 1,035,824 | 3.706 | 15.84 | 49.6 | 18.7 | 30,288 |
| 19 | `mm1m_r02_dense_mha` | mm1m | researched | 1,047,692 | 3.721 | 15.93 | 51.7 | 22.0 | 32,968 |
| 20 | `mm1m_r09_exp_dual_rope` | mm1m | novel-transformer | 1,033,624 | 3.723 | 15.22 | 49.6 | 21.3 | 32,266 |
| 21 | `mm1m_r11_exp_echo_ffn` | mm1m | novel-transformer | 1,043,109 | 3.978 | 16.41 | 51.5 | 21.3 | 27,304 |
| 22 | `arch_pureloop` | original | original-arch | 1,765,152 | 4.029 | 17.94 | 52.0 | 20.0 | 2,696 |
| 23 | `arch_loopcoda_glint` | original | original-arch | 1,710,048 | 4.635 | 18.51 | 52.1 | 16.7 | 4,805 |
| 24 | `arch_supra2` | original | original-arch | 1,738,016 | 4.992 | 19.21 | 50.6 | 19.3 | 12,957 |
| 25 | `mm1m_r13_loop_poisson` | mm1m | researched | 1,159,829 | 5.437 | 20.20 | 51.5 | 20.0 | 10,970 |

**Merge takeaway:** at equal tokens, the original ~1.7M dense still edges val
loss (3.266 vs 3.282), but several ~1.1M Mamba/hybrid candidates beat it on
WikiText byte-ppl and ARC-E, and beat every looped original. For an
*advertised-1M* Glint column, start from measured ranks 2–7. Design notes for
the twenty candidates live in [report 12](12_arch_1m_candidates.md).

## The consequence for beating Glint-2

Three things follow, and they shape the rest of the program:

- **For a fixed (small) training budget among ~1.7M transformers, a well-tuned
  dense model is the stronger contender** — so our head-to-head contender
  (report 09 / 11) is dense, not looped, and we spend the GPU budget on it.
- **At the advertised ~1M budget, selective SSM and Griffin hybrids are the
  strongest data-efficiency bets** on this harness; dense GQA remains the
  wall-clock winner on CPU/T4.
- **The looped architecture is a bet on inference-time compute and reasoning,
  not on per-token training efficiency.** That bet only pays if (a) you can
  afford the training and (b) the model can actually *use* variable loops at
  inference. Glint-2 fails (b) catastrophically (report 07): it works only at
  exactly 8 loops. Fixing that is our clearest architectural win, independent of
  the training-budget question.

```bash
# re-run / merge the mm1m wave
venv/bin/python research/experiments/bakeoff_1m_candidates.py --full
venv/bin/python research/experiments/bakeoff_1m_candidates.py --merge-only
venv/bin/python research/experiments/aggregate.py
```
