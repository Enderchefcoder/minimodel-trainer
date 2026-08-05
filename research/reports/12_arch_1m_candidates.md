# Report 12 — Twenty ~1M architectures to beat Glint-2

*Artifacts: `src/architectures/templates/mm1m_r*.yaml`,
`research/data/results/mm1m_r*.json`,
`research/data/results/arch_bakeoff_merged.json`,
`research/data/results/arch_1m_candidates.json` (smoke only),
`research/experiments/bakeoff_1m_candidates.py`.*

## Goal

Ship **20 architectures at ~1.0–1.2M parameters**, grade them with the **same
protocol as report 03**, and merge with the original five `arch_*` runs.

| Bucket | Count | What |
| --- | ---: | --- |
| Researched / optimized | 10 | Literature + reports 03/04/05/07/11 |
| Novel Transformers | 5 | Original wirings, still attention+FFN |
| Novel Mamba / SSM | 5 | Pure-PyTorch selective SSM compositions |

Glint-2 advertises **1.06M** but ships **1.71M** (report 10). These candidates
target the *advertised* budget; `dense_1_4m` (report 11) remains the GPU
contender matched to Glint's *real* size.

## Protocol (matched to report 03)

| Knob | Value |
| --- | --- |
| Corpus | TinyStories train / val, vocab 4096 byte-BPE |
| Budget | 300 steps × batch 32 × seq 256 = **2.46M tokens** |
| Optim | AdamW lr 3e-3, cosine, warmup 0.05 |
| Eval | BLiMP 15/paradigm, ARC-Easy 150, WikiText 10k tokens |
| Harness | same as Glint-2 / report 03 (`eval_harness.py`) |

```bash
venv/bin/python research/experiments/bakeoff_1m_candidates.py --full
venv/bin/python research/experiments/bakeoff_1m_candidates.py --merge-only
venv/bin/python research/experiments/aggregate.py
```

## Graded mm1m leaderboard (val_loss ↑)

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

### Findings at 2.46M tokens / ~1.05M params

1. **Selective SSM wins val loss** among the ~1M ladder (`mamba_conv_gate` /
   `mamba_pure` 3.282). At this short budget the recurrence fits TinyStories
   CE faster than dense attention — the opposite of report 03's *~1.7M dense
   vs looped* comparison, which is a different architecture family.
2. **`mamba_pure` ≡ `mamba_conv_gate` here**: both use `conv_kernel=4`, so the
   variants are architecturally identical at current templates (tie on every
   metric). Distinguish them later by setting `pure` to `conv_kernel=0`.
3. **Best WikiText byte-ppl and ARC-E among mm1m**: `hybrid_griffin` (14.51 /
   25.3) — strongest encyclopedic / MC signal at this budget.
4. **Dense GQA still solid** (val 3.554, ARC 24.0) and **~5× faster** than
   pure Mamba on CPU; preferred when wall-clock is the constraint (T4 Colab).
5. **Windowed dense ≈ full-window dense** at seq=256 (r01 ties r03): the local
   window never binds.
6. **Looped + Poisson is last on val loss** (5.437), matching report 03's
   finding that looping loses at fixed small token budgets.
7. **Filename `glint2_rank` stays the research prior**; measured ranks live in
   `arch_bakeoff_merged.json` (`mm1m_measured_rank`).

## Merged board: original 5 + 20 mm1m (25 runs)

Sorted by val_loss (then WikiText byte-ppl). Original `arch_*` models are
~1.7M params; mm1m are ~1.0–1.2M — same tokens and harness, different budget.

| Rank | Name | Bakeoff | Params | val↓ | byte-ppl↓ | BLiMP | ARC-E |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `arch_dense` | report03 original | 1,701,152 | **3.266** | 16.28 | 49.9 | 21.3 |
| 2 | `mm1m_r17_mamba_conv_gate` | mm1m report03 | 1,109,104 | 3.282 | 15.70 | 50.0 | 22.0 |
| 3 | `mm1m_r19_mamba_pure` | mm1m report03 | 1,109,104 | 3.282 | 15.70 | 50.0 | 22.0 |
| 4 | `mm1m_r16_mamba_multihead` | mm1m report03 | 1,163,344 | 3.300 | 16.46 | 49.4 | 23.3 |
| 5 | `mm1m_r10_mamba_attn_tail` | mm1m report03 | 1,057,394 | 3.393 | 15.43 | 50.5 | 22.0 |
| 6 | `mm1m_r04_hybrid_griffin` | mm1m report03 | 954,786 | 3.421 | **14.51** | 51.7 | **25.3** |
| 7 | `mm1m_r12_mamba_braid` | mm1m report03 | 1,097,075 | 3.447 | 15.98 | 48.0 | 20.0 |
| 8 | `arch_moe` | report03 original | 2,854,532 | 3.460 | 16.29 | 49.0 | 21.3 |
| 9 | `mm1m_r08_exp_braid` | mm1m report03 | 1,033,584 | 3.549 | 15.76 | 48.0 | 20.7 |
| 10 | `mm1m_r14_dense_wide` | mm1m report03 | 1,117,090 | 3.553 | 16.53 | 49.1 | 23.3 |
| 11 | `mm1m_r01_dense_gqa_vr` | mm1m report03 | 1,033,589 | 3.554 | 16.13 | 49.1 | 24.0 |
| 12 | `mm1m_r03_dense_window` | mm1m report03 | 1,033,589 | 3.554 | 16.13 | 49.1 | 24.0 |
| 13 | `mm1m_r06_exp_kv_inherit` | mm1m report03 | 1,033,609 | 3.562 | 16.08 | 48.9 | 20.7 |
| 14 | `mm1m_r18_dense_novr` | mm1m report03 | 1,033,584 | 3.562 | 16.08 | 48.9 | 20.7 |
| 15 | `mm1m_r20_dense_ffn4x` | mm1m report03 | 1,176,692 | 3.573 | 16.52 | 50.5 | 22.0 |
| 16 | `mm1m_r07_dense_deep` | mm1m report03 | 1,107,824 | 3.606 | 14.91 | 51.0 | 22.7 |
| 17 | `mm1m_r15_moe_micro` | mm1m report03 | 1,071,204 | 3.650 | 15.40 | 51.1 | 21.3 |
| 18 | `mm1m_r05_exp_resimix` | mm1m report03 | 1,035,824 | 3.706 | 15.84 | 49.6 | 18.7 |
| 19 | `mm1m_r02_dense_mha` | mm1m report03 | 1,047,692 | 3.721 | 15.93 | 51.7 | 22.0 |
| 20 | `mm1m_r09_exp_dual_rope` | mm1m report03 | 1,033,624 | 3.723 | 15.22 | 49.6 | 21.3 |
| 21 | `mm1m_r11_exp_echo_ffn` | mm1m report03 | 1,043,109 | 3.978 | 16.41 | 51.5 | 21.3 |
| 22 | `arch_pureloop` | report03 original | 1,765,152 | 4.029 | 17.94 | 52.0 | 20.0 |
| 23 | `arch_loopcoda_glint` | report03 original | 1,710,048 | 4.635 | 18.51 | 52.1 | 16.7 |
| 24 | `arch_supra2` | report03 original | 1,738,016 | 4.992 | 19.21 | 50.6 | 19.3 |
| 25 | `mm1m_r13_loop_poisson` | mm1m report03 | 1,159,829 | 5.437 | 20.20 | 51.5 | 20.0 |

**Merge takeaway:** at equal tokens, the old ~1.7M dense still edges val loss
(3.266 vs 3.282), but several ~1.1M Mamba/hybrid candidates beat it on
WikiText byte-ppl and ARC-E, and beat every looped original. For a fair
*advertised-1M* Glint column, start from measured ranks 1–6.

## Prior ladder (filenames)

`glint2_rank` / `mm1m_rXX_` remains the research-prior try order (report 11
dense-first). See the template YAML `glint2_rank` field or
`list_glint2_candidates()`.

## Novel ideas (unchanged)

**Transformers:** ResiMix, KV-inherit, braid attention, dual-RoPE, echo-FFN.

**Mamba:** attn_tail, braid, multihead, conv_gate, pure (pure currently shares
conv with conv_gate — see finding 2).

## Side fixes shipped with this work

- **RG-LRU**: sequential float32 scan (log-space parallel NaN'd past ~16–32
  tokens). Changes hybrid numerics vs earlier checkpoints.
- **Selective SSM**: custom autograd scan so CPU training is ~7k tok/s instead
  of ~300 tok/s through an unrolled Python graph.

## How to extend

```bash
# re-run one candidate
venv/bin/python research/experiments/bakeoff_1m_candidates.py --full \
  --only mm1m_r04_hybrid_griffin

# rebuild merged table + RESULTS.md
venv/bin/python research/experiments/bakeoff_1m_candidates.py --merge-only
venv/bin/python research/experiments/aggregate.py
```
