# Report 12 — Twenty ~1M architectures to beat Glint-2

*Artifacts: `src/architectures/templates/mm1m_r*.yaml`,
`research/data/results/arch_1m_candidates.json`,
`research/experiments/bakeoff_1m_candidates.py`.*

## Goal

Ship **20 architectures at ~1.0–1.2M parameters**, add them to the template
ladder in a **single ordered list**, and cover three buckets:

| Bucket | Count | What |
| --- | ---: | --- |
| Researched / optimized | 10 | Literature + reports 03/04/05/07/11 |
| Novel Transformers | 5 | Original wirings, still attention+FFN |
| Novel Mamba / SSM | 5 | Pure-PyTorch selective SSM compositions |

Glint-2 advertises **1.06M** but ships **1.71M** (report 10). These candidates
target the *advertised* budget so a fair leaderboard column is possible; the
locked 1.4M dense recipe (`dense_1_4m`, report 11) remains the GPU contender
matched to Glint's *real* size.

## Ordered ladder (prior rank = try order for Glint-2)

Canonical order is **`glint2_rank` 1→20** (also the `mm1m_rXX_` filename prefix
so `list_templates()` sorts them). Prior comes from the research program, not
from the short offline bake-off.

| Rank | Template | Class | Params | Idea |
| ---: | --- | --- | ---: | --- |
| 1 | `mm1m_r01_dense_gqa_vr` | researched | 1,033,589 | GQA + VR + QK-norm (report 11 shape) |
| 2 | `mm1m_r02_dense_mha` | researched | 1,047,692 | Full MHA (sandbox: ≈GQA) |
| 3 | `mm1m_r03_dense_window` | researched | 1,033,589 | Local/global window pattern |
| 4 | `mm1m_r04_hybrid_griffin` | researched | 954,786 | Griffin RG-LRU hybrid |
| 5 | `mm1m_r05_exp_resimix` | novel-transformer | 1,035,824 | Prenorm/postnorm residual mix |
| 6 | `mm1m_r06_exp_kv_inherit` | novel-transformer | 1,033,609 | Soft mix over all prior values |
| 7 | `mm1m_r07_dense_deep` | researched | 1,107,824 | Thin-deep L=8 |
| 8 | `mm1m_r08_exp_braid` | novel-transformer | 1,033,584 | Odd heads local / even global |
| 9 | `mm1m_r09_exp_dual_rope` | novel-transformer | 1,033,624 | Fast+slow RoPE mix |
| 10 | `mm1m_r10_mamba_attn_tail` | novel-mamba | 1,057,394 | SSM trunk + attention coda |
| 11 | `mm1m_r11_exp_echo_ffn` | novel-transformer | 1,043,109 | Tied double SwiGLU + LoRA bridge |
| 12 | `mm1m_r12_mamba_braid` | novel-mamba | 1,097,075 | Alternating SSM / attention |
| 13 | `mm1m_r13_loop_poisson` | researched | 1,159,829 | Stabilised loop + Poisson sampling |
| 14 | `mm1m_r14_dense_wide` | researched | 1,117,090 | Wide-shallow L=2 |
| 15 | `mm1m_r15_moe_micro` | researched | 1,071,204 | Micro-MoE (~0.85M active) |
| 16 | `mm1m_r16_mamba_multihead` | novel-mamba | 1,163,344 | Multi-head selective SSM |
| 17 | `mm1m_r17_mamba_conv_gate` | novel-mamba | 1,109,104 | Forced short-conv SSM |
| 18 | `mm1m_r18_dense_novr` | researched | 1,033,584 | Dense without value residual |
| 19 | `mm1m_r19_mamba_pure` | novel-mamba | 1,109,104 | Pure SSM stack |
| 20 | `mm1m_r20_dense_ffn4x` | researched | 1,176,692 | ~4× FFN width (report 05) |

```python
from minimodel.architectures import list_glint2_candidates, build_model
for row in list_glint2_candidates():
    print(row["rank"], row["name"], row["params"])
model = build_model("mm1m_r01_dense_gqa_vr")
```

## Novel Transformer ideas (still Transformers)

1. **ResiMix** — convex mix of prenorm residual and a post-normalised copy of
   the same branch via per-channel gates.
2. **KV-inherit** — value residual generalised to a softmax over *all* prior
   layer values (depth skip routing).
3. **Braid attention** — within one layer, even heads see full causal context;
   odd heads use a local window.
4. **Dual-RoPE** — each head (or the layer, under GQA) blends two rotary bases.
5. **Echo-FFN** — shared-weight SwiGLU applied twice with a tiny LoRA bridge
   (per-block mini-loop over the FFN only).

## Novel Mamba ideas (pure PyTorch)

No custom CUDA kernels. Selective scan + optional depthwise causal conv with
an explicit streamable pad state (cache-equivalent).

1. **attn_tail** — SSM trunk, attention coda for retrieval.
2. **braid** — alternate SSM / attention.
3. **multihead** — independent SSM heads over the expanded width.
4. **conv_gate** — forced short-conv Mamba block.
5. **pure** — all-SSM stack.

## Short offline bake-off (relative signal only)

Protocol: builtin pretrain corpus, AdamW 3e-3, 25 steps × batch 8 × seq 128
(~25.6K tokens). **Not** a Glint-2 claim — only a smoke ranking. Full results in
`arch_1m_candidates.json`.

Top measured losses on that toy budget were the Mamba variants and
`dense_wide` / `hybrid_griffin`. That disagrees with the research prior (dense
wins on TinyStories at matched tokens, report 03) and is expected: the bake-off
is too short for parametric knowledge, and SSMs can drop CE quickly on a tiny
repeated builtin set. **Use prior ranks for Glint-2 attack order;** use measured
ranks only as a training-smoke sanity check.

## Side fix

`GatedRecurrentUnit`'s log-space parallel scan NaN'd past ~16–32 tokens
(overflow of `exp(-cumsum(log a))`). It now uses a sequential float32 scan so
hybrid models stay finite and chunked streaming matches full forward. **This
changes hybrid recurrence numerics** relative to earlier checkpoints.

## How to train the ladder

```bash
# regenerate / verify templates
venv/bin/python scripts/generate_templates.py --check

# short offline bake-off
venv/bin/python research/experiments/bakeoff_1m_candidates.py --steps 40

# real attack: start at rank 1 with the crush-glint2 mixture
minimodel train configs/pretrain/crush_glint2_1.4m.yaml \
  --set model.template=mm1m_r01_dense_gqa_vr
```
