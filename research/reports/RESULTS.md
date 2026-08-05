# Experiment results (auto-aggregated)

_Generated from 34 result files. Glint-2 baseline included where relevant._

## Baseline

| name | params | blimp_acc | arc_easy_acc | wikitext_byte_ppl |
| --- | --- | --- | --- | --- |
| glint-2 | 1,710,049 | 66.36 | 36.78 | 3.179 |

## arch

| name | params | val_loss | wikitext_byte_ppl | blimp_acc | arc_easy_acc | arc_easy_acc_norm | tokens_per_second |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arch_dense | 1,701,152 | **3.266** | **16.28** | 49.85 | **21.33** | **23.33** | **3.245e+04** |
| arch_loopcoda_glint | 1,710,048 | 4.635 | 18.51 | **52.14** | 16.67 | 20 | 4805 |
| arch_moe | 2,854,532 | 3.46 | 16.29 | 49.05 | 21.33 | 21.33 | 2.258e+04 |
| arch_pureloop | 1,765,152 | 4.029 | 17.94 | 52.04 | 20 | 20.67 | 2696 |
| arch_supra2 | 1,738,016 | 4.992 | 19.21 | 50.65 | 19.33 | 22.67 | 1.296e+04 |

## contender

| name | params | val_loss | wikitext_byte_ppl | blimp_acc | arc_easy_acc | arc_easy_acc_norm | tokens_per_second |
| --- | --- | --- | --- | --- | --- | --- | --- |
| contender_dense | 1,701,152 | 2.021 | 16.27 | 57.52 | 26.6 | 25.34 | 2.531e+04 |

## ffn

| name | params | val_loss | wikitext_byte_ppl | blimp_acc | arc_easy_acc | arc_easy_acc_norm | tokens_per_second |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ffn_16x | 1,378,272 | **4.508** | 17.84 | 52.34 | **20.67** | 20 | 6164 |
| ffn_22x | 1,710,048 | 4.635 | 18.51 | 52.14 | 16.67 | 20 | 4825 |
| ffn_4x | 714,720 | 4.521 | **16.59** | 49.85 | 19.33 | **22** | **1.701e+04** |
| ffn_8x | 935,904 | 4.703 | 19.55 | **53.03** | 16.67 | 20.67 | 1.346e+04 |

## glint-2 (loops=8)

| name | params | wikitext_byte_ppl | blimp_acc | arc_easy_acc | arc_easy_acc_norm |
| --- | --- | --- | --- | --- | --- |
| glint-2 (loops=8) | 1,710,049 | 3.179 | 66.36 | 36.78 | 37.25 |

## mm1m

| name | params | val_loss | wikitext_byte_ppl | blimp_acc | arc_easy_acc | arc_easy_acc_norm | tokens_per_second |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mm1m_r01_dense_gqa_vr | 1,033,589 | 3.554 | 16.13 | 49.15 | 24 | 22.67 | 3.376e+04 |
| mm1m_r02_dense_mha | 1,047,692 | 3.721 | 15.93 | **51.74** | 22 | 21.33 | 3.297e+04 |
| mm1m_r03_dense_window | 1,033,589 | 3.554 | 16.13 | 49.15 | 24 | 22.67 | 3.283e+04 |
| mm1m_r04_hybrid_griffin | 954,786 | 3.421 | **14.51** | 51.74 | **25.33** | **24** | 2.319e+04 |
| mm1m_r05_exp_resimix | 1,035,824 | 3.706 | 15.84 | 49.65 | 18.67 | 20.67 | 3.029e+04 |
| mm1m_r06_exp_kv_inherit | 1,033,609 | 3.562 | 16.08 | 48.86 | 20.67 | 20 | 3.325e+04 |
| mm1m_r07_dense_deep | 1,107,824 | 3.606 | 14.91 | 51.04 | 22.67 | 20.67 | 2.547e+04 |
| mm1m_r08_exp_braid | 1,033,584 | 3.549 | 15.76 | 47.96 | 20.67 | 22.67 | 1.643e+04 |
| mm1m_r09_exp_dual_rope | 1,033,624 | 3.723 | 15.22 | 49.65 | 21.33 | 20 | 3.227e+04 |
| mm1m_r10_mamba_attn_tail | 1,057,394 | 3.393 | 15.43 | 50.45 | 22 | 21.33 | 1.035e+04 |
| mm1m_r11_exp_echo_ffn | 1,043,109 | 3.978 | 16.41 | 51.54 | 21.33 | 21.33 | 2.73e+04 |
| mm1m_r12_mamba_braid | 1,097,075 | 3.447 | 15.98 | 47.96 | 20 | 20 | 1e+04 |
| mm1m_r13_loop_poisson | 1,159,829 | 5.437 | 20.2 | 51.54 | 20 | 18 | 1.097e+04 |
| mm1m_r14_dense_wide | 1,117,090 | 3.553 | 16.53 | 49.15 | 23.33 | 23.33 | **5.673e+04** |
| mm1m_r15_moe_micro | 1,071,204 | 3.65 | 15.4 | 51.14 | 21.33 | 21.33 | 3.557e+04 |
| mm1m_r16_mamba_multihead | 1,163,344 | 3.3 | 16.46 | 49.35 | 23.33 | 21.33 | 6856 |
| mm1m_r17_mamba_conv_gate | 1,109,104 | **3.282** | 15.7 | 49.95 | 22 | 22 | 7193 |
| mm1m_r18_dense_novr | 1,033,584 | 3.562 | 16.08 | 48.86 | 20.67 | 20 | 3.273e+04 |
| mm1m_r19_mamba_pure | 1,109,104 | 3.282 | 15.7 | 49.95 | 22 | 22 | 7204 |
| mm1m_r20_dense_ffn4x | 1,176,692 | 3.573 | 16.52 | 50.45 | 22 | 22 | 3.401e+04 |

## opt

| name | params | val_loss | wikitext_byte_ppl | blimp_acc | arc_easy_acc | arc_easy_acc_norm | tokens_per_second |
| --- | --- | --- | --- | --- | --- | --- | --- |
| opt_adamw | 937,552 | 4.543 | 18.09 | **52.24** | 16.67 | **22.67** | 1.226e+04 |
| opt_lion | 937,552 | 5.38 | 18.81 | 50.55 | 16.67 | 22 | **1.277e+04** |
| opt_muon | 937,552 | **3.12** | **13.93** | 49.95 | **22.67** | 20 | 1.239e+04 |
