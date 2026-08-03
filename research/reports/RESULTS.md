# Experiment results (auto-aggregated)

_Generated from 15 result files. Glint-2 baseline included where relevant._

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

## head

| name |
| --- |
| head_to_head |

## opt

| name | params | val_loss | wikitext_byte_ppl | blimp_acc | arc_easy_acc | arc_easy_acc_norm | tokens_per_second |
| --- | --- | --- | --- | --- | --- | --- | --- |
| opt_adamw | 937,552 | 4.543 | 18.09 | **52.24** | 16.67 | **22.67** | 1.226e+04 |
| opt_lion | 937,552 | 5.38 | 18.81 | 50.55 | 16.67 | 22 | **1.277e+04** |
| opt_muon | 937,552 | **3.12** | **13.93** | 49.95 | **22.67** | 20 | 1.239e+04 |
