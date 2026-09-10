# Report 06 — Model & benchmark comparison (curated)

[RESULTS.md](RESULTS.md) is the *auto-generated* dump of every run. This report is
the *curated* view: what the numbers mean, which comparisons are fair, and what
to pick for which goal. All rows come from the shared harness
(`experiments/eval_harness.py`) under matched compute unless stated otherwise.

## How to read the benchmarks

| Benchmark | Metric | What it measures | Higher/lower better | Caveats |
| --- | --- | --- | --- | --- |
| **BLiMP** | macro-avg accuracy over 67 paradigms | grammatical acceptability judgements | higher | chance = 50%; tiny models cluster at 47–53%, so only differences > ~2 pts matter |
| **ARC-Easy** | accuracy (raw + length-normalised) | grade-school science QA | higher | chance = 25%; multiple-choice scoring is sensitive to length bias, so read `acc_norm` alongside |
| **WikiText-2** | byte-normalised perplexity (`2^bits_per_byte`) | language modelling | lower | **not** comparable to word-ppl numbers in model cards; byte-ppl ~16 corresponds to roughly word-ppl 30+ |
| **val_loss** | NLL on held-out corpus tokens | the training objective itself | lower | only comparable *within* a group sharing tokenizer + corpus |
| **exact-match (ablation)** | % of held-out problems whose final number is decoded exactly | task success, not likelihood | higher | the CoT variants must generate the trace before the answer (budget-forced), so this penalises verbosity |

The harness is model-agnostic: it wraps any model in a
`ModelAdapter` (`encode` + `forward`) and scores identically, so cross-family
comparisons (dense / looped / MoE / SSM hybrids) share one protocol. The Glint-2
baseline is scored with the same adapter, using the strict-loading
reimplementation in `baselines/` (the released public `generate.py` cannot load
its own checkpoint — see [report 10](10_glint2_verification.md)).

## The line to beat: Glint-2 (measured, not advertised)

Shipped weights are **1.71M params** (8-loop loop+coda), not the 1.06M the
upstream README claims ([report 01](01_baseline_validation.md),
[report 10](10_glint2_verification.md)):

| model | params | WikiText byte-ppl | BLiMP | ARC-Easy |
| --- | ---: | ---: | ---: | ---: |
| **glint-2 (loops=8)** | 1,710,049 | 3.179 | 66.36 | 36.78 |
| chance / random | — | — | 50.0 | 25.0 |

Every fair comparison below is compute-matched at the report-03 protocol and
targets ≥1.7M params. Where a run is at ~1.0M it is a *scaling* datapoint, not a
contender.

## Architecture bake-off (fixed budget, ~1.7M)

| arch | params | val loss ↓ | byte-ppl ↓ | BLiMP ↑ | ARC ↑ | tok/s ↑ | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense (GQA + value residual) | 1.70M | **3.266** | **16.28** | 49.85 | **21.33** | **32,450** | best LM quality and 7–12× faster decoding than loops |
| loop+coda (Glint-2 shape) | 1.71M | 4.635 | 18.51 | 52.14 | 16.67 | 4,805 | best BLiMP of the group; weak LM |
| MoE (top-k routing) | 2.85M | 3.460 | 16.29 | 49.05 | 21.33 | 22,580 | competitive but needs 1.7× the params |
| pure loop (8×) | 1.77M | 4.029 | 17.94 | 52.04 | 20.00 | 2,696 | slowest; latent compute doesn't pay here |
| supra2 (hand-annotated spec) | 1.74M | 4.992 | 19.21 | 50.65 | 19.33 | 12,960 | worst LM quality of the five |

**Takeaway.** Per *emitted token*, dense wins on loss and speed; the loop
families win a little on BLiMP but lose badly on the LM objective. This is the
core tension that [report 09](09_synthesis.md) resolves with a dense contender.

## ~1M candidates (report 03 protocol, 20 runs)

Top-5 of 20 by val loss (full table in [RESULTS.md](RESULTS.md), `mm1m` group;
family notes in [report 12](12_arch_1m_candidates.md)):

| rank | arch | params | val loss ↓ | byte-ppl ↓ | BLiMP ↑ | ARC ↑ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | r17/r19 mamba (conv-gate / pure) | 1.11M | **3.282** | 15.70 | 49.95 | 22.00 |
| 2 | r16 mamba-multihead | 1.16M | 3.300 | 16.46 | 49.35 | 23.33 |
| 3 | r10 mamba-attn-tail (hybrid) | 1.06M | 3.393 | 15.43 | 50.45 | 22.00 |
| 4 | r04 hybrid-griffin | 0.95M | 3.421 | **14.51** | 51.74 | **25.33** |
| 5 | r12 mamba-braid (hybrid) | 1.10M | 3.447 | 15.98 | 47.96 | 20.00 |

**Takeaway.** At ~1M, SSM-hybrid shapes beat dense on loss, and
hybrid-griffin wins the *benchmark-normalised* row (best ppl and ARC at the
smallest param count). Dense GQA remains the throughput king. r13 (poisson
looping) is last by a wide margin — consistent with the bake-off.

## Optimizers (compute-matched, 937k)

| optimizer | val loss ↓ | byte-ppl ↓ | note |
| --- | ---: | ---: | --- |
| **Muon** | **3.12** | **13.93** | decisive winner; adopted for all contenders |
| AdamW | 4.543 | 18.09 | baseline |
| Lion | 5.380 | 18.81 | worst |

## FFN ratio (is Glint-2's 22× allocation right?)

| FFN | params | val loss ↓ | byte-ppl ↓ | BLiMP ↑ |
| --- | ---: | ---: | ---: | ---: |
| 16× | 1.38M | **4.508** | 17.84 | **52.34** |
| 22× | 1.71M | 4.635 | 18.51 | 52.14 |
| 4× | 0.71M | 4.521 | **16.59** | 49.85 |
| 8× | 0.94M | 4.703 | 19.55 | **53.03** |

**Takeaway.** 22× is not justified: 16× matches it with 20% fewer params. The
differences inside this group are small overall.

## Loop robustness (train loops ≠ eval loops)

Trained-at-8-loop models evaluated across loop counts break badly off-distribution
(ours span 1.04–1.19× quality change and *improve* with more loops; Glint-2
degrades 35×) — full data in [report 07](07_loop_robustness.md) and
`results/loops_scaling.json`.

## Inference-time scaling (architecture-agnostic)

The effort ladder + quality probe ([report 08](08_inference_wins.md)) buys
measurable gains at zero training cost; `results/head_to_head.json` holds the
contender-vs-glint numbers.

## What to choose

| Goal | Pick | Why |
| --- | --- | --- |
| Best loss/quality per FLOP at ≤2M | dense GQA + value residual + QK-norm, Muon, FFN 16× | reports 03/04/05/09 |
| Best benchmark mix at ~1M | hybrid-griffin or mamba-conv-gate | report 12 |
| Latent reasoning (extra compute at inference) | looped — but train with variable/Poisson loop sampling and expect weak LM loss | reports 03/07, report 13 |
| Inference-only gains on an existing model | effort ladder / quality probe | report 08 |
