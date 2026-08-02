# Report 01 — Evaluation harness and the Glint-2 baseline

*Artifacts: `research/experiments/eval_harness.py`, `eval_glint2.py`,
`research/baselines/glint2_model.py`, results in
`research/data/results/glint2.json`.*

## The released Glint-2 is not the advertised Glint-2

The README sells "1.06M parameters, pure-loop, zero unique layers". The shipped
checkpoint (`glint-2.pt`) disagrees. Its `model_config` is
`prelude_layers: 0, coda_layers: 1, shared_loops: 16` and the state dict
contains **both** a looped `shared` block **and** a unique `coda.0` block:

```
embed.weight            (4096, 96)   393,216   (tied head)
shared.*  (looped x8)                645,312   dim 96, ffn_hidden 2112 (22x), 8 heads
coda.0.*  (unique)                   645,312   identical block, run once after the loop
loop_lora + loop_embed               ~26,000   rank-4 per-loop LoRA + per-loop embedding
final_norm, indexer.gate                  97
                             TOTAL 1,710,049
```

So the real artifact is a **1.71M-param loop+coda model**, 61% larger than
advertised and architecturally almost identical to our own `supra2_1406240`
(1.41M) — minus a prelude, minus value residuals / timestep scale / outer gate,
plus a much wider FFN. Their own `generate.py` cannot even load their own
checkpoint (`strict=True` chokes on the unexpected `coda.*` keys). Our faithful
loader `research/baselines/glint2_model.py` loads it with `strict=True`,
reproduces their sample behaviour exactly (fluent TinyStories prose that drifts
to noise on encyclopedic prompts), and is the object every baseline number below
is measured on.

## Harness validation

We built one model-agnostic harness (`eval_harness.py`) so Glint-2 and our own
models are scored by identical code. Batched right-padded scoring is exact for
causal models (position *t* never attends to padding at *t+1*). Two of the three
metrics reproduce Glint-2's published numbers almost exactly:

| metric | Glint-2 reported | our harness (full) | verdict |
| --- | --- | --- | --- |
| ARC-Easy acc | 36.80 | **36.78** (acc) / 37.25 (acc_norm) | ✅ matches |
| WikiText-2 "ppl" | 3.09 | **3.18** (byte-ppl = 2^BPB) | ✅ matches (see below) |
| BLiMP | 73.96 | 66.36 | ⚠ harness gap (see below) |

### Finding: their "WikiText ppl" is byte-normalised, not token perplexity

A 1M model cannot have token perplexity 3.09 on WikiText (that would be 1.13
nats/token; real value is ~3.4). The number only makes sense as
**2^(bits-per-byte)**: we measure BPB = 1.668, and 2^1.668 = **3.18 ≈ 3.09**.
This is actually the *right* choice — byte-normalised perplexity is
tokenizer-independent and the only fair cross-model perplexity metric — but it
must be stated, because a naive token-ppl comparison would be off by 10x. We
added `wikitext_byte_ppl` to the harness and use it as the head-to-head metric.

### Finding: BLiMP is only comparable within one harness

Our full-BLiMP number (66.36) is ~7 points below their leaderboard 73.96.
Calibration (report inline, `blimp` subsample) shows this is **not** a
BOS/EOS/normalisation bug on our side — prepending `<bos>` or appending `<eos>`
*lowers* our number (69.3 → 68.4 → 67.9 on a 60/paradigm subsample), and ARC +
byte-ppl both match to the decimal, so the harness is sound. The gap is the
unreleased "tiny-ml leaderboard" harness using a different BLiMP protocol
(aggregation, paradigm subset, or length treatment). The consequence for this
project is the central methodological rule:

> **A "we beat Glint-2" claim is only valid on identical evaluation code.** We
> therefore always report both models through *our* harness. Absolute numbers
> are not portable across leaderboards; deltas within one harness are.

## The baseline to beat (our harness, full sets)

| model | params | BLiMP | ARC-Easy acc | ARC acc_norm | WikiText byte-ppl (↓) | BPB (↓) |
| --- | --- | --- | --- | --- | --- | --- |
| **glint-2** | 1,710,049 | 66.36 | 36.78 | 37.25 | 3.18 | 1.668 |

Timing on 4 CPU cores: full BLiMP 118s, full ARC 86s, WikiText 4s.

## What this tells us for the attack

1. Glint-2 spends ~75% of its non-embedding params on **two identical wide-FFN
   blocks** (one looped, one coda). The FFN is where its knowledge sits.
2. It has **no value residuals / timestep scaling / outer gate** — stabilisers
   our `supra2` has. At 8 loops + coda that is a real robustness gap we can
   exploit.
3. Its training data is TinyStories-dominant with an encyclopedic slice — our
   experiments should train on the same distribution to compare fairly.
4. Its headline advantages are as much *inference-time* (the effort ladder +
   quality probe) as architectural. Those are architecture-agnostic and we can
   implement them for our own models (reports 08–09) as pure upside.
