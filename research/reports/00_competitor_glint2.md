# Competitor analysis: Glint-Research / Glint-2

*Source: <https://huggingface.co/Glint-Research/Glint-2> (README, `generate.py`,
`effort.py`), retrieved 2026-08-02. All quotes and numbers are from that repo.*

Glint-2 is the model we are explicitly trying to beat. It is a **1.06M-parameter
pure-loop transformer** — the same architectural family as our
`supra2_1406240`, but taken to a minimalist extreme.

## Reported scores (the target)

From the "tiny-ml leaderboard", base model, single checkpoint, no soup:

| model | params | BLiMP | ARC-Easy | WikiText-2 ppl |
| --- | --- | --- | --- | --- |
| glint-0.3 | 1M | 47.3 | 25.5 | 7.87 |
| glint-0.4 | 1M | 58.5 | 31.0 | 5.01 |
| glint-1 | 1M | 61.2 | 32.0 | 4.45 |
| glint-1.3 (merged) | 982K | 68.7 | 32.5 | 3.08 |
| **glint-2** | **1.06M** | **73.96** | **36.80** | **3.09** |

BLiMP chance is 50; ARC-Easy chance ≈ 25. WikiText ppl is *per-token on a
4096-vocab BPE*, so it is not comparable to word-level GPT-2 numbers (~29) and
**not comparable across tokenizers** — this motivates a bits-per-byte metric on
our side (see report 02).

## Exact architecture (reconstructed from `generate.py`)

```
embed (tied)                       # [vocab=4096, dim=96]
for i in range(8):                 # LOOPS = 8, table capacity 16 (running at 16 = gibberish)
    gated = x + loop_embed[i]      # loop_embed: [16, 96]
    delta = loop_lora_up[i](loop_lora_down[i](gated))   # rank-r LoRA on the QKV delta
    x = shared_block(gated, cos, sin, qkv_delta=delta)  # ONE block, reused every loop
x = final_norm(x)                  # RMSNorm, torch default eps=1e-5
logits = x @ embed.weight.T        # tied head
```

`shared_block` = pre-norm attention (RoPE, causal, **window 256**, `head_dim=12`)
+ pre-norm SwiGLU FFN. **`ffn_hidden = 2112` on `dim = 96` is a 22× ratio** — the
block is ~94% FFN. The single block is ~645K params; tied embeddings
(4096×96 ≈ 393K) are most of the rest.

Notable *absences* versus our `supra2`:

- **No prelude or coda.** "zero unique layers before or after the loop. the loop
  is the model."
- **No value residual, no per-loop timestep scale, no outer-residual gate.** Our
  supra2 has all three. (An `Indexer.gate=0.1` parameter exists but is unused in
  the forward pass — likely a training-only or vestigial hook.)
- Simpler LoRA: plain per-loop `nn.Linear` pairs rather than parameter tables
  (mathematically equivalent).

So Glint-2 spends its entire block budget on one **very wide-FFN** block and
loops it. Our supra2 spreads the budget across prelude + 2 shared + coda with a
narrower FFN (544 on dim 128, ~4×) plus stabilisers.

## Two inference-time innovations we do not yet have

### 1. The effort ladder (`effort.py`)

Six levels that scale **search compute, not the model** (loops stays 8):

| level | search | notes |
| --- | --- | --- |
| low | 1 sample, temp 0.25, top-k 8, rp 1.15 | |
| medium | 1 sample, temp 0.3 | best single-shot |
| high | 6 full continuations, reranked | best-of-N |
| xhigh | 8 instances × 2 beams, 24-token chunks | chunked beam, resync top-2 each round |
| max | 8 instances × 4 beams | top-4 |
| ultra | 10 independent max searches, best final | ~26s CPU |

The rerank score is:
`mean token log-prob − 2.0·(4-gram repetition fraction) − 1.5·(length shortfall) + 2.0·P(real)`.
The repetition and shortfall penalties are cheap and general; we should adopt
them. Their honest framing of the ceiling: *"the ceiling does not move, but the
model reaches it far more often."*

### 2. The corrective quality probe (`effort_probe.pt`, 3.5 KB)

One linear layer over the model's **mean-pooled final hidden state** →
`P(text is real)`, trained to separate real fineweb-edu passages from the
model's own generations. Standardises the feature (`(feat-mean)/std`) then
`sigmoid(w·z + b)`. Blended as `+2.0·P(real)` into the rerank score from `high`
upward.

**Why it works** (their words): self-logprob reranking Goodharts under heavy
search — the model drifts toward "confident boilerplate" (memorised WikiText
section headers). The probe is trained against exactly that failure, so
"confident garbage stops winning". Measured `P(real)`: 0.575 on a real sentence
vs 0.000 on the model's own continuation.

This is a genuinely clever, near-free technique. It is architecture-agnostic and
we can implement + train it against our own models.

## How we beat it — the plan

Absolute SOTA training (BLiMP 74) needs billions of tokens on GPU; the sandbox
here is 4 CPU cores. So this research splits into two provable tracks:

1. **Architecture-agnostic inference wins** we implement and measure directly:
   bits-per-byte eval, the effort ladder, and the quality probe. These transfer
   to *any* of our models and are pure upside.
2. **A controlled experimental program** (reports 03–07) that measures, at small
   scale on real data, the relative value of each design lever — pure-loop vs
   prelude/coda, FFN ratio, value residuals, optimizer (Muon vs AdamW), schedule
   (WSD vs cosine), loop-count test-time scaling, vocab size — and turns the
   winners into a recipe that, trained to Glint-2's token budget on GPU, is
   projected to beat it. Every claim is backed by a logged run in
   `research/experiments/`.

We also download the real Glint-2 checkpoint and run it through *our* eval
harness (report 01) to (a) validate our harness against their published numbers
and (b) get an apples-to-apples baseline on identical code.
