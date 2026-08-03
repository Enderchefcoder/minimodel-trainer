# Report 05 — FFN width: is Glint-2's 22× the right allocation?

*Artifacts: `research/experiments/ablations.py` (group `ffn`),
`research/data/results/ffn_*.json`.*

Glint-2 puts ~94% of each block into the feed-forward network — `ffn_hidden=2112`
on `dim=96`, a **22× ratio** (a normal SwiGLU is ~2.7–4×). We swept the ratio on
the loop+coda shape, everything else fixed, same 300-step / 2.46M-token
TinyStories budget (AdamW; note report 04 shows Muon would lift all of these,
but the *relative* comparison holds).

| FFN ratio | ffn_hidden | params | val loss ↓ | WikiText byte-ppl ↓ | BLiMP |
| --- | --- | --- | --- | --- | --- |
| **4×** | 384 | **714,720** | 4.52 | **16.59** | 49.9 |
| 8× | 768 | 935,904 | 4.70 | 19.56 | 53.0 |
| 16× | 1,536 | 1,378,272 | 4.51 | 17.84 | 52.3 |
| 22× (Glint-2) | 2,112 | 1,710,048 | 4.63 | 18.51 | 52.1 |

## Finding: at a small budget, wider FFN buys nothing

byte-ppl does **not** improve with width — the **4× ratio (714K params) beats the
22× ratio (1.71M params) by 10%** (16.59 vs 18.51) while using **2.4× fewer
parameters**, and val loss is flat across the sweep (4.5–4.7). The extra 1M
parameters Glint-2 spends on a 22× FFN are essentially dead weight at this
training scale: a wide FFN is a large lookup table that needs many tokens to
populate, and 2.46M tokens is nowhere near enough.

This is the same lesson as the architecture bake-off (report 03), sharpened: the
22× FFN is a **bet on training scale**. With billions of tokens it may pay off
(more capacity to memorise), but per training token — the regime that matters
when compute is the constraint — it is strictly wasteful. A well-trained model on
a fixed budget should spend its parameters on a *narrower* FFN (and, per report
03, probably on being *dense* rather than looped at all).

## Consequence for the recipe

- Do **not** copy Glint-2's 22× FFN at small scale. Use a standard ~4× SwiGLU.
- The parameters saved (2.4× here) are better spent on more tokens, a dense
  stack, or (if looped) on being trainable with Muon.
- Glint-2's headline 1.71M size is inflated by this choice: a 4×-FFN model of
  ~0.7M params reaches lower byte-ppl at the same budget. Its parameter count is
  spent on capacity it has not learned to use.
