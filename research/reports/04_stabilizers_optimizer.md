# Report 04 — Optimizer and stabilisers

*Artifacts: `research/experiments/ablations.py` (group `optimizer`),
`research/data/results/opt_*.json`; stabiliser evidence from report 07.*

## Optimizer: Muon dominates on the looped model

Same looped model (~0.94M, loop+coda, 8× FFN, all stabilisers on), same 300-step
(2.46M-token) TinyStories budget, three optimizers:

| optimizer | lr | val loss ↓ | WikiText byte-ppl ↓ |
| --- | --- | --- | --- |
| AdamW | 3e-3 | 4.54 | 18.09 |
| **Muon** | 2e-2 (+AdamW 3e-3 aux) | **3.12** | **13.93** |
| Lion | 4e-4 | 5.38 | 18.81 |

**Muon reaches in 300 steps what AdamW does not** — a 1.4-point val-loss gap and
a 23% lower byte-ppl. This is consistent with Muon's design (orthogonalised
momentum on 2-D parameters): the looped model *reuses* the same matrices eight
times, so well-conditioned updates to those matrices compound across iterations.
Lion underperformed at this LR (it is sensitive to LR/weight-decay tuning and we
did not sweep it).

**This reframes the architecture bake-off (report 03).** That bake-off used
AdamW for every architecture, which disadvantaged the looped models specifically
— they are the ones that benefit most from Muon. Under Muon a looped model
closes much of the per-token gap to dense. The honest combined recommendation:

- If you will train briefly and want the best model per token: **dense + AdamW**
  (simplest, fastest, report 03).
- If you are committed to the looped architecture (for test-time scaling):
  **train it with Muon**, which nearly erases the looped training penalty.

## Stabilisers: value residuals, timestep scale, outer-residual gate

Our looped model carries three stabilisers Glint-2 omits: value residuals
(Zhou et al., 2024), a per-loop timestep scale, and an outer-residual gate that
re-injects the post-embedding stream each iteration. Their decisive effect is on
**loop robustness** (report 07): with them, a model trained at a *fixed* 8 loops
still degrades only 1.19× out to 16 loops, where Glint-2 (no stabilisers)
explodes 35×. The stabilisers keep the recurrent dynamics bounded across depths
the model was never trained at — the difference between a model that can spend
test-time compute and one that shatters when you ask it to.

At fixed FFN width the stabilisers also help quality modestly (the report-07
`fixed8` model, 8× FFN + stabilisers, reaches byte-ppl 15.6 at 6.5M tokens; the
report-03 Glint-shape without stabilisers is at 18.5 — though the FFN widths
differ, so the clean isolation is the robustness result). We keep them on by
default for any looped model and expose each as a config flag for ablation.

## Takeaways folded into the recipe (report 09)

1. Use **Muon** for looped models; it is the single biggest per-token lever here.
2. Keep the **stabilisers** on for any looped model — they are what make
   test-time loop scaling actually work.
3. For a fixed small budget, **dense + AdamW** is still the simplest strong
   baseline; the contender uses it and wins the matched head-to-head (report 09).
