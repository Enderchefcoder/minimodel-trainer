# Report 02 — SLM techniques and the literature behind our attack

A focused review of the techniques that matter for beating a ~1M looped model,
with what we adopt and why. Sources are linked inline.

## Looped / recurrent-depth transformers (the Glint-2 family)

Weight sharing across depth originates with the **Universal Transformer**
(Dehghani et al., 2018, [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)):
iterate a shared block to get effective depth at fixed parameters. The idea has
returned as a route to *latent* reasoning and test-time compute:

- **Huginn / "Scaling up Test-Time Compute with Latent Reasoning"** (Geiping et
  al., 2025, [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)) — a 3.5B
  recurrent-depth model trained on 800B tokens that scales test-time compute by
  unrolling a recurrent block, reaching the reasoning performance of a ~50B
  dense model. Crucially for us, it **samples the iteration count during
  training from a log-normal Poisson distribution** so the model works at
  iteration counts it never saw at any single step.
- **Ouro** (Zhu et al., 2025) — a LoopLM scaled to compete with mainstream open
  models.
- **Iso-Depth Scaling Laws** (2026,
  [arXiv:2604.21106](https://arxiv.org/abs/2604.21106)) — fits looped and
  non-looped models under one "recurrence-equivalence exponent" φ. Reports the
  key compute-matched pattern: **looped models trail non-looped baselines on
  validation loss and parametric-knowledge tasks, but match or beat them on
  reasoning**. Also quantifies **truncated backpropagation** (gradients only
  through the last few loops → ~30% training-FLOP savings) and hyperconnections.

### The instability that Glint-2 exhibits — and how to fix it

Glint-2's README states the model "trains at 8 loops … run it at 16 and you get
gibberish". This is a documented failure mode, not a quirk:

- **"Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning"**
  (2026, [arXiv:2605.26733](https://arxiv.org/abs/2605.26733)) finds that LoopLM
  performance "typically peaks at a specific iteration depth and declines
  sharply beyond it", most severely in Ouro, and proposes **Jacobian Spectral
  Radius Regularization (JSRR)** combined with **random loop sampling** to make
  scaling monotone.
- **Parcae: Scaling Laws for Stable Looped Language Models** (Prairie et al.,
  2026) treats the loop as a dynamical system and stabilises recurrence against
  residual explosion / loss spikes.
- **"Loop, Think & Generalize"** (2026,
  [arXiv:2604.07822](https://arxiv.org/abs/2604.07822)) compares fixed vs
  dynamic iteration; dynamic sampling `R ~ clip(Poisson(λ), Rmin, Rmax)` plus
  Fixup-style init (Zhang et al., 2019) supports "stable optimization under
  unbounded unrolling".

**Consequence for us.** Glint-2 uses a *fixed* 8-loop schedule with a 16-slot
table it never learned to use — precisely the brittle regime the literature
warns against. Our looped model already samples loops during training; we
upgrade the sampler from uniform to **log-normal Poisson** (Huginn's choice) and
verify robustness directly in report 07 (loop-scaling), where Glint-2 should
degrade past 8 loops while our model does not. This is a concrete, literature-
grounded architectural advantage at equal parameters.

## The small-model fundamentals (report 00 of the main docs covers these)

The design levers our ablations test are each grounded:

- **QK-norm** (Henry et al., 2020, [arXiv:2010.04245](https://arxiv.org/abs/2010.04245);
  adopted by Gemma 2 / OLMo 2) — stabilises attention logits at high LR. Glint-2
  omits it; we test whether adding it helps at fixed budget.
- **Value residuals** (Zhou et al., 2024,
  [arXiv:2410.17897](https://arxiv.org/abs/2410.17897)) — shorten the gradient
  path through a deep weight-shared stack. Glint-2 omits these; report 04
  measures their effect on the *identical* shape.
- **Muon optimizer** (Jordan et al., 2024) — orthogonalised momentum; the
  small-model speedrun optimizer. Report 05.
- **WSD schedule** (MiniCPM, Hu et al., 2024,
  [arXiv:2404.06395](https://arxiv.org/abs/2404.06395)). Report 06.
- **Data quality** — TinyStories (Eldan & Li, 2023,
  [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)) is Glint-2's evident
  training distribution (its samples are pure TinyStories). Textbook/synthetic
  data (phi, [arXiv:2306.11644](https://arxiv.org/abs/2306.11644)) is the
  highest-signal-per-token source at this scale.

## Inference-time compute (the other half of Glint-2's advantage)

Glint-2's headline results lean heavily on its **effort ladder** (best-of-N +
chunked beam search) and a **quality probe** (a linear P(real) head that stops
self-log-prob reranking from Goodharting toward boilerplate). This mirrors the
broader test-time-compute literature (Snell et al., 2024,
[arXiv:2408.03314](https://arxiv.org/abs/2408.03314), "scaling test-time compute
can beat scaling parameters"). Both techniques are architecture-agnostic; we
implemented them for our own models in `src/inference/search.py` and
`src/inference/quality_probe.py` (report 08) — pure upside, no retraining.

## What we adopt

| Technique | Source | Where |
| --- | --- | --- |
| Log-normal Poisson loop sampling | Huginn 2025 | `LoopedTransformer` sampler (this report's change) |
| Truncated backprop through loops | Iso-depth 2026 | training-efficiency option |
| Effort ladder (best-of-N + chunked beam) | Glint-2 / Snell 2024 | `inference/search.py` |
| Quality / correctness probe | Glint-2 | `inference/quality_probe.py` |
| Value residuals, QK-norm, Muon, WSD | as cited | ablations, reports 03–06 |
| Byte-normalised perplexity for fair comparison | (our methodology) | `eval_harness.py` |

The experimental reports (03–07) measure each lever on real runs; report 09
assembles the winning recipe and projects it to Glint-2's training budget.
