# Small language models: a reading list

## Existence proofs (what small models can do)

- **TinyStories** (Eldan & Li, 2023, arXiv:2305.07759) — 1–30M models produce
  coherent stories when the *data* is capped to a child's vocabulary. The
  founding document of "data difficulty should match model capacity"; our
  `tiny-curriculum` mixture is this idea.
- **Textbooks Are All You Need / phi series** (Gunasekar et al., 2023,
  arXiv:2306.11644; phi-1.5 arXiv:2309.05463) — filtered + synthetic
  "textbook" data lets 1.3B behave like 7B. Scaled-down, this is why
  `cosmopedia-v2` sits in every default mixture.
- **SmolLM / SmolLM2** (HuggingFaceTB, 2024–25, arXiv:2502.02737) — the
  best-documented public recipe at 135M–1.7B: FineWeb-Edu + Cosmopedia,
  over-trained far past Chinchilla. Our `smollm-corpus` mixture and token
  budgets copy the shape of this work.
- **MobileLLM** (Liu et al., 2024, arXiv:2402.14905) — sub-1B design study:
  deep-and-thin wins, embedding sharing matters, GQA is free. Informs our
  template proportions.
- **BabyLM Challenge** (Warstadt et al., 2023, arXiv:2301.11796 context) —
  what is learnable from 10–100M *words*; why BLiMP is our headline small
  benchmark.

## Data

- **FineWeb / FineWeb-Edu** (Penedo et al., 2024, arXiv:2406.17557) — the
  filtering pipeline; the -edu classifier is the single highest-leverage data
  decision for small models.
- **Cosmopedia** (Ben Allal et al., 2024) — recipe for 25B synthetic tokens;
  the seed-diversity trick is the transferable insight.
- **DataComp-LM** (Li et al., 2024, arXiv:2406.11794) — controlled data
  ablations; evidence that data curation beats architecture tweaks at fixed
  compute (we cite this when tempted to add exotic layers instead of cleaning
  data).

## Scaling and budgets

- **Chinchilla** (Hoffmann et al., 2022, arXiv:2203.15556) — compute-optimal
  ≈20 tok/param. We deliberately violate it (300–1000 tok/param): small models
  are inference-bound, so over-training is the correct economics.
- **Scaling Data-Constrained LMs** (Muennighoff et al., 2023,
  arXiv:2305.16264) — up to ~4 epochs of repeated data ≈ fresh data. License
  to sweep small corpora multiple times.

## Post-training at small scale

- **LIMA** (Zhou et al., 2023, arXiv:2305.11206) — 1K good SFT examples beat
  50K mediocre ones. Why `limo`/`s1k`/`no-robots` are in the registry despite
  their size.
- **Zephyr** (Tunstall et al., 2023, arXiv:2310.16944) — the SFT→DPO pipeline
  our `configs/sft` + `configs/rl/dpo.yaml` reproduce in miniature.

Per-topic depth: [architecture-notes.md](architecture-notes.md),
[post-training-notes.md](post-training-notes.md).
