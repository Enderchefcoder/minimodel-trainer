# Architecture notes: what we use and why

Each entry: the source, what we took, what we skipped.

## The dense baseline

- **RMSNorm** (Zhang & Sennrich, 2019, arXiv:1910.07467): cheaper than
  LayerNorm, no mean subtraction. Taken wholesale; eps=1e-6 everywhere.
- **RoPE** (Su et al., 2021, arXiv:2104.09864): relative positions via
  rotation. We use the even/odd interleave convention — documented in
  `layers.apply_rope` because *convention mismatch silently corrupts loaded
  weights*.
- **SwiGLU** (Shazeer, 2020, arXiv:2002.05202): the 8/3 ratio keeps params
  equal to a 4× GELU FFN. No bias terms anywhere (PaLM, arXiv:2204.02311).
- **GQA** (Ainslie et al., 2023, arXiv:2305.13245): KV heads < Q heads. At our
  sizes quality cost ≈ 0 and the cache shrinks 2–6×; on by default.
- **QK-norm** (Henry et al., 2020, arXiv:2010.04245; adopted by Gemma/OLMo 2):
  normalise Q,K before the dot product. The cheapest fix for logit blow-ups at
  high LR — we treat it as mandatory, not optional.
- **Sliding window + periodic global layers** (Mistral, arXiv:2310.06825;
  Gemma 2's 1:1 pattern, ours is configurable `window_pattern`): most of the
  quality, fraction of the cache.
- **Depth-scaled init** (GPT-2, Radford et al. 2019): residual projections ×
  1/√(2L). Skipped: µP (arXiv:2203.03466) — right idea, but its benefit at
  our scale didn't justify the config surface.

## Looped / recurrent depth

- **Universal Transformers** (Dehghani et al., 2018, arXiv:1807.03819) — the
  origin of weight tying across depth.
- **Looped transformers can be programmed** (Giannou et al., 2023,
  arXiv:2301.13196) and **depth-recurrence for latent reasoning** (Geiping et
  al., 2025, arXiv:2502.05171) — looping as adjustable compute; our
  `loops=` inference argument is this.
- **Per-iteration conditioning**: loop embeddings + rank-4 LoRA deltas
  (LoRA, arXiv:2106.09685, repurposed per-step) + per-step FFN gains. This
  quartet is what stops shared blocks collapsing to a fixed point; ablate any
  one and the 1.4M model visibly regresses.
- **Value residuals** (Zhou et al., 2024, arXiv:2410.17897) — per-head blend
  with the previous layer's values; stabilises deep weight-shared stacks.

## Mixture of experts

- **Switch/expert basics** (Fedus et al., 2021, arXiv:2101.03961).
- **Shared expert + fine-grained experts** (DeepSeekMoE, 2024,
  arXiv:2401.06066) — taken: one always-on expert.
- **Aux-loss-free balancing** (Wang et al., 2024, arXiv:2408.15664; used in
  DeepSeek-V3) — bias on router logits for *selection only*, updated by a
  controller. Taken because an aux loss visibly taxed LM loss at 30M scale.
  Skipped: expert parallelism (out of scope for mini).

## Hybrid recurrence

- **RG-LRU / Griffin** (De et al., 2024, arXiv:2402.19427) — the gated
  real-diagonal recurrence and the 2:1 recurrent:attention pattern, both taken
  directly. Our parallel form uses a log-space cumsum scan.
- **Mamba** (Gu & Dao, 2023, arXiv:2312.00752) — implemented as ``mamba-lm``
  in pure PyTorch (sequential selective scan + streamable depthwise conv) for
  the ~1M Glint-2 candidate ladder; still no custom CUDA kernel.

## Optimizers & schedules

- **Muon** (Jordan et al., 2024, kellerjordan.github.io/posts/muon) —
  Newton–Schulz-orthogonalised momentum for 2-D params; the speedrun-proven
  small-model optimizer. Embeddings/head stay on AdamW (sparse gradients
  shouldn't be orthogonalised).
- **WSD schedule** (MiniCPM, Hu et al., 2024, arXiv:2404.06395) — plateau you
  can extend + branchable checkpoints; our default for exploratory runs.
- **Lion** (Chen et al., 2023, arXiv:2302.06675) — kept as the low-memory
  option.

## Merging

Model soups (arXiv:2203.05482), SLERP folklore, task arithmetic
(arXiv:2212.04089), TIES (arXiv:2306.01708), DARE (arXiv:2311.03099) — all
five implemented in `merging/slerp.py`; guidance in docs/merging.md.
