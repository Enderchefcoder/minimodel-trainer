# Design notes: the supra2 looped family

`src/architectures/templates/supra2_1406240.yaml` is the specification; this
file is the reasoning behind it.

## The bet

At 1–20M parameters, the scarce resource is parameters, not FLOPs. Weight
sharing spends FLOPs (re-running blocks) to buy effective depth without
parameter cost: supra2 gets 18 effective layers from 4 blocks' weights, and
15.6% of its budget sits in the embedding factorization alone.

## Why each mechanism exists

| Piece | Failure it prevents |
| --- | --- |
| loop_embed (16×dim) | iterations computing the *same* function (fixed point) |
| loop_lora rank-4 on QKV | attention identical across iterations |
| timestep_scale on FFN | later iterations over-writing earlier structure |
| outer_gate ×x0 | input becoming unreachable after many iterations |
| value residuals | gradient attenuation through the shared stack |
| variable loops U{4..8} | over-fitting to one depth; enables test-time scaling |

Factorized embedding (E[V,r]·proj[r,d], both factors tied into the head)
because a full 4096×128 tie would still be 524K of a 1.4M budget; the
factorization is 270K.

## Invariants the code must keep (tests enforce most)

1. Total params = 1,406,240 exactly for the reference config.
2. `model(tokens, loops=K)` honours K; training samples U{min..train}.
3. Loops beyond `max_loops_table` clamp to the last table entry.
4. LoRA up-projection initialises to zero (iteration-0 = plain block).
5. Init order: generic normal(0,0.02) first, then re-assert specials
   (lora_up=0, outer_gate=0.1, τ=1); RMSNorm gains and v_λ are never touched
   by the generic pass.
6. RoPE convention: even/odd interleave, cos/sin shape [1,1,T,hd/2].

## Open questions worth an experiment

- Loop-wise early exit (stop when the residual delta is small) — free
  adaptive compute at inference.
- KV-cache sharing across iterations (currently each call site caches
  separately; correct but memory-linear in loops).
- Distilling a dense teacher into a looped student at matched params.
