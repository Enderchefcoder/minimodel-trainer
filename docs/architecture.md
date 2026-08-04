# Architectures

Four core language-model families share one interface
(`minimodel.architectures.BaseLanguageModel`), one decoding cache protocol and
one trainer — plus two experimental families aimed at the ~1M Glint-2 budget
(`experimental-transformer`, `mamba-lm`). This page explains what each is, why
it exists, and how to pick.

## The families

### `dense-transformer` - the baseline done properly

A standard decoder-only stack, with the small-model best practices already
switched on:

- **RMSNorm** (eps `1e-6`) pre-norm everywhere;
- **RoPE** position encoding;
- **SwiGLU** FFN at the 8/3 ratio (parameter-matched to a 4x GELU FFN);
- **Grouped-query attention** (`n_kv_heads < n_heads`), shrinking the KV cache
  by the grouping factor at negligible quality cost;
- **QK-norm** - RMSNorm on queries and keys before the dot product. This is
  the single cheapest fix for attention-logit explosions, which are the most
  common cause of loss spikes when small models train at high learning rates;
- **Tied embeddings** (input embedding doubles as the LM head);
- optional **windowed attention with a global pattern** (`window_pattern: 4` =
  every fourth layer sees full context) - most of the quality of full
  attention at a fraction of the cache;
- GPT-2-style **depth-scaled residual init** (`1/sqrt(2*n_layers)` on residual
  projections).

Pick it by default. Everything else in this list should be justified against
it.

### `looped-transformer` - depth without parameters

The flagship (`supra2_1406240.yaml` is its fully annotated specification). A
prelude block, `N` *shared* blocks run for `L` loop iterations (schedule
`ABAB...`), and a coda block. 4 blocks of weights yield `2 + L` effective
layers - the 1.4M-param config runs 18 effective layers.

Naive weight sharing collapses into a fixed point, so each iteration is
conditioned four cheap ways:

| Mechanism | Cost | Effect |
| --- | --- | --- |
| `loop_embed` | `16*dim` params | tells the block which iteration it is in |
| `loop_lora` (rank 4 on QKV) | `16*(4*dim + 4*3*dim)` | per-iteration attention weights |
| `timestep_scale` | `16*dim` | per-iteration FFN output gain |
| `outer_gate` residual | `dim` | re-injects the embedding stream every iteration |

Value residuals (a learned per-head blend with the previous layer's values)
stabilise the deep shared stack, and the loop count is sampled
`U{min_loops..train_loops}` during training, so at inference you can trade
quality for latency by asking for more or fewer loops:

```python
model(tokens, loops=4)   # faster
model(tokens, loops=8)   # better
```

Pick it when parameters are the binding constraint: embedded targets, weight
budgets, "how good can 5MB get" experiments.

The looped model is fully configurable to span the design space: `prelude_layers`
/ `coda_layers` (0+ each — set both to 0 for a pure-loop model), `embedding_type`
(`factorized` or `tied`), and independent switches for the value-residual,
timestep-scale and outer-residual stabilisers. Loop-count sampling supports
`uniform` or **log-normal `poisson`** (Huginn 2025 — trains the upper loop range
so the model keeps improving past its centre instead of collapsing), and
`backprop_loops` enables truncated backprop through the last *N* iterations for
~30% backward-FLOP savings. These choices are studied empirically in
`research/reports/` (07 shows Poisson sampling + stabilisers make test-time loop
scaling actually work).

### `moe-transformer` - parameters without FLOPs

Sparse mixture-of-experts: top-2 routing over 8-16 SwiGLU experts per layer,
plus one **shared expert** that is always active (so no token is left with
only a badly-routed transformation), with layer 0 kept dense (early
representations route poorly).

Load balancing is **auxiliary-loss-free**: a per-expert bias is added to the
router logits *for selection only* and nudged toward uniform load by a
controller rule. Unlike an auxiliary loss, this cannot trade
language-modelling quality for balance - the gradient never sees it. Watch
`routing_stats()` during training: `max_over_mean` should settle in 1.0-1.3.

Pick it when you have memory to spare and training FLOPs are the constraint:
`moe_28m_a16m` trains at dense-16M cost and beats dense-16M clearly.

### `hybrid-recurrent` - long context on a budget

Griffin-style: two thirds of the layers replace attention with an **RG-LRU**
(real-diagonal gated linear recurrence, evaluated with a parallel log-space
scan - O(T) time, O(1) state), and every third layer is local attention so the
model can still do exact retrieval, the one thing pure recurrences are bad at.

The decode-time state is *constant-size* regardless of context length, which
is the whole point: `hybrid_35m` holds an 8K context with the KV cache of a
~2.7K-context dense model.

Pick it for long documents, streaming, or memory-constrained decoding.

### `experimental-transformer` - novel wirings at ~1M

Five original Transformer compositions for Glint-2 bake-offs (ResiMix residual
mix, KV-inherit depth routing, braid local/global heads, dual-RoPE, echo-FFN).
Not production defaults — see `research/reports/12_arch_1m_candidates.md`.

### `mamba-lm` - selective SSM in pure PyTorch

Mamba-style selective state space without custom kernels, with hybrid variants
(attention tail, SSM/attention braid, multi-head SSM). Streamable via
`KVCache.recurrent_states`.

## Choosing, in one table

| Constraint | Choice |
| --- | --- |
| "Just train me a good model" | `dense` |
| Model file size / parameter budget | `looped` |
| Training compute budget, memory is fine | `moe` |
| Long context or tiny decode memory | `hybrid` / `mamba-lm` |
| Novel ~1M Glint-2 experiments | `experimental` / `mm1m_r*` templates |

The ordered ~1M candidate ladder is `list_glint2_candidates()` (ranks 1–20).

## Templates

Templates are YAML files mapping 1:1 onto an architecture config, with the
declared `params:` verified against the built model by tests. List them:

```bash
minimodel models            # names, families, exact counts
minimodel models --verify   # re-build each and check the counts now
```

Build one in code:

```python
from minimodel.architectures import build_model

model = build_model("dense_30m", overrides={"vocab_size": 16384})
model.parameter_breakdown()   # params per top-level module
```

Two template layouts exist:

- **generated** (everything except supra2): a flat `arch:` block, produced by
  `scripts/generate_templates.py`. Edit the generator, not the YAML.
- **annotated** (`supra2_1406240.yaml`): the full written specification of the
  looped architecture - shapes, formulas, init order, parameter budget - which
  the builder knows how to read directly. The spec *is* the config.

## Adding an architecture

1. Implement `BaseLanguageModel`: `__init__(config)`, `forward(tokens, *,
   cache=None, return_hidden=False, **kwargs) -> logits`, `from_config`.
   Support the `KVCache` call-order protocol (see `layers.KVCache`); the test
   suite requires full-sequence and incremental decoding to agree.
2. Register it in `architectures/registry.py` with any aliases.
3. Add a size to `scripts/generate_templates.py` and regenerate.
4. `pytest tests/test_architectures.py` - the shared behavioural tests
   (shapes, cache equivalence, save/load) run against every registered family
   automatically once you add a case to the parametrised fixture.

## The supporting layers

All in `minimodel/architectures/layers.py`, all documented in-source:
`RMSNorm`, `RotaryEmbedding`/`apply_rope` (even/odd interleave),
`CausalLocalAttention` (GQA + QK-norm + windows + value residual + per-call
QKV delta), `SwiGLUFeedForward`, `MoEFeedForward`, `GatedRecurrentUnit`
(RG-LRU), `FactorizedEmbedding` (the shared-factor embedding/LM-head used by
the looped family), `KVCache`.
