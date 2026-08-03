# architectures/

Language-model families behind one interface (`BaseLanguageModel`): build with
`build_model(<template>)`, decode with the shared `KVCache`, save/load with
`save_pretrained`/`load_model`.

- `layers.py` — RMSNorm, RoPE, GQA attention (+QK-norm, windows, value
  residual, per-call QKV delta), SwiGLU, MoE FFN, RG-LRU, factorized
  embedding, KVCache.
- `dense.py` / `looped.py` / `moe.py` / `hybrid.py` — the four families.
- `builder.py` — templates → models, with parameter-budget verification.
- `registry.py` — name → class; register yours here.
- `templates/` — size ladder (1.4M–343M). **Generated** by
  `scripts/generate_templates.py`, except `supra2_1406240.yaml`, which is the
  hand-annotated looped-transformer specification that the builder reads
  directly.

Choosing a family and adding one: [docs/architecture.md](../../docs/architecture.md).
