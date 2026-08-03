# inference/

- `sampling.py` — the decoding engine: temperature / top-k / top-p / min-p /
  repetition & presence penalties, seeded sampling, stop tokens, KV-cached
  `generate`, `stream_generate`, left-padded `generate_batch`.
- `run.py` — user-facing layer: `load_for_inference` (model+tokenizer+
  template), `complete`, `stream_completion` (UTF-8-safe chunks),
  `generate_with_reasoning` (separate think/answer budgets), `chat_loop`.

Docs: [docs/inference.md](../../docs/inference.md).
