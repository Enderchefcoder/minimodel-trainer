# cardgen/

`modelcard_autogen.py` builds a model card from run artifacts —
`run_metadata.json`, `metrics.jsonl`, benchmark JSON, the dataset registry —
instead of from memory. Anything underivable becomes an explicit `[TODO: …]`.
Output is HF-hub-compatible Markdown with YAML frontmatter.

`example_generation/example_1.md` is the original hand-written target format
this generator was built to fill.

Docs: [docs/model-cards.md](../../docs/model-cards.md).
