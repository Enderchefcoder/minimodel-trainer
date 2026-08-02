# Tokenization

## What ships

A self-contained **byte-level BPE** tokenizer
(`minimodel.tokenization.BPETokenizer`) - training, encoding, decoding and
JSON persistence in pure Python, with an optional fast path through the Rust
`tokenizers` package for training only (identical output, ~100x faster on
large corpora; `pip install 'minimodel-trainer[hf]'`, then `--backend fast`).

Byte-level means **no unknown tokens, ever**: the base alphabet is all 256
byte values (via the GPT-2 byte<->unicode bijection, which keeps the vocab
JSON readable), so emoji, code, and any language all encode. Pre-tokenization
uses a GPT-style split pattern that is provably *total* - every character of
the input lands in exactly one piece (there is a test for this), because a
pattern that silently drops characters corrupts the corpus without an error.

## Training one

```bash
# On a registered dataset:
minimodel tokenizer train --dataset fineweb-edu-10bt --limit 100000 \
    --vocab-size 16384 -o artifacts/tokenizer.json

# On your own files (txt or jsonl with a `text` field):
minimodel tokenizer train --input corpus/*.jsonl --vocab-size 8192 -o artifacts/tokenizer.json
```

The reported `bytes_per_token` is the number to look at: ~4.0+ for English
means the vocabulary fits the corpus; under ~3.0 means the vocab is too small
(or the corpus is very non-English) and every context window is carrying
fewer words than it should.

### Choosing a vocabulary size

The embedding table costs `vocab x dim` parameters (tied models pay it once).
For small models that is a large fraction of the budget, which is why the
templates pair sizes:

| Model params | Vocab |
| --- | --- |
| < 5M | 4096 (or factorized embeddings, as supra2 does) |
| 5-20M | 8192 |
| 20-80M | 16384 |
| 80M+ | 32000 |

## Special tokens

Every tokenizer reserves these ids up front - **including for base models** -
so instruction tuning and reasoning distillation never resize the embedding:

```
<|endoftext|>  <|pad|>  <|system|>  <|user|>  <|assistant|>  <|end|>
<|think|>  <|/think|>  <|tool|>  <|/tool|>
```

`encode(text, allow_special=False)` treats marker text in *user input* as
plain text - use it for anything untrusted, or a user can inject role
markers.

## The chat template

`ChatTemplate` renders message lists to token ids **and the training label
mask** in one place (prompt positions are `-100`):

```python
from minimodel.tokenization import BPETokenizer, ChatTemplate

tok = BPETokenizer.load("artifacts/tokenizer.json")
tpl = ChatTemplate(tok, default_system="Be brief.")

rendered = tpl.render({"instruction": "Hi", "output": "Hello!"})
rendered.input_ids, rendered.labels, rendered.n_supervised

prompt_ids = tpl.render_prompt([{"role": "user", "content": "Hi"}])  # ends with <|assistant|>
```

It accepts every common dataset shape (Alpaca fields, `messages`, ShareGPT
`conversations`, prompt/response pairs) via `normalize_messages`, renders
`reasoning` fields inside `<|think|>...<|/think|>`, and exposes
`supervise_reasoning=False` / `train_on_prompt=True` switches. Options worth
knowing: `stop_token_ids()` gives the generation stops for this format.

## Interop

`BPETokenizer.load()` also reads Hugging Face `tokenizers` JSON files with a
byte-level BPE model, so you can adopt an existing tokenizer instead of
training one. `save()` writes a single `tokenizer.json` that is bundled next
to model weights by every export path in this repo.

## Inspecting

```bash
minimodel tokenizer inspect artifacts/tokenizer.json                    # stats
minimodel tokenizer inspect artifacts/tokenizer.json --text "hello!"    # ids + pieces + roundtrip check
minimodel tokenizer inspect artifacts/tokenizer.json --ids 12 400 7     # decode
```
