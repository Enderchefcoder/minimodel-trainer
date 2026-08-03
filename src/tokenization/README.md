# tokenization/

- `tokenize.py` — self-contained byte-level BPE: `BPETokenizer.train/encode/
  decode/save/load`, plus `train_tokenizer` which uses the Rust `tokenizers`
  backend when installed (training only; runtime is identical either way).
  The split pattern is total — a test asserts no character can be dropped.
- `chat.py` — `ChatTemplate`: messages → token ids **and** the SFT label mask
  (prompt positions = -100), reasoning spans in `<|think|>…<|/think|>`,
  `normalize_messages` for Alpaca/ShareGPT/messages shapes.

Docs: [docs/tokenization.md](../../docs/tokenization.md).
