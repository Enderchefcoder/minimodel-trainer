"""Train the research tokenizer and tokenize corpora into shards.

Produces, under research/artifacts/:
  tokenizer_v{VOCAB}.json           byte-level BPE (fast backend)
  tokenized/<name>_v{VOCAB}/        packed uint16 token shards (our shard format)

The default vocab is 4096, matching Glint-2, so byte-per-token and model-budget
comparisons line up.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from minimodel.datasets.tokenize_datasets import tokenize_text_records
from minimodel.tokenization.tokenize import BPETokenizer, train_tokenizer as _train_bpe

ART = Path("research/artifacts")
TRAIN = Path("research/data/train")


def iter_documents(path: Path):
    """Yield documents from a `<eos>`-delimited text file."""
    buf: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip() == "<eos>":
            if buf:
                yield {"text": "\n".join(buf)}
                buf = []
        else:
            buf.append(line)
    if buf:
        yield {"text": "\n".join(buf)}


def train_tokenizer(sources: list[Path], vocab: int, sample_docs: int) -> BPETokenizer:
    texts: list[str] = []
    for src in sources:
        for i, doc in enumerate(iter_documents(src)):
            if i >= sample_docs:
                break
            texts.append(doc["text"])
    t0 = time.perf_counter()
    tok = _train_bpe(texts, vocab_size=vocab, backend="fast")
    print(f"trained {vocab}-vocab tokenizer on {len(texts):,} docs in {time.perf_counter()-t0:.1f}s")
    print(f"  bytes/token on sample: {tok.compression_ratio(texts[:500]):.2f}")
    return tok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--tokenizer-sample-docs", type=int, default=60000)
    parser.add_argument(
        "--tokenize",
        nargs="*",
        default=["tinystories", "tinystories_val"],
        help="dataset stems under research/data/train to tokenize",
    )
    parser.add_argument("--train-tokenizer-on", nargs="*", default=["tinystories"])
    args = parser.parse_args()

    ART.mkdir(parents=True, exist_ok=True)
    tok_path = ART / f"tokenizer_v{args.vocab}.json"
    if tok_path.exists():
        tok = BPETokenizer.load(tok_path)
        print(f"loaded existing tokenizer {tok_path} (vocab {tok.vocab_size})")
    else:
        sources = [TRAIN / f"{s}.txt" for s in args.train_tokenizer_on]
        tok = train_tokenizer(sources, args.vocab, args.tokenizer_sample_docs)
        tok.save(tok_path)
        print(f"saved -> {tok_path}")

    for stem in args.tokenize:
        src = TRAIN / f"{stem}.txt"
        if not src.exists():
            print(f"skip {stem}: {src} not found")
            continue
        out = ART / "tokenized" / f"{stem}_v{args.vocab}"
        if (out / "index.json").exists():
            print(f"skip {stem}: already tokenized at {out}")
            continue
        t0 = time.perf_counter()
        stats = tokenize_text_records(iter_documents(src), tok, out, source=stem)
        print(f"{stem}: {stats['n_tokens']:,} tokens in {time.perf_counter()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
