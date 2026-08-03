"""Pull training corpora for the SLM experiments, cache as plain text shards.

TinyStories is the obvious base: Glint-2's samples ("Once upon a time, there
was a little girl named Lily") are pure TinyStories. We also pull a small
fineweb-edu sample for a knowledge/encyclopedic slice (Glint-2 clearly saw some,
given its WikiText behaviour and the probe trained on fineweb-edu).
"""
from __future__ import annotations
import sys
from pathlib import Path
from datasets import load_dataset

OUT = Path("research/data/train"); OUT.mkdir(parents=True, exist_ok=True)

def dump(name, ds, field, max_docs, max_chars):
    path = OUT / f"{name}.txt"
    n = chars = 0
    with path.open("w", encoding="utf-8") as f:
        for r in ds:
            t = r.get(field) or ""
            if not t.strip():
                continue
            f.write(t.strip() + "\n<eos>\n")
            n += 1; chars += len(t)
            if n >= max_docs or chars >= max_chars:
                break
    print(f"{name}: {n:,} docs, {chars:,} chars -> {path}")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "tinystories"):
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        dump("tinystories", ds, "text", max_docs=200_000, max_chars=120_000_000)
    if which in ("all", "tinystories_val"):
        ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
        dump("tinystories_val", ds, "text", max_docs=2_000, max_chars=2_000_000)
    if which in ("all", "fineweb"):
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        dump("fineweb_edu", ds, "text", max_docs=40_000, max_chars=40_000_000)
