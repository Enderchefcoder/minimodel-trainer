"""Pull the benchmark datasets Glint-2 reports on, cache them as JSONL.

Targets: BLiMP (all 67 paradigms, minimal pairs), ARC-Easy (test, multiple
choice), WikiText-2 (test, raw text for perplexity / bits-per-byte). Cached
under research/data/eval/ so every later experiment is offline and reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset

OUT = Path("research/data/eval")
OUT.mkdir(parents=True, exist_ok=True)


def _write(name: str, rows: list[dict]) -> None:
    path = OUT / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows):>7,} rows -> {path}")


def pull_blimp() -> None:
    configs = get_dataset_config_names("nyu-mll/blimp")
    print(f"BLiMP has {len(configs)} paradigms")
    rows: list[dict] = []
    for config in configs:
        ds = load_dataset("nyu-mll/blimp", config, split="train")
        for r in ds:
            rows.append(
                {
                    "paradigm": config,
                    "field": r.get("field"),
                    "good": r["sentence_good"],
                    "bad": r["sentence_bad"],
                }
            )
    _write("blimp", rows)


def pull_arc_easy() -> None:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    rows = []
    for r in ds:
        rows.append(
            {
                "question": r["question"],
                "choices": r["choices"]["text"],
                "labels": r["choices"]["label"],
                "answerKey": r["answerKey"],
            }
        )
    _write("arc_easy", rows)


def pull_wikitext() -> None:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(r["text"] for r in ds)
    (OUT / "wikitext2_test.txt").write_text(text, encoding="utf-8")
    print(f"wrote {len(text):,} chars -> {OUT / 'wikitext2_test.txt'}")


if __name__ == "__main__":
    pull_blimp()
    pull_arc_easy()
    pull_wikitext()
