"""Evaluate the released Glint-2 checkpoint with our harness.

This validates the harness against Glint-2's published numbers (BLiMP 73.96,
ARC-Easy 36.80, WikiText-2 ppl 3.09) and produces the head-to-head baseline all
later experiments are measured against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))

from eval_harness import ModelAdapter, run_all  # noqa: E402
from glint2_model import load_glint2  # noqa: E402


def build_adapter(loops: int = 8, max_len: int = 256, batch_size: int = 32) -> ModelAdapter:
    model = load_glint2("research/baselines/glint2/glint-2.pt", loops=loops)
    tok = Tokenizer.from_file("research/baselines/glint2/tokenizer.json")

    def encode(text: str) -> list[int]:
        return list(tok.encode(text).ids)

    def forward(tokens: torch.Tensor) -> torch.Tensor:
        return model(tokens, loops=loops)

    def n_bytes(ids: list[int]) -> int:
        return len(tok.decode(ids).encode("utf-8"))

    return ModelAdapter(
        name=f"glint-2 (loops={loops})",
        encode=encode,
        forward=forward,
        n_bytes=n_bytes,
        max_len=max_len,
        batch_size=batch_size,
        params=model.num_parameters(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loops", type=int, default=8)
    parser.add_argument("--blimp-per-paradigm", type=int, default=None)
    parser.add_argument("--arc-limit", type=int, default=None)
    parser.add_argument("--wikitext-max-tokens", type=int, default=None)
    parser.add_argument("--wikitext-stride", type=int, default=256)
    parser.add_argument("-o", "--output", default="research/data/results/glint2.json")
    args = parser.parse_args()

    adapter = build_adapter(loops=args.loops)
    torch.manual_seed(0)
    result = run_all(
        adapter,
        blimp_per_paradigm=args.blimp_per_paradigm,
        arc_limit=args.arc_limit,
        wikitext_max_tokens=args.wikitext_max_tokens,
    )
    payload = result.to_dict()
    payload.pop("blimp_per_paradigm", None)  # keep the top-line file compact
    print(json.dumps(payload, indent=2))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
