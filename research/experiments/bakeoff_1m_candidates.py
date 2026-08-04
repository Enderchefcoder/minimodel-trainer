#!/usr/bin/env python3
"""Short offline bake-off of the 20 ~1M Glint-2 candidate architectures.

Trains each ``mm1m_r*`` template for a few hundred thousand tokens on the
builtin corpus and ranks them by final training loss. Writes
``research/data/results/arch_1m_candidates.json`` so the ordered leaderboard
stays machine-readable.

Usage::

    python research/experiments/bakeoff_1m_candidates.py [--steps 40]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from minimodel.architectures.builder import build_model, list_glint2_candidates  # noqa: E402
from minimodel.datasets.builtin import builtin_records  # noqa: E402
from minimodel.datasets.loader import (  # noqa: E402
    PackedTextDataset,
    build_dataloader,
    infinite_loader,
)
from minimodel.datasets.tokenize_datasets import tokenize_text_records  # noqa: E402
from minimodel.tokenization.tokenize import BPETokenizer  # noqa: E402
from minimodel.training.optim import build_optimizer  # noqa: E402

RESULTS = REPO_ROOT / "research" / "data" / "results"


def _prepare_corpus(tmp: Path, vocab: int = 1024) -> Path:
    """Tokenize the builtin pretrain texts into a packed shard directory."""
    out_dir = tmp / "pretrain"
    if out_dir.exists() and any(out_dir.glob("*.bin")):
        return out_dir
    records = builtin_records("pretrain", repeat=40)
    texts = []
    for record in records:
        text = str(record.get("text") or record.get("content") or "")
        if text:
            texts.append(text)
    tokenizer = BPETokenizer.train(texts, vocab_size=vocab)
    tokenizer.save(tmp / "tokenizer.json")
    # Re-wrap so extract_text finds the field consistently.
    tokenize_text_records(({"text": t} for t in texts), tokenizer, out_dir)
    return out_dir


def _train_one(name: str, shard: Path, *, steps: int, seq_len: int, batch: int) -> dict:
    model = build_model(name, verify_budget=True)
    device = torch.device("cpu")
    model.to(device)
    ds = PackedTextDataset(shard, seq_len=seq_len, seed=0)
    loader = build_dataloader(ds, batch_size=batch, seed=0, drop_last=True)
    it = infinite_loader(loader)
    opt = build_optimizer(model, "adamw", lr=3e-3, weight_decay=0.1)
    model.train()
    losses: list[float] = []
    t0 = time.perf_counter()
    tokens = 0
    for _ in range(steps):
        batch_data = next(it)
        inp = batch_data["input_ids"].to(device)
        tgt = batch_data["labels"].to(device)
        out = model.forward_with_loss(inp, tgt)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss.detach()))
        tokens += int(inp.numel())
    elapsed = time.perf_counter() - t0
    return {
        "name": name,
        "params": model.num_parameters(),
        "final_loss": round(sum(losses[-10:]) / min(10, len(losses)), 4),
        "train_tokens": tokens,
        "train_seconds": round(elapsed, 2),
        "tok_per_s": round(tokens / max(elapsed, 1e-6)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS / "arch_1m_candidates.json",
    )
    args = parser.parse_args()

    candidates = list_glint2_candidates()
    if len(candidates) != 20:
        raise SystemExit(f"expected 20 glint2 candidates, found {len(candidates)}")

    tmp = REPO_ROOT / "research" / "artifacts" / "mm1m_bakeoff"
    tmp.mkdir(parents=True, exist_ok=True)
    shard = _prepare_corpus(tmp)

    measured: list[dict] = []
    for row in candidates:
        print(f"==> {row['rank']:02d} {row['name']}", flush=True)
        result = _train_one(
            row["name"], shard, steps=args.steps, seq_len=args.seq_len, batch=args.batch
        )
        result.update(
            {
                "prior_rank": row["rank"],
                "candidate_class": row["candidate_class"],
                "family": row["family"],
                "description": row["description"],
            }
        )
        print(
            f"    loss={result['final_loss']:.4f}  params={result['params']:,}  "
            f"{result['tok_per_s']:,} tok/s",
            flush=True,
        )
        measured.append(result)

    by_loss = sorted(
        measured,
        key=lambda r: (
            1 if r["final_loss"] != r["final_loss"] else 0,  # NaN last
            r["final_loss"] if r["final_loss"] == r["final_loss"] else 0.0,
        ),
    )
    for i, row in enumerate(by_loss, start=1):
        row["measured_rank"] = i

    payload = {
        "protocol": {
            "steps": args.steps,
            "seq_len": args.seq_len,
            "batch": args.batch,
            "corpus": "builtin pretrain (tokenized)",
            "optimizer": "adamw lr=3e-3",
            "note": (
                "Short-budget ranking for relative signal only; not a claim of "
                "final Glint-2 win. Prior ranks come from reports 03/04/07/11."
            ),
        },
        "ordered_by_prior": measured,
        "ordered_by_measured_loss": by_loss,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print("Measured order (loss ↑):")
    for row in by_loss:
        print(
            f"  {row['measured_rank']:2d}. {row['name']}  "
            f"loss={row['final_loss']:.4f}  prior={row['prior_rank']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
