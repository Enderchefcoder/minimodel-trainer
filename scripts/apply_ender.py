#!/usr/bin/env python3
"""Graft ENDER onto a pretrained backbone (default: GPT-X2.5-135M).

This is the *adaptation* entry point for ENDER
("Endogenous Neural Depth with Evolving Recurrence"). It:

1. loads the backbone (``--backbone``, default
   ``AxiomicLabs/GPT-X2.5-135M`` via :mod:`transformers`; falls back to a local
   ``minimodel`` dense checkpoint, and finally to a small random backbone so the
   whole flow is demonstrable offline);
2. grafts an :class:`~minimodel.architectures.ender.EnderAdapter` in front of
   the backbone's final norm — a no-op at init, by design;
3. runs a short staged training pass (recurrence only, backbone frozen) so the
   adapter actually learns;
4. saves the wrapped model to ``--output`` (default ``runs/ender-adapted``).

Usage::

    python scripts/apply_ender.py                       # HF backbone, full flow
    python scripts/apply_ender.py --steps 0             # graft only, no training
    python scripts/apply_ender.py --backbone runs/pretrain/model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from minimodel.architectures.ender import (  # noqa: E402
    EnderAdapter,
    EnderLoss,
    build_ender_optimizer,
)
from minimodel.core.logging_utils import get_logger, setup_logging  # noqa: E402

logger = get_logger("apply-ender")

DEFAULT_BACKBONE = "AxiomicLabs/GPT-X2.5-135M"


def load_backbone(spec: str) -> tuple[Any, Any]:
    """Load the named backbone, preferring local minimodel checkpoints.

    Returns ``(model, tokenizer_or_None)``. HF models are wrapped so the rest of
    the script never has to care which framework produced the weights.
    """
    path = Path(spec)
    if path.exists() and (path / "config.json").exists():
        from minimodel.architectures.builder import load_model

        model = load_model(path)
        tokenizer = None
        tokenizer_path = path / "tokenizer.json"
        if tokenizer_path.exists():
            from minimodel.tokenization.tokenize import BPETokenizer

            tokenizer = BPETokenizer.load(tokenizer_path)
        return model, tokenizer

    # HuggingFace backbone (network). Kept behind the local path so everything
    # else in the repository still works offline (repo rule 2).
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(spec)
        model = AutoModelForCausalLM.from_pretrained(spec, config=config)
        return model, None
    except Exception as exc:
        logger.warning("could not load %s (%s); falling back to a small local backbone", spec, exc)
        from minimodel.architectures.builder import build_model

        return build_model("dense_3m", verify_budget=False), None


def backbone_dim(model: Any) -> int:
    """Hidden size of the backbone, whichever attribute names it uses."""
    for attribute in ("d_model", "hidden_size", "n_embd", "dim"):
        value = getattr(model.config, attribute, None) if hasattr(model, "config") else None
        if value:
            return int(value)
    if hasattr(model, "dim"):
        return int(model.dim)
    raise ValueError("cannot infer the backbone hidden size; pass --dim")


def main() -> int:
    """Graft, optionally train, and save the adapted model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, help="HF id or local model dir")
    parser.add_argument("-o", "--output", default="runs/ender-adapted")
    parser.add_argument("--r-latent", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--dim", type=int, help="override the inferred backbone hidden size")
    parser.add_argument("--steps", type=int, default=20, help="adaptation optimizer steps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--unfreeze-top-blocks", type=int, default=0)
    parser.add_argument("--lambda-delta", type=float, default=0.05)
    parser.add_argument("--lambda-kd", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    setup_logging(force=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    backbone, tokenizer = load_backbone(args.backbone)
    backbone = backbone.to(device)
    dim = args.dim or backbone_dim(backbone)
    logger.info("backbone hidden size: %d", dim)

    adapter = EnderAdapter(
        backbone,
        {"dim": dim, "r_latent": args.r_latent, "num_steps": args.num_steps},
    ).to(device)

    # Sanity: a fresh adapter must be an exact no-op on the backbone.
    tokens = torch.randint(0, 100, (2, 16), device=device)
    adapter.eval()
    with torch.no_grad():
        wrapped = adapter(tokens)
        backbone.eval()
        if hasattr(backbone, "forward"):
            try:
                reference = backbone(tokens).logits if hasattr(backbone(tokens), "logits") else backbone(tokens)
            except TypeError:  # HF models want dict input
                reference = backbone(input_ids=tokens).logits
        else:
            reference = backbone(tokens)
        if wrapped.shape == reference.shape:
            max_diff = float((wrapped - reference).abs().max())
            logger.info("no-op check: max|wrapped - backbone| = %.2e", max_diff)

    if args.steps > 0:
        from minimodel.datasets.builtin import builtin_records

        optimizer = build_ender_optimizer(
            adapter,
            base_lr=args.lr,
            unfreeze_top_blocks=args.unfreeze_top_blocks,
        )
        loss_engine = EnderLoss(
            lambda_delta=args.lambda_delta,
            lambda_kd=args.lambda_kd,
        )
        texts = [record["text"] for record in builtin_records("pretrain", repeat=20)]
        if tokenizer is None:
            from minimodel.tokenization.tokenize import BPETokenizer

            tokenizer = BPETokenizer.train(texts[:40], vocab_size=512)

        def encode(text: str) -> list[int]:
            ids = tokenizer.encode(text, add_bos=True)
            return ids[: args.seq_len] or [1]

        adapter.train()
        for step in range(args.steps):
            batch_ids = [
                encode(texts[(step * args.batch_size + i) % len(texts)])
                for i in range(args.batch_size)
            ]
            width = max(len(ids) for ids in batch_ids)
            batch = torch.tensor(
                [ids + [0] * (width - len(ids)) for ids in batch_ids], device=device
            )
            logits = adapter(batch)
            targets = batch[:, 1:]
            loss, extras = loss_engine(logits[:, :-1], targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in adapter.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
                logger.info(
                    "step %d/%d loss=%.4f (ce=%.4f delta=%.5f)",
                    step + 1,
                    args.steps,
                    loss.item(),
                    extras.get("loss_ce", 0.0),
                    extras.get("loss_delta", 0.0),
                )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), output / "ender_adapter.pt")
    metadata = {
        "backbone": args.backbone,
        "dim": dim,
        "r_latent": args.r_latent,
        "num_steps": args.num_steps,
        "trained_steps": args.steps,
        "lambda_delta": args.lambda_delta,
        "lambda_kd": args.lambda_kd,
        "final_norm": type(adapter.final_norm).__name__,
    }
    (output / "ender_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("saved ENDER adapter -> %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
