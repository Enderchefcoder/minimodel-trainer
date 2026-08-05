"""Train one model config and evaluate it on the shared harness.

Every ablation calls `run_experiment(...)` with a name, an architecture config,
and a training budget. Output is a compact JSON in research/data/results/ with
the model's parameter count, final/val loss, throughput, and BLiMP / ARC-Easy /
WikiText byte-ppl measured by the *same* harness used on Glint-2 — so numbers
are directly comparable to the baseline.

Designed for a 4-core CPU: models ~1-2M params, short runs for relative signal,
subsampled evals during ablation (full evals reserved for the final contender).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))

from eval_harness import ModelAdapter, eval_arc_easy, eval_blimp, eval_wikitext  # noqa: E402

from minimodel.architectures.registry import ARCHITECTURES  # noqa: E402
from minimodel.datasets.loader import PackedTextDataset, build_dataloader, infinite_loader  # noqa: E402
from minimodel.tokenization.tokenize import BPETokenizer  # noqa: E402
from minimodel.training.optim import build_optimizer  # noqa: E402
from minimodel.training.schedules import build_scheduler, resolve_warmup  # noqa: E402

ART = Path("research/artifacts")
RESULTS = Path("research/data/results")
RESULTS.mkdir(parents=True, exist_ok=True)


@dataclass
class ExpConfig:
    name: str
    family: str = "looped_transformer"
    arch: dict[str, Any] = field(default_factory=dict)
    # data / budget
    vocab: int = 4096
    train_corpus: str = "tinystories_v4096"
    val_corpus: str = "tinystories_val_v4096"
    seq_len: int = 256
    batch_size: int = 32
    grad_accum: int = 1
    max_steps: int = 2000
    # optim
    optimizer: str = "adamw"
    lr: float = 3e-3
    weight_decay: float = 0.1
    warmup: float = 0.03
    schedule: str = "cosine"
    grad_clip: float = 1.0
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    schedule_kwargs: dict[str, Any] = field(default_factory=dict)
    seed: int = 1234
    # eval
    eval_loops: int | None = None
    blimp_per_paradigm: int | None = 40
    arc_limit: int | None = 400
    wikitext_max_tokens: int | None = 20000
    log_every: int = 200


def build_model_from(cfg: ExpConfig):
    arch = dict(cfg.arch)
    arch.setdefault("vocab_size", cfg.vocab)
    model_cls = ARCHITECTURES.get(cfg.family)
    return model_cls.from_config(arch)


def train(model, cfg: ExpConfig, device: torch.device) -> dict[str, Any]:
    ds = PackedTextDataset(ART / "tokenized" / cfg.train_corpus, seq_len=cfg.seq_len, seed=cfg.seed)
    loader = build_dataloader(ds, batch_size=cfg.batch_size, seed=cfg.seed, drop_last=True)
    it = infinite_loader(loader)

    optimizer = build_optimizer(
        model, cfg.optimizer, lr=cfg.lr, weight_decay=cfg.weight_decay, **cfg.optimizer_kwargs
    )
    scheduler = build_scheduler(
        optimizer,
        cfg.schedule,
        total_steps=cfg.max_steps,
        warmup_steps=resolve_warmup(cfg.warmup, cfg.max_steps),
        **cfg.schedule_kwargs,
    )
    model.train()
    losses: list[float] = []
    tokens = 0
    t0 = time.perf_counter()
    for step in range(1, cfg.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(cfg.grad_accum):
            batch = next(it)
            inp = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)
            out = model.forward_with_loss(inp, tgt)
            (out.loss / cfg.grad_accum).backward()
            total += float(out.loss.detach())
            tokens += inp.numel()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        losses.append(total / cfg.grad_accum)
        if step % cfg.log_every == 0 or step == cfg.max_steps:
            recent = sum(losses[-cfg.log_every :]) / len(losses[-cfg.log_every :])
            print(f"  step {step:>5}/{cfg.max_steps}  loss {recent:.4f}  "
                  f"{tokens / (time.perf_counter() - t0):,.0f} tok/s", flush=True)
    elapsed = time.perf_counter() - t0
    return {
        "final_loss": round(sum(losses[-50:]) / min(50, len(losses)), 4),
        "train_seconds": round(elapsed, 1),
        "train_tokens": tokens,
        "tokens_per_second": round(tokens / elapsed, 1),
    }


@torch.no_grad()
def val_loss(model, cfg: ExpConfig, device: torch.device, n_batches: int = 20) -> float:
    ds = PackedTextDataset(
        ART / "tokenized" / cfg.val_corpus, seq_len=cfg.seq_len, seed=cfg.seed + 1
    )
    loader = build_dataloader(ds, batch_size=cfg.batch_size, seed=cfg.seed + 1, drop_last=True)
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        out = model.forward_with_loss(batch["input_ids"].to(device), batch["labels"].to(device))
        total += float(out.loss)
        n += 1
    return round(total / max(1, n), 4)


def make_adapter(model, tok: BPETokenizer, cfg: ExpConfig) -> ModelAdapter:
    model.eval()
    loops = cfg.eval_loops
    family_key = ARCHITECTURES.normalize(cfg.family)
    forward_kwargs = {"loops": loops} if (loops and family_key == "looped_transformer") else {}

    def encode(text: str) -> list[int]:
        return tok.encode(text, allow_special=False)

    def forward(tokens: torch.Tensor) -> torch.Tensor:
        return model(tokens, **forward_kwargs)

    def n_bytes(ids: list[int]) -> int:
        return len(tok.decode(ids).encode("utf-8"))

    return ModelAdapter(
        name=cfg.name, encode=encode, forward=forward, n_bytes=n_bytes,
        max_len=256, batch_size=32, params=model.num_parameters(),
    )


def run_experiment(cfg: ExpConfig, *, save: bool = True, do_eval: bool = True) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    device = torch.device("cpu")
    tok = BPETokenizer.load(ART / f"tokenizer_v{cfg.vocab}.json")

    model = build_model_from(cfg).to(device)
    params = model.num_parameters()
    print(f"\n=== {cfg.name} ({cfg.family}, {params:,} params) ===", flush=True)

    train_stats = train(model, cfg, device)
    result: dict[str, Any] = {
        "name": cfg.name,
        "family": cfg.family,
        "params": params,
        "config": asdict(cfg),
        **train_stats,
        "val_loss": val_loss(model, cfg, device),
    }

    if do_eval:
        adapter = make_adapter(model, tok, cfg)
        t0 = time.perf_counter()
        result.update(eval_blimp(adapter, per_paradigm=cfg.blimp_per_paradigm))
        result.update(eval_arc_easy(adapter, limit=cfg.arc_limit))
        result.update(eval_wikitext(adapter, max_tokens=cfg.wikitext_max_tokens))
        result.pop("blimp_per_paradigm", None)
        result["eval_seconds"] = round(time.perf_counter() - t0, 1)

    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2), flush=True)
    if save:
        out = RESULTS / f"{cfg.name}.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"saved -> {out}", flush=True)
    return result
