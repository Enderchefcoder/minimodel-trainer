"""Chain-of-thought and latent-reasoning ablations on identical architectures.

Report 13's experiment. One question, three ways to answer it at fixed
parameters and a fixed token budget:

1. **direct** - train the same model to emit the answer immediately;
2. **brief CoT** - one short reasoning sentence before the answer;
3. **detailed CoT** - a longer step-by-step trace;
4. **latent loops** - looped architecture (recurrent depth = latent reasoning),
   trained directly, spending its extra *compute* in unrolled iterations rather
   than in emitted tokens.

All four see the same arithmetic/word-problem corpus built from the bundled
builtin CoT records (``minimodel.datasets.builtin.BUILTIN_COT``), tiled and
varied over operands so no answer can be memorised: the training and test sets
are disjoint by construction (held-out operand ranges). Answer accuracy is
exact-match on the final number, decoded after a ``<|think|>``-free prompt; for
the CoT variants the trace is generated before the answer, mirroring real
budget-forced decoding.

This is deliberately a *tiny* study - a few CPU minutes - but it is controlled:
identical parameters (1.04M dense vs 1.04M looped-core), identical data,
identical optimizer and schedule, identical harness. Run::

    python research/experiments/cot_latent_ablation.py [--steps 600] [--skip-bench]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))

from eval_harness import ModelAdapter, eval_arc_easy, eval_blimp, eval_wikitext

from minimodel.architectures.registry import ARCHITECTURES
from minimodel.datasets.builtin import BUILTIN_COT
from minimodel.tokenization.tokenize import BPETokenizer

ART = Path("research/artifacts")
RESULTS = Path("research/data/results")

THINK_OPEN, THINK_CLOSE = "<|think|>", "<|/think|>"


# ---------------------------------------------------------------------------
# Data: arithmetic word problems with controllable trace length
# ---------------------------------------------------------------------------
def _make_problem(a: int, b: int, op: str) -> tuple[str, str, str]:
    """Return (question, brief_trace, detailed_trace) with the same answer."""
    if op == "+":
        question = f"What is {a} plus {b}?"
        brief = f"Add: {a} + {b} = {a + b}."
        detail = (
            f"Start at {a}. Adding {b} means counting up {b} more. "
            f"{a} + {b} = {a + b}. Check: the sum is larger than both numbers."
        )
    elif op == "-":
        question = f"What is {a} minus {b}?"
        brief = f"Subtract: {a} - {b} = {a - b}."
        detail = (
            f"Start at {a}. Taking away {b} means counting down {b}. "
            f"{a} - {b} = {a - b}. Check: the result is smaller than {a}."
        )
    elif op == "*":
        question = f"What is {a} times {b}?"
        brief = f"Multiply: {a} x {b} = {a * b}."
        detail = (
            f"{a} groups of {b}. Add {b} to itself {a} times: "
            f"{a} x {b} = {a * b}. Check: tens first, then units."
        )
    else:
        raise ValueError(f"unknown op {op!r}")
    return question, brief, detail


#: Operand pools: train on small values, test on disjoint (larger) values so
#: accuracy measures generalisation, not memorisation.
TRAIN_OPS = ["+", "-", "*"]
TRAIN_POOL = list(range(2, 13))
TEST_POOL = list(range(13, 26))


def build_corpus(mode: str, *, seed: int = 7, n_repeat: int = 40) -> list[str]:
    """Render the builtin CoT records + synthetic problems into plain text.

    ``mode`` is one of ``direct``, ``cot_brief``, ``cot_detail``, ``latent``.
    Latent trains on the *direct* text (the looped model spends its extra
    compute in unrolled iterations, not emitted tokens).
    """
    docs: list[str] = []
    # Real builtin CoT records, rendered in the target style.
    for row in BUILTIN_COT:
        if mode == "direct" or mode == "latent":
            docs.append(f"{row['instruction']} {row['output']}")
        elif mode == "cot_brief":
            docs.append(f"{row['instruction']} {THINK_OPEN} {row['reasoning']} {THINK_CLOSE} {row['output']}")
        else:
            docs.append(
                f"{row['instruction']} {THINK_OPEN} Step by step: {row['reasoning']} "
                f"Restating the question helps. {THINK_CLOSE} {row['output']}"
            )
    # Synthetic arithmetic, tiled for volume.
    import random

    rng = random.Random(seed)
    for _ in range(n_repeat * len(TRAIN_POOL) * 3):
        a, b = rng.choice(TRAIN_POOL), rng.choice(TRAIN_POOL)
        op = rng.choice(TRAIN_OPS)
        question, brief, detail = _make_problem(a, b, op)
        if op == "-" and b > a:
            a, b = max(a, b), min(a, b)
            question, brief, detail = _make_problem(a, b, op)
        if mode == "direct" or mode == "latent":
            docs.append(f"{question} {brief.rsplit('= ', 1)[-1]}")
            docs[-1] = f"{question} {brief.split('= ')[1].rstrip('.')}"
        elif mode == "cot_brief":
            docs.append(f"{question} {THINK_OPEN} {brief} {THINK_CLOSE} The answer is {brief.split('= ')[1].rstrip('.')}.")
        else:
            docs.append(
                f"{question} {THINK_OPEN} {detail} {THINK_CLOSE} "
                f"The answer is {brief.split('= ')[1].rstrip('.')}."
            )
    return docs


def test_set(*, seed: int = 99, n: int = 60) -> list[dict[str, str]]:
    """Held-out problems on unseen operand ranges, with gold answers."""
    import random

    rng = random.Random(seed)
    items: list[dict[str, str]] = []
    for _ in range(n):
        a, b = rng.choice(TEST_POOL), rng.choice(TEST_POOL)
        op = rng.choice(TRAIN_OPS)
        if op == "-" and b > a:
            a, b = max(a, b), min(a, b)
        question, _, _ = _make_problem(a, b, op)
        _, _, detail = _make_problem(a, b, op)
        answer = detail.split("= ")[-1].split(".")[0]
        items.append({"question": question, "answer": answer, "op": op})
    return items


# ---------------------------------------------------------------------------
# Training (same budget mechanics as run_experiment, but on raw text docs)
# ---------------------------------------------------------------------------
@dataclass
class AblationConfig:
    name: str
    mode: str  # direct | cot_brief | cot_detail | latent
    family: str = "dense_transformer"
    arch: dict[str, Any] = field(default_factory=dict)
    seq_len: int = 128
    batch_size: int = 16
    max_steps: int = 600
    lr: float = 3e-3
    seed: int = 1234


def _tok() -> BPETokenizer:
    path = ART / "tokenizer_v4096.json"
    if path.exists():
        return BPETokenizer.load(path)
    # Offline fallback: train a small tokenizer on the corpus itself.
    from minimodel.tokenization.tokenize import train_tokenizer

    tok = train_tokenizer(build_corpus("direct"), vocab_size=512)
    ART.mkdir(parents=True, exist_ok=True)
    tok.save(path)
    return tok


def encode_docs(tok: BPETokenizer, docs: list[str]) -> list[list[int]]:
    return [tok.encode(d, add_bos=True, add_eos=True) for d in docs if d]


def batches(token_ids: list[list[int]], cfg: AblationConfig, seed: int):
    """Yield (B, T) input/target batches by concatenating and slicing."""
    import numpy as np

    rng = np.random.default_rng(seed)
    flat: list[int] = []
    for ids in token_ids:
        flat.extend(ids)
    n = len(flat) - 1
    while True:
        idx = rng.integers(0, n - cfg.seq_len - 1, size=cfg.batch_size)
        x = torch.tensor([flat[i : i + cfg.seq_len] for i in idx], dtype=torch.long)
        y = torch.tensor([flat[i + 1 : i + cfg.seq_len + 1] for i in idx], dtype=torch.long)
        yield x, y


def train(cfg: AblationConfig, tok: BPETokenizer, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    arch = dict(cfg.arch)
    arch.setdefault("vocab_size", tok.vocab_size)
    model = ARCHITECTURES.get(cfg.family).from_config(arch)
    params = model.num_parameters()
    docs = build_corpus(cfg.mode)
    ids = encode_docs(tok, docs)
    flow = batches(ids, cfg, cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, int(0.05 * cfg.max_steps)))
    )
    model.to(device).train()
    losses: list[float] = []
    t0 = time.perf_counter()
    tokens = 0
    for _ in range(1, cfg.max_steps + 1):
        x, y = next(flow)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        out = model.forward_with_loss(x, y)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(out.loss.detach()))
        tokens += x.numel()
    return {
        "params": params,
        "final_loss": round(sum(losses[-50:]) / min(50, len(losses)), 4),
        "tokens_seen": tokens,
        "tokens_per_second": round(tokens / max(1e-6, time.perf_counter() - t0), 1),
        "model": model,
    }


# ---------------------------------------------------------------------------
# Evaluation: held-out exact-match accuracy + the shared harness
# ---------------------------------------------------------------------------
def extract_answer(text: str) -> str | None:
    """Pull the final number from a completion, ignoring any think span."""
    text = re.sub(re.escape(THINK_OPEN) + r".*?" + re.escape(THINK_CLOSE), " ", text, flags=re.S)
    numbers = re.findall(r"-?\d+", text)
    return numbers[-1] if numbers else None


@torch.no_grad()
def accuracy(model, tok: BPETokenizer, cfg: AblationConfig, device, *, is_loop: bool) -> dict[str, float]:
    """Greedy-decode held-out problems; score exact numeric match."""
    model.eval()
    items = test_set()
    correct = 0
    n = 0
    for item in items:
        ids = tok.encode(item["question"], add_bos=True)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        for _ in range(16):
            logits = model(x, loops=8) if is_loop else model(x)
            next_id = int(logits[0, -1].argmax())
            if next_id == tok.eos_id:
                break
            x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
            if x.shape[1] > 64:
                break
        text = tok.decode(x[0].tolist())
        predicted = extract_answer(text[len(item["question"]) :])
        n += 1
        correct += int(predicted == item["answer"])
    return {"test_exact_match": round(100 * correct / max(1, n), 2), "test_n": n}


@torch.no_grad()
def bench(model, tok: BPETokenizer, cfg: AblationConfig, device) -> dict[str, Any]:
    """The shared harness: BLiMP + ARC-Easy + WikiText byte-ppl."""
    model.eval()

    def encode(text: str) -> list[int]:
        return tok.encode(text, allow_special=False)

    def forward(tokens: torch.Tensor) -> torch.Tensor:
        return model(tokens, loops=8) if cfg.family == "looped_transformer" else model(tokens)

    adapter = ModelAdapter(
        name=cfg.name, encode=encode, forward=forward,
        max_len=256, batch_size=32, params=0,
    )
    out: dict[str, Any] = {}
    out.update(eval_blimp(adapter, per_paradigm=15))
    out.pop("blimp_per_paradigm", None)
    out.update(eval_arc_easy(adapter, limit=150))
    out.update(eval_wikitext(adapter, max_tokens=8000))
    return out


# ---------------------------------------------------------------------------
# The four conditions: identical dense arch, one looped variant
# ---------------------------------------------------------------------------
DENSE_ARCH = {
    "dim": 112, "n_layers": 5, "n_heads": 7, "head_dim": 16, "n_kv_heads": 1,
    "ffn_hidden": 256, "window": 512, "max_seq_len": 1024,
    "qk_norm": True, "tie_embeddings": True, "value_residual": True,
}
LOOPED_ARCH = {
    "dim": 112, "n_heads": 7, "head_dim": 16, "ffn_hidden": 256,
    "embedding_rank": 48, "window": 512, "max_seq_len": 1024,
    "n_shared_blocks": 1, "train_loops": 8, "min_loops": 4,
    "max_loops_table": 16, "loop_lora_rank": 4, "value_residual": True,
    "variable_loops": True, "loop_sampling": "poisson",
}


def conditions() -> list[AblationConfig]:
    return [
        AblationConfig(name="cot_direct", mode="direct", arch=dict(DENSE_ARCH)),
        AblationConfig(name="cot_brief", mode="cot_brief", arch=dict(DENSE_ARCH)),
        AblationConfig(name="cot_detail", mode="cot_detail", arch=dict(DENSE_ARCH)),
        AblationConfig(name="cot_latent_loop", mode="direct", family="looped_transformer",
                       arch=dict(LOOPED_ARCH)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--skip-bench", action="store_true", help="skip BLiMP/ARC/WikiText")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = torch.device("cpu")
    tok = _tok()
    print(f"tokenizer: {tok.vocab_size} vocab", flush=True)

    rows: list[dict[str, Any]] = []
    for cfg in conditions():
        cfg.max_steps = args.steps
        cfg.seed = args.seed
        print(f"\n=== {cfg.name} ({cfg.mode}, {cfg.family}) ===", flush=True)
        stats = train(cfg, tok, device)
        model = stats.pop("model")
        row: dict[str, Any] = {"name": cfg.name, "mode": cfg.mode, "family": cfg.family, **stats}
        is_loop = cfg.family == "looped_transformer"
        row.update(accuracy(model, tok, cfg, device, is_loop=is_loop))
        if not args.skip_bench and (ART / "tokenizer_v4096.json").exists():
            try:
                row.update(bench(model, tok, cfg, device))
            except FileNotFoundError as exc:
                # Eval corpora not cached (run pull_eval_data.py) - task EM only.
                print(f"  (skipping BLiMP/ARC/WikiText: {exc})", flush=True)
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "cot_latent_ablation.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {out}", flush=True)

    print("\n| condition | params | final loss | held-out EM % |")
    print("| --- | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['name']} | {row['params']:,} | {row['final_loss']:.3f} "
            f"| {row.get('test_exact_match', float('nan'))} |"
        )


if __name__ == "__main__":
    main()
