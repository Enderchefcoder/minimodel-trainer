#!/usr/bin/env python3
"""Benchmark GPT-X2.5-135M before and after a short ENDER adaptation.

Scores the *real* model (``AxiomicLabs/GPT-X2.5-135M``) on seeded slices of the
benchmarks the user cares about — BLiMP, ARC-Easy, HellaSwag and PIQA (the
Open_SLM_Leaderboard's core tasks) — with the plain backbone and then with an
:class:`~minimodel.architectures.ender.EnderAdapter` trained briefly on real
text with the model's own tokenizer. Writes a JSON report and prints a
side-by-side table.

This is a *measurement* entry point: it deliberately reuses the repository's
ENDER machinery (``EnderAdapter``, ``EnderLoss``, ``build_ender_optimizer``)
without adding new abstractions, so the numbers it prints are exactly what the
library would produce on a GPU box. Run it as::

    python scripts/bench_gptx2_ender.py --train-steps 60

The full leaderboard (MMLU, GSM8K, Winogrande, ...) needs GPU-scale time; pass
``--tasks arc_easy hellaswag piqa`` etc. to trim the slice budget.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

BACKBONE_ID = "AxiomicLabs/GPT-X2.5-135M"

#: BLiMP subtasks spanning distinct syntactic phenomena (cloze minimal pairs).
BLIMP_SUBTASKS = [
    "anaphor_gender_agreement",
    "determiner_noun_agreement_with_adj_2",
    "ellipsis_n_bar_1",
    "irregular_past_participle_adjectives",
    "sentential_negation_npi_scope",
    "tough_vs_raising_1",
]

#: Seeded slice sizes (examples) per task; small enough for CPU, large enough
#: to show a directional signal. ``blimp`` counts *pairs per subtask*.
SLICES = {"arc_easy": 120, "hellaswag": 80, "piqa": 80, "blimp": 120}


def pick_rows(dataset: Any, size: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic sample of rows (seed * k + index pattern from repo rule 7)."""
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(dataset), size=min(size, len(dataset)), replace=False))
    return [dataset[int(i)] for i in indices]


def sequence_logprobs(model: Any, tokenizer: Any, texts: list[str], device: torch.device) -> list[tuple[float, int]]:
    """(Sum of token log-probs, token count) per text, choices batched together.

    Padding is masked out so every sequence is scored on its own tokens only.
    """
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = encoded.input_ids.to(device)
    attention = encoded.attention_mask.to(device)
    with torch.no_grad():
        try:
            logits = model(input_ids=input_ids, attention_mask=attention).logits
        except TypeError:
            logits = model(input_ids=input_ids).logits
    scores: list[tuple[float, int]] = []
    for i in range(len(texts)):
        seq = input_ids[i]
        mask = attention[i].bool()
        length = int(mask.sum())
        target = seq[1 : length + 1]
        pred = logits[i, :length, :]
        token_logprobs = torch.log_softmax(pred.float(), dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
        scores.append((float(token_logprobs.sum()), length - 1))
    return scores


def choice_scores(model: Any, tokenizer: Any, prompt: str, choices: list[str], device: torch.device) -> list[float]:
    """Mean log-prob of each choice's *continuation* after the shared prompt.

    Averaging over the full prompt+choice dilutes the choice signal (a choice
    that is one easy token longer wins by raw length); subtracting the prompt's
    own log-prob and normalising by the choice length is the standard
    multiple-choice protocol and matches the leaderboards' numbers far better.
    """
    texts = [prompt + c for c in choices]
    (prompt_sum, prompt_len) = sequence_logprobs(model, tokenizer, [prompt], device)[0]
    per_choice = sequence_logprobs(model, tokenizer, texts, device)
    return [(full_sum - prompt_sum) / (full_len - prompt_len) for full_sum, full_len in per_choice]


def score_task(model: Any, tokenizer: Any, task: str, slice_size: int, seed: int, device: torch.device) -> dict[str, float]:
    """Accuracy of the model on a seeded slice of one task."""
    from datasets import load_dataset

    if task == "blimp":
        correct = total = 0
        per_subtask: dict[str, float] = {}
        for subtask in BLIMP_SUBTASKS:
            ds = load_dataset("lukaemon/blimp", data_files=f"{subtask}.jsonl", split="train")
            rows = pick_rows(ds, slice_size, seed)
            good = 0
            for row in rows:
                good_text, bad_text = row["sentence_good"], row["sentence_bad"]
                (good_sum, good_len), (bad_sum, bad_len) = sequence_logprobs(
                    model, tokenizer, [good_text, bad_text], device
                )
                good += 1 if good_sum / good_len > bad_sum / bad_len else 0
            acc = good / len(rows)
            per_subtask[subtask] = acc
            correct += good
            total += len(rows)
        return {"acc": correct / total, **per_subtask}

    if task == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        rows = pick_rows(ds, slice_size, seed)
        correct = 0
        for row in rows:
            choices = row["choices"]
            lps = choice_scores(model, tokenizer, row["question"] + "\n", list(choices["text"]), device)
            label = choices["label"].index(row["answerKey"])
            correct += 1 if int(np.argmax(lps)) == label else 0
        return {"acc": correct / len(rows)}

    if task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        rows = pick_rows(ds, slice_size, seed)
        correct = 0
        for row in rows:
            lps = choice_scores(model, tokenizer, row["ctx"], list(row["endings"]), device)
            correct += 1 if int(np.argmax(lps)) == int(row["label"]) else 0
        return {"acc": correct / len(rows)}

    if task == "piqa":
        ds = load_dataset("ybisk/piqa", split="validation")
        rows = pick_rows(ds, slice_size, seed)
        correct = 0
        for row in rows:
            lps = choice_scores(model, tokenizer, row["goal"] + " ", [row["sol1"], row["sol2"]], device)
            correct += 1 if int(np.argmax(lps)) == int(row["label"]) else 0
        return {"acc": correct / len(rows)}

    raise ValueError(f"unknown task {task!r}; expected one of arc_easy, hellaswag, piqa, blimp")


def train_ender(
    adapter: Any,
    tokenizer: Any,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    unfreeze_top_blocks: int,
    seed: int,
    device: torch.device,
) -> float:
    """Short staged adaptation on a slice of TinyStories, returns final loss."""
    from datasets import load_dataset
    from minimodel.architectures.ender import EnderLoss, build_ender_optimizer

    torch.manual_seed(seed)
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    texts: list[str] = []
    for _i, row in enumerate(ds):
        texts.append(row["text"])
        if len(texts) >= 400:
            break

    optimizer = build_ender_optimizer(adapter, base_lr=lr, unfreeze_top_blocks=unfreeze_top_blocks)
    loss_engine = EnderLoss(lambda_delta=0.05, lambda_kd=0.0)
    adapter.train()
    final_loss = float("nan")
    start = time.time()
    for step in range(steps):
        chunk = texts[(step * batch_size) % len(texts) : (step * batch_size) % len(texts) + batch_size]
        while len(chunk) < batch_size:  # wrap around the corpus
            chunk += texts[: batch_size - len(chunk)]
        ids_list = [tokenizer.encode(t)[:seq_len] or [1] for t in chunk]
        width = max(len(ids) for ids in ids_list)
        batch = torch.tensor(
            [ids + [0] * (width - len(ids)) for ids in ids_list], device=device
        )
        logits = adapter(batch)
        loss, extras = loss_engine(logits[:, :-1], batch[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        final_loss = float(loss)
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            elapsed = time.time() - start
            print(
                f"  train step {step + 1}/{steps} loss={final_loss:.4f} "
                f"ce={float(extras.get('loss_ce', 0.0)):.4f} "
                f"delta={float(extras.get('loss_delta', 0.0)):.5f} "
                f"({elapsed:.0f}s so far)",
                flush=True,
            )
    return final_loss


def load_backbone(device: torch.device) -> tuple[Any, Any]:
    """The real GPT-X2.5-135M and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BACKBONE_ID, trust_remote_code=True)
    model.eval()
    return model.to(device), tokenizer


def main() -> int:
    """Run the before/after benchmark and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["arc_easy", "hellaswag", "piqa", "blimp"])
    parser.add_argument("--train-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--unfreeze-top-blocks", type=int, default=2)
    parser.add_argument("--r-latent", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("-o", "--output", default="runs/ender-gptx2-benchmark.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    for task in args.tasks:
        if task not in SLICES:
            raise ValueError(f"unknown task {task!r}; expected one of {sorted(SLICES)}")

    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"device={device} tasks={args.tasks} slices={SLICES} train_steps={args.train_steps}", flush=True)

    backbone, tokenizer = load_backbone(device)
    from minimodel.architectures.ender import EnderAdapter

    report: dict[str, Any] = {"backbone": BACKBONE_ID, "seed": args.seed, "config": vars(args)}

    print("\n--- BEFORE (plain backbone) ---", flush=True)
    before: dict[str, dict[str, float]] = {}
    for task in args.tasks:
        t0 = time.time()
        before[task] = score_task(backbone, tokenizer, task, SLICES[task], args.seed, device)
        print(f"  {task}: acc={before[task]['acc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    report["before"] = before

    print("\n--- ENDER adaptation ---", flush=True)
    dim = int(backbone.config.hidden_size)
    adapter = EnderAdapter(
        backbone, {"dim": dim, "r_latent": args.r_latent, "num_steps": args.num_steps}
    ).to(device)
    final_loss = train_ender(
        adapter,
        tokenizer,
        steps=args.train_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        unfreeze_top_blocks=args.unfreeze_top_blocks,
        seed=args.seed,
        device=device,
    )
    adapter.eval()
    report["train"] = {"final_loss": final_loss, "steps": args.train_steps}
    print(f"  adaptation done (final loss {final_loss:.4f})", flush=True)

    print(f"\n--- AFTER (ENDER, {args.train_steps} steps) ---", flush=True)
    after: dict[str, dict[str, float]] = {}
    for task in args.tasks:
        t0 = time.time()
        after[task] = score_task(adapter, tokenizer, task, SLICES[task], args.seed, device)
        print(f"  {task}: acc={after[task]['acc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    report["after"] = after

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n=== GPT-X2.5-135M before/after ENDER (seeded slices, CPU) ===", flush=True)
    print(f"{'task':<12} {'before':>8} {'after':>8} {'delta':>8}", flush=True)
    for task in args.tasks:
        b, a = before[task]["acc"], after[task]["acc"]
        print(f"{task:<12} {b:>8.4f} {a:>8.4f} {a - b:>+8.4f}", flush=True)
    print(f"\nsaved -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
