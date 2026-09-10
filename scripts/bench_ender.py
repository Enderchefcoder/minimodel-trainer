#!/usr/bin/env python3
"""Benchmark a model before and after ENDER adaptation.

Scores the requested tasks — BLiMP, ARC-Easy and the full
``AxiomicLabs/Open_SLM_Leaderboard`` set by default — with the plain backbone,
then again with the ENDER adapter grafted (and optionally trained), and writes a
side-by-side comparison.

By default the tasks come from this repository's benchmark suite (the bundled
offline demo tasks plus any pulled leaderboard data), so the script always
produces a result even with no network. To score the real leaderboard datasets,
pull them first::

    minimodel data pull blimp && minimodel data pull arc-easy && ...

Usage::

    python scripts/bench_ender.py --adapter runs/ender-adapted
    python scripts/bench_ender.py --before runs/pretrain/model --adapter runs/ender-adapted
    python scripts/bench_ender.py --leaderboard   # only leaderboard tasks, needs pulled data
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

from minimodel.benchmarking.bench import run_suite  # noqa: E402
from minimodel.core.logging_utils import get_logger, setup_logging  # noqa: E402

logger = get_logger("bench-ender")

#: The leaderboard's exact task names, in a stable reporting order.
LEADERBOARD_ORDER = (
    "blimp",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
    "mmlu",
    "gsm8k",
)


def build_task_list(leaderboard_only: bool, limit: int | None) -> tuple[list[Any], Any]:
    """Tasks to score, plus the optional perplexity corpus."""
    if leaderboard_only:
        from minimodel.benchmarking.leaderboard import load_leaderboard_suite

        tasks, corpus = load_leaderboard_suite(limit=limit)
        if not tasks:
            raise SystemExit(
                "no leaderboard data found; pull datasets first, e.g.\n"
                "  minimodel data pull blimp\n"
                "  minimodel data pull arc-easy\n"
                "(or drop --leaderboard to use the bundled offline tasks)"
            )
        return tasks, corpus
    from minimodel.benchmarking.leaderboard import load_leaderboard_tasks
    from minimodel.benchmarking.tasks import BUILTIN_TASKS
    from minimodel.datasets.shards import TokenizedCorpus

    tasks = list(BUILTIN_TASKS.values())
    pulled = load_leaderboard_tasks(REQUIRED := ("blimp", "arc_easy"), limit=limit)
    tasks.extend(pulled.values())
    if len(pulled) < len(REQUIRED):
        logger.warning(
            "pulled leaderboard coverage incomplete (%s present); "
            "bundled demo tasks are included so the run still works offline",
            sorted(pulled) or "none",
        )
    corpus_path = Path("data/tokenized/pretrain")
    corpus = TokenizedCorpus(corpus_path) if corpus_path.exists() else None
    return tasks, corpus


def load_ender(adapter_dir: Path, backbone: Any) -> Any:
    """Re-create an :class:`EnderAdapter` with trained parameters loaded."""
    from minimodel.architectures.ender import EnderAdapter

    metadata = json.loads((adapter_dir / "ender_metadata.json").read_text(encoding="utf-8"))
    config = {
        "dim": metadata["dim"],
        "r_latent": metadata["r_latent"],
        "num_steps": metadata["num_steps"],
    }
    adapter = EnderAdapter(backbone, config)
    state = torch.load(adapter_dir / "ender_adapter.pt", map_location="cpu", weights_only=True)
    adapter.load_state_dict(state, strict=False)
    return adapter


def headline(result: Any) -> dict[str, float]:
    """Primary metric per task, plus the adapter's step count if present."""
    scores = dict(result.headline())
    return scores


def main() -> int:
    """Run the before/after benchmark and print a comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", default=None, help="backbone model dir (default: dense_3m)")
    parser.add_argument(
        "--adapter",
        default=None,
        help="ENDER adapter dir from scripts/apply_ender.py; omit to benchmark an untrained graft",
    )
    parser.add_argument("-o", "--output", default="runs/ender-benchmark.json")
    parser.add_argument("--leaderboard", action="store_true", help="only leaderboard tasks")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--r-latent", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-throughput", action="store_true")
    args = parser.parse_args()

    setup_logging(force=True)
    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tasks, perplexity_corpus = build_task_list(args.leaderboard, args.limit)

    from minimodel.architectures.builder import build_model

    def load_backbone() -> Any:
        if args.before:
            from minimodel.architectures.builder import load_model

            return load_model(args.before)
        return build_model("dense_3m", verify_budget=False)

    results: dict[str, Any] = {}

    backbone = load_backbone().to(device)
    before_name = args.before or "dense_3m (stand-in backbone)"
    results["before"] = run_suite(
        backbone,
        _tokenizer_for(backbone),
        tasks=tasks,
        perplexity_corpus=perplexity_corpus,
        device=device,
        limit=args.limit,
        include_throughput=not args.no_throughput,
        model_name=f"before: {before_name}",
    )
    logger.info("before: %s", results["before"].headline())

    if args.adapter:
        adapted = load_ender(Path(args.adapter), load_backbone()).to(device)
    else:
        from minimodel.architectures.ender import EnderAdapter

        dim = int(getattr(backbone, "dim", getattr(backbone.config, "hidden_size", 192)))
        adapted = EnderAdapter(
            load_backbone(), {"dim": dim, "r_latent": args.r_latent, "num_steps": args.num_steps}
        ).to(device)
    results["after"] = run_suite(
        adapted,
        _tokenizer_for(adapted.backbone),
        tasks=tasks,
        perplexity_corpus=perplexity_corpus,
        device=device,
        limit=args.limit,
        include_throughput=not args.no_throughput,
        model_name=f"after: ENDER r={args.r_latent} steps={args.num_steps}",
    )
    logger.info("after: %s", results["after"].headline())

    comparison = {
        "before": {
            name: headline(results["before"]).get(name)
            for name in _leaderboard_then_rest(results["before"].tasks)
        },
        "after": {
            name: headline(results["after"]).get(name)
            for name in _leaderboard_then_rest(results["after"].tasks)
        },
        "delta": {},
    }
    for name in _leaderboard_then_rest(results["after"].tasks):
        b, a = comparison["before"].get(name), comparison["after"].get(name)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            comparison["delta"][name] = round(a - b, 6)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "before": results["before"].to_dict(),
        "after": results["after"].to_dict(),
        "comparison": comparison,
    }
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n=== ENDER before/after ===")
    print(f"{'task':<18} {'before':>10} {'after':>10} {'delta':>10}")
    for name in _leaderboard_then_rest(comparison["after"]):
        b = comparison["before"].get(name)
        a = comparison["after"][name]
        d = comparison["delta"].get(name)
        print(
            f"{name:<18} "
            f"{'-' if b is None else f'{float(b):.4f}':>10} "
            f"{float(a):>10.4f} "
            f"{'-' if d is None else f'{d:+.4f}':>10}"
        )
    print(f"\nsaved -> {output}")
    return 0


def _leaderboard_then_rest(names: Any) -> list[str]:
    """Sort task names leaderboard-first, everything else after, stable."""
    names = list(names)
    ordered = [n for n in LEADERBOARD_ORDER if n in names]
    return ordered + [n for n in names if n not in LEADERBOARD_ORDER]


def _tokenizer_for(model: Any) -> Any:
    """Best-effort tokenizer: saved alongside the model, else a fresh tiny one."""
    from minimodel.tokenization.tokenize import BPETokenizer

    directory = None
    if hasattr(model, "config") and isinstance(getattr(model, "_model_dir", None), Path):
        directory = model._model_dir
    if directory and (Path(directory) / "tokenizer.json").exists():
        return BPETokenizer.load(Path(directory) / "tokenizer.json")
    texts = [r["text"] for r in __import__("minimodel.datasets.builtin", fromlist=["x"]).builtin_records("pretrain", repeat=4)]
    return BPETokenizer.train(texts, vocab_size=400, min_frequency=2)


if __name__ == "__main__":
    raise SystemExit(main())
