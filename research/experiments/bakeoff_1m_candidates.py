#!/usr/bin/env python3
"""Report-03-protocol bake-off for the 20 ~1M Glint-2 candidates.

Trains each ``mm1m_r*`` template on TinyStories with the same budget as the
original five ``arch_*`` runs (300 steps x batch 32 x seq 256 = 2.46M tokens),
evaluates BLiMP / ARC-Easy / WikiText on the shared harness, then merges with
the original ``arch_*.json`` results into one ranked leaderboard.

Usage::

    python research/experiments/bakeoff_1m_candidates.py --full
    python research/experiments/bakeoff_1m_candidates.py --full --only mm1m_r01_dense_gqa_vr
    python research/experiments/bakeoff_1m_candidates.py --merge-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO_ROOT))

from run_experiment import ExpConfig, run_experiment  # noqa: E402

from minimodel.architectures.builder import (  # noqa: E402
    list_glint2_candidates,
    load_template,
    template_to_model_config,
)

RESULTS = REPO_ROOT / "research" / "data" / "results"
ORIGINAL_ARCH = [
    "arch_dense",
    "arch_moe",
    "arch_pureloop",
    "arch_loopcoda_glint",
    "arch_supra2",
]

#: Matched to research/experiments/ablations.py ``group_arch`` / report 03.
PROTOCOL = {
    "vocab": 4096,
    "train_corpus": "tinystories_v4096",
    "val_corpus": "tinystories_val_v4096",
    "seq_len": 256,
    "batch_size": 32,
    "max_steps": 300,
    "lr": 3e-3,
    "optimizer": "adamw",
    "schedule": "cosine",
    "warmup": 0.05,
    "eval_loops": 8,
    "blimp_per_paradigm": 15,
    "arc_limit": 150,
    "wikitext_max_tokens": 10000,
    "log_every": 100,
}


def _configs(only: list[str] | None = None) -> list[ExpConfig]:
    """Build ExpConfigs from the ordered mm1m templates."""
    out: list[ExpConfig] = []
    for row in list_glint2_candidates():
        name = row["name"]
        if only and name not in only:
            continue
        template = load_template(name).to_dict()
        family, arch = template_to_model_config(template)
        # Drop bookkeeping keys that are not model config.
        arch = {k: v for k, v in arch.items() if k not in {"glint2_rank", "candidate_class"}}
        cfg = ExpConfig(
            name=name,
            family=family,
            arch=arch,
            **PROTOCOL,
        )
        out.append(cfg)
    return out


def _load_json(name: str) -> dict[str, Any] | None:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    """Sort key: val_loss ↑, then WikiText byte-ppl ↑, then -BLiMP."""
    val = row.get("val_loss")
    bppl = row.get("wikitext_byte_ppl")
    blimp = row.get("blimp_acc")
    return (
        float(val) if isinstance(val, (int, float)) and math.isfinite(val) else 1e9,
        float(bppl) if isinstance(bppl, (int, float)) and math.isfinite(bppl) else 1e9,
        -float(blimp) if isinstance(blimp, (int, float)) and math.isfinite(blimp) else 0.0,
    )


def merge_results() -> dict[str, Any]:
    """Combine original arch_* + mm1m_* into one ranked payload."""
    original: list[dict[str, Any]] = []
    for name in ORIGINAL_ARCH:
        data = _load_json(name)
        if data is None:
            continue
        row = {k: v for k, v in data.items() if k != "config"}
        row["bakeoff"] = "report03_original"
        row["candidate_class"] = "original-arch"
        original.append(row)

    mm1m: list[dict[str, Any]] = []
    meta_by_name = {r["name"]: r for r in list_glint2_candidates()}
    for row_meta in list_glint2_candidates():
        data = _load_json(row_meta["name"])
        if data is None:
            continue
        row = {k: v for k, v in data.items() if k != "config"}
        row["bakeoff"] = "mm1m_report03"
        row["candidate_class"] = row_meta.get("candidate_class") or meta_by_name.get(
            row_meta["name"], {}
        ).get("candidate_class", "")
        row["prior_rank"] = row_meta.get("rank")
        mm1m.append(row)

    combined = original + mm1m
    ranked = sorted(combined, key=_rank_key)
    for i, row in enumerate(ranked, start=1):
        row["measured_rank"] = i

    mm1m_ranked = sorted(mm1m, key=_rank_key)
    for i, row in enumerate(mm1m_ranked, start=1):
        row["mm1m_measured_rank"] = i

    payload = {
        "protocol": {
            **PROTOCOL,
            "train_tokens_per_run": PROTOCOL["max_steps"]
            * PROTOCOL["batch_size"]
            * PROTOCOL["seq_len"],
            "note": (
                "Same budget as report 03 arch bake-off. mm1m models are ~1.0-1.2M; "
                "original arch_* models are ~1.7M. Absolute numbers are comparable "
                "on the harness; parameter budgets differ."
            ),
        },
        "original_arch": original,
        "mm1m_candidates": mm1m_ranked,
        "merged_ranked": ranked,
        "n_original": len(original),
        "n_mm1m": len(mm1m),
        "n_merged": len(ranked),
    }
    out = RESULTS / "arch_bakeoff_merged.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(ranked)} rows)")
    return payload


def _print_table(rows: list[dict[str, Any]], title: str) -> None:
    print(f"\n=== {title} ===")
    print(
        f"{'rk':>3} {'name':36} {'params':>10} {'val↓':>7} {'bppl↓':>7} "
        f"{'blimp':>6} {'arc':>6} {'tok/s':>8}"
    )
    for row in rows:
        print(
            f"{row.get('measured_rank', row.get('mm1m_measured_rank', 0)):3d} "
            f"{row['name']:36} {row.get('params', 0):10,} "
            f"{row.get('val_loss', float('nan')):7.3f} "
            f"{row.get('wikitext_byte_ppl', float('nan')):7.2f} "
            f"{row.get('blimp_acc', float('nan')):6.1f} "
            f"{row.get('arc_easy_acc', float('nan')):6.1f} "
            f"{row.get('tokens_per_second', float('nan')):8.0f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the report-03 protocol on all (or --only) mm1m templates",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="rebuild arch_bakeoff_merged.json from existing result JSONs",
    )
    parser.add_argument("--only", nargs="*", default=None, help="subset of template names")
    parser.add_argument("--steps", type=int, default=None, help="override max_steps")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip templates that already have a result JSON",
    )
    args = parser.parse_args()

    if args.merge_only and not args.full:
        payload = merge_results()
        _print_table(payload["merged_ranked"], "Merged ranking (val_loss)")
        return 0

    if not args.full:
        parser.error("pass --full to run the bake-off, or --merge-only")

    configs = _configs(args.only)
    if not configs:
        raise SystemExit("no candidates selected")

    for cfg in configs:
        if args.steps:
            cfg.max_steps = args.steps
        if args.skip_existing and (RESULTS / f"{cfg.name}.json").exists():
            print(f"skip existing {cfg.name}", flush=True)
            continue
        # Fresh copy so mutations in run_experiment cannot leak.
        run_experiment(deepcopy(cfg), save=True, do_eval=True)

    payload = merge_results()
    _print_table(payload["mm1m_candidates"], "mm1m measured ranking")
    _print_table(payload["merged_ranked"], "Merged ranking (original + mm1m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
