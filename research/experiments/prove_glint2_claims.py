"""Reproducible proof: Glint-2 ships 1.71M loop+coda, not advertised 1.06M pure-loop.

Downloads (or reuses) the live Hugging Face release, counts parameters from the
checkpoint, and shows that the repo's own ``generate.py`` cannot
``load_state_dict`` the shipped weights because of unexpected ``coda.*`` keys.

Usage::

    venv/bin/python research/experiments/prove_glint2_claims.py
    venv/bin/python research/experiments/prove_glint2_claims.py --local-dir /path/to/Glint-2

Writes JSON evidence under ``research/data/results/glint2_proof.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))

from glint2_model import load_glint2

ADVERTISED_PARAMS = 1_065_000
REPO_ID = "Glint-Research/Glint-2"
RESULTS_PATH = Path(__file__).resolve().parents[1] / "data" / "results" / "glint2_proof.json"


def _download(local_dir: Path) -> Path:
    """Fetch the public release into ``local_dir`` if the checkpoint is missing."""
    ckpt = local_dir / "checkpoints" / "glint-2.pt"
    if ckpt.is_file() and (local_dir / "generate.py").is_file():
        return local_dir
    # Optional dependency: only needed when the release is not already on disk.
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(local_dir),
        allow_patterns=[
            "checkpoints/glint-2.pt",
            "generate.py",
            "README.md",
            "config.json",
            "tokenizer.json",
        ],
    )
    return local_dir


def _load_generate_module(generate_path: Path):
    """Import the release ``generate.py`` without putting it on sys.path permanently."""
    spec = importlib.util.spec_from_file_location("glint2_generate_release", generate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {generate_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def audit(local_dir: Path) -> dict:
    """Return a structured proof dict from a local Glint-2 tree."""
    ckpt_path = local_dir / "checkpoints" / "glint-2.pt"
    generate_path = local_dir / "generate.py"
    readme = (local_dir / "README.md").read_text(encoding="utf-8")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    sd = ckpt["model"]

    by_prefix: dict[str, int] = {}
    total = 0
    for name, tensor in sd.items():
        n = int(tensor.numel())
        total += n
        by_prefix[name.split(".")[0]] = by_prefix.get(name.split(".")[0], 0) + n

    coda_params = by_prefix.get("coda", 0)
    excl_coda = total - coda_params

    gen = _load_generate_module(generate_path)
    their_model = gen.Glint2(cfg, max_loops=sd["loop_embed.weight"].shape[0])
    their_n = sum(p.numel() for p in their_model.parameters())

    strict_error = None
    try:
        their_model.load_state_dict(sd, strict=True)
        strict_ok = True
    except RuntimeError as exc:
        strict_ok = False
        strict_error = str(exc)

    _missing, unexpected = their_model.load_state_dict(sd, strict=False)

    faithful = load_glint2(ckpt_path)
    faithful_n = sum(p.numel() for p in faithful.parameters())

    advertised_quotes = [
        line.strip()
        for line in readme.splitlines()
        if any(
            needle in line.lower()
            for needle in ("1.06", "1,065", "zero unique", "pure-loop", "real scores")
        )
    ]

    prior = Path(__file__).resolve().parents[1] / "data" / "results" / "glint2.json"
    bench = json.loads(prior.read_text()) if prior.is_file() else {}

    return {
        "repo_id": REPO_ID,
        "checkpoint": str(ckpt_path),
        "file_bytes": ckpt_path.stat().st_size,
        "model_config": {
            "prelude_layers": cfg.get("prelude_layers"),
            "coda_layers": cfg.get("coda_layers"),
            "shared_loops": cfg.get("shared_loops"),
            "dim": cfg.get("dim"),
            "ffn_hidden": cfg.get("ffn_hidden"),
            "vocab_size": cfg.get("vocab_size"),
            "n_heads": cfg.get("n_heads"),
        },
        "training_step": ckpt.get("step"),
        "advertised_params": ADVERTISED_PARAMS,
        "actual_params": total,
        "params_excluding_coda": excl_coda,
        "params_coda_only": coda_params,
        "by_prefix": by_prefix,
        "readme_claims_sample": advertised_quotes[:8],
        "generate_py": {
            "constructed_params": their_n,
            "strict_load_ok": strict_ok,
            "strict_error": strict_error,
            "non_strict_unexpected_keys": list(unexpected),
            "coda_params_silently_dropped_if_non_strict": sum(
                int(sd[k].numel()) for k in unexpected if k.startswith("coda")
            ),
        },
        "faithful_loader": {
            "params": faithful_n,
            "matches_checkpoint": faithful_n == total,
        },
        "benchmark_labeling_from_prior_eval": {
            "source": str(prior) if prior.is_file() else None,
            "advertised_wikitext_ppl": 3.09,
            "measured_token_ppl": bench.get("wikitext_ppl"),
            "measured_byte_ppl_2bpb": bench.get("wikitext_byte_ppl"),
            "measured_bpb": bench.get("wikitext_bits_per_byte"),
            "advertised_arc_easy": 36.80,
            "measured_arc_easy": bench.get("arc_easy_acc"),
            "advertised_blimp": 73.96,
            "measured_blimp": bench.get("blimp_acc"),
            "advertised_params_on_leaderboard": "1.06M",
            "actual_params_evaluated": bench.get("params"),
        },
        "verdicts": {
            "param_count_misreported": total > ADVERTISED_PARAMS + 100_000,
            "not_pure_loop": bool(cfg.get("coda_layers", 0)) and coda_params > 0,
            "generate_py_cannot_strict_load": not strict_ok,
            "wikitext_ppl_is_byte_normalised": (
                bench.get("wikitext_byte_ppl") is not None
                and abs(float(bench["wikitext_byte_ppl"]) - 3.09) < 0.15
                and abs(float(bench["wikitext_ppl"]) - 3.09) > 10
            ),
            "arc_reproducible": (
                bench.get("arc_easy_acc") is not None
                and abs(float(bench["arc_easy_acc"]) - 36.80) < 0.1
            ),
            "blimp_not_reproduced_in_independent_harness": (
                bench.get("blimp_acc") is not None and float(bench["blimp_acc"]) < 70.0
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("/tmp/glint2-fresh"),
        help="Directory to download or reuse the Glint-2 release",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_PATH,
        help="Where to write the JSON proof artifact",
    )
    args = parser.parse_args(argv)

    local_dir = _download(args.local_dir)
    evidence = audit(local_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    v = evidence["verdicts"]
    print(f"actual_params={evidence['actual_params']:,}  advertised={ADVERTISED_PARAMS:,}")
    print(
        f"coda_layers={evidence['model_config']['coda_layers']}  "
        f"coda_params={evidence['params_coda_only']:,}"
    )
    print(f"strict_load_ok={evidence['generate_py']['strict_load_ok']}")
    print(f"wrote {args.out}")
    print("verdicts:", json.dumps(v, indent=2))
    # Tripwire: core architectural claims must hold against the live release.
    if not (
        v["param_count_misreported"]
        and v["not_pure_loop"]
        and v["generate_py_cannot_strict_load"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
