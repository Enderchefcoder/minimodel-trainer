"""Test-time loop scaling: does more inference compute help, and how robustly?

Glint-2 trains at a fixed 8 loops and admits "run it at 16 and you get
gibberish" — its per-loop tables have capacity 16 but it never learned to use
them. Our looped models train with *variable* loops (U{min..train}), which
should make them degrade gracefully — or improve — when asked for more loops at
inference.

This script evaluates byte-ppl + BLiMP at a range of loop counts for:
  * the released Glint-2 (baseline), and
  * one of our variable-loop models (trained here or loaded).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))

from eval_harness import ModelAdapter, eval_blimp, eval_wikitext  # noqa: E402
from run_experiment import ExpConfig, build_model_from, train  # noqa: E402

from minimodel.tokenization.tokenize import BPETokenizer  # noqa: E402

ART = Path("research/artifacts")
RESULTS = Path("research/data/results")
LOOP_GRID = [2, 4, 6, 8, 10, 12, 16]


def eval_at_loops(name, encode, model, n_bytes, loop_grid, *, blimp_pp=15, wt=10000):
    rows = []
    for loops in loop_grid:
        def forward(tokens, _loops=loops):
            return model(tokens, loops=_loops)

        adapter = ModelAdapter(name=f"{name}@{loops}", encode=encode, forward=forward,
                               n_bytes=n_bytes, max_len=256, batch_size=32,
                               params=sum(p.numel() for p in model.parameters()))
        t0 = time.perf_counter()
        wik = eval_wikitext(adapter, max_tokens=wt)
        blimp = eval_blimp(adapter, per_paradigm=blimp_pp)
        rows.append({"loops": loops, "wikitext_byte_ppl": wik["wikitext_byte_ppl"],
                     "blimp_acc": blimp["blimp_acc"], "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  {name} loops={loops:>2}: byte_ppl {wik['wikitext_byte_ppl']:.3f}  "
              f"blimp {blimp['blimp_acc']:.1f}", flush=True)
    return rows


def run_glint() -> dict:
    from glint2_model import load_glint2  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415

    model = load_glint2("research/baselines/glint2/glint-2.pt", loops=8)
    tok = Tokenizer.from_file("research/baselines/glint2/tokenizer.json")
    encode = lambda t: list(tok.encode(t).ids)  # noqa: E731
    n_bytes = lambda ids: len(tok.decode(ids).encode("utf-8"))  # noqa: E731
    print("Glint-2 loop scaling:")
    return {"name": "glint-2", "rows": eval_at_loops("glint-2", encode, model, n_bytes, LOOP_GRID)}


def run_ours(steps: int) -> dict:
    torch.manual_seed(1234)
    cfg = ExpConfig(
        name="ours_varloop", family="looped_transformer",
        arch=dict(dim=96, n_heads=8, head_dim=12, ffn_hidden=768, embedding_type="tied",
                  prelude_layers=0, coda_layers=1, n_shared_blocks=1, value_residual=True,
                  use_timestep_scale=True, use_outer_residual=True, loop_lora_rank=4,
                  max_loops_table=16, train_loops=8, min_loops=4, variable_loops=True,
                  norm_eps=1e-5, window=256, max_seq_len=512, vocab_size=4096),
        seq_len=256, batch_size=32, max_steps=steps, lr=3e-3,
    )
    tok = BPETokenizer.load(ART / "tokenizer_v4096.json")
    model = build_model_from(cfg)
    print(f"\nTraining ours_varloop ({model.num_parameters():,} params) for {steps} steps...")
    train(model, cfg, torch.device("cpu"))
    model.eval()
    encode = lambda t: tok.encode(t, allow_special=False)  # noqa: E731
    n_bytes = lambda ids: len(tok.decode(ids).encode("utf-8"))  # noqa: E731
    print("Ours (variable-loop) loop scaling:")
    return {"name": "ours_varloop", "params": model.num_parameters(),
            "rows": eval_at_loops("ours", encode, model, n_bytes, LOOP_GRID)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["glint", "ours", "both"], default="both")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    out: dict = {"loop_grid": LOOP_GRID}
    if args.which in ("glint", "both"):
        out["glint"] = run_glint()
    if args.which in ("ours", "both"):
        out["ours"] = run_ours(args.steps)

    path = RESULTS / "loops_scaling.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
