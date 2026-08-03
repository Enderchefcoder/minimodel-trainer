"""Controlled test of loop-count robustness — Glint-2's key weakness.

Glint-2 has a razor-sharp quality optimum at its trained 8 loops and collapses
to gibberish by 16 (measured in loops_scaling.py; confirmed by its own README).
This is the documented LoopLM failure mode (arXiv:2605.26733): quality peaks at
one depth and declines sharply beyond.

Here we train three otherwise-identical ~1.7M looped models that differ only in
how the loop count is chosen during training, and evaluate each across a grid of
inference loop counts:

  fixed8   - always 8 loops (Glint-2's regime; expected sharp optimum)
  uniform  - U{4..8} per step (our previous default)
  poisson  - log-normal Poisson centred on 8, clipped to [4, 16] (Huginn 2025)

Claim under test: variable-loop training — especially Poisson, which actually
samples the upper table slots — flattens the loop-quality curve and lets the
model use test-time compute (loops > 8) instead of breaking.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from eval_harness import ModelAdapter, eval_blimp, eval_wikitext  # noqa: E402
from run_experiment import ExpConfig, build_model_from, train  # noqa: E402

from minimodel.tokenization.tokenize import BPETokenizer  # noqa: E402

ART = Path("research/artifacts")
RESULTS = Path("research/data/results")
LOOP_GRID = [4, 6, 8, 10, 12, 16]


def base_arch(**over):
    arch = dict(
        dim=96, n_heads=8, head_dim=12, ffn_hidden=768, embedding_type="tied",
        prelude_layers=0, coda_layers=1, n_shared_blocks=1, value_residual=True,
        use_timestep_scale=True, use_outer_residual=True, loop_lora_rank=4,
        max_loops_table=16, norm_eps=1e-5, window=256, max_seq_len=512, vocab_size=4096,
    )
    arch.update(over)
    return arch


VARIANTS = {
    "fixed8": base_arch(train_loops=8, min_loops=8, variable_loops=False),
    "uniform": base_arch(train_loops=8, min_loops=4, variable_loops=True, loop_sampling="uniform"),
    "poisson": base_arch(train_loops=8, min_loops=4, variable_loops=True, loop_sampling="poisson"),
}


def eval_across_loops(model, tok) -> list[dict]:
    model.eval()
    encode = lambda t: tok.encode(t, allow_special=False)  # noqa: E731
    n_bytes = lambda ids: len(tok.decode(ids).encode("utf-8"))  # noqa: E731
    rows = []
    for loops in LOOP_GRID:
        def forward(tokens, _loops=loops):
            return model(tokens, loops=_loops)

        adapter = ModelAdapter(name=f"@{loops}", encode=encode, forward=forward, n_bytes=n_bytes,
                               max_len=256, batch_size=32, params=model.num_parameters())
        wik = eval_wikitext(adapter, max_tokens=10000)
        blimp = eval_blimp(adapter, per_paradigm=12)
        rows.append({"loops": loops, "byte_ppl": wik["wikitext_byte_ppl"],
                     "blimp": blimp["blimp_acc"]})
        print(f"    loops={loops:>2}: byte_ppl {wik['wikitext_byte_ppl']:>8.2f}  blimp {blimp['blimp_acc']:.1f}",
              flush=True)
    return rows


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    tok = BPETokenizer.load(ART / "tokenizer_v4096.json")
    out: dict = {"loop_grid": LOOP_GRID, "steps": steps, "variants": {}}
    for name, arch in VARIANTS.items():
        torch.manual_seed(1234)
        cfg = ExpConfig(name=f"looprobust_{name}", family="looped_transformer", arch=arch,
                        seq_len=256, batch_size=32, max_steps=steps, lr=3e-3)
        model = build_model_from(cfg)
        print(f"\n=== {name} ({model.num_parameters():,} params, {steps} steps) ===", flush=True)
        t0 = time.perf_counter()
        stats = train(model, cfg, torch.device("cpu"))
        print(f"  trained in {time.perf_counter()-t0:.0f}s, final_loss {stats['final_loss']}", flush=True)
        rows = eval_across_loops(model, tok)
        best = min(rows, key=lambda r: r["byte_ppl"])
        out["variants"][name] = {
            "params": model.num_parameters(), "final_loss": stats["final_loss"],
            "rows": rows,
            "best_loops": best["loops"], "best_byte_ppl": best["byte_ppl"],
            # Robustness: ratio of worst to best byte-ppl across the grid (1.0 = flat).
            "ppl_spread": round(max(r["byte_ppl"] for r in rows) / best["byte_ppl"], 2),
        }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "loop_robustness.json").write_text(json.dumps(out, indent=2))
    print("\nsaved -> research/data/results/loop_robustness.json")
    for name, v in out["variants"].items():
        print(f"  {name:8s} best@{v['best_loops']} ppl {v['best_byte_ppl']:.2f}  spread x{v['ppl_spread']}")


if __name__ == "__main__":
    main()
