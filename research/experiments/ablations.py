"""The controlled experimental program for beating Glint-2.

Groups of compute-matched runs (fixed ~6M-token budget unless noted), each
producing a result JSON in research/data/results/. Run a group with:

    python research/experiments/ablations.py <group> [--steps N] [--full-eval]

Groups: arch, stabilizers, optimizer, schedule, ffn, vocab, loops, all.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_experiment import ExpConfig, build_model_from, run_experiment  # noqa: E402

# ~1.7M looped block at Glint-2's shape; other configs match its budget.
GLINT_ARCH = dict(
    dim=96, n_heads=8, head_dim=12, ffn_hidden=2112, embedding_type="tied",
    prelude_layers=0, coda_layers=1, n_shared_blocks=1,
    use_timestep_scale=False, use_outer_residual=False, value_residual=False,
    loop_lora_rank=4, max_loops_table=16, train_loops=8, min_loops=8,
    variable_loops=False, norm_eps=1e-5, window=256, max_seq_len=512,
)

DEFAULTS = dict(
    vocab=4096, seq_len=256, batch_size=32, max_steps=300, lr=3e-3,
    optimizer="adamw", schedule="cosine", warmup=0.05, eval_loops=8,
    blimp_per_paradigm=15, arc_limit=150, wikitext_max_tokens=10000, log_every=100,
)


def _looped(**arch_over):
    arch = deepcopy(GLINT_ARCH)
    arch.update(arch_over)
    return arch


# Faster base (8x FFN instead of 22x) for ablations that measure *relative*
# effects (stabilisers, optimizer, schedule); these transfer across FFN width,
# and the 22x FFN is ~3x slower on CPU. FFN width itself is studied in group_ffn.
def _fast(**arch_over):
    return _looped(ffn_hidden=768, **arch_over)


def group_arch() -> list[ExpConfig]:
    """Architecture bake-off at a matched ~1.6-1.7M budget on TinyStories."""
    return [
        # Glint-2's exact shape: looped shared block + 1 coda, tied embed.
        ExpConfig(name="arch_loopcoda_glint", family="looped_transformer",
                  arch=_looped(), **DEFAULTS),
        # Pure loop: no coda (Glint-2's *advertised* design). FFN widened to match budget.
        ExpConfig(name="arch_pureloop", family="looped_transformer",
                  arch=_looped(coda_layers=0, ffn_hidden=4544), **DEFAULTS),
        # Our supra2 shape: prelude + 2 shared + coda, factorized embed, all
        # stabilisers, widened to ~1.74M to match Glint-2's budget.
        ExpConfig(name="arch_supra2", family="looped_transformer",
                  arch=_looped(embedding_type="factorized", embedding_rank=64,
                               prelude_layers=1, coda_layers=1, n_shared_blocks=2,
                               use_timestep_scale=True, use_outer_residual=True,
                               value_residual=True, ffn_hidden=760, dim=128, head_dim=16,
                               train_loops=8, min_loops=4, variable_loops=True,
                               norm_eps=1e-6), **DEFAULTS),
        # Dense transformer at matched budget (no looping): ~1.70M.
        ExpConfig(name="arch_dense", family="dense_transformer",
                  arch=dict(dim=160, n_layers=3, n_heads=5, head_dim=32, n_kv_heads=5,
                            ffn_hidden=512, window=256, qk_norm=True, tie_embeddings=True,
                            max_seq_len=512), **DEFAULTS),
        # Sparse MoE at matched *active* budget (~1.67M active, 2.85M total).
        ExpConfig(name="arch_moe", family="moe_transformer",
                  arch=dict(dim=128, n_layers=4, n_heads=4, head_dim=32, n_kv_heads=1,
                            ffn_hidden=256, n_routed_experts=6, top_k=2, first_moe_layer=1,
                            window=256, qk_norm=True, tie_embeddings=True, max_seq_len=512),
                  **DEFAULTS),
    ]


def group_stabilizers() -> list[ExpConfig]:
    """Do our supra2 stabilisers beat Glint-2's minimalism at the SAME shape?

    Uses the 8x-FFN fast base; the stabiliser effect is orthogonal to FFN width.
    """
    base = dict(family="looped_transformer")
    return [
        ExpConfig(name="stab_none", **base, arch=_fast(), **DEFAULTS),
        ExpConfig(name="stab_vr", **base, arch=_fast(value_residual=True), **DEFAULTS),
        ExpConfig(name="stab_vr_ts", **base,
                  arch=_fast(value_residual=True, use_timestep_scale=True), **DEFAULTS),
        ExpConfig(name="stab_all", **base,
                  arch=_fast(value_residual=True, use_timestep_scale=True,
                             use_outer_residual=True), **DEFAULTS),
    ]


def group_optimizer() -> list[ExpConfig]:
    """AdamW vs Muon vs Lion on the loop+coda+stabilisers model (fast base)."""
    arch = _fast(value_residual=True, use_timestep_scale=True, use_outer_residual=True)
    return [
        ExpConfig(name="opt_adamw", arch=arch, **{**DEFAULTS, "optimizer": "adamw", "lr": 3e-3}),
        ExpConfig(name="opt_muon", arch=arch,
                  **{**DEFAULTS, "optimizer": "muon", "lr": 2e-2,
                     "optimizer_kwargs": {"adamw_lr": 3e-3}}),
        ExpConfig(name="opt_lion", arch=arch, **{**DEFAULTS, "optimizer": "lion", "lr": 4e-4,
                                                 "weight_decay": 1.0}),
    ]


def group_schedule() -> list[ExpConfig]:
    """Cosine vs WSD (fast base)."""
    arch = _fast(value_residual=True, use_timestep_scale=True, use_outer_residual=True)
    return [
        ExpConfig(name="sched_cosine", arch=arch, **{**DEFAULTS, "schedule": "cosine"}),
        ExpConfig(name="sched_wsd", arch=arch,
                  **{**DEFAULTS, "schedule": "wsd",
                     "schedule_kwargs": {"decay_ratio": 0.2, "decay_shape": "sqrt"}}),
    ]


def group_ffn() -> list[ExpConfig]:
    """FFN ratio sweep on loop+coda (Glint-2 uses 22x)."""
    return [
        ExpConfig(name="ffn_4x", arch=_looped(ffn_hidden=384), **DEFAULTS),
        ExpConfig(name="ffn_8x", arch=_looped(ffn_hidden=768), **DEFAULTS),
        ExpConfig(name="ffn_16x", arch=_looped(ffn_hidden=1536), **DEFAULTS),
        ExpConfig(name="ffn_22x", arch=_looped(ffn_hidden=2112), **DEFAULTS),
    ]


def group_vocab() -> list[ExpConfig]:
    """Vocabulary-size sweep (needs matching tokenized corpora)."""
    out = []
    for v in (2048, 4096, 8192):
        out.append(
            ExpConfig(name=f"vocab_{v}", arch=_looped(value_residual=True), vocab=v,
                      train_corpus=f"tinystories_v{v}", val_corpus=f"tinystories_val_v{v}",
                      **{k: val for k, val in DEFAULTS.items() if k != "vocab"})
        )
    return out


GROUPS = {
    "arch": group_arch,
    "stabilizers": group_stabilizers,
    "optimizer": group_optimizer,
    "schedule": group_schedule,
    "ffn": group_ffn,
    "vocab": group_vocab,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=[*GROUPS, "all", "list"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--full-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print param counts only")
    args = parser.parse_args()

    groups = GROUPS if args.group == "all" else {args.group: GROUPS[args.group]}
    if args.group == "list":
        for gname, fn in GROUPS.items():
            for cfg in fn():
                print(f"{gname:12s} {cfg.name:24s} {build_model_from(cfg).num_parameters():>10,}")
        return

    for _gname, fn in groups.items():
        for cfg in fn():
            if args.steps:
                cfg.max_steps = args.steps
            if args.full_eval:
                cfg.blimp_per_paradigm = None
                cfg.arc_limit = None
                cfg.wikitext_max_tokens = None
            if args.dry_run:
                print(f"{cfg.name:24s} {build_model_from(cfg).num_parameters():>10,} params")
                continue
            run_experiment(cfg)


if __name__ == "__main__":
    main()
