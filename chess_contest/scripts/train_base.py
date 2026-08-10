"""Train and export base Stigmergy weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python chess_contest/scripts/train_base.py` without install.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chess_contest.stigmergy.train import TrainConfig, train_base  # noqa: E402
from chess_contest.stigmergy.uniqueness import score_uniqueness  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train Stigmergy-DPFE base weights (CPU/CUDA).")
    p.add_argument("--out", default="chess_contest/weights/base_weights.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--imitation-steps", type=int, default=200)
    p.add_argument("--selfplay-rounds", type=int, default=24)
    p.add_argument("--games-per-round", type=int, default=4)
    p.add_argument("--move-time-ms", type=int, default=80)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--max-plies", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--quick", action="store_true", help="Tiny smoke training for CI.")
    args = p.parse_args(argv)

    if args.quick:
        cfg = TrainConfig(
            seed=args.seed,
            imitation_steps=20,
            selfplay_rounds=2,
            games_per_round=2,
            move_time_ms=40,
            max_depth=2,
            max_plies=30,
            lr=args.lr,
            device=args.device,
            out_path=args.out,
        )
    else:
        cfg = TrainConfig(
            seed=args.seed,
            imitation_steps=args.imitation_steps,
            selfplay_rounds=args.selfplay_rounds,
            games_per_round=args.games_per_round,
            move_time_ms=args.move_time_ms,
            max_depth=args.max_depth,
            max_plies=args.max_plies,
            lr=args.lr,
            device=args.device,
            out_path=args.out,
        )

    print(f"Training Stigmergy-DPFE → {cfg.out_path} (device={cfg.device})")
    weights = train_base(cfg)
    uniq = score_uniqueness(weights.to_dict()["uniquenessFingerprint"])
    print(json.dumps({"trainingMeta": weights.training_meta, "uniqueness": uniq.to_dict()}, indent=2))
    print(f"Wrote {cfg.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
