"""Estimate Elo for a Stigmergy weights file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chess_contest.stigmergy.elo import estimate_elo_from_file, save_report  # noqa: E402
from chess_contest.stigmergy.uniqueness import (  # noqa: E402
    composite_contest_score,
    score_uniqueness,
)
from chess_contest.stigmergy.weights import load_weights  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Estimate Stigmergy Elo vs reference ladder.")
    p.add_argument("--weights", default="chess_contest/weights/base_weights.json")
    p.add_argument("--games-per-opponent", type=int, default=6)
    p.add_argument("--move-time-ms", type=int, default=250)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--max-plies", type=int, default=80)
    p.add_argument("--out", default="chess_contest/weights/elo_report.json")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    if args.quick:
        args.games_per_opponent = 2
        args.move_time_ms = 60
        args.max_depth = 2
        args.max_plies = 40

    report = estimate_elo_from_file(
        args.weights,
        games_per_opponent=args.games_per_opponent,
        move_time_ms=args.move_time_ms,
        max_depth=args.max_depth,
        max_plies=args.max_plies,
    )
    weights = load_weights(args.weights)
    uniq = score_uniqueness(weights.to_dict().get("uniquenessFingerprint"))
    composite = composite_contest_score(report.engine_elo, bracket_winrate=0.5, uniqueness=uniq.score)
    payload = {
        **report.to_dict(),
        "uniqueness": uniq.to_dict(),
        "composite": composite,
    }
    save_report(report, args.out)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Estimated Elo: {report.engine_elo:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
