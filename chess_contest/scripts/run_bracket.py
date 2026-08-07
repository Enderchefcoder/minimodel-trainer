"""Run a tiny offline bracket demo (Stigmergy vs references)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chess  # noqa: E402

from chess_contest.stigmergy.bracket import BracketPlayer, round_robin, winrate_for  # noqa: E402
from chess_contest.stigmergy.engine import StigmergyEngine  # noqa: E402
from chess_contest.stigmergy.opponents import (  # noqa: E402
    ClassicPSTOpponent,
    GreedyMaterialOpponent,
    RandomOpponent,
)
from chess_contest.stigmergy.uniqueness import (  # noqa: E402
    composite_contest_score,
    score_uniqueness,
)
from chess_contest.stigmergy.weights import load_weights  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Contest bracket demo.")
    p.add_argument("--weights", default="chess_contest/weights/base_weights.json")
    p.add_argument("--games-per-pair", type=int, default=2)
    p.add_argument("--move-time-ms", type=int, default=120)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--out", default="chess_contest/weights/bracket_report.json")
    args = p.parse_args(argv)

    weights = load_weights(args.weights)
    engine = StigmergyEngine(weights)

    def stig(board: chess.Board) -> chess.Move:
        res = engine.choose_move(board, time_ms=args.move_time_ms, max_depth=args.max_depth)
        assert res.move is not None
        return res.move

    players = [
        BracketPlayer("stigmergy", stig),
        BracketPlayer("random", RandomOpponent(1).choose),
        BracketPlayer("greedy_material", GreedyMaterialOpponent().choose),
        BracketPlayer("classic_pst_d2", ClassicPSTOpponent(2).choose),
    ]
    bracket = round_robin(players, games_per_pair=args.games_per_pair, max_plies=60)
    wr = winrate_for("stigmergy", bracket)
    uniq = score_uniqueness(weights.to_dict().get("uniquenessFingerprint"))
    # Elo unknown here — placeholder from uniqueness-only blend.
    composite = composite_contest_score(elo=1400.0, bracket_winrate=wr, uniqueness=uniq.score)
    payload = {"bracket": bracket, "stigmergy_winrate": wr, "uniqueness": uniq.to_dict(), "composite": composite}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"ranking": bracket["ranking"], "stigmergy_winrate": wr, "uniqueness": uniq.score}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
