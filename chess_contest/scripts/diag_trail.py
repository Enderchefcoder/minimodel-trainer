from __future__ import annotations

import sys
from pathlib import Path

import chess

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from chess_contest.scripts.flash_3000 import _log, converge_target
from chess_contest.stigmergy.engine import StigmergyEngine
from chess_contest.stigmergy.search import Searcher, set_policy_sprint, set_swarm
from chess_contest.stigmergy.stockfish_uci import StockfishConfig, StockfishEngine
from chess_contest.stigmergy.weights import load_weights, trail_key

def main() -> None:
    log = Path("chess_contest/weights/gm/diag.log")
    weights = load_weights("chess_contest/weights/gm/latest.json")
    set_swarm(None)
    set_policy_sprint(False)
    teacher = StockfishEngine(StockfishConfig(threads=2, hash_mb=256))
    teacher.set_elo(None)
    opponent = StockfishEngine(StockfishConfig(threads=1, hash_mb=128))
    try:
        _log(log, "diag converge")
        converge_target(
            weights,
            teacher,
            opponent,
            log,
            target=2200,
            rounds=2,
            depth=12,
            stig_ms=40,
            stig_depth=3,
            hit_goal=0.99,
            fanout_all_replies=True,
            fanout_depth=10,
        )
        eng = StigmergyEngine(weights, load_swarm=False)
        set_swarm(None)
        searcher = Searcher(weights)
        board = chess.Board()
        for ply in range(60):
            if board.is_game_over(claim_draw=True):
                break
            if board.turn == chess.WHITE:
                tr = searcher.trail_move(board)
                chosen = eng.choose_move(board, time_ms=40, max_depth=3).move
                _log(
                    log,
                    f"ply{ply} trail={tr} chosen={chosen} same={tr == chosen} "
                    f"has_slot={trail_key(board) in weights.trails}",
                )
                if tr is None:
                    _log(log, f"MISS fen={board.fen()}")
                    break
                board.push(chosen)
            else:
                opponent.set_elo(2200)
                board.push(opponent.choose(board, movetime_ms=50))
        _log(log, f"result={board.result()}")
    finally:
        teacher.close()
        opponent.close()
        set_swarm(None)

if __name__ == "__main__":
    main()
