#!/usr/bin/env python3
"""Quick strength + policy-match check with v2 net + neural beam."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import chess

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from chess_contest.stigmergy.engine import StigmergyEngine
from chess_contest.stigmergy.search import set_swarm
from chess_contest.stigmergy.stockfish_uci import StockfishConfig, StockfishEngine
from chess_contest.stigmergy.swarm_net import try_load_swarm
from chess_contest.stigmergy.weights import default_weights


def main() -> int:
    net = try_load_swarm("chess_contest/weights/gm/swarm_net.pt")
    assert net is not None
    set_swarm(net)
    w = default_weights()
    w.book.clear()
    w.trails.clear()
    eng = StigmergyEngine(w, load_swarm=False)
    set_swarm(net)
    sf = StockfishEngine(StockfishConfig(threads=2, hash_mb=128))

    match = 0
    n = 80
    tried = 0
    for _ in range(n):
        b = chess.Board()
        for _ in range(random.randint(0, 20)):
            ms = list(b.legal_moves)
            if not ms or b.is_game_over():
                break
            b.push(random.choice(ms))
        if b.is_game_over():
            continue
        sf.set_elo(None)
        tops = sf.analyse_top(b, multipv=1, depth=10)
        if not tops:
            continue
        try:
            sfm = chess.Move.from_uci(tops[0]["uci"])
        except ValueError:
            continue
        tried += 1
        if net.choose(b) == sfm:
            match += 1
    print(f"policy_match_sf_d10={match}/{tried} ({match / max(1, tried):.0%})", flush=True)

    try:
        for target, games, ms in [
            (1320, 4, 2000),
            (1600, 4, 3000),
            (1900, 4, 4000),
            (2200, 4, 6000),
        ]:
            sc = 0.0
            for gi in range(games):
                b = chess.Board()
                stig_white = gi % 2 == 0
                for _ in range(100):
                    if b.is_game_over(claim_draw=True):
                        break
                    if (b.turn == chess.WHITE) == stig_white:
                        b.push(eng.choose_move(b, time_ms=ms, max_depth=12).move)
                    else:
                        sf.set_elo(target)
                        b.push(sf.choose(b, movetime_ms=60))
                res = b.result(claim_draw=True)
                if b.is_checkmate():
                    res = "0-1" if b.turn == chess.WHITE else "1-0"
                if res == "1-0":
                    s = 1.0 if stig_white else 0.0
                elif res == "0-1":
                    s = 0.0 if stig_white else 1.0
                else:
                    s = 0.5
                sc += s
            print(f"vs {target}: {sc}/{games} ({sc / games:.0%}) think={ms}ms", flush=True)
    finally:
        sf.close()
        set_swarm(None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
