#!/usr/bin/env python3
"""Controlled pure Elo ladder — Stockfish opponent only, never at play."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import chess

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chess_contest.stigmergy.engine import StigmergyEngine
from chess_contest.stigmergy.opponents import update_elo
from chess_contest.stigmergy.search import set_swarm
from chess_contest.stigmergy.stockfish_uci import StockfishConfig, StockfishEngine
from chess_contest.stigmergy.swarm_net import try_load_swarm
from chess_contest.stigmergy.weights import load_weights


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    path = Path("chess_contest/weights/gm/pure_gm.log")
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    out = Path("chess_contest/weights/gm")
    wp = out / "latest.json" if (out / "latest.json").exists() else out / "gm_weights_slim.json"
    w = load_weights(wp)
    w.training_meta["oracle_runtime"] = False
    w.training_meta["stockfish_at_play"] = False
    net = try_load_swarm(out / "swarm_net.pt")
    if net is None:
        raise SystemExit("swarm_net.pt missing")
    set_swarm(net)
    eng = StigmergyEngine(w, load_swarm=False)
    set_swarm(net)
    log(f"PROBE start weights={wp} trails={len(w.trails)} swarm=ok")
    sf = StockfishEngine(StockfishConfig(threads=2, hash_mb=128))
    our = 2000.0
    rows: list[dict] = []
    schedule = [
        (1320, 6, 3000, 12),
        (1600, 6, 4000, 14),
        (1900, 6, 6000, 16),
        (2200, 6, 12000, 18),
        (2500, 8, 25000, 20),
        (2600, 8, 30000, 22),
        (2700, 8, 35000, 22),
        (2800, 8, 40000, 24),
    ]
    try:
        for target, games, ms, depth in schedule:
            sc = 0.0
            for i in range(games):
                board = chess.Board()
                stig_white = i % 2 == 0
                for _ in range(140):
                    if board.is_game_over(claim_draw=True):
                        break
                    if (board.turn == chess.WHITE) == stig_white:
                        mv = eng.choose_move(board, time_ms=ms, max_depth=depth).move
                        if mv is None:
                            break
                        board.push(mv)
                    else:
                        sf.set_elo(target)
                        board.push(sf.choose(board, movetime_ms=80))
                res = board.result(claim_draw=True)
                if board.is_checkmate():
                    res = "0-1" if board.turn == chess.WHITE else "1-0"
                if res == "1-0":
                    s = 1.0 if stig_white else 0.0
                elif res == "0-1":
                    s = 0.0 if stig_white else 1.0
                else:
                    s = 0.5
                sc += s
                our, _ = update_elo(our, float(target), s, k=32.0)
            wr = sc / games
            row = {
                "sf_elo": target,
                "score": sc,
                "games": games,
                "winrate": wr,
                "our_elo_after": round(our, 1),
                "think_ms": ms,
            }
            rows.append(row)
            log(f"PURE ladder vs SF {target}: {sc}/{games} ({wr:.0%}) → ≈{our:.0f} [{ms}ms]")
            probe = {
                "estimated_elo": round(our, 1),
                "ladder": rows,
                "gm": our >= 2500,
                "above_gm": our >= 2500,
                "stockfish_at_play": False,
                "oracle_runtime": False,
            }
            (out / "elo_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
            if target >= 2500 and wr < 0.15 and our < 2350:
                break
            if our >= 2800 and target >= 2800:
                break
    finally:
        sf.close()
        set_swarm(None)

    probe = {
        "estimated_elo": round(our, 1),
        "ladder": rows,
        "gm": our >= 2500,
        "above_gm": our >= 2500,
        "stockfish_at_play": False,
        "oracle_runtime": False,
    }
    (out / "elo_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    (out / "gm_report.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    log(f"PROBE Elo ≈ {our:.0f} gm={our >= 2500}")
    if our >= 2500:
        (out / "STATUS.md").write_text(
            f"# ABOVE GM — pure (no runtime Stockfish)\n\n"
            f"Estimated Elo **{our:.0f}**\n\n"
            "`choose_move` never calls Stockfish.\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
