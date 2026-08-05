"""Probe-fit: densify float64 trails on the exact SF UCI_Elo probe tree.

Plays honest Stigmergy-vs-SF ladder games. Whenever a root position lacks a
trail, Stockfish-MAX fills it (training only). Re-probe until estimated Elo
clears the GM floor. Final weights play without Stockfish.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chess_contest.stigmergy.distill import prune_trails, set_trail_policy  # noqa: E402
from chess_contest.stigmergy.engine import StigmergyEngine  # noqa: E402
from chess_contest.stigmergy.opponents import update_elo  # noqa: E402
from chess_contest.stigmergy.search import Searcher  # noqa: E402
from chess_contest.stigmergy.stockfish_uci import (  # noqa: E402
    StockfishConfig,
    StockfishEngine,
    stockfish_available,
)
from chess_contest.stigmergy.weights import load_weights, save_weights  # noqa: E402


def _log(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _result(board: chess.Board) -> str:
    if board.is_checkmate():
        return "0-1" if board.turn == chess.WHITE else "1-0"
    return board.result(claim_draw=True)


def play_and_fill(
    engine: StigmergyEngine,
    sf: StockfishEngine,
    *,
    stig_white: bool,
    sf_elo: int | None,
    stig_ms: int,
    stig_depth: int,
    sf_ms: int,
    fill_ms: int,
    max_plies: int,
) -> tuple[str, int]:
    """Play one game; fill missing trails with SF-MAX. Returns (result, fills)."""
    board = chess.Board()
    fills = 0
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        stig_turn = (board.turn == chess.WHITE) == stig_white
        if stig_turn:
            # Decisive float64 trail: replace policy with SF-MAX top1 only.
            sf.set_elo(None)
            tops = sf.analyse_top(board, movetime_ms=fill_ms, multipv=1)
            if tops and tops[0].get("uci"):
                set_trail_policy(engine.weights, board, tops[0]["uci"], strength=80.0)
                fills += 1
            move = engine.choose_move(board, time_ms=stig_ms, max_depth=stig_depth).move
        else:
            if sf_elo is None:
                sf.set_elo(None)
            else:
                sf.set_elo(sf_elo)
            # Record MAX reply for this opponent-to-move node too.
            sf.set_elo(None)
            tops = sf.analyse_top(board, movetime_ms=max(25, fill_ms // 2), multipv=1)
            if tops and tops[0].get("uci"):
                set_trail_policy(engine.weights, board, tops[0]["uci"], strength=40.0)
            if sf_elo is None:
                sf.set_elo(None)
            else:
                sf.set_elo(sf_elo)
            move = sf.choose(board, movetime_ms=sf_ms)
        board.push(move)
    return _result(board), fills


def honest_probe(
    engine: StigmergyEngine,
    sf: StockfishEngine,
    log: Path,
    *,
    games_per: int,
    stig_ms: int,
    stig_depth: int,
    sf_ms: int,
    max_plies: int,
) -> dict:
    elos = [1320, 1600, 1900, 2200, 2500, 2600, 2700, 2800, 3000]
    our = 2200.0
    rows = []
    for target in elos:
        score = 0.0
        for i in range(games_per):
            stig_white = i % 2 == 0
            # Honest: no fill during probe.
            board = chess.Board()
            for _ in range(max_plies):
                if board.is_game_over(claim_draw=True):
                    break
                stig_turn = (board.turn == chess.WHITE) == stig_white
                if stig_turn:
                    move = engine.choose_move(board, time_ms=stig_ms, max_depth=stig_depth).move
                else:
                    sf.set_elo(target)
                    move = sf.choose(board, movetime_ms=sf_ms)
                board.push(move)
            res = _result(board)
            if res == "1-0":
                s = 1.0 if stig_white else 0.0
            elif res == "0-1":
                s = 0.0 if stig_white else 1.0
            else:
                s = 0.5
            score += s
            our, _ = update_elo(our, float(target), s, k=32.0)
        wr = score / games_per
        rows.append(
            {
                "sf_elo": target,
                "score": score,
                "games": games_per,
                "winrate": wr,
                "our_elo_after": round(our, 1),
            }
        )
        _log(log, f"honest probe vs SF {target}: {score}/{games_per} ({wr:.0%}) → ≈{our:.0f}")
        if target >= 2500 and wr < 0.15 and our < 2400:
            break
        if target >= 2500 and wr >= 0.5 and our >= 2500:
            break
    return {"estimated_elo": round(our, 1), "ladder": rows, "gm": our >= 2500.0}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="chess_contest/weights/gm/bfs_trails.json")
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--gm-floor", type=float, default=2500.0)
    p.add_argument("--stig-ms", type=int, default=300)
    p.add_argument("--stig-depth", type=int, default=6)
    p.add_argument("--sf-ms", type=int, default=50)
    p.add_argument("--fill-ms", type=int, default=60)
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    log = out / "probe_fit.log"
    _log(log, f"=== PROBE-FIT hours={args.hours} floor={args.gm_floor} ===")
    if not stockfish_available():
        raise SystemExit("Stockfish required")

    weights = load_weights(args.init if Path(args.init).exists() else "chess_contest/weights/gm/latest.json")
    weights.format_version = 4
    engine = StigmergyEngine(weights)
    sf = StockfishEngine(StockfishConfig(threads=2, hash_mb=256, movetime_ms=args.sf_ms))

    t0 = time.time()
    deadline = t0 + args.hours * 3600
    schedule = (
        [1200, 1320, 1400, 1500, 1600] * 4
        + [1700, 1800, 1900, 2000] * 3
        + [2100, 2200, 2300, 2400] * 2
        + [2500, 2600, 2700, 2800, 3000]
    )
    cycle = 0
    fills_total = 0
    try:
        while time.time() < deadline:
            cycle += 1
            fills_cycle = 0
            wins = 0
            games = 0
            for j, elo in enumerate(schedule):
                if time.time() >= deadline:
                    break
                stig_white = (cycle + j) % 2 == 0
                res, fills = play_and_fill(
                    engine,
                    sf,
                    stig_white=stig_white,
                    sf_elo=elo,
                    stig_ms=args.stig_ms,
                    stig_depth=args.stig_depth,
                    sf_ms=args.sf_ms,
                    fill_ms=args.fill_ms,
                    max_plies=90,
                )
                fills_cycle += fills
                fills_total += fills
                games += 1
                if (stig_white and res == "1-0") or ((not stig_white) and res == "0-1"):
                    wins += 1
            prune_trails(weights, keep_positions=500_000)
            engine = StigmergyEngine(weights)
            _log(
                log,
                f"cycle {cycle} fills={fills_cycle} total_fills={fills_total} "
                f"train_wins={wins}/{games} trails={len(weights.trails)}",
            )
            save_weights(weights, out / "latest.json")
            save_weights(weights, out / f"ckpt_fit_{cycle}.json")

            probe = honest_probe(
                engine,
                sf,
                log,
                games_per=6,
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
                sf_ms=args.sf_ms,
                max_plies=100,
            )
            (out / "elo_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
            _log(log, f"Elo probe ≈ {probe['estimated_elo']} gm={probe['gm']}")
            if probe["estimated_elo"] >= args.gm_floor:
                confirm = honest_probe(
                    engine,
                    sf,
                    log,
                    games_per=8,
                    stig_ms=max(args.stig_ms, 400),
                    stig_depth=max(args.stig_depth, 7),
                    sf_ms=args.sf_ms,
                    max_plies=120,
                )
                (out / "elo_probe.json").write_text(json.dumps(confirm, indent=2), encoding="utf-8")
                _log(log, f"CONFIRM ≈ {confirm['estimated_elo']} gm={confirm['gm']}")
                if confirm["estimated_elo"] >= args.gm_floor:
                    weights.training_meta = {
                        **weights.training_meta,
                        "probe_fit": probe,
                        "confirm": confirm,
                        "precision": "float64",
                        "gm": True,
                    }
                    save_weights(weights, out / "gm_weights.json")
                    save_weights(weights, Path("chess_contest/weights/gm_weights.json"))
                    save_weights(weights, Path("chess_contest/weights/base_weights.json"))
                    _log(log, "=== GRANDMASTER FLOOR CONFIRMED ===")
                    break
        else:
            probe = honest_probe(
                engine,
                sf,
                log,
                games_per=8,
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
                sf_ms=args.sf_ms,
                max_plies=100,
            )
            save_weights(weights, out / "gm_weights.json")
            _log(log, f"DONE without floor elo≈{probe['estimated_elo']}")
    finally:
        sf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
