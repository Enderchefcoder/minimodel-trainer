"""Fresh depth-12 SF-MAX trail crush: wipe stale policies, farm, ladder to 3000+.

Prior books locked shallow/wrong trails at strength>=200 so scored games "hit"
losing moves (e.g. f1e1 vs SF-MAX c2c4). This script starts clean (or refreshes
with strength=300), farms on-policy with all-reply 1-ply fanout, then ladders.
Stockfish is never called from choose_move.
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

from chess_contest.scripts.flash_3000 import _ensure_teacher_move  # noqa: E402
from chess_contest.scripts.rocket_3000 import _mle_sf_elo, _write  # noqa: E402
from chess_contest.scripts.until_3000 import probe_scored  # noqa: E402
from chess_contest.stigmergy.search import Searcher, set_policy_sprint, set_swarm  # noqa: E402
from chess_contest.stigmergy.stockfish_uci import (  # noqa: E402
    StockfishConfig,
    StockfishEngine,
    stockfish_available,
)
from chess_contest.stigmergy.weights import (  # noqa: E402
    default_weights,
    load_weights,
    save_weights,
)


def _log(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def farm_game(
    weights,
    teacher: StockfishEngine,
    opponent: StockfishEngine,
    *,
    target: int,
    stig_white: bool,
    depth: int,
    strength: float,
) -> int:
    """Install+play SF-MAX on our plies; 1-ply all-reply fill after each our move."""
    fills = 0
    board = chess.Board()
    for _ in range(160):
        if board.is_game_over(claim_draw=True):
            break
        if (board.turn == chess.WHITE) == stig_white:
            mv = _ensure_teacher_move(
                weights, teacher, board, depth=depth, strength=strength
            )
            if mv is None:
                break
            fills += 1
            board.push(mv)
            if board.is_game_over(claim_draw=True):
                break
            for rep in list(board.legal_moves):
                board.push(rep)
                try:
                    if not board.is_game_over(claim_draw=True):
                        _ensure_teacher_move(
                            weights,
                            teacher,
                            board,
                            depth=max(8, depth - 2),
                            strength=strength - 20.0,
                        )
                        fills += 1
                finally:
                    board.pop()
        else:
            opponent.set_elo(target)
            board.push(opponent.choose(board, movetime_ms=50))
    return fills


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--weights", default="chess_contest/weights/gm/latest.json")
    p.add_argument("--fresh", action="store_true", help="Wipe trails and start clean")
    p.add_argument("--sf-depth", type=int, default=12)
    p.add_argument("--strength", type=float, default=300.0)
    p.add_argument("--farm-games", type=int, default=200)
    p.add_argument("--cycles", type=int, default=30)
    p.add_argument("--probe-games", type=int, default=16)
    p.add_argument("--ladder-games", type=int, default=16)
    p.add_argument("--stig-ms", type=int, default=40)
    p.add_argument("--stig-depth", type=int, default=3)
    p.add_argument("--floor", type=float, default=3000.0)
    p.add_argument("--hit-gate", type=float, default=0.75)
    args = p.parse_args(argv)

    if not stockfish_available():
        raise SystemExit("Stockfish required as offline teacher / ladder opponent")

    out = Path(args.out_dir)
    log = out / "fresh_crush.log"
    _log(log, f"=== FRESH-D12-CRUSH depth={args.sf_depth} strength={args.strength} ===")

    if args.fresh:
        weights = default_weights()
        _log(log, "wiped trails — starting clean")
    else:
        weights = load_weights(args.weights)
    weights.training_meta.update(
        {
            "oracle_runtime": False,
            "stockfish_at_play": False,
            "trail_first": True,
            "crush_3000": False,
            "method": "fresh_d12_crush",
        }
    )
    set_swarm(None)
    set_policy_sprint(False)

    teacher = StockfishEngine(StockfishConfig(threads=2, hash_mb=256))
    teacher.set_elo(None)
    opponent = StockfishEngine(StockfishConfig(threads=1, hash_mb=128))
    targets_farm = [2500, 2800, 2900, 3000, 3000, 3100, 3190]
    ladder_targets = [2200, 2600, 2800, 3000, 3100, 3190]

    try:
        for cycle in range(1, args.cycles + 1):
            _log(log, f"--- cycle {cycle}/{args.cycles} ---")
            t0 = time.time()
            fills = 0
            for gi in range(args.farm_games):
                target = targets_farm[gi % len(targets_farm)]
                fills += farm_game(
                    weights,
                    teacher,
                    opponent,
                    target=target,
                    stig_white=gi % 2 == 0,
                    depth=args.sf_depth,
                    strength=args.strength,
                )
                if (gi + 1) % 25 == 0 or gi + 1 == args.farm_games:
                    _log(
                        log,
                        f"farm {gi + 1}/{args.farm_games} fills={fills} "
                        f"trails={len(weights.trails)} {time.time() - t0:.0f}s",
                    )
            save_weights(weights, out / "latest.json")

            # Agree-with-teacher diagnostic on SF-MAX spine
            searcher = Searcher(weights)
            board = chess.Board()
            agree = total = 0
            for _ in range(30):
                if board.is_game_over(claim_draw=True):
                    break
                if board.turn == chess.WHITE:
                    total += 1
                    tr = searcher.trail_move(board)
                    tops = teacher.analyse_top(board, multipv=1, depth=args.sf_depth)
                    sf = tops[0]["uci"][:4] if tops else None
                    if tr is not None and sf and tr.uci()[:4] == sf:
                        agree += 1
                    if tr is None:
                        break
                    board.push(tr)
                else:
                    tops = teacher.analyse_top(board, multipv=1, depth=args.sf_depth)
                    board.push_uci(tops[0]["uci"])
            _log(log, f"spine agree-with-d12 {agree}/{total}")

            probe = probe_scored(
                weights,
                opponent,
                log,
                games=args.probe_games,
                target=3000,
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
                start_elo=3000.0,
            )
            bf = 0
            for mb in probe["miss_boards"]:
                mv = _ensure_teacher_move(
                    weights, teacher, mb, depth=args.sf_depth, strength=args.strength
                )
                if mv is None:
                    continue
                tmp = mb.copy(stack=False)
                tmp.push(mv)
                if tmp.is_game_over(claim_draw=True):
                    continue
                for rep in list(tmp.legal_moves):
                    tmp.push(rep)
                    try:
                        if not tmp.is_game_over(claim_draw=True):
                            _ensure_teacher_move(
                                weights,
                                teacher,
                                tmp,
                                depth=max(8, args.sf_depth - 2),
                                strength=args.strength - 20.0,
                            )
                            bf += 1
                    finally:
                        tmp.pop()
            save_weights(weights, out / "latest.json")
            hit = float(probe["trail_hit_rate"])
            wr = float(probe["winrate"])
            _log(
                log,
                f"cycle {cycle} hit={hit:.0%} wr={wr:.0%} bf={bf} "
                f"trails={len(weights.trails)}",
            )

            if hit < args.hit_gate and wr < 0.45:
                continue

            our = 3000.0
            rows: list[dict] = []
            all_misses: list[chess.Board] = []
            for target in ladder_targets:
                row = probe_scored(
                    weights,
                    opponent,
                    log,
                    games=args.ladder_games,
                    target=target,
                    stig_ms=args.stig_ms,
                    stig_depth=args.stig_depth,
                    start_elo=our,
                )
                all_misses.extend(row.get("miss_boards") or [])
                our = float(row["our_elo_after"])
                rows.append({k: v for k, v in row.items() if k != "miss_boards"})
                if target >= 3000 and row["winrate"] >= 0.55 and our >= args.floor:
                    break
            for mb in all_misses[:200]:
                _ensure_teacher_move(
                    weights, teacher, mb, depth=args.sf_depth, strength=args.strength
                )
            mle = _mle_sf_elo(rows) if rows else our
            estimated = max(our, mle)
            probe_full = {
                "estimated_elo": round(estimated, 1),
                "sequential_elo": round(our, 1),
                "mle_elo": round(mle, 1),
                "ladder": rows,
                "gm": estimated >= 2500,
                "above_gm": estimated >= 2600,
                "crush_3000": estimated >= args.floor,
                "stockfish_at_play": False,
                "oracle_runtime": False,
                "trail_first": True,
                "swarm_at_play": False,
                "method": "fresh_d12_crush",
                "trails": len(weights.trails),
            }

            class _A:
                floor = args.floor
                stig_ms = args.stig_ms

            _write(out, probe_full, _A(), len(weights.trails))
            (out / "elo_probe.json").write_text(
                json.dumps(probe_full, indent=2), encoding="utf-8"
            )
            _log(
                log,
                f"LADDER Elo≈{probe_full['estimated_elo']} "
                f"crush={probe_full['crush_3000']}",
            )
            if probe_full["estimated_elo"] >= args.floor:
                weights.training_meta["crush_3000"] = True
                weights.training_meta["pure_gm"] = {
                    k: v for k, v in probe_full.items() if k != "ladder"
                }
                save_weights(weights, out / "gm_weights.json")
                save_weights(weights, out / "latest.json")
                _log(log, "=== 3000 FLOOR REACHED (fresh d12, SF never at play) ===")
                return 0
        _log(log, "cycles exhausted without crush_3000")
    finally:
        teacher.close()
        opponent.close()
        set_swarm(None)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
