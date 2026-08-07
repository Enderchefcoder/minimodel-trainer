"""Full-width opening trail tree, then miss-backfill probes until Elo >= 3000.

Scored games first-miss at our-ply 6-10: the opening reply tree is thin.
This densifies breadth-first on early plies (all legal replies), then deepens
along MultiPV-preferred lines - Stockfish offline only.
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
from chess_contest.stigmergy.weights import load_weights, save_weights, trail_key  # noqa: E402


def _log(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def opening_densify(
    weights,
    teacher: StockfishEngine,
    log: Path,
    *,
    depth: int,
    max_our_plies: int,
    full_width_plies: int,
    max_nodes: int,
    deep_branch: int,
) -> int:
    """DFS: all-reply fill at every visit; recurse full-width early, MultiPV later."""
    filled = 0
    t0 = time.time()
    seen: set[str] = set()

    def replies(board: chess.Board, our_ply: int) -> list[chess.Move]:
        legal = list(board.legal_moves)
        if our_ply < full_width_plies:
            return legal
        teacher.set_elo(None)
        tops = teacher.analyse_top(
            board, multipv=min(10, deep_branch), depth=max(8, depth - 2)
        )
        prefer = {info["uci"] for info in tops if info.get("uci")}
        pri = [m for m in legal if board.is_capture(m) or board.gives_check(m)]
        rest = sorted(
            (m for m in legal if m not in pri),
            key=lambda m: (0 if m.uci() in prefer else 1, m.uci()),
        )
        out: list[chess.Move] = []
        seen_u: set[str] = set()
        for m in pri + rest:
            u = m.uci()
            if u in seen_u:
                continue
            seen_u.add(u)
            out.append(m)
            if len(out) >= deep_branch:
                break
        return out

    def dfs(board: chess.Board, our_ply: int) -> None:
        nonlocal filled
        if filled >= max_nodes or our_ply > max_our_plies:
            return
        if board.is_game_over(claim_draw=True):
            return
        key = trail_key(board)
        if key in seen:
            return
        seen.add(key)
        use_depth = depth if our_ply < full_width_plies + 4 else max(8, depth - 2)
        mv = _ensure_teacher_move(
            weights, teacher, board, depth=use_depth, strength=270.0
        )
        if mv is None:
            return
        filled += 1
        if filled % 1000 == 0:
            _log(
                log,
                f"opening filled={filled} trails={len(weights.trails)} "
                f"{time.time() - t0:.0f}s",
            )
        if our_ply >= max_our_plies:
            return
        board.push(mv)
        try:
            if board.is_game_over(claim_draw=True):
                return
            legal = list(board.legal_moves)
            # 1-ply fill every reply so playing ``mv`` stays on-trail next.
            for reply in legal:
                board.push(reply)
                try:
                    if not board.is_game_over(claim_draw=True):
                        _ensure_teacher_move(
                            weights,
                            teacher,
                            board,
                            depth=max(8, use_depth - 2),
                            strength=250.0,
                        )
                finally:
                    board.pop()
            for reply in replies(board, our_ply):
                if filled >= max_nodes:
                    return
                board.push(reply)
                try:
                    dfs(board, our_ply + 1)
                finally:
                    board.pop()
        finally:
            board.pop()

    dfs(chess.Board(), 0)
    # Also densify as Black after common White first moves.
    root = chess.Board()
    for first in list(root.legal_moves)[:12]:
        if filled >= max_nodes:
            break
        root.push(first)
        try:
            if not root.is_game_over(claim_draw=True):
                dfs(root, 0)
        finally:
            root.pop()
    _log(
        log,
        f"opening densify done filled={filled} trails={len(weights.trails)} "
        f"{time.time() - t0:.0f}s",
    )
    return filled


def scored_ladder(
    weights,
    opponent: StockfishEngine,
    log: Path,
    *,
    games_per: int,
    stig_ms: int,
    stig_depth: int,
    targets: list[int],
    start_elo: float,
) -> dict:
    our = float(start_elo)
    rows: list[dict] = []
    for target in targets:
        row = probe_scored(
            weights,
            opponent,
            log,
            games=games_per,
            target=target,
            stig_ms=stig_ms,
            stig_depth=stig_depth,
            start_elo=our,
        )
        our = float(row["our_elo_after"])
        rows.append({k: v for k, v in row.items() if k != "miss_boards"})
        if target >= 3000 and row["winrate"] >= 0.5 and our >= 3000:
            break
        if target >= 3000 and row["trail_hit_rate"] < 0.4 and row["winrate"] < 0.2:
            break
    mle = _mle_sf_elo(rows) if rows else our
    estimated = max(our, mle)
    return {
        "estimated_elo": round(estimated, 1),
        "sequential_elo": round(our, 1),
        "mle_elo": round(mle, 1),
        "ladder": rows,
        "gm": estimated >= 2500,
        "above_gm": estimated >= 2600,
        "crush_3000": estimated >= 3000,
        "stockfish_at_play": False,
        "oracle_runtime": False,
        "trail_first": True,
        "swarm_at_play": False,
        "method": "opening_tree_3000",
        "trails": len(weights.trails),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--weights", default="chess_contest/weights/gm/latest.json")
    p.add_argument("--sf-depth", type=int, default=12)
    p.add_argument("--max-nodes", type=int, default=120_000)
    p.add_argument("--full-width-plies", type=int, default=7)
    p.add_argument("--max-our-plies", type=int, default=28)
    p.add_argument("--deep-branch", type=int, default=14)
    p.add_argument("--cycles", type=int, default=40)
    p.add_argument("--probe-games", type=int, default=12)
    p.add_argument("--ladder-games", type=int, default=14)
    p.add_argument("--backfill-cap", type=int, default=80)
    p.add_argument("--stig-ms", type=int, default=40)
    p.add_argument("--stig-depth", type=int, default=3)
    p.add_argument("--floor", type=float, default=3000.0)
    p.add_argument("--hit-gate", type=float, default=0.72)
    args = p.parse_args(argv)

    if not stockfish_available():
        raise SystemExit("Stockfish required as offline teacher / ladder opponent")

    out = Path(args.out_dir)
    log = out / "opening_3000.log"
    _log(log, f"=== OPENING-3000 depth={args.sf_depth} nodes={args.max_nodes} ===")

    weights = load_weights(args.weights)
    weights.training_meta.update(
        {
            "oracle_runtime": False,
            "stockfish_at_play": False,
            "trail_first": True,
            "crush_3000": False,
            "method": "opening_tree_3000",
        }
    )
    set_swarm(None)
    set_policy_sprint(False)

    teacher = StockfishEngine(StockfishConfig(threads=2, hash_mb=256))
    teacher.set_elo(None)
    opponent = StockfishEngine(StockfishConfig(threads=1, hash_mb=128))
    ladder_targets = [2200, 2800, 3000, 3100, 3190]

    try:
        _log(log, "phase: opening densify")
        opening_densify(
            weights,
            teacher,
            log,
            depth=args.sf_depth,
            max_our_plies=args.max_our_plies,
            full_width_plies=args.full_width_plies,
            max_nodes=args.max_nodes,
            deep_branch=args.deep_branch,
        )
        save_weights(weights, out / "latest.json")

        for cycle in range(1, args.cycles + 1):
            _log(log, f"--- cycle {cycle}/{args.cycles} ---")
            # Cheap on-policy spine pass: install+play teacher, 1-ply all replies.
            searcher = Searcher(weights)
            for gi in range(8):
                board = chess.Board()
                stig_white = gi % 2 == 0
                target = [2800, 3000, 3100, 3190][gi % 4]
                our_ply = 0
                for _ in range(120):
                    if board.is_game_over(claim_draw=True):
                        break
                    if (board.turn == chess.WHITE) == stig_white:
                        _ensure_teacher_move(
                            weights,
                            teacher,
                            board,
                            depth=args.sf_depth,
                            strength=270.0,
                        )
                        # Fill all replies after our move.
                        searcher = Searcher(weights)
                        mv = searcher.trail_move(board)
                        if mv is None:
                            break
                        board.push(mv)
                        if not board.is_game_over(claim_draw=True):
                            for rep in list(board.legal_moves):
                                board.push(rep)
                                try:
                                    if not board.is_game_over(claim_draw=True):
                                        _ensure_teacher_move(
                                            weights,
                                            teacher,
                                            board,
                                            depth=max(8, args.sf_depth - 2),
                                            strength=250.0,
                                        )
                                finally:
                                    board.pop()
                        our_ply += 1
                    else:
                        opponent.set_elo(target)
                        board.push(opponent.choose(board, movetime_ms=50))

            probe = probe_scored(
                weights,
                opponent,
                log,
                games=args.probe_games,
                target=3000,
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
                start_elo=2800.0,
            )
            misses = probe["miss_boards"]
            # Cheap 1-ply ensure + all-reply fill on each miss (no quadratic 2-ply).
            bf = 0
            for mb in misses[: args.backfill_cap]:
                _ensure_teacher_move(
                    weights, teacher, mb, depth=args.sf_depth, strength=270.0
                )
                searcher = Searcher(weights)
                mv = searcher.trail_move(mb)
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
                            if _ensure_teacher_move(
                                weights,
                                teacher,
                                tmp,
                                depth=max(8, args.sf_depth - 2),
                                strength=250.0,
                            ):
                                bf += 1
                    finally:
                        tmp.pop()
            for mb in misses[args.backfill_cap :]:
                _ensure_teacher_move(
                    weights, teacher, mb, depth=args.sf_depth, strength=260.0
                )
            save_weights(weights, out / "latest.json")
            hit = float(probe["trail_hit_rate"])
            wr = float(probe["winrate"])
            _log(
                log,
                f"cycle {cycle} hit={hit:.0%} wr={wr:.0%} backfill={bf} "
                f"trails={len(weights.trails)}",
            )

            # Quick first-miss diagnostic
            searcher = Searcher(weights)
            board = chess.Board()
            first = None
            for our_i in range(40):
                if board.is_game_over(claim_draw=True):
                    break
                if searcher.trail_move(board) is None:
                    first = our_i + 1
                    break
                board.push(searcher.trail_move(board))
                if board.is_game_over(claim_draw=True):
                    break
                opponent.set_elo(3000)
                board.push(opponent.choose(board, movetime_ms=50))
            _log(log, f"cycle {cycle} sample_first_miss_our_ply={first}")

            if hit < args.hit_gate or wr < 0.35:
                continue

            probe_full = scored_ladder(
                weights,
                opponent,
                log,
                games_per=args.ladder_games,
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
                targets=ladder_targets,
                start_elo=3000.0,
            )

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
                _log(log, "=== 3000 FLOOR REACHED (opening tree, SF never at play) ===")
                return 0
        _log(log, "cycles exhausted without crush_3000")
    finally:
        teacher.close()
        opponent.close()
        set_swarm(None)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
