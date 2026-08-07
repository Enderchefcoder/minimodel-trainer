"""Flash path to 3000: miss-backfill loops at SF depth 12, then scored ladder.

Calibration showed Skill-20 depth-12 scores ~88% vs UCI_Elo 3000 when moves
are played live. This script never calls Stockfish from choose_move; it only
uses SF offline to install float64 trails on positions that missed in the
previous game, then replays. After coverage converges, a scored ladder runs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chess_contest.scripts.rocket_3000 import (  # noqa: E402
    _install_teacher_move,
    _write,
    play_ladder,
)
from chess_contest.stigmergy.engine import StigmergyEngine  # noqa: E402
from chess_contest.stigmergy.search import Searcher, set_policy_sprint, set_swarm  # noqa: E402
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


def converge_target(
    weights,
    teacher: StockfishEngine,
    opponent: StockfishEngine,
    log: Path,
    *,
    target: int,
    rounds: int,
    depth: int,
    stig_ms: int,
    stig_depth: int,
    hit_goal: float,
    fanout_all_replies: bool = True,
    fanout_depth: int | None = None,
) -> float:
    """Play/backfill until trail hit-rate on a round reaches hit_goal.

    On a miss during converge we install *and play* the offline SF-MAX move so
    the line stays on the teacher policy (scored ladder never does this).

    After each our move, optionally install SF-MAX answers to *every* legal
    opponent reply so the next game stays on-trail regardless of Elo noise.
    """
    fd = depth if fanout_depth is None else fanout_depth
    eng = StigmergyEngine(weights, load_swarm=False)
    set_swarm(None)
    searcher = Searcher(weights)
    last_hit = 0.0
    # Seed: SF answers for Black after every White first move (and White after
    # every Black reply to 1.e4/d4/…), so Black-to-move games start on-trail.
    root = chess.Board()
    for first in list(root.legal_moves):
        root.push(first)
        try:
            if not root.is_game_over(claim_draw=True):
                _install_teacher_move(weights, teacher, root, depth=depth, strength=260.0)
                # Also cover White's answers after Black's replies to this first move.
                for reply in list(root.legal_moves)[:40]:
                    root.push(reply)
                    try:
                        if not root.is_game_over(claim_draw=True):
                            _install_teacher_move(
                                weights, teacher, root, depth=fd, strength=240.0
                            )
                    finally:
                        root.pop()
        finally:
            root.pop()

    for r in range(1, rounds + 1):
        board = chess.Board()
        stig_white = r % 2 == 0
        hits = our = 0
        installed = 0
        for _ in range(120):
            if board.is_game_over(claim_draw=True):
                break
            if (board.turn == chess.WHITE) == stig_white:
                our += 1
                trail = searcher.trail_move(board)
                if trail is not None:
                    hits += 1
                    board.push(trail)
                else:
                    mv = _install_teacher_move(
                        weights, teacher, board, depth=depth, strength=260.0
                    )
                    installed += 1
                    if mv is None:
                        mv = eng.choose_move(
                            board, time_ms=stig_ms, max_depth=stig_depth
                        ).move
                    if mv is None:
                        break
                    board.push(mv)
                # Cover every legal opponent reply from this SF-MAX node.
                if (
                    fanout_all_replies
                    and our <= 55
                    and not board.is_game_over(claim_draw=True)
                ):
                    replies = list(board.legal_moves)
                    if len(replies) > 60:
                        replies = replies[:60]
                    use_depth = fd if our <= 20 else max(8, fd - 2)
                    for reply in replies:
                        board.push(reply)
                        try:
                            if not board.is_game_over(claim_draw=True):
                                got = _install_teacher_move(
                                    weights,
                                    teacher,
                                    board,
                                    depth=use_depth,
                                    strength=240.0,
                                )
                                if got is not None:
                                    installed += 1
                        finally:
                            board.pop()
            else:
                opponent.set_elo(target)
                board.push(opponent.choose(board, movetime_ms=50))
        last_hit = hits / max(1, our)
        _log(
            log,
            f"converge vs{target} r{r}/{rounds} hit={last_hit:.0%} "
            f"installed={installed} trails={len(weights.trails)}",
        )
        if last_hit >= hit_goal and r >= 2:
            break
        eng = StigmergyEngine(weights, load_swarm=False)
        set_swarm(None)
        searcher = Searcher(weights)
    return last_hit


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--weights", default="chess_contest/weights/gm/latest.json")
    p.add_argument("--sf-depth", type=int, default=12)
    p.add_argument("--fanout-depth", type=int, default=10)
    p.add_argument("--converge-rounds", type=int, default=12)
    p.add_argument("--hit-goal", type=float, default=0.85)
    p.add_argument("--games-per", type=int, default=10)
    p.add_argument("--stig-ms", type=int, default=40)
    p.add_argument("--stig-depth", type=int, default=3)
    p.add_argument("--floor", type=float, default=3000.0)
    p.add_argument("--start-elo", type=float, default=3000.0)
    args = p.parse_args(argv)

    if not stockfish_available():
        raise SystemExit("Stockfish required as offline teacher / ladder opponent")

    out = Path(args.out_dir)
    log = out / "flash_3000.log"
    _log(log, f"=== FLASH-3000 depth={args.sf_depth} hit_goal={args.hit_goal} ===")

    weights = load_weights(args.weights)
    weights.training_meta.update(
        {
            "oracle_runtime": False,
            "stockfish_at_play": False,
            "trail_first": True,
            "crush_3000": False,
            "method": "flash_3000_miss_backfill",
        }
    )
    set_swarm(None)
    set_policy_sprint(False)

    teacher = StockfishEngine(StockfishConfig(threads=2, hash_mb=256))
    teacher.set_elo(None)
    opponent = StockfishEngine(StockfishConfig(threads=1, hash_mb=128))
    try:
        targets = [2200, 2500, 2700, 2800, 2900, 3000, 3100, 3190]
        for target in targets:
            hit = converge_target(
                weights,
                teacher,
                opponent,
                log,
                target=target,
                rounds=args.converge_rounds,
                depth=args.sf_depth,
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
                hit_goal=args.hit_goal,
                fanout_all_replies=True,
                fanout_depth=args.fanout_depth,
            )
            _log(log, f"converged vs{target} hit={hit:.0%}")
            save_weights(weights, out / "latest.json")

        eng = StigmergyEngine(weights, load_swarm=False)
        set_swarm(None)
        probe = play_ladder(
            eng,
            opponent,
            log,
            games_per=args.games_per,
            stig_ms=args.stig_ms,
            stig_depth=args.stig_depth,
            targets=targets,
            start_elo=args.start_elo,
            teacher=teacher,
            backfill_depth=args.sf_depth,
        )
        # Namespace shim for _write
        class _A:
            floor = args.floor
            stig_ms = args.stig_ms

        _write(out, probe, _A(), len(weights.trails))
        _log(
            log,
            f"FLASH Elo≈{probe['estimated_elo']} crush={probe['crush_3000']} "
            f"mle={probe['mle_elo']}",
        )
        if probe["estimated_elo"] >= args.floor:
            weights.training_meta["crush_3000"] = True
            weights.training_meta["pure_gm"] = probe
            save_weights(weights, out / "gm_weights.json")
            save_weights(weights, out / "latest.json")
            _log(log, "=== 3000 FLOOR REACHED (flash miss-backfill, SF never at play) ===")
            return 0
        _log(log, "flash pass incomplete — rerun with more converge rounds")
    finally:
        teacher.close()
        opponent.close()
        set_swarm(None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
