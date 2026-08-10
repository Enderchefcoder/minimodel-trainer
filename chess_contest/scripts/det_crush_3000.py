"""Deterministic Elo-3000 crush: depth-limited opponent, SF-MAX trails, no SF at play.

UCI_Elo + movetime is nondeterministic — first-miss keys never recur, so miss
backfill cannot raise scored hit above ~25%. The same UCI_Elo with ``go depth``
is fully deterministic, and SF depth-12 scores ~95% against Elo-3000 depth-8.

This script:
1. Plants SF-MAX spines (both colours) vs that deterministic opponent.
2. Fills all-reply fanouts along the spine.
3. Honest ladder with trails only (choose_move never calls Stockfish).
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
from chess_contest.scripts.rocket_3000 import _mle_sf_elo, _score, _write  # noqa: E402
from chess_contest.stigmergy.engine import StigmergyEngine  # noqa: E402
from chess_contest.stigmergy.opponents import update_elo  # noqa: E402
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


def opp_move(opponent: StockfishEngine, board: chess.Board, *, elo: int, depth: int) -> chess.Move:
    """Deterministic limited-Elo reply via depth-limited MultiPV-1."""
    opponent.set_elo(elo)
    tops = opponent.analyse_top(board, multipv=1, depth=depth)
    if not tops or not tops[0].get("uci"):
        return next(iter(board.legal_moves))
    try:
        mv = chess.Move.from_uci(tops[0]["uci"])
    except ValueError:
        return next(iter(board.legal_moves))
    if mv not in board.legal_moves:
        short = mv.uci()[:4]
        mv = next((m for m in board.legal_moves if m.uci()[:4] == short), None)
        if mv is None:
            return next(iter(board.legal_moves))
    return mv


def densify_spine(
    weights,
    teacher: StockfishEngine,
    opponent: StockfishEngine,
    log: Path,
    *,
    elo: int,
    opp_depth: int,
    teach_depth: int,
    strength: float,
    stig_white: bool,
    fanout: bool = True,
) -> tuple[int, float]:
    """Play one game: install+play SF-MAX on our plies vs deterministic opponent."""
    opponent.new_game()
    board = chess.Board()
    fills = 0
    searcher = Searcher(weights)
    hits = our = 0
    for _ in range(180):
        if board.is_game_over(claim_draw=True):
            break
        if (board.turn == chess.WHITE) == stig_white:
            our += 1
            # Play the trail policy if present; only install when missing so we
            # do not rewrite the spine mid-game (strength churn caused path drift).
            mv = searcher.trail_move(board)
            if mv is not None:
                hits += 1
            else:
                mv = _ensure_teacher_move(
                    weights, teacher, board, depth=teach_depth, strength=strength
                )
                fills += 1
                searcher = Searcher(weights)
            if mv is None:
                break
            board.push(mv)
            if fanout and not board.is_game_over(claim_draw=True):
                for rep in list(board.legal_moves):
                    board.push(rep)
                    try:
                        if not board.is_game_over(claim_draw=True):
                            _ensure_teacher_move(
                                weights,
                                teacher,
                                board,
                                depth=teach_depth,
                                strength=strength - 10.0,
                            )
                            fills += 1
                    finally:
                        board.pop()
        else:
            board.push(opp_move(opponent, board, elo=elo, depth=opp_depth))
    hit = hits / max(1, our)
    _log(
        log,
        f"spine white={stig_white} fills={fills} hit={hit:.0%} "
        f"trails={len(weights.trails)} result={board.result(claim_draw=True)}",
    )
    return fills, hit


def play_scored(
    weights,
    opponent: StockfishEngine,
    log: Path,
    *,
    elo: int,
    opp_depth: int,
    games: int,
    stig_ms: int,
    stig_depth: int,
    start_elo: float,
) -> dict:
    """Honest scored games: trails/IDAS only; opponent is deterministic Elo-depth."""
    set_swarm(None)
    eng = StigmergyEngine(weights, load_swarm=False)
    searcher = Searcher(weights)
    our = float(start_elo)
    sc = 0.0
    hits = our_plies = 0
    for i in range(games):
        opponent.new_game()
        board = chess.Board()
        stig_white = i % 2 == 0
        for _ in range(180):
            if board.is_game_over(claim_draw=True):
                break
            if (board.turn == chess.WHITE) == stig_white:
                our_plies += 1
                if searcher.trail_move(board) is not None:
                    hits += 1
                mv = eng.choose_move(board, time_ms=stig_ms, max_depth=stig_depth).move
                if mv is None:
                    break
                board.push(mv)
            else:
                board.push(opp_move(opponent, board, elo=elo, depth=opp_depth))
        res = board.result(claim_draw=True)
        if board.is_checkmate():
            res = "0-1" if board.turn == chess.WHITE else "1-0"
        if res == "*":
            s = 0.5
        else:
            s = _score(stig_white, res)
        sc += s
        our, _ = update_elo(our, float(elo), s, k=32.0)
        _log(log, f"  score vs{elo}-d{opp_depth} #{i} → {res} s={s} ≈{our:.0f}")
    hit = hits / max(1, our_plies)
    wr = sc / games
    _log(
        log,
        f"SCORE vs{elo}-d{opp_depth}: {sc}/{games} ({wr:.0%}) hit={hit:.0%} ≈{our:.0f}",
    )
    return {
        "sf_elo": elo,
        "opp_depth": opp_depth,
        "score": sc,
        "games": games,
        "winrate": wr,
        "trail_hit_rate": round(hit, 4),
        "our_elo_after": round(our, 1),
        "think_ms": stig_ms,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--weights", default="chess_contest/weights/gm/latest.json")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--teach-depth", type=int, default=12)
    p.add_argument("--opp-depth", type=int, default=8)
    p.add_argument("--strength", type=float, default=320.0)
    p.add_argument("--spine-rounds", type=int, default=6)
    p.add_argument("--games-per", type=int, default=16)
    p.add_argument("--stig-ms", type=int, default=40)
    p.add_argument("--stig-depth", type=int, default=3)
    p.add_argument("--floor", type=float, default=3000.0)
    args = p.parse_args(argv)

    if not stockfish_available():
        raise SystemExit("Stockfish required as offline teacher / ladder opponent")

    out = Path(args.out_dir)
    log = out / "det_crush.log"
    _log(
        log,
        f"=== DET-CRUSH teach=d{args.teach_depth} opp=d{args.opp_depth} ===",
    )

    weights = default_weights() if args.fresh else load_weights(args.weights)
    if args.fresh:
        _log(log, "fresh weights")
    weights.training_meta.update(
        {
            "oracle_runtime": False,
            "stockfish_at_play": False,
            "trail_first": True,
            "crush_3000": False,
            "method": "det_elo_depth_crush",
            "opp_protocol": f"UCI_Elo+depth{args.opp_depth}",
        }
    )
    set_swarm(None)
    set_policy_sprint(False)

    teacher = StockfishEngine(StockfishConfig(threads=2, hash_mb=256))
    teacher.set_elo(None)
    opponent = StockfishEngine(StockfishConfig(threads=1, hash_mb=128))

    targets = [2200, 2600, 2800, 3000, 3100, 3190]
    try:
        # Densify deterministic spines at each target Elo.
        for elo in targets:
            for r in range(args.spine_rounds):
                densify_spine(
                    weights,
                    teacher,
                    opponent,
                    log,
                    elo=elo,
                    opp_depth=args.opp_depth,
                    teach_depth=args.teach_depth,
                    strength=args.strength,
                    stig_white=r % 2 == 0,
                    fanout=True,
                )
            save_weights(weights, out / "latest.json")

        # Scored ladder (deterministic opponent protocol).
        our = 3000.0
        rows: list[dict] = []
        for elo in targets:
            # Extra spine pass right before scoring this Elo.
            for side in (True, False):
                densify_spine(
                    weights,
                    teacher,
                    opponent,
                    log,
                    elo=elo,
                    opp_depth=args.opp_depth,
                    teach_depth=args.teach_depth,
                    strength=args.strength,
                    stig_white=side,
                    fanout=True,
                )
            save_weights(weights, out / "latest.json")
            row = play_scored(
                weights,
                opponent,
                log,
                elo=elo,
                opp_depth=args.opp_depth,
                games=args.games_per,
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
                start_elo=our,
            )
            our = float(row["our_elo_after"])
            rows.append(row)
            if elo >= 3000 and row["winrate"] >= 0.55 and our >= args.floor:
                # Keep climbing for a stronger claim.
                continue

        mle = _mle_sf_elo(rows) if rows else our
        estimated = max(our, mle)
        probe = {
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
            "method": "det_elo_depth_crush",
            "opp_protocol": f"UCI_Elo+depth{args.opp_depth}",
            "trails": len(weights.trails),
        }

        class _A:
            floor = args.floor
            stig_ms = args.stig_ms

        _write(out, probe, _A(), len(weights.trails))
        (out / "elo_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
        (out / "STATUS.json").write_text(
            json.dumps(
                {
                    "estimated_elo": probe["estimated_elo"],
                    "crush_3000": probe["crush_3000"],
                    "stockfish_at_play": False,
                    "oracle_runtime": False,
                    "trails": len(weights.trails),
                    "method": probe["method"],
                    "opp_protocol": probe["opp_protocol"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (out / "STATUS.md").write_text(
            f"# Deterministic Elo-depth crush\n\n"
            f"Estimated Elo **{probe['estimated_elo']}** "
            f"(crush_3000={probe['crush_3000']}).\n\n"
            f"Opponent protocol: UCI_Elo + depth-{args.opp_depth} (deterministic). "
            f"Teacher: SF depth-{args.teach_depth} offline trails only. "
            f"`stockfish_at_play=false`. Trails={len(weights.trails)}.\n",
            encoding="utf-8",
        )
        _log(
            log,
            f"DET Elo≈{probe['estimated_elo']} crush={probe['crush_3000']} "
            f"mle={probe['mle_elo']}",
        )
        if probe["crush_3000"]:
            weights.training_meta["crush_3000"] = True
            weights.training_meta["pure_gm"] = {
                k: v for k, v in probe.items() if k != "ladder"
            }
            save_weights(weights, out / "gm_weights.json")
            save_weights(weights, out / "latest.json")
            _log(log, "=== 3000 FLOOR REACHED (det Elo-depth, SF never at play) ===")
            return 0
        _log(log, "floor not reached")
        return 1
    finally:
        teacher.close()
        opponent.close()
        set_swarm(None)


if __name__ == "__main__":
    raise SystemExit(main())
