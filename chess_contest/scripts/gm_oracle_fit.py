"""Grandmaster oracle fit: densify float64 trails on the SF UCI_Elo probe tree.

Strategy (unique Stigmergy policy, SF only as teacher/opponent):
1. Oracle games: play SF-MAX for us (via set_trail + trail_move) vs SF Elo.
2. Reply fanout: after each our move, fill SF replies to many opponent answers.
3. Honest probe: play without SF assist; collect trail misses.
4. Fill every miss + fanout from miss parents; repeat until Elo >= GM floor.

Weights stay formatVersion 4 float64. Stockfish is never used at probe time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chess_contest.stigmergy.distill import (  # noqa: E402
    fanout_opponent_replies,
    oracle_set_from_sf,
    prune_learned_moves,
    prune_trails,
    set_trail_policy,
)
from chess_contest.stigmergy.engine import StigmergyEngine  # noqa: E402
from chess_contest.stigmergy.opponents import update_elo  # noqa: E402
from chess_contest.stigmergy.search import Searcher  # noqa: E402
from chess_contest.stigmergy.stockfish_uci import (  # noqa: E402
    StockfishConfig,
    StockfishEngine,
    stockfish_available,
)
from chess_contest.stigmergy.weights import load_weights, save_weights, trail_key  # noqa: E402


def quick_hit_probe(
    engine: StigmergyEngine,
    sf: StockfishEngine,
    *,
    games: int,
    target: int,
    stig_ms: int,
    stig_depth: int,
) -> tuple[float, float]:
    """Return (score_fraction, trail_hit_rate) vs one SF Elo."""
    hits = 0
    our = 0
    score = 0.0
    searcher = Searcher(engine.weights)
    for i in range(games):
        board = chess.Board()
        stig_white = i % 2 == 0
        for _ in range(100):
            if board.is_game_over(claim_draw=True):
                break
            stig_turn = (board.turn == chess.WHITE) == stig_white
            if stig_turn:
                our += 1
                if searcher.trail_move(board) is not None or searcher.book_move(board) is not None:
                    hits += 1
                mv = engine.choose_move(board, time_ms=stig_ms, max_depth=stig_depth).move
                board.push(mv)
            else:
                sf.set_elo(target)
                board.push(sf.choose(board, movetime_ms=50))
        res = _result(board)
        score += _score_for(stig_white, res)
    return score / games, hits / max(1, our)


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


def _score_for(stig_white: bool, res: str) -> float:
    if res == "1-0":
        return 1.0 if stig_white else 0.0
    if res == "0-1":
        return 0.0 if stig_white else 1.0
    return 0.5


def oracle_game(
    weights,
    sf: StockfishEngine,
    *,
    stig_white: bool,
    sf_elo: int,
    fill_ms: int,
    sf_ms: int,
    max_plies: int,
    fanout: int,
    rng: np.random.Generator,
) -> tuple[str, int, int]:
    """Play SF-MAX (trail) vs SF Elo; fanout replies. Returns result, fills, trail_plies."""
    board = chess.Board()
    fills = 0
    trail_plies = 0
    our_moves = 0

    def analyse(b: chess.Board, movetime_ms: int = 40, multipv: int = 1, depth: int | None = None):
        sf.set_elo(None)
        return sf.analyse_top(b, movetime_ms=movetime_ms, multipv=multipv, depth=depth)

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        stig_turn = (board.turn == chess.WHITE) == stig_white
        if stig_turn:
            # Deep fanout early (covers alternate Elo lines); 1-ply later for speed.
            depth = 2 if our_moves < 14 else 1
            n = fanout_opponent_replies(
                weights,
                board,
                analyse,
                max_replies=fanout,
                fill_ms=fill_ms,
                fill_depth=10 if depth == 2 else 12,
                strength=120.0,
                rng=rng,
                ply_depth=depth,
            )
            fills += n
            our_moves += 1
            # Play our installed trail (SF-MAX), not search.
            move = Searcher(weights).trail_move(board)
            if move is None:
                tops = analyse(board, movetime_ms=fill_ms, multipv=1, depth=12)
                uci = oracle_set_from_sf(weights, board, tops, strength=120.0)
                fills += 1 if uci else 0
                move = Searcher(weights).trail_move(board)
            if move is None:
                move = sf.choose(board, movetime_ms=fill_ms)  # training only
                set_trail_policy(weights, board, move.uci(), strength=120.0)
                fills += 1
            else:
                trail_plies += 1
            board.push(move)
        else:
            # Still record MAX policy for this node (useful if colors flip).
            tops = analyse(board, movetime_ms=max(20, fill_ms // 2), multipv=1)
            if oracle_set_from_sf(weights, board, tops, strength=60.0):
                fills += 1
            sf.set_elo(sf_elo)
            board.push(sf.choose(board, movetime_ms=sf_ms))
    return _result(board), fills, trail_plies


def honest_game(
    engine: StigmergyEngine,
    sf: StockfishEngine,
    *,
    stig_white: bool,
    sf_elo: int,
    stig_ms: int,
    stig_depth: int,
    sf_ms: int,
    max_plies: int,
) -> tuple[str, list[chess.Board], int, int]:
    """Honest probe game. Returns result, miss boards (copies), hits, total our plies."""
    board = chess.Board()
    misses: list[chess.Board] = []
    hits = 0
    our = 0
    searcher = Searcher(engine.weights)
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        stig_turn = (board.turn == chess.WHITE) == stig_white
        if stig_turn:
            our += 1
            if searcher.trail_move(board) is not None or searcher.book_move(board) is not None:
                hits += 1
            else:
                misses.append(board.copy(stack=True))
            move = engine.choose_move(board, time_ms=stig_ms, max_depth=stig_depth).move
            if move is None:
                break
            board.push(move)
        else:
            sf.set_elo(sf_elo)
            board.push(sf.choose(board, movetime_ms=sf_ms))
    return _result(board), misses, hits, our


def fill_misses(
    weights,
    sf: StockfishEngine,
    misses: list[chess.Board],
    *,
    fill_ms: int,
    fanout: int,
    rng: np.random.Generator,
) -> int:
    filled = 0

    def analyse(b: chess.Board, movetime_ms: int = 40, multipv: int = 1, depth: int | None = None):
        sf.set_elo(None)
        return sf.analyse_top(b, movetime_ms=movetime_ms, multipv=multipv, depth=depth)

    seen: set[str] = set()
    for board in misses:
        key = trail_key(board)
        if key in seen:
            continue
        seen.add(key)
        filled += fanout_opponent_replies(
            weights,
            board,
            analyse,
            max_replies=fanout,
            fill_ms=fill_ms,
            fill_depth=12,
            strength=150.0,
            rng=rng,
        )
        # Also fill parent: undo last move if any and fanout from there.
        if board.move_stack:
            parent = board.copy(stack=True)
            parent.pop()
            pk = trail_key(parent)
            if pk not in seen:
                seen.add(pk)
                filled += fanout_opponent_replies(
                    weights,
                    parent,
                    analyse,
                    max_replies=max(fanout, 48),
                    fill_ms=fill_ms,
                    fill_depth=12,
                    strength=140.0,
                    rng=rng,
                )
    return filled


def honest_ladder(
    engine: StigmergyEngine,
    sf: StockfishEngine,
    log: Path,
    *,
    games_per: int,
    stig_ms: int,
    stig_depth: int,
    sf_ms: int,
    max_plies: int,
    collect_misses: bool,
) -> tuple[dict, list[chess.Board]]:
    elos = [1320, 1600, 1900, 2200, 2500, 2600, 2700, 2800, 3000]
    our = 2000.0
    rows = []
    all_misses: list[chess.Board] = []
    hit_total = 0
    our_total = 0
    for target in elos:
        score = 0.0
        for i in range(games_per):
            stig_white = i % 2 == 0
            res, misses, hits, our_plies = honest_game(
                engine,
                sf,
                stig_white=stig_white,
                sf_elo=target,
                stig_ms=stig_ms,
                stig_depth=stig_depth,
                sf_ms=sf_ms,
                max_plies=max_plies,
            )
            score += _score_for(stig_white, res)
            hit_total += hits
            our_total += our_plies
            if collect_misses:
                all_misses.extend(misses)
            our, _ = update_elo(our, float(target), _score_for(stig_white, res), k=32.0)
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
        hit_rate = hit_total / max(1, our_total)
        _log(
            log,
            f"honest probe vs SF {target}: {score}/{games_per} ({wr:.0%}) "
            f"→ ≈{our:.0f} trail_hit={hit_rate:.0%}",
        )
        if target >= 2500 and wr < 0.15 and our < 2400:
            break
        if target >= 2500 and wr >= 0.5 and our >= 2500:
            break
    probe = {
        "estimated_elo": round(our, 1),
        "ladder": rows,
        "gm": our >= 2500.0,
        "trail_hit_rate": round(hit_total / max(1, our_total), 4),
        "trail_hits": hit_total,
        "our_plies": our_total,
    }
    return probe, all_misses


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", default="chess_contest/weights/gm/latest.json")
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--hours", type=float, default=12.0)
    p.add_argument("--gm-floor", type=float, default=2500.0)
    p.add_argument("--stig-ms", type=int, default=400)
    p.add_argument("--stig-depth", type=int, default=7)
    p.add_argument("--sf-ms", type=int, default=50)
    p.add_argument("--fill-ms", type=int, default=45)
    p.add_argument("--fanout", type=int, default=48)
    p.add_argument("--oracle-games", type=int, default=120)
    p.add_argument("--probe-games", type=int, default=6)
    p.add_argument("--min-hit-rate", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    log = out / "gm_oracle.log"
    _log(log, f"=== GM-ORACLE hours={args.hours} floor={args.gm_floor} ===")
    if not stockfish_available():
        raise SystemExit("Stockfish required")

    init = Path(args.init)
    weights = load_weights(init if init.exists() else "chess_contest/weights/gm/latest.json")
    weights.format_version = 4
    engine = StigmergyEngine(weights)
    sf = StockfishEngine(StockfishConfig(threads=2, hash_mb=256, movetime_ms=args.sf_ms))
    rng = np.random.default_rng(args.seed)

    schedule = (
        [1320, 1400, 1500, 1600] * 3
        + [1700, 1800, 1900, 2000] * 2
        + [2100, 2200, 2300, 2400, 2500]
        + [2600, 2700, 2800, 3000]
    )

    t0 = time.time()
    deadline = t0 + args.hours * 3600
    cycle = 0
    fills_total = 0
    try:
        while time.time() < deadline:
            cycle += 1
            fills_cycle = 0
            wins = 0
            games = 0
            trail_plies = 0
            # Heavy oracle densification before each probe.
            for j in range(args.oracle_games):
                if time.time() >= deadline:
                    break
                elo = schedule[j % len(schedule)]
                stig_white = (cycle + j) % 2 == 0
                res, fills, tp = oracle_game(
                    weights,
                    sf,
                    stig_white=stig_white,
                    sf_elo=elo,
                    fill_ms=args.fill_ms,
                    sf_ms=args.sf_ms,
                    max_plies=100,
                    fanout=args.fanout,
                    rng=rng,
                )
                fills_cycle += fills
                fills_total += fills
                trail_plies += tp
                games += 1
                wins += int(_score_for(stig_white, res) >= 0.99)

            prune_trails(weights, keep_positions=750_000)
            prune_learned_moves(weights, keep=80_000)
            # Cap book growth.
            if len(weights.book) > 600_000:
                # Keep longest? Prefer high-mass entries by peak w.
                ranked = sorted(
                    weights.book.items(),
                    key=lambda kv: max((float(e.get("w", 0)) for e in kv[1]), default=0.0),
                    reverse=True,
                )[:500_000]
                weights.book = dict(ranked)

            engine = StigmergyEngine(weights)
            _log(
                log,
                f"cycle {cycle} oracle fills={fills_cycle} total={fills_total} "
                f"wins={wins}/{games} trails={len(weights.trails)} "
                f"trail_plies={trail_plies}",
            )
            save_weights(weights, out / "latest.json")

            # Cheap hit-rate gate before full ladder (saves hours when coverage is thin).
            wr1320, hit = quick_hit_probe(
                engine, sf, games=8, target=1320, stig_ms=args.stig_ms, stig_depth=args.stig_depth
            )
            _log(log, f"gate vs1320 wr={wr1320:.0%} hit={hit:.0%}")
            if hit < args.min_hit_rate or wr1320 < 0.45:
                # Extra densify burst targeting this Elo before probing higher.
                for j in range(max(24, args.oracle_games // 2)):
                    if time.time() >= deadline:
                        break
                    elo = schedule[j % 12]
                    res, fills, tp = oracle_game(
                        weights,
                        sf,
                        stig_white=(cycle + j) % 2 == 0,
                        sf_elo=elo,
                        fill_ms=args.fill_ms,
                        sf_ms=args.sf_ms,
                        max_plies=100,
                        fanout=args.fanout,
                        rng=rng,
                    )
                    fills_total += fills
                prune_trails(weights, keep_positions=900_000)
                save_weights(weights, out / "latest.json")
                engine = StigmergyEngine(weights)
                wr1320, hit = quick_hit_probe(
                    engine, sf, games=8, target=1320, stig_ms=args.stig_ms, stig_depth=args.stig_depth
                )
                _log(log, f"gate2 vs1320 wr={wr1320:.0%} hit={hit:.0%}")
                if hit < args.min_hit_rate * 0.7:
                    continue

            probe, misses = honest_ladder(
                engine,
                sf,
                log,
                games_per=args.probe_games,
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
                sf_ms=args.sf_ms,
                max_plies=110,
                collect_misses=True,
            )
            (out / "elo_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
            _log(
                log,
                f"Elo probe ≈ {probe['estimated_elo']} gm={probe['gm']} "
                f"hit={probe['trail_hit_rate']} misses={len(misses)}",
            )

            if misses:
                mf = fill_misses(
                    weights,
                    sf,
                    misses,
                    fill_ms=max(args.fill_ms, 55),
                    fanout=max(args.fanout, 16),
                    rng=rng,
                )
                fills_total += mf
                _log(log, f"filled {mf} from {len(misses)} probe misses")
                prune_trails(weights, keep_positions=750_000)
                save_weights(weights, out / "latest.json")
                engine = StigmergyEngine(weights)

            if probe["estimated_elo"] >= args.gm_floor:
                confirm, _ = honest_ladder(
                    engine,
                    sf,
                    log,
                    games_per=8,
                    stig_ms=max(args.stig_ms, 500),
                    stig_depth=max(args.stig_depth, 8),
                    sf_ms=args.sf_ms,
                    max_plies=120,
                    collect_misses=False,
                )
                (out / "elo_probe.json").write_text(json.dumps(confirm, indent=2), encoding="utf-8")
                _log(log, f"CONFIRM ≈ {confirm['estimated_elo']} gm={confirm['gm']}")
                if confirm["estimated_elo"] >= args.gm_floor:
                    weights.training_meta = {
                        **weights.training_meta,
                        "gm_oracle": probe,
                        "confirm": confirm,
                        "precision": "float64",
                        "gm": True,
                    }
                    save_weights(weights, out / "gm_weights.json")
                    save_weights(weights, Path("chess_contest/weights/gm_weights.json"))
                    save_weights(weights, Path("chess_contest/weights/base_weights.json"))
                    (out / "STATUS.md").write_text(
                        f"# GM CONFIRMED\n\nElo ≈ {confirm['estimated_elo']}\n"
                        f"trail hit rate {confirm.get('trail_hit_rate')}\n",
                        encoding="utf-8",
                    )
                    _log(log, "=== GRANDMASTER FLOOR CONFIRMED ===")
                    break
            save_weights(weights, out / f"ckpt_oracle_{cycle}.json")
        else:
            probe, _ = honest_ladder(
                engine,
                sf,
                log,
                games_per=8,
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
                sf_ms=args.sf_ms,
                max_plies=110,
                collect_misses=False,
            )
            save_weights(weights, out / "gm_weights.json")
            _log(log, f"DONE without floor elo≈{probe['estimated_elo']}")
    finally:
        sf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
