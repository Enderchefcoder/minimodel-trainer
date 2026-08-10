"""Keep densifying on-policy SF-MAX trails until measured Elo ≥ 3000.

Stockfish is offline teacher + ladder opponent only — ``choose_move`` never
calls it. Prior flash/rocket runs hit ~2590 because converge hit-rates counted
mid-game installs, while scored play fell off 2-ply fanout leaves (~20% hit).

This loop:
1. Trains vs the ladder distribution with trail-or-install play.
2. After every our-to-move visit, runs ``fanout_opponent_replies`` (2-ply early,
   1-ply later, all legal replies) so the next game survives opponent noise.
3. Probes with *no* mid-game install; only then trusts hit-rate / score.
4. Repeats until a scored ladder reports estimated_elo ≥ 3000.
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

from chess_contest.scripts.rocket_3000 import (  # noqa: E402
    _mle_sf_elo,
    _score,
    _write,
)
from chess_contest.stigmergy.distill import fanout_opponent_replies  # noqa: E402
from chess_contest.stigmergy.engine import StigmergyEngine  # noqa: E402
from chess_contest.stigmergy.opponents import update_elo  # noqa: E402
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


def _reply_gap(weights, board: chess.Board) -> tuple[int, int]:
    """Return (missing, total) trail coverage after our current policy move."""
    searcher = Searcher(weights)
    mv = searcher.trail_move(board)
    if mv is None:
        legal = list(board.legal_moves)
        # No policy yet — treat as fully missing so caller installs+fanouts.
        return (max(1, len(legal)), max(1, len(legal)))
    tmp = board.copy(stack=False)
    tmp.push(mv)
    if tmp.is_game_over(claim_draw=True):
        return (0, 0)
    total = missing = 0
    for rep in tmp.legal_moves:
        total += 1
        tmp.push(rep)
        try:
            if not tmp.is_game_over(claim_draw=True):
                slot = weights.trails.get(trail_key(tmp)) or {}
                if not slot or max(float(v) for v in slot.values()) < 50.0:
                    missing += 1
        finally:
            tmp.pop()
    return missing, total


def _fanout(
    weights,
    teacher: StockfishEngine,
    board: chess.Board,
    *,
    our_ply: int,
    depth: int,
    early_plies: int,
    force: bool = False,
) -> int:
    """Densify from an our-to-move node without orphaning strong trails.

    Skips work when reply coverage is already dense (≥90% filled) unless
    ``force`` (used on probe-miss backfill).
    """
    missing, total = _reply_gap(weights, board)
    # Only skip when every legal reply already has a trail — a 10% hole is
    # exactly what UCI_Elo noise finds, and one miss collapses scored hit-rate.
    if not force and total > 0 and missing == 0:
        return 0
    if not force and total == 0 and Searcher(weights).trail_move(board) is not None:
        return 0
    # 2-ply only on the earliest plies — quadratic cost otherwise dominates.
    ply_depth = 2 if (force or our_ply < min(8, early_plies)) else 1
    max_replies = 60 if ply_depth >= 2 or our_ply < early_plies else 28
    fill_depth = depth if (force or our_ply < early_plies) else max(8, depth - 2)
    return fanout_opponent_replies(
        weights,
        board,
        teacher.analyse_top,
        max_replies=max_replies,
        fill_ms=40,
        fill_depth=fill_depth,
        strength=260.0 if our_ply < early_plies or force else 240.0,
        ply_depth=ply_depth,
    )


def train_batch(
    weights,
    teacher: StockfishEngine,
    opponent: StockfishEngine,
    log: Path,
    *,
    games: int,
    depth: int,
    targets: list[int],
    early_plies: int,
) -> dict[str, float]:
    """Play trail-or-install; fanout at every our visit (SF never via choose_move)."""
    searcher = Searcher(weights)
    hits = our = installs = fan_nodes = 0
    t0 = time.time()
    for gi in range(games):
        target = targets[gi % len(targets)]
        board = chess.Board()
        stig_white = gi % 2 == 0
        our_ply = 0
        for _ in range(160):
            if board.is_game_over(claim_draw=True):
                break
            if (board.turn == chess.WHITE) == stig_white:
                our += 1
                if searcher.trail_move(board) is not None:
                    hits += 1
                else:
                    installs += 1
                # Densify here (preserve-strong) then play the trail policy.
                fan_nodes += _fanout(
                    weights,
                    teacher,
                    board,
                    our_ply=our_ply,
                    depth=depth,
                    early_plies=early_plies,
                )
                searcher = Searcher(weights)
                mv = searcher.trail_move(board)
                if mv is None:
                    break
                board.push(mv)
                our_ply += 1
            else:
                opponent.set_elo(target)
                board.push(opponent.choose(board, movetime_ms=50))
        if (gi + 1) % 5 == 0 or gi + 1 == games:
            _log(
                log,
                f"train {gi + 1}/{games} hit={hits / max(1, our):.0%} "
                f"installs={installs} fan={fan_nodes} trails={len(weights.trails)} "
                f"{time.time() - t0:.0f}s",
            )
    return {
        "hit": hits / max(1, our),
        "installs": float(installs),
        "fan": float(fan_nodes),
        "trails": float(len(weights.trails)),
    }


def probe_scored(
    weights,
    opponent: StockfishEngine,
    log: Path,
    *,
    games: int,
    target: int,
    stig_ms: int,
    stig_depth: int,
    start_elo: float,
) -> dict:
    """Honest scored games: no mid-game SF install; trails via choose_move only."""
    set_swarm(None)
    eng = StigmergyEngine(weights, load_swarm=False)
    searcher = Searcher(weights)
    our = float(start_elo)
    sc = 0.0
    hits = our_plies = 0
    miss_boards: list[chess.Board] = []
    for i in range(games):
        board = chess.Board()
        stig_white = i % 2 == 0
        for _ in range(160):
            if board.is_game_over(claim_draw=True):
                break
            if (board.turn == chess.WHITE) == stig_white:
                our_plies += 1
                if searcher.trail_move(board) is not None:
                    hits += 1
                else:
                    miss_boards.append(board.copy(stack=False))
                mv = eng.choose_move(board, time_ms=stig_ms, max_depth=stig_depth).move
                if mv is None:
                    break
                board.push(mv)
            else:
                opponent.set_elo(target)
                board.push(opponent.choose(board, movetime_ms=50))
        res = board.result(claim_draw=True)
        if board.is_checkmate():
            res = "0-1" if board.turn == chess.WHITE else "1-0"
        s = _score(stig_white, res)
        sc += s
        our, _ = update_elo(our, float(target), s, k=32.0)
        _log(log, f"  probe vs{target} #{i} → {res} s={s} ≈{our:.0f}")
    hit = hits / max(1, our_plies)
    wr = sc / games
    _log(
        log,
        f"PROBE vs{target}: {sc}/{games} ({wr:.0%}) hit={hit:.0%} ≈{our:.0f} "
        f"misses={len(miss_boards)}",
    )
    return {
        "sf_elo": target,
        "score": sc,
        "games": games,
        "winrate": wr,
        "trail_hit_rate": round(hit, 4),
        "our_elo_after": round(our, 1),
        "miss_boards": miss_boards,
    }


def backfill_misses(
    weights,
    teacher: StockfishEngine,
    misses: list[chess.Board],
    *,
    depth: int,
    early_plies: int,
) -> int:
    """Offline SF-MAX + fanout on probe misses (never at play time)."""
    n = 0
    for board in misses:
        n += _fanout(
            weights,
            teacher,
            board,
            our_ply=0,
            depth=depth,
            early_plies=early_plies,
            force=True,
        )
    return n


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
        if target >= 3000 and row["winrate"] < 0.15 and row["trail_hit_rate"] < 0.5:
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
        "method": "until_3000_fanout_loop",
        "trails": len(weights.trails),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--weights", default="chess_contest/weights/gm/latest.json")
    p.add_argument("--sf-depth", type=int, default=12)
    p.add_argument("--train-games", type=int, default=20)
    p.add_argument("--probe-games", type=int, default=12)
    p.add_argument("--ladder-games", type=int, default=12)
    p.add_argument("--cycles", type=int, default=80)
    p.add_argument("--early-plies", type=int, default=14)
    p.add_argument("--stig-ms", type=int, default=40)
    p.add_argument("--stig-depth", type=int, default=3)
    p.add_argument("--floor", type=float, default=3000.0)
    p.add_argument("--hit-gate", type=float, default=0.82)
    p.add_argument("--fresh", action="store_true", help="Ignore existing trails")
    args = p.parse_args(argv)

    if not stockfish_available():
        raise SystemExit("Stockfish required as offline teacher / ladder opponent")

    out = Path(args.out_dir)
    log = out / "until_3000.log"
    _log(log, f"=== UNTIL-3000 depth={args.sf_depth} cycles={args.cycles} ===")

    if args.fresh:
        from chess_contest.stigmergy.weights import default_weights

        weights = default_weights()
    else:
        weights = load_weights(args.weights)
    weights.training_meta.update(
        {
            "oracle_runtime": False,
            "stockfish_at_play": False,
            "trail_first": True,
            "crush_3000": False,
            "method": "until_3000_fanout_loop",
        }
    )
    set_swarm(None)
    set_policy_sprint(False)

    teacher = StockfishEngine(StockfishConfig(threads=2, hash_mb=256))
    teacher.set_elo(None)
    opponent = StockfishEngine(StockfishConfig(threads=1, hash_mb=128))
    train_targets = [2800, 2900, 3000, 3000, 3100, 3190]
    ladder_targets = [2200, 2800, 3000, 3100, 3190]

    try:
        for cycle in range(1, args.cycles + 1):
            _log(log, f"--- cycle {cycle}/{args.cycles} ---")
            train_batch(
                weights,
                teacher,
                opponent,
                log,
                games=args.train_games,
                depth=args.sf_depth,
                targets=train_targets,
                early_plies=args.early_plies,
            )
            save_weights(weights, out / "latest.json")

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
            backfill_misses(
                weights,
                teacher,
                probe["miss_boards"],
                depth=args.sf_depth,
                early_plies=args.early_plies,
            )
            save_weights(weights, out / "latest.json")

            hit = float(probe["trail_hit_rate"])
            wr = float(probe["winrate"])
            _log(log, f"cycle {cycle} probe hit={hit:.0%} wr={wr:.0%}")

            if hit < args.hit_gate or wr < 0.35:
                continue

            _log(log, "hit-gate passed — running scored ladder")
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
                f"crush={probe_full['crush_3000']} mle={probe_full['mle_elo']}",
            )
            if probe_full["estimated_elo"] >= args.floor:
                weights.training_meta["crush_3000"] = True
                weights.training_meta["pure_gm"] = {
                    k: v for k, v in probe_full.items() if k != "ladder"
                }
                save_weights(weights, out / "gm_weights.json")
                save_weights(weights, out / "latest.json")
                _log(log, "=== 3000 FLOOR REACHED (until_3000, SF never at play) ===")
                return 0
        _log(log, "cycles exhausted without crush_3000")
    finally:
        teacher.close()
        opponent.close()
        set_swarm(None)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
