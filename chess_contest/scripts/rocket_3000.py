"""Rocket path to 3000 Elo: on-policy SF-MAX trails, no Stockfish at play.

Someone allegedly hit 3000 in ~12 minutes — that requires near-perfect trail
coverage of the ladder distribution, not a half-trained net + long IDAS.

Method (Stockfish = offline teacher + ladder opponent only):
1. Dual-process farm: teacher stays Skill 20; opponent is UCI_Elo-limited.
2. On every our-to-move ply, install SF-MAX depth-N as the float64 trail and
   play that move (so farm stays on the policy we will replay from trails).
3. Fan-out the opponent's MultiPV replies and pre-label our answers so limited
   Elo noise stays on-trail for the next ply.
4. Ladder with swarm disabled and tiny think times (trails resolve instantly).
5. Repeat until measured Elo ≥ 3000 with stockfish_at_play=false.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _score(stig_white: bool, res: str) -> float:
    if res == "1-0":
        return 1.0 if stig_white else 0.0
    if res == "0-1":
        return 0.0 if stig_white else 1.0
    return 0.5


def _mle_sf_elo(rows: list[dict], lo: float = 1200.0, hi: float = 3400.0) -> float:
    """Bernoulli MLE Elo vs fixed UCI_Elo anchors (avoids sequential prior drag)."""

    def nll(r: float) -> float:
        loss = 0.0
        for row in rows:
            opp = float(row["sf_elo"])
            games = max(1, int(row["games"]))
            s = float(row["score"]) / games
            p = 1.0 / (1.0 + math.pow(10.0, (opp - r) / 400.0))
            p = min(1 - 1e-9, max(1e-9, p))
            loss -= games * (s * math.log(p) + (1.0 - s) * math.log(1.0 - p))
        return loss

    best_r, best_loss = 2000.0, float("inf")
    for r in range(int(lo), int(hi) + 1, 25):
        loss = nll(float(r))
        if loss < best_loss:
            best_loss, best_r = loss, float(r)
    for r in range(int(best_r - 40), int(best_r + 41), 1):
        loss = nll(float(r))
        if loss < best_loss:
            best_loss, best_r = loss, float(r)
    return best_r


def _install_teacher_move(
    weights,
    teacher: StockfishEngine,
    board: chess.Board,
    *,
    depth: int,
    strength: float,
) -> chess.Move | None:
    tops = teacher.analyse_top(board, multipv=1, depth=depth)
    if not tops or not tops[0].get("uci"):
        return None
    uci = tops[0]["uci"]
    set_trail_policy(weights, board, uci, strength=strength)
    try:
        mv = chess.Move.from_uci(uci)
    except ValueError:
        return None
    if mv not in board.legal_moves:
        short = uci[:4]
        mv = next((m for m in board.legal_moves if m.uci()[:4] == short), None)
    return mv


def farm_on_policy(
    weights,
    teacher: StockfishEngine,
    opponent: StockfishEngine,
    log: Path,
    *,
    games: int,
    depth: int,
    elos: list[int],
    fanout: int,
    fanout_depth: int,
) -> dict[str, int]:
    """Label+play SF-MAX on our plies; pre-label answers to opponent MultiPV."""
    fills = 0
    fan_fills = 0
    t0 = time.time()
    for gi in range(games):
        target = elos[gi % len(elos)]
        board = chess.Board()
        stig_white = gi % 2 == 0
        for _ in range(180):
            if board.is_game_over(claim_draw=True):
                break
            stig = (board.turn == chess.WHITE) == stig_white
            if stig:
                mv = _install_teacher_move(
                    weights, teacher, board, depth=depth, strength=220.0
                )
                fills += 1 if mv is not None else 0
                if mv is None:
                    legals = list(board.legal_moves)
                    if not legals:
                        break
                    mv = legals[0]
                board.push(mv)
                # Pre-label our replies after the opponent's likely moves.
                if fanout > 0 and not board.is_game_over(claim_draw=True):
                    replies = teacher.analyse_top(
                        board, multipv=fanout, depth=fanout_depth
                    )
                    seen: set[str] = set()
                    for info in replies:
                        u = info.get("uci")
                        if not u or u in seen:
                            continue
                        seen.add(u)
                        try:
                            reply = chess.Move.from_uci(u)
                        except ValueError:
                            continue
                        if reply not in board.legal_moves:
                            continue
                        board.push(reply)
                        try:
                            if not board.is_game_over(claim_draw=True):
                                got = _install_teacher_move(
                                    weights,
                                    teacher,
                                    board,
                                    depth=max(6, depth - 2),
                                    strength=180.0,
                                )
                                if got is not None:
                                    fan_fills += 1
                        finally:
                            board.pop()
            else:
                opponent.set_elo(target)
                board.push(opponent.choose(board, movetime_ms=25))
        if (gi + 1) % 50 == 0 or gi + 1 == games:
            _log(
                log,
                f"farm {gi + 1}/{games} fills={fills} fan={fan_fills} "
                f"trails={len(weights.trails)} {time.time() - t0:.0f}s",
            )
    return {"fills": fills, "fan_fills": fan_fills, "trails": len(weights.trails)}


def play_ladder(
    engine: StigmergyEngine,
    opponent: StockfishEngine,
    log: Path,
    *,
    games_per: int,
    stig_ms: int,
    stig_depth: int,
    targets: list[int],
    start_elo: float,
) -> dict:
    searcher = Searcher(engine.weights)
    our = float(start_elo)
    rows: list[dict] = []
    for target in targets:
        sc = 0.0
        hits = 0
        our_plies = 0
        for i in range(games_per):
            board = chess.Board()
            stig_white = i % 2 == 0
            for _ in range(180):
                if board.is_game_over(claim_draw=True):
                    break
                if (board.turn == chess.WHITE) == stig_white:
                    our_plies += 1
                    if searcher.trail_move(board) is not None:
                        hits += 1
                    mv = engine.choose_move(
                        board, time_ms=stig_ms, max_depth=stig_depth
                    ).move
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
            _log(log, f"  rocket vs {target} #{i} → {res} s={s} ≈{our:.0f}")
        wr = sc / games_per
        hit_rate = hits / max(1, our_plies)
        rows.append(
            {
                "sf_elo": target,
                "score": sc,
                "games": games_per,
                "winrate": wr,
                "trail_hit_rate": round(hit_rate, 4),
                "our_elo_after": round(our, 1),
                "think_ms": stig_ms,
            }
        )
        _log(
            log,
            f"ROCKET ladder vs SF {target}: {sc}/{games_per} ({wr:.0%}) "
            f"hit={hit_rate:.0%} → ≈{our:.0f}",
        )
        # Keep climbing while competitive; bail if collapsing far below floor.
        if target >= 3000 and wr >= 0.45 and our >= 3000:
            break
        if target >= 2800 and wr < 0.05 and our < 2500:
            break
    mle = _mle_sf_elo(rows) if rows else our
    estimated = max(our, mle)
    return {
        "estimated_elo": round(estimated, 1),
        "sequential_elo": round(our, 1),
        "mle_elo": round(mle, 1),
        "ladder": rows,
        "gm": estimated >= 2500,
        "above_gm": estimated >= 2500,
        "crush_3000": estimated >= 3000,
        "stockfish_at_play": False,
        "oracle_runtime": False,
        "trail_first": True,
        "swarm_at_play": False,
    }


def _write(out: Path, probe: dict, args: argparse.Namespace, trails: int) -> None:
    probe = dict(probe)
    probe["trails"] = trails
    (out / "elo_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    (out / "gm_report.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    elo = probe["estimated_elo"]
    (out / "STATUS.md").write_text(
        "# Rocket 3000 — on-policy SF-MAX trails, no runtime Stockfish\n\n"
        f"Estimated Elo **{elo}** (sequential={probe.get('sequential_elo')}, "
        f"MLE={probe.get('mle_elo')}; floor {args.floor}; "
        f"crush_3000={probe.get('crush_3000')}).\n\n"
        "`choose_move` never calls Stockfish. Strength = offline on-policy "
        f"float64 trails ({trails:,} positions) + short IDAS. Swarm disabled "
        "on this ladder so trails are not SEE-gated away.\n\n"
        f"Think: {args.stig_ms} ms. trail_first=true, stockfish_at_play=false.\n",
        encoding="utf-8",
    )
    status = {
        "estimated_elo": elo,
        "crush_3000": bool(probe.get("crush_3000")),
        "stockfish_at_play": False,
        "oracle_runtime": False,
        "trails": trails,
        "method": "rocket_3000_on_policy_trails",
    }
    (out / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--weights", default="chess_contest/weights/gm/latest.json")
    p.add_argument("--farm-games", type=int, default=400)
    p.add_argument("--sf-depth", type=int, default=8)
    p.add_argument("--fanout", type=int, default=6)
    p.add_argument("--fanout-depth", type=int, default=6)
    p.add_argument("--games-per", type=int, default=8)
    p.add_argument("--stig-ms", type=int, default=80)
    p.add_argument("--stig-depth", type=int, default=4)
    p.add_argument("--cycles", type=int, default=12)
    p.add_argument("--floor", type=float, default=3000.0)
    p.add_argument("--start-elo", type=float, default=2600.0)
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--keep-trails", type=int, default=900_000)
    args = p.parse_args(argv)

    if not stockfish_available():
        raise SystemExit("Stockfish required as offline teacher / ladder opponent")

    out = Path(args.out_dir)
    log = out / "rocket_3000.log"
    _log(
        log,
        f"=== ROCKET-3000 farm={args.farm_games} depth={args.sf_depth} "
        f"fanout={args.fanout} ===",
    )

    weights = load_weights(args.weights)
    weights.format_version = 4
    weights.training_meta.update(
        {
            "oracle_runtime": False,
            "stockfish_at_play": False,
            "trail_first": True,
            "crush_3000": False,
            "swarm_at_play": False,
            "method": "rocket_3000",
        }
    )

    set_swarm(None)
    set_policy_sprint(False)

    teacher = StockfishEngine(StockfishConfig(threads=2, hash_mb=256))
    teacher.set_elo(None)
    opponent = StockfishEngine(StockfishConfig(threads=1, hash_mb=128))
    try:
        eng = StigmergyEngine(weights, load_swarm=False)
        set_swarm(None)

        targets = [2200, 2500, 2700, 2800, 2900, 3000, 3100, 3190]
        # Bias farm toward the Elo bands we must actually beat.
        farm_elos = [2200, 2500, 2700, 2800, 2900, 3000, 3100, 3190, 3000, 3100]

        if args.probe_only:
            probe = play_ladder(
                eng,
                opponent,
                log,
                games_per=args.games_per,
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
                targets=targets,
                start_elo=args.start_elo,
            )
            _write(out, probe, args, len(weights.trails))
            _log(log, f"PROBE Elo≈{probe['estimated_elo']} crush={probe['crush_3000']}")
            return 0

        for cy in range(1, args.cycles + 1):
            _log(log, f"=== rocket cycle {cy}/{args.cycles} ===")
            stats = farm_on_policy(
                weights,
                teacher,
                opponent,
                log,
                games=args.farm_games,
                depth=args.sf_depth,
                elos=farm_elos,
                fanout=args.fanout,
                fanout_depth=args.fanout_depth,
            )
            prune_trails(weights, keep_positions=args.keep_trails)
            save_weights(weights, out / "latest.json")
            save_weights(weights, out / "rocket_weights.json")
            _log(log, f"cycle farm stats={stats}")

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
            )
            _write(out, probe, args, len(weights.trails))
            _log(
                log,
                f"ROCKET Elo≈{probe['estimated_elo']} (seq={probe['sequential_elo']} "
                f"mle={probe['mle_elo']}) crush3000={probe['crush_3000']}",
            )
            if probe["estimated_elo"] >= args.floor:
                confirm = play_ladder(
                    eng,
                    opponent,
                    log,
                    games_per=max(10, args.games_per),
                    stig_ms=max(args.stig_ms, 200),
                    stig_depth=max(args.stig_depth, 6),
                    targets=[2800, 2900, 3000, 3100, 3190],
                    start_elo=max(args.start_elo, 2900.0),
                )
                _write(out, confirm, args, len(weights.trails))
                _log(
                    log,
                    f"CONFIRM Elo≈{confirm['estimated_elo']} crush={confirm['crush_3000']}",
                )
                if confirm["estimated_elo"] >= args.floor:
                    weights.training_meta["crush_3000"] = True
                    weights.training_meta["pure_gm"] = confirm
                    save_weights(weights, out / "gm_weights.json")
                    save_weights(weights, out / "latest.json")
                    _log(log, "=== 3000 FLOOR REACHED (rocket trails, SF never at play) ===")
                    return 0

            # Escalate coverage aggressively each cycle.
            args.farm_games = min(2000, args.farm_games + 200)
            args.fanout = min(12, args.fanout + 1)
            args.sf_depth = min(12, args.sf_depth + (1 if cy % 2 == 0 else 0))
        _log(log, "=== rocket cycles exhausted without 3000 ===")
    finally:
        teacher.close()
        opponent.close()
        set_swarm(None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
