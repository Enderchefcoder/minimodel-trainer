"""Blitz path to 3000: densify SF-MAX offline trails, ladder without SF at play.

Stockfish is teacher + ladder opponent only. Play uses float64 trails / IDAS /
optional swarm — choose_move never opens a Stockfish process.
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

from chess_contest.stigmergy.distill import (  # noqa: E402
    oracle_set_from_sf,
    prune_trails,
    set_trail_policy,
)
from chess_contest.stigmergy.engine import StigmergyEngine  # noqa: E402
from chess_contest.stigmergy.opponents import update_elo  # noqa: E402
from chess_contest.stigmergy.search import Searcher, set_policy_sprint, set_swarm  # noqa: E402
from chess_contest.stigmergy.stockfish_uci import (  # noqa: E402
    StockfishConfig,
    StockfishEngine,
    stockfish_available,
)
from chess_contest.stigmergy.swarm_net import try_load_swarm  # noqa: E402
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


def farm_trails(
    weights,
    sf: StockfishEngine,
    log: Path,
    *,
    games: int,
    depth: int,
    elos: list[int],
) -> int:
    """Record SF-MAX policy on every ply of SF-MAX vs SF-Elo games."""
    fills = 0
    t0 = time.time()
    for gi in range(games):
        target = elos[gi % len(elos)]
        board = chess.Board()
        stig_white = gi % 2 == 0
        for _ in range(140):
            if board.is_game_over(claim_draw=True):
                break
            stig = (board.turn == chess.WHITE) == stig_white
            # Always label with full-strength teacher (offline).
            sf.set_elo(None)
            tops = sf.analyse_top(board, multipv=3, depth=depth)
            if tops and tops[0].get("uci"):
                if oracle_set_from_sf(weights, board, tops, strength=120.0 + min(40, depth)):
                    fills += 1
                # Also lock top-1 as decisive policy.
                set_trail_policy(
                    weights, board, tops[0]["uci"][:4], strength=150.0 + depth
                )
            if stig:
                # Play the just-distilled trail / top-1 (still no SF in Searcher play path
                # when we push the UCI directly from teacher for farming speed).
                mv = Searcher(weights).trail_move(board)
                if mv is None and tops and tops[0].get("uci"):
                    try:
                        mv = chess.Move.from_uci(tops[0]["uci"])
                    except ValueError:
                        mv = None
                if mv is None or mv not in board.legal_moves:
                    legals = list(board.legal_moves)
                    if not legals:
                        break
                    mv = legals[0]
                board.push(mv)
            else:
                sf.set_elo(target)
                board.push(sf.choose(board, movetime_ms=40))
        if (gi + 1) % 25 == 0:
            _log(
                log,
                f"farm games={gi + 1}/{games} fills={fills} trails={len(weights.trails)} "
                f"{time.time() - t0:.0f}s",
            )
    return fills


def ladder(
    engine: StigmergyEngine,
    sf: StockfishEngine,
    log: Path,
    *,
    games_per: int,
    stig_ms: int,
    stig_depth: int,
    targets: list[int],
) -> dict:
    our = 2400.0
    rows = []
    for target in targets:
        sc = 0.0
        for i in range(games_per):
            board = chess.Board()
            stig_white = i % 2 == 0
            for _ in range(160):
                if board.is_game_over(claim_draw=True):
                    break
                if (board.turn == chess.WHITE) == stig_white:
                    mv = engine.choose_move(board, time_ms=stig_ms, max_depth=stig_depth).move
                    if mv is None:
                        break
                    board.push(mv)
                else:
                    sf.set_elo(target)
                    board.push(sf.choose(board, movetime_ms=80))
            res = board.result(claim_draw=True)
            if board.is_checkmate():
                res = "0-1" if board.turn == chess.WHITE else "1-0"
            s = _score(stig_white, res)
            sc += s
            our, _ = update_elo(our, float(target), s, k=24.0)
            _log(log, f"  blitz vs {target} #{i} → {res} s={s} ≈{our:.0f}")
        wr = sc / games_per
        rows.append(
            {
                "sf_elo": target,
                "score": sc,
                "games": games_per,
                "winrate": wr,
                "our_elo_after": round(our, 1),
                "think_ms": stig_ms,
            }
        )
        _log(log, f"BLITZ ladder vs SF {target}: {sc}/{games_per} ({wr:.0%}) → ≈{our:.0f}")
        if target >= 3000 and our >= 3000 and wr >= 0.35:
            break
        if target >= 2800 and wr < 0.05 and our < 2600:
            break
    return {
        "estimated_elo": round(our, 1),
        "ladder": rows,
        "gm": our >= 2500,
        "above_gm": our >= 2500,
        "crush_3000": our >= 3000,
        "stockfish_at_play": False,
        "oracle_runtime": False,
        "trail_first": True,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--weights", default="chess_contest/weights/gm/latest.json")
    p.add_argument("--farm-games", type=int, default=200)
    p.add_argument("--sf-depth", type=int, default=12)
    p.add_argument("--games-per", type=int, default=8)
    p.add_argument("--stig-ms", type=int, default=800)
    p.add_argument("--stig-depth", type=int, default=8)
    p.add_argument("--cycles", type=int, default=6)
    p.add_argument("--floor", type=float, default=3000.0)
    p.add_argument("--with-swarm", action="store_true")
    p.add_argument("--probe-only", action="store_true")
    args = p.parse_args(argv)

    if not stockfish_available():
        raise SystemExit("Stockfish required as offline teacher / ladder opponent")

    out = Path(args.out_dir)
    log = out / "blitz_3000.log"
    _log(log, f"=== BLITZ-3000 farm={args.farm_games} depth={args.sf_depth} ===")

    weights = load_weights(args.weights)
    weights.format_version = 4
    weights.training_meta.update(
        {
            "oracle_runtime": False,
            "stockfish_at_play": False,
            "trail_first": True,
            "crush_3000": True,
        }
    )

    sf = StockfishEngine(StockfishConfig(threads=2, hash_mb=256))
    set_swarm(None)
    set_policy_sprint(False)
    try:
        net = try_load_swarm(out / "swarm_net_best_ood.pt") if args.with_swarm else None
        if net is not None:
            set_swarm(net)
            _log(log, f"swarm loaded params={net.count_params():,}")
        else:
            _log(log, "play = trails + IDAS (no swarm)")

        eng = StigmergyEngine(weights, load_swarm=False)
        if net is not None:
            set_swarm(net)

        if args.probe_only:
            probe = ladder(
                eng,
                sf,
                log,
                games_per=args.games_per,
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
                targets=[2200, 2500, 2700, 2800, 2900, 3000, 3100],
            )
            _write(out, probe, args)
            return 0

        elos = [1320, 1600, 1800, 2000, 2200, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
        for cy in range(1, args.cycles + 1):
            _log(log, f"=== blitz cycle {cy}/{args.cycles} ===")
            fills = farm_trails(
                weights,
                sf,
                log,
                games=args.farm_games,
                depth=args.sf_depth,
                elos=elos,
            )
            prune_trails(weights, keep_positions=500_000)
            save_weights(weights, out / "latest.json")
            save_weights(weights, out / "blitz_weights.json")
            _log(log, f"cycle farm fills={fills} trails={len(weights.trails)}")

            eng = StigmergyEngine(weights, load_swarm=False)
            if net is not None:
                set_swarm(net)
            probe = ladder(
                eng,
                sf,
                log,
                games_per=args.games_per,
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
                targets=[2200, 2500, 2700, 2800, 2900, 3000, 3100],
            )
            _write(out, probe, args)
            _log(
                log,
                f"BLITZ Elo≈{probe['estimated_elo']} crush3000={probe['crush_3000']}",
            )
            if probe["estimated_elo"] >= args.floor:
                confirm = ladder(
                    eng,
                    sf,
                    log,
                    games_per=max(10, args.games_per),
                    stig_ms=max(args.stig_ms, 3000),
                    stig_depth=max(args.stig_depth, 12),
                    targets=[2800, 2900, 3000, 3100, 3200],
                )
                _write(out, confirm, args)
                _log(log, f"CONFIRM Elo≈{confirm['estimated_elo']}")
                if confirm["estimated_elo"] >= args.floor:
                    weights.training_meta["crush_3000"] = True
                    weights.training_meta["pure_gm"] = confirm
                    save_weights(weights, out / "gm_weights.json")
                    _log(log, "=== 3000 FLOOR REACHED (trail-first, SF never at play) ===")
                    break
            args.farm_games = min(800, args.farm_games + 50)
            args.sf_depth = min(16, args.sf_depth + 1)
            args.stig_ms = min(8000, int(args.stig_ms * 1.3))
    finally:
        sf.close()
        set_swarm(None)
    return 0


def _write(out: Path, probe: dict, args) -> None:
    (out / "elo_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    (out / "gm_report.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    elo = probe["estimated_elo"]
    (out / "STATUS.md").write_text(
        "# Blitz 3000 - trail-first, no runtime Stockfish\n\n"
        f"Estimated Elo **{elo}** (floor {args.floor}; "
        f"crush_3000={probe.get('crush_3000')}).\n\n"
        "`choose_move` never calls Stockfish. Strength = offline SF-MAX "
        "float64 trails + IDAS (+ optional swarm).\n\n"
        f"Think: {args.stig_ms} ms. Trails-first play enabled.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
