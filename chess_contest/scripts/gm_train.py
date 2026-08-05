"""Grandmaster push: high-precision float64 trails + SF-oracle distillation.

Fills continuous zobrist pheromone trails from Stockfish multipv, spars against
UCI_Elo-limited SF, and probes until estimated Elo >= GM_FLOOR (2500).

Architecture stays Stigmergy-DPFE (trails + field + tactical floor). Stockfish
is oracle-only during training — probes use weights alone.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chess_contest.stigmergy.distill import (  # noqa: E402
    clip_field_params,
    distill_game,
    distill_stockfish_top,
    imitation_toward_move,
    prune_learned_moves,
    prune_trails,
)
from chess_contest.stigmergy.engine import StigmergyEngine  # noqa: E402
from chess_contest.stigmergy.evaluate import clear_eval_cache, evaluate_board  # noqa: E402
from chess_contest.stigmergy.stockfish_uci import (  # noqa: E402
    StockfishConfig,
    StockfishEngine,
    stockfish_available,
)
from chess_contest.stigmergy.uniqueness import score_uniqueness  # noqa: E402
from chess_contest.stigmergy.weights import (  # noqa: E402
    StigmergyWeights,
    default_field_params,
    load_weights,
    save_weights,
    trail_key,
)

GM_FLOOR = 2500.0
GM_TARGET = 2800.0


@dataclass
class GMConfig:
    hours: float = 10.0
    seed: int = 42
    init_weights: str = "chess_contest/weights/overnight/latest.json"
    out_dir: str = "chess_contest/weights/gm"
    stockfish_path: str = "/usr/games/stockfish"
    stig_movetime_ms: int = 400
    stig_max_depth: int = 8
    sf_movetime_ms: int = 80
    analyse_ms: int = 120
    max_plies: int = 90
    trail_build_positions: int = 8000
    analyse_per_cycle: int = 200
    sf_batch: int = 16
    probe_every_min: float = 20.0
    checkpoint_every_min: float = 8.0
    gm_floor: float = GM_FLOOR
    reset_saturated_field: bool = True


def _log(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _result_of(board: chess.Board) -> str:
    if board.is_checkmate():
        return "0-1" if board.turn == chess.WHITE else "1-0"
    return "1/2-1/2"


def play_recorded(white_choose, black_choose, max_plies: int) -> tuple[str, list[chess.Move]]:
    board = chess.Board()
    moves: list[chess.Move] = []
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        chooser = white_choose if board.turn == chess.WHITE else black_choose
        move = chooser(board)
        if move not in board.legal_moves:
            return ("0-1" if board.turn == chess.WHITE else "1-0"), moves
        board.push(move)
        moves.append(move)
        plies += 1
    return _result_of(board), moves


def migrate_book_to_float_w(weights: StigmergyWeights) -> int:
    """Ensure every book entry has a continuous float weight ``w``."""
    n = 0
    for _path, entries in weights.book.items():
        for e in entries:
            if "w" not in e:
                code = int(e.get("code", 0))
                count = float(e.get("count", 1) or 1)
                e["w"] = float((code + 2) * count)  # -1,0,1 → positive mass
                n += 1
            else:
                e["w"] = float(e["w"])
    return n


def field_is_saturated(weights: StigmergyWeights) -> bool:
    d = weights.field.deposit
    return bool(np.mean(np.abs(d) >= 14.5) > 0.4)


def prepare_weights(cfg: GMConfig, log: Path) -> StigmergyWeights:
    if Path(cfg.init_weights).exists():
        weights = load_weights(cfg.init_weights)
        _log(log, f"Loaded {cfg.init_weights}")
    else:
        from chess_contest.stigmergy.weights import default_weights

        weights = default_weights()
        _log(log, "Using default weights")

    weights.format_version = 4
    if weights.field.field_head is None:
        weights.field.field_head = np.zeros(24, dtype=np.float64)

    migrated = migrate_book_to_float_w(weights)
    _log(log, f"Migrated book float w on {migrated} entries; book_size={len(weights.book)}")

    if cfg.reset_saturated_field and field_is_saturated(weights):
        _log(log, "Field deposit saturated at clip — resetting field to defaults (keep book/trails)")
        book = weights.book
        trails = weights.trails
        learned = weights.learned_moves
        meta = weights.training_meta
        weights.field = default_field_params()
        weights.book = book
        weights.trails = trails
        weights.learned_moves = learned
        weights.training_meta = meta
        clip_field_params(weights.field)

    clear_eval_cache()
    ev = evaluate_board(chess.Board(), weights)
    _log(log, f"Startpos eval={ev:.3f} trails={len(weights.trails)} learned={len(weights.learned_moves)}")
    return weights


def build_trails_from_sf(
    weights: StigmergyWeights,
    sf: StockfishEngine,
    n_positions: int,
    analyse_ms: int,
    log: Path,
    rng: np.random.Generator,
) -> int:
    """Walk openings + random middlegames; distill SF multipv into float trails."""
    _log(log, f"Trail build: {n_positions} positions @ {analyse_ms}ms multipv")
    reinforced = 0
    for i in range(n_positions):
        b = chess.Board()
        depth = int(rng.integers(0, 18))
        for _ in range(depth):
            if b.is_game_over():
                break
            # Prefer existing trails/book for realistic positions.
            key = trail_key(b)
            if weights.trails.get(key):
                uci = max(weights.trails[key], key=weights.trails[key].get)
                try:
                    mv = chess.Move.from_uci(uci if len(uci) >= 4 else uci + "q")
                    if mv in b.legal_moves:
                        b.push(mv)
                        continue
                except ValueError:
                    pass
            legal = list(b.legal_moves)
            if not legal:
                break
            b.push(rng.choice(legal))
        if b.is_game_over():
            continue
        try:
            tops = sf.analyse_top(b, movetime_ms=analyse_ms, multipv=4)
        except Exception as e:
            if i < 5:
                _log(log, f"analyse fail: {e}")
            continue
        reinforced += distill_stockfish_top(weights, b, tops, boost=2.2)
        # Occasional light field nudge only — full imitation every position is too slow.
        if tops and (i % 40 == 0):
            try:
                mv = chess.Move.from_uci(tops[0]["uci"])
                imitation_toward_move(weights, b, mv, rng, lr=0.02)
            except Exception:
                pass
        if (i + 1) % 200 == 0:
            _log(
                log,
                f"  trails {i+1}/{n_positions} reinforced={reinforced} "
                f"positions={len(weights.trails)}",
            )
    return reinforced


def sf_vs_sf_distill(
    weights: StigmergyWeights,
    sf: StockfishEngine,
    games: int,
    movetime_ms: int,
    max_plies: int,
    log: Path,
) -> int:
    """Play SF vs SF (oracle vs oracle) and distill BOTH sides into trails."""
    _log(log, f"SF-vs-SF distill games={games}")
    n = 0
    sf.set_elo(None)
    for g in range(games):
        board = chess.Board()
        moves: list[chess.Move] = []
        for _ply in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break
            # Capture multipv before the move.
            try:
                tops = sf.analyse_top(board, movetime_ms=movetime_ms, multipv=3)
                n += distill_stockfish_top(weights, board, tops, boost=2.5)
            except Exception:
                tops = []
            move = sf.choose(board, movetime_ms=movetime_ms)
            if move not in board.legal_moves:
                break
            board.push(move)
            moves.append(move)
        result = _result_of(board)
        distill_game(weights, moves, result, winner_boost=1.4, loser_penalty=0.2, book_plies=24)
        if (g + 1) % 5 == 0:
            _log(log, f"  sf-sf {g+1}/{games} trails={len(weights.trails)}")
    return n


def probe_sf_ladder(
    weights: StigmergyWeights,
    sf: StockfishEngine,
    cfg: GMConfig,
    log: Path,
    *,
    games_per: int = 6,
) -> dict:
    """Honest probe: Stigmergy alone vs SF UCI_Elo ladder (no online SF help)."""
    engine = StigmergyEngine(weights)
    elos = [1320, 1600, 1900, 2200, 2500, 2600, 2700, 2800, 3000, 3190]
    our = 2000.0
    rows = []
    from chess_contest.stigmergy.opponents import update_elo

    for target in elos:
        sf.set_elo(target)
        score = 0.0
        for i in range(games_per):
            stig_white = i % 2 == 0

            def stig(b, _e=engine):
                return _e.choose_move(
                    b, time_ms=cfg.stig_movetime_ms, max_depth=cfg.stig_max_depth
                ).move

            def sfc(b, _sf=sf):
                return _sf.choose(b, movetime_ms=cfg.sf_movetime_ms)

            if stig_white:
                result, _ = play_recorded(stig, sfc, cfg.max_plies)
                s = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
            else:
                result, _ = play_recorded(sfc, stig, cfg.max_plies)
                s = {"1-0": 0.0, "0-1": 1.0, "1/2-1/2": 0.5}[result]
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
        _log(log, f"probe vs SF Elo {target}: score {score}/{games_per} ({wr:.0%}) → our≈{our:.0f}")
        # Early exit if crushed far below floor.
        if wr < 0.1 and target >= 2500 and our < cfg.gm_floor - 200:
            break
        # Success short-circuit: clear GM vs 2500+.
        if target >= 2500 and wr >= 0.5 and our >= cfg.gm_floor:
            break
    return {"estimated_elo": round(our, 1), "ladder": rows, "gm": our >= cfg.gm_floor}


def run_gm(cfg: GMConfig) -> Path:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "gm_train.log"
    _log(log, f"=== GM PUSH hours={cfg.hours} floor={cfg.gm_floor} ===")

    if not stockfish_available(cfg.stockfish_path):
        raise RuntimeError(f"Stockfish not found at {cfg.stockfish_path}")

    weights = prepare_weights(cfg, log)
    sf = StockfishEngine(
        StockfishConfig(
            path=cfg.stockfish_path,
            threads=2,
            hash_mb=256,
            skill_level=20,
            movetime_ms=cfg.sf_movetime_ms,
        )
    )
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    deadline = t0 + cfg.hours * 3600
    last_ckpt = t0
    last_probe = 0.0
    stats: dict = {
        "trail_reinforce": 0,
        "sf_games": 0,
        "stig_wins": 0,
        "probes": [],
    }

    try:
        # Phase A: massive trail distillation.
        sf.set_elo(None)
        stats["trail_reinforce"] += build_trails_from_sf(
            weights, sf, cfg.trail_build_positions, cfg.analyse_ms, log, rng
        )
        stats["trail_reinforce"] += sf_vs_sf_distill(
            weights, sf, games=80, movetime_ms=max(50, cfg.analyse_ms // 2), max_plies=70, log=log
        )
        # Mid-phase checkpoint before probe.
        save_weights(weights, out / "ckpt_trails_raw.json")
        # Anchor root opening from SF-MAX so we never open with garbage book lines.
        sf.set_elo(None)
        root = chess.Board()
        tops = sf.analyse_top(root, movetime_ms=max(200, cfg.analyse_ms), multipv=5)
        distill_stockfish_top(weights, root, tops, boost=8.0)
        weights.book[""] = [
            {"m": (info.get("uci") or "")[:4], "w": 50.0 - 5.0 * i, "code": 1 if i == 0 else 0}
            for i, info in enumerate(tops)
            if info.get("uci")
        ]
        _log(log, f"Root book set from SF: {[e['m'] for e in weights.book['']]}")
        prune_learned_moves(weights, keep=80000)
        prune_trails(weights, keep_positions=250000)
        save_weights(weights, out / "ckpt_trails.json")
        save_weights(weights, out / "latest.json")
        _log(log, f"Phase A done trails={len(weights.trails)} book={len(weights.book)}")

        # Immediate probe after trail fill.
        probe = probe_sf_ladder(weights, sf, cfg, log, games_per=4)
        stats["probes"].append(probe)
        (out / "elo_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
        if probe.get("gm"):
            _log(log, f"GM FLOOR REACHED early ≈{probe['estimated_elo']}")
        last_probe = time.time()

        # Phase B: sparring + online trail fill from game positions.
        cycle = 0
        # Heavy bias to winnable bands once trails exist; still sample high.
        schedule = (
            [1200, 1320, 1400, 1500, 1600, 1700, 1800] * 5
            + [1900, 2000, 2100, 2200] * 3
            + [2300, 2400, 2500] * 2
            + [2600, 2700, 2800, 3000]
        )

        while time.time() < deadline:
            cycle += 1
            clear_eval_cache()
            engine = StigmergyEngine(weights)

            # High-throughput analysis into trails.
            sf.set_elo(None)
            hits = 0
            for _ in range(cfg.analyse_per_cycle):
                b = chess.Board()
                for _ply in range(int(rng.integers(0, 20))):
                    if b.is_game_over():
                        break
                    key = trail_key(b)
                    if weights.trails.get(key):
                        uci = max(weights.trails[key], key=weights.trails[key].get)
                        try:
                            mv = chess.Move.from_uci(uci)
                            if mv in b.legal_moves:
                                b.push(mv)
                                continue
                        except ValueError:
                            pass
                    b.push(rng.choice(list(b.legal_moves)))
                if b.is_game_over():
                    continue
                tops = sf.analyse_top(b, movetime_ms=cfg.analyse_ms, multipv=4)
                stats["trail_reinforce"] += distill_stockfish_top(weights, b, tops, boost=2.0)
                if tops:
                    hits += 1
                    if rng.random() < 0.05:
                        with contextlib.suppress(Exception):
                            imitation_toward_move(
                                weights, b, chess.Move.from_uci(tops[0]["uci"]), rng, lr=0.02
                            )
            _log(
                log,
                f"cycle {cycle} analyse hits~{hits}/{cfg.analyse_per_cycle} "
                f"trails={len(weights.trails)}",
            )

            # Sparring batch.
            for j in range(cfg.sf_batch):
                target = schedule[(cycle * cfg.sf_batch + j) % len(schedule)]
                use_max = (cycle + j) % 5 == 0
                if use_max:
                    sf.set_elo(None)
                    label = "MAX"
                else:
                    sf.set_elo(target)
                    label = str(target)

                stig_white = (cycle + j) % 2 == 0

                def stig_choose(b, _e=engine, _sf=sf, _w=weights):
                    # Guided trail fill during TRAINING only (probes stay pure).
                    from chess_contest.stigmergy.search import Searcher

                    searcher = Searcher(_w)
                    if searcher.trail_move(b) is None and searcher.book_move(b) is None:
                        try:
                            _sf.set_elo(None)
                            tops = _sf.analyse_top(
                                b, movetime_ms=max(50, cfg.analyse_ms // 2), multipv=3
                            )
                            distill_stockfish_top(_w, b, tops, boost=3.0)
                        except Exception:
                            pass
                    return _e.choose_move(
                        b, time_ms=cfg.stig_movetime_ms, max_depth=cfg.stig_max_depth
                    ).move

                def sf_choose(b, _sf=sf, _w=weights):
                    # While training, also distill the position we face.
                    try:
                        tops = _sf.analyse_top(b, movetime_ms=max(40, cfg.sf_movetime_ms), multipv=2)
                        distill_stockfish_top(_w, b, tops, boost=1.8)
                    except Exception:
                        pass
                    return _sf.choose(b, movetime_ms=cfg.sf_movetime_ms)

                if stig_white:
                    result, moves = play_recorded(stig_choose, sf_choose, cfg.max_plies)
                    won = result == "1-0"
                else:
                    result, moves = play_recorded(sf_choose, stig_choose, cfg.max_plies)
                    won = result == "0-1"

                distill_game(
                    weights,
                    moves,
                    result,
                    winner_boost=2.0 if not won else 1.2,
                    loser_penalty=0.45,
                    book_plies=20,
                )
                # Midgame MAX PV.
                if moves:
                    mid = chess.Board()
                    cut = max(1, min(20, len(moves) // 2))
                    for m in moves[:cut]:
                        mid.push(m)
                    if not mid.is_game_over():
                        sf.set_elo(None)
                        tops = sf.analyse_top(mid, movetime_ms=cfg.analyse_ms, multipv=3)
                        stats["trail_reinforce"] += distill_stockfish_top(
                            weights, mid, tops, boost=2.2
                        )

                stats["sf_games"] += 1
                if won:
                    stats["stig_wins"] += 1

            _log(
                log,
                f"cycle {cycle} stig_wins={stats['stig_wins']}/{stats['sf_games']} last_vs={label}",
            )

            clip_field_params(weights.field)
            prune_learned_moves(weights, keep=100000)
            prune_trails(weights, keep_positions=300000)

            # Health.
            pev = abs(float(evaluate_board(chess.Board(), weights)))
            if pev > 1e5 or not np.isfinite(pev):
                _log(log, f"HEALTH FAIL eval={pev}; reload latest")
                weights = load_weights(out / "latest.json")
                continue

            now = time.time()
            if now - last_ckpt >= cfg.checkpoint_every_min * 60:
                ckpt = out / f"ckpt_{int(now - t0)}s.json"
                weights.training_meta = {
                    **weights.training_meta,
                    "gm_train": stats,
                    "elapsed_hours": round((now - t0) / 3600, 3),
                }
                save_weights(weights, ckpt)
                save_weights(weights, out / "latest.json")
                _log(log, f"checkpoint {ckpt.name}")
                last_ckpt = now

            if now - last_probe >= cfg.probe_every_min * 60:
                try:
                    probe = probe_sf_ladder(weights, sf, cfg, log, games_per=6)
                    stats["probes"].append(probe)
                    (out / "elo_probe.json").write_text(
                        json.dumps(probe, indent=2), encoding="utf-8"
                    )
                    _log(log, f"Elo probe ≈ {probe['estimated_elo']} gm={probe.get('gm')}")
                    if probe.get("estimated_elo", 0) >= cfg.gm_floor:
                        # Confirm with a second tougher probe.
                        confirm = probe_sf_ladder(weights, sf, cfg, log, games_per=8)
                        stats["probes"].append(confirm)
                        _log(log, f"GM confirm ≈ {confirm['estimated_elo']}")
                        if confirm.get("estimated_elo", 0) >= cfg.gm_floor:
                            _log(log, "=== GRANDMASTER FLOOR CONFIRMED ===")
                            break
                except Exception:
                    _log(log, "probe failed:\n" + traceback.format_exc())
                last_probe = now

        # Final.
        probe = probe_sf_ladder(weights, sf, cfg, log, games_per=8)
        stats["final_probe"] = probe
        weights.format_version = 4
        weights.training_meta = {
            **weights.training_meta,
            "gm_train": stats,
            "elapsed_hours": round((time.time() - t0) / 3600, 3),
            "uniqueness": score_uniqueness(
                weights.to_dict()["uniquenessFingerprint"]
            ).to_dict(),
            "precision": "float64",
            "goal": "grandmaster_sf_uci_elo_ladder",
        }
        final = out / "gm_weights.json"
        save_weights(weights, final)
        save_weights(weights, out / "latest.json")
        save_weights(weights, Path("chess_contest/weights/base_weights.json"))
        save_weights(weights, Path("chess_contest/weights/gm_weights.json"))
        (out / "final_report.json").write_text(
            json.dumps({"stats": stats, "probe": probe, "config": asdict(cfg)}, indent=2),
            encoding="utf-8",
        )
        _log(
            log,
            f"DONE elo≈{probe.get('estimated_elo')} gm={probe.get('gm')} → {final}",
        )
        return final
    finally:
        sf.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Grandmaster Stigmergy float64 trail trainer")
    p.add_argument("--hours", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init", default="chess_contest/weights/overnight/latest.json")
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--stockfish", default="/usr/games/stockfish")
    p.add_argument("--stig-ms", type=int, default=400)
    p.add_argument("--stig-depth", type=int, default=8)
    p.add_argument("--sf-ms", type=int, default=80)
    p.add_argument("--analyse-ms", type=int, default=120)
    p.add_argument("--trail-positions", type=int, default=8000)
    p.add_argument("--analyse-per-cycle", type=int, default=200)
    p.add_argument("--sf-batch", type=int, default=16)
    p.add_argument("--gm-floor", type=float, default=GM_FLOOR)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    cfg = GMConfig(
        hours=0.2 if args.quick else args.hours,
        seed=args.seed,
        init_weights=args.init,
        out_dir=args.out_dir,
        stockfish_path=args.stockfish,
        stig_movetime_ms=120 if args.quick else args.stig_ms,
        stig_max_depth=4 if args.quick else args.stig_depth,
        sf_movetime_ms=40 if args.quick else args.sf_ms,
        analyse_ms=50 if args.quick else args.analyse_ms,
        trail_build_positions=80 if args.quick else args.trail_positions,
        analyse_per_cycle=20 if args.quick else args.analyse_per_cycle,
        sf_batch=2 if args.quick else args.sf_batch,
        probe_every_min=3.0 if args.quick else 20.0,
        checkpoint_every_min=2.0 if args.quick else 8.0,
        gm_floor=args.gm_floor,
    )
    run_gm(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
