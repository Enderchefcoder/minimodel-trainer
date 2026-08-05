"""Overnight GM push: heavy self-play + Stockfish-max winner distillation.

Keeps Stigmergy-DPFE unique (pheromone fields / ternary trails / swarm head).
Stockfish is only an oracle opponent — we train on the winner's moves.
"""

from __future__ import annotations

import argparse
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
    distill_stockfish_pv,
    imitation_toward_move,
    prune_learned_moves,
)
from chess_contest.stigmergy.engine import StigmergyEngine  # noqa: E402
from chess_contest.stigmergy.evaluate import clear_eval_cache  # noqa: E402
from chess_contest.stigmergy.opponents import (  # noqa: E402
    ClassicPSTOpponent,
    GreedyMaterialOpponent,
    play_game,
    update_elo,
)
from chess_contest.stigmergy.stockfish_uci import (  # noqa: E402
    StockfishConfig,
    StockfishEngine,
    stockfish_available,
)
from chess_contest.stigmergy.uniqueness import score_uniqueness  # noqa: E402
from chess_contest.stigmergy.weights import (  # noqa: E402
    StigmergyWeights,
    default_weights,
    load_weights,
    mutate_field,
    save_weights,
)


@dataclass
class OvernightConfig:
    hours: float = 8.0
    seed: int = 7
    init_weights: str = "chess_contest/weights/base_weights.json"
    out_dir: str = "chess_contest/weights/overnight"
    stockfish_path: str = "/usr/games/stockfish"
    stig_movetime_ms: int = 120
    stig_max_depth: int = 4
    sf_movetime_ms: int = 80
    max_plies: int = 70
    checkpoint_every_min: float = 12.0
    probe_every_min: float = 30.0
    book_build_positions: int = 400
    selfplay_batch: int = 2
    sf_batch: int = 10
    analyse_per_cycle: int = 80
    skip_book_if_ckpt: bool = True


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


def play_recorded(
    white_choose,
    black_choose,
    max_plies: int,
) -> tuple[str, list[chess.Move]]:
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


def build_sf_opening_book(weights, sf: StockfishEngine, n_positions: int, log: Path, rng: np.random.Generator) -> int:
    """Query SF multipv from many reachable positions to seed ternary trails."""
    _log(log, f"SF book build: {n_positions} positions")
    reinforced = 0
    # Random-walk openings + SF PVs.
    for i in range(n_positions):
        b = chess.Board()
        depth = int(rng.integers(0, 10))
        for _ in range(depth):
            if b.is_game_over():
                break
            # Mix random legal with previous book bias.
            legal = list(b.legal_moves)
            b.push(rng.choice(legal))
        if b.is_game_over():
            continue
        try:
            tops = sf.analyse_top(b, movetime_ms=80, multipv=3)
        except Exception as e:
            _log(log, f"SF analyse failed: {e}")
            continue
        for info in tops:
            pv = info.get("pv") or []
            if pv:
                reinforced += distill_stockfish_pv(weights, b, pv, boost=1.5 if info is tops[0] else 0.8)
        if (i + 1) % 50 == 0:
            _log(log, f"  book build {i+1}/{n_positions}, reinforced={reinforced}")
    return reinforced


def probe_sf_ladder(weights, sf: StockfishEngine, cfg: OvernightConfig, log: Path) -> dict:
    """Play short matches vs SF at escalating UCI_Elo; estimate our Elo."""
    engine = StigmergyEngine(weights)
    elos = [1320, 1600, 1900, 2200, 2500, 2800, 3000, 3190]
    our = 1800.0
    rows = []
    for target in elos:
        sf.set_elo(target)
        score = 0.0
        games = 4
        for i in range(games):
            stig_white = i % 2 == 0

            def stig(b, _e=engine):
                return _e.choose_move(b, time_ms=cfg.stig_movetime_ms, max_depth=cfg.stig_max_depth).move

            def sfc(b, _sf=sf):
                return _sf.choose(b, movetime_ms=cfg.sf_movetime_ms)

            if stig_white:
                result, _ = play_recorded(stig, sfc, cfg.max_plies)
                s = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
            else:
                result, _ = play_recorded(sfc, stig, cfg.max_plies)
                s = {"1-0": 0.0, "0-1": 1.0, "1/2-1/2": 0.5}[result]
            score += s
            our, _ = update_elo(our, float(target), s, k=48.0)
        wr = score / games
        rows.append({"sf_elo": target, "score": score, "games": games, "winrate": wr, "our_elo_after": round(our, 1)})
        _log(log, f"probe vs SF Elo {target}: score {score}/{games} ({wr:.0%}) → our≈{our:.0f}")
        if wr < 0.15 and target >= 2500:
            # No point burning time on higher if we're crushed.
            break
    return {"estimated_elo": round(our, 1), "ladder": rows}


def run_overnight(cfg: OvernightConfig) -> Path:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "overnight.log"
    _log(log, f"Starting overnight GM push hours={cfg.hours} seed={cfg.seed}")

    if Path(cfg.init_weights).exists():
        weights = load_weights(cfg.init_weights)
        _log(log, f"Loaded {cfg.init_weights}")
    else:
        weights = default_weights()
        _log(log, "Using default weights")
    weights.format_version = 3
    weights.diffusion_steps = max(weights.diffusion_steps, 4)
    if weights.field.field_head is None:
        weights.field.field_head = np.zeros(24, dtype=np.float64)

    if not stockfish_available(cfg.stockfish_path):
        raise RuntimeError(f"Stockfish not found at {cfg.stockfish_path}")

    sf = StockfishEngine(
        StockfishConfig(
            path=cfg.stockfish_path,
            threads=1,
            hash_mb=128,
            skill_level=20,
            movetime_ms=cfg.sf_movetime_ms,
        )
    )
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    deadline = t0 + cfg.hours * 3600
    last_ckpt = t0
    last_probe = t0
    stats = {
        "sf_games": 0,
        "selfplay_games": 0,
        "sf_wins_as_stig": 0,
        "distill_moves": 0,
        "phases": [],
    }

    try:
        book_ckpt = out / "ckpt_book.json"
        init_path = Path(cfg.init_weights).resolve()
        if cfg.skip_book_if_ckpt and book_ckpt.exists() and (
            init_path == book_ckpt.resolve() or "ckpt_book" in cfg.init_weights
        ):
            _log(log, "Skipping book rebuild (resuming from SF book checkpoint)")
        else:
            sf.set_elo(None)
            n = build_sf_opening_book(weights, sf, cfg.book_build_positions, log, rng)
            stats["distill_moves"] += n
            stats["phases"].append({"name": "sf_book", "reinforced": n})
            save_weights(weights, book_ckpt)

        engine = StigmergyEngine(weights)
        cycle = 0
        # Escalating SF Elo schedule (ends at max 3190).
        sf_schedule = (
            [1320, 1500, 1700, 1900, 2100] * 2
            + [2300, 2500, 2700, 2800] * 3
            + [2900, 3000, 3100, 3190] * 10
        )

        while time.time() < deadline:
            cycle += 1
            clear_eval_cache()
            engine = StigmergyEngine(weights)

            # --- Fast SF analysis distillation (high throughput) ---
            sf.set_elo(None)
            analysed = 0
            for _ in range(cfg.analyse_per_cycle):
                b = chess.Board()
                for _ply in range(int(rng.integers(0, 16))):
                    if b.is_game_over():
                        break
                    b.push(rng.choice(list(b.legal_moves)))
                if b.is_game_over():
                    continue
                tops = sf.analyse_top(b, movetime_ms=max(40, cfg.sf_movetime_ms // 2), multipv=3)
                for info in tops:
                    if info.get("pv"):
                        stats["distill_moves"] += distill_stockfish_pv(
                            weights, b, info["pv"], boost=1.6 if info is tops[0] else 0.9
                        )
                    try:
                        mv = chess.Move.from_uci(info["uci"])
                        if imitation_toward_move(weights, b, mv, rng, lr=0.06):
                            analysed += 1
                    except Exception:
                        pass
            _log(log, f"cycle {cycle} analysed≈{cfg.analyse_per_cycle} top1_hits~{analysed}")

            # --- SF sparring + winner distillation (volume over depth) ---
            for j in range(cfg.sf_batch):
                target = sf_schedule[(cycle * cfg.sf_batch + j) % len(sf_schedule)]
                if (cycle + j) % 4 == 0:
                    sf.set_elo(None)
                    target_label = "MAX"
                else:
                    sf.set_elo(target)
                    target_label = str(target)

                stig_white = (cycle + j) % 2 == 0

                def stig_choose(b, _e=engine):
                    res = _e.choose_move(
                        b, time_ms=cfg.stig_movetime_ms, max_depth=cfg.stig_max_depth
                    )
                    return res.move

                def sf_choose(b, _sf=sf):
                    return _sf.choose(b, movetime_ms=cfg.sf_movetime_ms)

                if stig_white:
                    result, moves = play_recorded(stig_choose, sf_choose, cfg.max_plies)
                    stig_won = result == "1-0"
                else:
                    result, moves = play_recorded(sf_choose, stig_choose, cfg.max_plies)
                    stig_won = result == "0-1"

                dstat = distill_game(
                    weights,
                    moves,
                    result,
                    winner_boost=1.8 if not stig_won else 1.1,
                    loser_penalty=0.55,
                )
                stats["sf_games"] += 1
                stats["distill_moves"] += dstat.moves_reinforced
                if stig_won:
                    stats["sf_wins_as_stig"] += 1

                # Always pull a MAX PV from a midgame node of this game.
                if moves:
                    mid = chess.Board()
                    cut = max(1, min(16, len(moves) // 2))
                    for m in moves[:cut]:
                        mid.push(m)
                    if not mid.is_game_over():
                        sf.set_elo(None)
                        tops = sf.analyse_top(mid, movetime_ms=90, multipv=2)
                        for info in tops:
                            if info.get("pv"):
                                stats["distill_moves"] += distill_stockfish_pv(
                                    weights, mid, info["pv"], boost=1.5
                                )

            _log(
                log,
                f"cycle {cycle} SF games stig_wins={stats['sf_wins_as_stig']}/{stats['sf_games']} "
                f"last_vs={target_label}",
            )

            # --- Light self-play ES (every other cycle) ---
            if cycle % 2 == 0 and cfg.selfplay_batch > 0:
                ref = ClassicPSTOpponent(depth=2)
                greedy = GreedyMaterialOpponent()

                def score_weights(w, _ref=ref, _greedy=greedy) -> float:
                    eng = StigmergyEngine(w)
                    pts = 0.0
                    for i in range(2):
                        def ch(board_pos, _eng=eng):
                            return _eng.choose_move(board_pos, time_ms=50, max_depth=3).move

                        opp = _ref.choose if i % 2 == 0 else _greedy.choose
                        if i % 2 == 0:
                            result = play_game(ch, opp, max_plies=40)
                            pts += {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
                        else:
                            result = play_game(opp, ch, max_plies=40)
                            pts += {"1-0": 0.0, "0-1": 1.0, "1/2-1/2": 0.5}[result]
                    return pts / 2.0

                baseline = score_weights(weights)
                adopted = 0
                for _ in range(cfg.selfplay_batch):
                    cand = mutate_field(weights.field, rng, sigma=0.05)
                    trial = StigmergyWeights(
                        field=cand,
                        book=weights.book,
                        learned_moves=dict(weights.learned_moves),
                        diffusion_steps=weights.diffusion_steps,
                        format_version=3,
                    )
                    sc = score_weights(trial)
                    stats["selfplay_games"] += 2
                    if sc > baseline + 0.05:
                        weights.field = cand
                        clip_field_params(weights.field)
                        baseline = sc
                        adopted += 1
                        clear_eval_cache()
                _log(log, f"cycle {cycle} self-play adopted={adopted}/{cfg.selfplay_batch}")

            clip_field_params(weights.field)
            prune_learned_moves(weights, keep=20000)
            engine = StigmergyEngine(weights)

            now = time.time()
            # Health check: abort cycle updates if eval exploded.
            try:
                from chess_contest.stigmergy.evaluate import evaluate_board as _ev

                probe_eval = abs(float(_ev(chess.Board(), weights)))
                if probe_eval > 1e6 or not np.isfinite(probe_eval):
                    _log(log, f"HEALTH FAIL eval={probe_eval}; reloading last good checkpoint")
                    latest = out / "latest.json"
                    book = out / "ckpt_book.json"
                    reload_path = latest if latest.exists() else book
                    weights = load_weights(reload_path)
                    clip_field_params(weights.field)
                    clear_eval_cache()
                    continue
            except Exception as e:
                _log(log, f"health check error: {e}")

            if now - last_ckpt >= cfg.checkpoint_every_min * 60:
                ckpt = out / f"ckpt_{int(now - t0)}s.json"
                weights.training_meta = {
                    **weights.training_meta,
                    "overnight": stats,
                    "elapsed_hours": round((now - t0) / 3600, 3),
                    "uniqueness": score_uniqueness(
                        weights.to_dict()["uniquenessFingerprint"]
                    ).to_dict(),
                }
                save_weights(weights, ckpt)
                save_weights(weights, out / "latest.json")
                _log(log, f"checkpoint {ckpt.name}")
                last_ckpt = now

            if now - last_probe >= cfg.probe_every_min * 60:
                try:
                    probe = probe_sf_ladder(weights, sf, cfg, log)
                    (out / "elo_probe.json").write_text(
                        json.dumps(probe, indent=2), encoding="utf-8"
                    )
                    stats["last_probe"] = probe
                    _log(log, f"Elo probe ≈ {probe['estimated_elo']}")
                except Exception:
                    _log(log, "probe failed:\n" + traceback.format_exc())
                last_probe = now

        # Final max-strength distillation burst.
        _log(log, "Final SF-MAX distillation burst")
        sf.set_elo(None)
        for i in range(20):
            engine = StigmergyEngine(weights)
            stig_white = i % 2 == 0

            def stig_choose(b, _e=engine):
                return _e.choose_move(b, time_ms=cfg.stig_movetime_ms, max_depth=cfg.stig_max_depth).move

            def sf_choose(b, _sf=sf):
                return _sf.choose(b, movetime_ms=max(200, cfg.sf_movetime_ms))

            if stig_white:
                result, moves = play_recorded(stig_choose, sf_choose, cfg.max_plies)
            else:
                result, moves = play_recorded(sf_choose, stig_choose, cfg.max_plies)
            distill_game(weights, moves, result, winner_boost=1.6, loser_penalty=0.5)
            stats["sf_games"] += 1

        probe = probe_sf_ladder(weights, sf, cfg, log)
        stats["final_probe"] = probe
        weights.training_meta = {
            **weights.training_meta,
            "overnight": stats,
            "elapsed_hours": round((time.time() - t0) / 3600, 3),
            "uniqueness": score_uniqueness(weights.to_dict()["uniquenessFingerprint"]).to_dict(),
            "goal": "near_or_over_3000_vs_sf_uci_elo_ladder",
        }
        final = out / "gm_weights.json"
        save_weights(weights, final)
        save_weights(weights, Path("chess_contest/weights/base_weights.json"))
        (out / "final_report.json").write_text(
            json.dumps({"stats": stats, "probe": probe, "config": asdict(cfg)}, indent=2),
            encoding="utf-8",
        )
        _log(log, f"DONE final_elo≈{probe.get('estimated_elo')} → {final}")
        return final
    finally:
        sf.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Overnight Stigmergy GM push vs Stockfish")
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--init", default="chess_contest/weights/base_weights.json")
    p.add_argument("--out-dir", default="chess_contest/weights/overnight")
    p.add_argument("--stockfish", default="/usr/games/stockfish")
    p.add_argument("--stig-ms", type=int, default=120)
    p.add_argument("--stig-depth", type=int, default=4)
    p.add_argument("--sf-ms", type=int, default=80)
    p.add_argument("--book-positions", type=int, default=400)
    p.add_argument("--analyse-per-cycle", type=int, default=80)
    p.add_argument("--sf-batch", type=int, default=10)
    p.add_argument("--selfplay-batch", type=int, default=2)
    p.add_argument("--quick", action="store_true", help="~10min smoke overnight")
    args = p.parse_args(argv)

    cfg = OvernightConfig(
        hours=0.15 if args.quick else args.hours,
        seed=args.seed,
        init_weights=args.init,
        out_dir=args.out_dir,
        stockfish_path=args.stockfish,
        stig_movetime_ms=60 if args.quick else args.stig_ms,
        stig_max_depth=3 if args.quick else args.stig_depth,
        sf_movetime_ms=40 if args.quick else args.sf_ms,
        book_build_positions=30 if args.quick else args.book_positions,
        checkpoint_every_min=2.0 if args.quick else 12.0,
        probe_every_min=4.0 if args.quick else 30.0,
        selfplay_batch=1 if args.quick else args.selfplay_batch,
        sf_batch=2 if args.quick else args.sf_batch,
        analyse_per_cycle=15 if args.quick else args.analyse_per_cycle,
        skip_book_if_ckpt=True,
    )
    run_overnight(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
