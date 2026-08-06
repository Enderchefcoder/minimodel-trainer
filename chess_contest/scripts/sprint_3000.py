"""2-hour sprint: warm-train SwarmNet hard, policy-sprint ladder vs SF Elo.

Play never calls Stockfish. Teacher/opponent SF only. Target crush ≥3000.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chess_contest.scripts.crush_3000 import (  # noqa: E402
    generate_dataset,
    train_swarm,
    write_reports,
)
from chess_contest.stigmergy.engine import StigmergyEngine  # noqa: E402
from chess_contest.stigmergy.opponents import update_elo  # noqa: E402
from chess_contest.stigmergy.search import set_policy_sprint, set_swarm  # noqa: E402
from chess_contest.stigmergy.stockfish_uci import (  # noqa: E402
    StockfishConfig,
    StockfishEngine,
    stockfish_available,
)
from chess_contest.stigmergy.swarm_net import IN_CH, SwarmNet, try_load_swarm  # noqa: E402
from chess_contest.stigmergy.weights import default_weights  # noqa: E402


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


def policy_match(net: SwarmNet, sf: StockfishEngine, n: int = 100, depth: int = 10) -> float:
    """OOD top-1 match vs full-strength SF (honest teacher agreement)."""
    hits = 0
    board = chess.Board()
    for _ in range(n):
        if board.is_game_over(claim_draw=True) or len(board.move_stack) > 60:
            board = chess.Board()
            for _ in range(int(np.random.randint(0, 12))):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(moves[int(np.random.randint(0, len(moves)))])
            continue
        sf.set_elo(None)
        tops = sf.analyse_top(board, multipv=1, depth=depth)
        if not tops or not tops[0].get("uci"):
            board = chess.Board()
            continue
        try:
            teacher = chess.Move.from_uci(tops[0]["uci"])
        except ValueError:
            board = chess.Board()
            continue
        ours = net.choose(board)
        if ours == teacher:
            hits += 1
        board.push(teacher if teacher in board.legal_moves else next(iter(board.legal_moves)))
    return hits / max(1, n)


def sprint_ladder(
    engine: StigmergyEngine,
    sf: StockfishEngine,
    log: Path,
    *,
    games_per: int,
    targets: list[int],
    stig_ms: int = 500,
    stig_depth: int = 6,
    use_policy_sprint: bool = False,
) -> dict:
    """Fast ladder; SF is opponent only. Default = short IDAS + swarm order."""
    our = 2000.0
    rows = []
    set_policy_sprint(use_policy_sprint)
    try:
        for target in targets:
            sc = 0.0
            for i in range(games_per):
                board = chess.Board()
                stig_white = i % 2 == 0
                for _ in range(140):
                    if board.is_game_over(claim_draw=True):
                        break
                    if (board.turn == chess.WHITE) == stig_white:
                        mv = engine.choose_move(
                            board, time_ms=stig_ms, max_depth=stig_depth
                        ).move
                        if mv is None:
                            break
                        board.push(mv)
                    else:
                        sf.set_elo(target)
                        board.push(sf.choose(board, movetime_ms=50))
                res = board.result(claim_draw=True)
                if board.is_checkmate():
                    res = "0-1" if board.turn == chess.WHITE else "1-0"
                s = _score(stig_white, res)
                sc += s
                our, _ = update_elo(our, float(target), s, k=32.0)
                _log(log, f"  sprint vs {target} #{i} → {res} s={s} ≈{our:.0f}")
            wr = sc / games_per
            rows.append(
                {
                    "sf_elo": target,
                    "score": sc,
                    "games": games_per,
                    "winrate": wr,
                    "our_elo_after": round(our, 1),
                    "think_ms": stig_ms,
                    "stig_depth": stig_depth,
                    "policy_sprint": use_policy_sprint,
                }
            )
            _log(
                log,
                f"SPRINT ladder vs SF {target}: {sc}/{games_per} ({wr:.0%}) → ≈{our:.0f}",
            )
            if target >= 2500 and wr < 0.05 and our < 2200:
                break
            if target >= 3000 and our >= 3000 and wr >= 0.35:
                break
    finally:
        set_policy_sprint(False)
    return {
        "estimated_elo": round(our, 1),
        "ladder": rows,
        "gm": our >= 2500,
        "above_gm": our >= 2500,
        "crush_3000": our >= 3000,
        "stockfish_at_play": False,
        "oracle_runtime": False,
        "policy_sprint": use_policy_sprint,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--extra-positions", type=int, default=20000)
    p.add_argument("--sf-depth", type=int, default=8)
    p.add_argument("--games-per", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--stig-ms", type=int, default=500)
    p.add_argument("--stig-depth", type=int, default=6)
    p.add_argument("--policy-sprint", action="store_true")
    p.add_argument("--skip-gen", action="store_true")
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--floor", type=float, default=3000.0)
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    log = out / "sprint_3000.log"
    data_path = out / "crush_dataset.npz"
    net_path = out / "swarm_net.pt"
    _log(log, f"=== SPRINT-3000 ep={args.epochs} extra={args.extra_positions} ===")

    if not stockfish_available():
        raise SystemExit("Stockfish required as offline teacher / ladder opponent")

    # Clear trails for sprint play — ablation: trails hurt vs SF.
    weights = default_weights()
    weights.format_version = 4
    weights.book.clear()
    weights.trails.clear()
    weights.training_meta.update(
        {
            "oracle_runtime": False,
            "stockfish_at_play": False,
            "swarm_net": str(net_path),
            "policy_sprint": True,
            "crush_3000": True,
        }
    )
    if (out / "latest.json").exists():
        # Keep meta path only; do not load fat trails into play.
        pass

    sf = StockfishEngine(StockfishConfig(threads=2, hash_mb=256))
    try:
        if not data_path.is_file():
            raise SystemExit(f"missing {data_path} — run crush_3000 dataset gen first")

        blob = np.load(data_path)
        data = [
            (np.asarray(pl, dtype=np.float32), np.asarray(s, dtype=np.float32), float(v))
            for pl, s, v in zip(blob["planes"], blob["soft"], blob["values"], strict=True)
        ]
        _log(log, f"loaded crush dataset {len(data)}")

        net = try_load_swarm(net_path)
        if net is None:
            net = SwarmNet(channels=192, blocks=10, in_ch=IN_CH)
            _log(log, "cold-start SwarmNet 192x10")
        else:
            _log(log, f"warm-start {net_path} ch={net.channels} bl={net.blocks}")

        if args.probe_only:
            set_swarm(net)
            eng = StigmergyEngine(weights, load_swarm=False)
            set_swarm(net)
            probe = sprint_ladder(
                eng,
                sf,
                log,
                games_per=args.games_per,
                targets=[1320, 1600, 2000, 2200, 2500, 2700, 2800, 2900, 3000, 3100],
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
                use_policy_sprint=args.policy_sprint,
            )
            match = policy_match(net, sf, n=80)
            _log(log, f"OOD policy match={match:.0%}")
            probe["policy_match_ood"] = round(match, 4)
            write_reports(out, probe, args)
            return 0

        # Expand teacher set first (diversity), then short fine-tunes with OOD early-stop.
        # Long train on a fixed 80k overfits (65% train-top1 → 25% OOD).
        if not args.skip_gen and args.extra_positions > 0:
            extra = generate_dataset(
                sf, log, n_positions=args.extra_positions, depth=args.sf_depth, weights=None
            )
            data = (data + extra)[-160_000:]
            np.savez_compressed(
                data_path,
                planes=np.stack([d[0] for d in data]),
                soft=np.stack([d[1] for d in data]),
                values=np.asarray([d[2] for d in data], dtype=np.float32),
            )
            _log(log, f"expanded dataset → {len(data)}")

        best_path = out / "swarm_net_best_ood.pt"
        best_ood = policy_match(net, sf, n=80)
        _log(log, f"baseline OOD match={best_ood:.0%}")
        net.save(best_path)
        net.save(net_path)

        # Few epochs; keep the checkpoint with best OOD (not train top1).
        for ep in range(1, args.epochs + 1):
            train_swarm(net, data, log, epochs=1, batch_size=args.batch_size, lr=1.0e-4)
            ood = policy_match(net, sf, n=100)
            _log(log, f"epoch-ood {ep}/{args.epochs} OOD={ood:.0%} best={best_ood:.0%}")
            if ood >= best_ood - 0.005:
                if ood > best_ood:
                    best_ood = ood
                    net.save(best_path)
                    net.save(net_path)
                    _log(log, f"new best OOD={best_ood:.0%} saved")
            else:
                # Restore best and stop — further train hurts generalization.
                _log(log, f"OOD dropped ({ood:.0%} < {best_ood:.0%}); early-stop")
                net.load(best_path)
                break

        match = best_ood
        net.load(best_path)
        set_swarm(net)
        eng = StigmergyEngine(weights, load_swarm=False)
        set_swarm(net)
        probe = sprint_ladder(
            eng,
            sf,
            log,
            games_per=args.games_per,
            targets=[1320, 1600, 2000, 2200, 2500, 2700, 2800, 2900, 3000, 3100],
            stig_ms=args.stig_ms,
            stig_depth=args.stig_depth,
            use_policy_sprint=args.policy_sprint,
        )
        probe["policy_match_ood"] = round(match, 4)
        write_reports(out, probe, args)
        _log(
            log,
            f"SPRINT Elo ≈ {probe['estimated_elo']} crush3000={probe['crush_3000']} "
            f"match={match:.0%}",
        )

        if probe["estimated_elo"] < args.floor:
            _log(log, "below floor — longer-think confirm ladder")
            probe2 = sprint_ladder(
                eng,
                sf,
                log,
                games_per=max(8, args.games_per),
                targets=[2000, 2200, 2500, 2700, 2800, 2900, 3000, 3100],
                stig_ms=max(args.stig_ms, 2000),
                stig_depth=max(args.stig_depth, 10),
                use_policy_sprint=False,
            )
            probe2["policy_match_ood"] = round(match, 4)
            write_reports(out, probe2, args)
            _log(log, f"SPRINT2 Elo ≈ {probe2['estimated_elo']} match={match:.0%}")
    finally:
        sf.close()
        set_swarm(None)
        set_policy_sprint(False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
