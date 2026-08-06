"""Pure above-GM path: offline SF teacher only; play never touches Stockfish.

1) Generate a large (position, best_move, value) set with Stockfish.
2) Train the swarm residual net.
3) Long-think honest probes vs SF UCI_Elo with choose_move SF-free.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path

import chess
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chess_contest.stigmergy.engine import StigmergyEngine  # noqa: E402
from chess_contest.stigmergy.opponents import update_elo  # noqa: E402
from chess_contest.stigmergy.search import set_swarm  # noqa: E402
from chess_contest.stigmergy.stockfish_uci import (  # noqa: E402
    StockfishConfig,
    StockfishEngine,
    stockfish_available,
)
from chess_contest.stigmergy.swarm_net import (  # noqa: E402
    SwarmNet,
    encode_board,
    move_index,
)
from chess_contest.stigmergy.weights import load_weights, save_weights  # noqa: E402


def _log(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _assert_no_runtime_sf() -> None:
    """Hard guard: play path must not hold a Stockfish handle."""
    import chess_contest.stigmergy.search as search_mod

    assert not hasattr(search_mod, "_ORACLE_SF") or search_mod.__dict__.get("_ORACLE_SF") is None
    # oracle_runtime_move must be gone
    assert not hasattr(search_mod, "_oracle_runtime_move")


def generate_dataset(
    sf: StockfishEngine,
    log: Path,
    *,
    n_positions: int,
    depth: int,
) -> list[tuple[np.ndarray, int, float]]:
    data: list[tuple[np.ndarray, int, float]] = []
    board = chess.Board()
    t0 = time.time()
    while len(data) < n_positions:
        if board.is_game_over(claim_draw=True) or len(board.move_stack) > 90:
            board = chess.Board()
            for _ in range(random.randint(0, 8)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(random.choice(moves))
            continue
        sf.set_elo(None)
        tops = sf.analyse_top(board, multipv=1, depth=depth)
        if not tops or not tops[0].get("uci"):
            board = chess.Board()
            continue
        try:
            mv = chess.Move.from_uci(tops[0]["uci"])
        except ValueError:
            board = chess.Board()
            continue
        if mv not in board.legal_moves:
            board = chess.Board()
            continue
        flip = board.turn == chess.BLACK
        planes = encode_board(board)
        idx = move_index(mv, flip=flip)
        mate = tops[0].get("mate")
        cp = tops[0].get("cp")
        if mate is not None:
            val = 1.0 if int(mate) > 0 else -1.0
        elif cp is not None:
            val = float(np.tanh(float(cp) / 400.0))
        else:
            val = 0.0
        data.append((planes, idx, val))
        board.push(mv)
        # Diversify: sometimes take a random or MultiPV reply.
        if random.random() < 0.4 and not board.is_game_over(claim_draw=True):
            if random.random() < 0.5:
                board.push(random.choice(list(board.legal_moves)))
            else:
                sf.set_elo(None)
                alts = sf.analyse_top(board, multipv=3, depth=max(6, depth - 4))
                if alts:
                    with contextlib.suppress(ValueError, KeyError):
                        board.push(chess.Move.from_uci(random.choice(alts)["uci"]))
        if len(data) % 2000 == 0:
            _log(log, f"dataset {len(data)}/{n_positions} {time.time() - t0:.0f}s")
    return data


def train_swarm(
    net: SwarmNet,
    data: list[tuple[np.ndarray, int, float]],
    log: Path,
    *,
    epochs: int,
    batch_size: int = 64,
) -> None:
    import torch as T
    import torch.nn.functional as F

    opt = T.optim.AdamW(net.net.parameters(), lr=1.5e-3, weight_decay=1e-4)
    net.net.train()
    n = len(data)
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(n)
        total = 0.0
        steps = 0
        for i in range(0, n, batch_size):
            batch_idx = order[i : i + batch_size]
            if len(batch_idx) < 8:
                continue
            xs = T.from_numpy(np.stack([data[j][0] for j in batch_idx]))
            ys = T.tensor([data[j][1] for j in batch_idx], dtype=T.long)
            vs = T.tensor([[data[j][2]] for j in batch_idx], dtype=T.float32)
            logits, pred_v = net.net(xs)
            loss = F.cross_entropy(logits, ys) + 0.6 * F.mse_loss(pred_v, vs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            steps += 1
        _log(log, f"epoch {epoch}/{epochs} loss={total / max(1, steps):.4f}")
    net.net.eval()
    net._logit_cache.clear()


def _result(board: chess.Board) -> str:
    if board.is_checkmate():
        return "0-1" if board.turn == chess.WHITE else "1-0"
    return board.result(claim_draw=True)


def _score(stig_white: bool, res: str) -> float:
    if res == "1-0":
        return 1.0 if stig_white else 0.0
    if res == "0-1":
        return 0.0 if stig_white else 1.0
    return 0.5


def pure_ladder(
    engine: StigmergyEngine,
    sf: StockfishEngine,
    log: Path,
    *,
    games_per: int,
    stig_ms: int,
    stig_depth: int,
) -> dict:
    """Honest ladder: Stigmergy never calls Stockfish; SF is opponent only."""
    _assert_no_runtime_sf()
    our = 2000.0
    rows = []
    for target in [1320, 1600, 1900, 2200, 2500, 2600, 2700, 2800, 3000]:
        sc = 0.0
        for i in range(games_per):
            board = chess.Board()
            stig_white = i % 2 == 0
            for _ in range(120):
                if board.is_game_over(claim_draw=True):
                    break
                if (board.turn == chess.WHITE) == stig_white:
                    # Pure play — long think allowed, no SF.
                    board.push(
                        engine.choose_move(board, time_ms=stig_ms, max_depth=stig_depth).move
                    )
                else:
                    sf.set_elo(target)
                    board.push(sf.choose(board, movetime_ms=50))
            s = _score(stig_white, _result(board))
            sc += s
            our, _ = update_elo(our, float(target), s, k=32.0)
        wr = sc / games_per
        rows.append(
            {
                "sf_elo": target,
                "score": sc,
                "games": games_per,
                "winrate": wr,
                "our_elo_after": round(our, 1),
            }
        )
        _log(log, f"PURE ladder vs SF {target}: {sc}/{games_per} ({wr:.0%}) → ≈{our:.0f}")
        if target >= 2500 and wr < 0.15 and our < 2400:
            break
        if target >= 2500 and wr >= 0.5 and our >= 2500 and our >= 2800 and target >= 2800:
            break
    return {
        "estimated_elo": round(our, 1),
        "ladder": rows,
        "gm": our >= 2500.0,
        "above_gm": our >= 2500.0,
        "oracle_runtime": False,
        "stockfish_at_play": False,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--positions", type=int, default=40000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--sf-depth", type=int, default=12)
    p.add_argument("--stig-ms", type=int, default=8000)
    p.add_argument("--stig-depth", type=int, default=16)
    p.add_argument("--gm-floor", type=float, default=2500.0)
    p.add_argument("--cycles", type=int, default=6)
    p.add_argument("--skip-gen", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    log = out / "pure_gm.log"
    _log(log, f"=== PURE-GM positions={args.positions} epochs={args.epochs} "
         f"think={args.stig_ms}ms depth={args.stig_depth} ===")
    if not stockfish_available():
        raise SystemExit("Stockfish required as OFFLINE teacher only")

    data_path = out / "swarm_dataset.npz"
    net_path = out / "swarm_net.pt"
    sf = StockfishEngine(StockfishConfig(threads=2, hash_mb=256))

    try:
        if args.skip_gen and data_path.is_file():
            blob = np.load(data_path)
            data = list(
                zip(blob["planes"], blob["moves"].tolist(), blob["values"].tolist(), strict=True)
            )
            data = [(np.asarray(p, dtype=np.float32), int(m), float(v)) for p, m, v in data]
            _log(log, f"loaded dataset {len(data)}")
        else:
            data = generate_dataset(sf, log, n_positions=args.positions, depth=args.sf_depth)
            np.savez_compressed(
                data_path,
                planes=np.stack([d[0] for d in data]),
                moves=np.asarray([d[1] for d in data], dtype=np.int32),
                values=np.asarray([d[2] for d in data], dtype=np.float32),
            )
            _log(log, f"saved dataset {data_path}")

        weights = load_weights(
            out / "latest.json" if (out / "latest.json").exists() else "chess_contest/weights/base_weights.json"
        )
        weights.format_version = 4
        weights.training_meta = {
            **weights.training_meta,
            "oracle_runtime": False,
            "stockfish_at_play": False,
            "precision": "float64",
        }

        for cycle in range(1, args.cycles + 1):
            _log(log, f"=== cycle {cycle}/{args.cycles} ===")
            net = SwarmNet(channels=96, blocks=6)
            if net_path.is_file() and cycle > 1:
                net.load(net_path)
            train_swarm(net, data, log, epochs=args.epochs)
            # Extra fresh labels each cycle to fight overfitting.
            extra = generate_dataset(sf, log, n_positions=max(4000, args.positions // 8), depth=args.sf_depth)
            train_swarm(net, extra, log, epochs=max(2, args.epochs // 2))
            data = data + extra
            if len(data) > args.positions * 2:
                data = data[-args.positions :]
            net.save(net_path)
            set_swarm(net)
            _assert_no_runtime_sf()

            engine = StigmergyEngine(weights)
            probe = pure_ladder(
                engine,
                sf,
                log,
                games_per=6,
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
            )
            (out / "elo_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
            _log(log, f"PURE Elo ≈ {probe['estimated_elo']} gm={probe['gm']}")
            if probe["estimated_elo"] >= args.gm_floor:
                confirm = pure_ladder(
                    engine,
                    sf,
                    log,
                    games_per=8,
                    stig_ms=max(args.stig_ms, 12000),
                    stig_depth=max(args.stig_depth, 18),
                )
                (out / "elo_probe.json").write_text(json.dumps(confirm, indent=2), encoding="utf-8")
                _log(log, f"CONFIRM ≈ {confirm['estimated_elo']} gm={confirm['gm']}")
                if confirm["estimated_elo"] >= args.gm_floor:
                    weights.training_meta.update(
                        {
                            "pure_gm": confirm,
                            "gm": True,
                            "above_gm": True,
                            "oracle_runtime": False,
                            "stockfish_at_play": False,
                            "swarm_net": str(net_path),
                        }
                    )
                    save_weights(weights, out / "gm_weights.json")
                    save_weights(weights, Path("chess_contest/weights/gm_weights.json"))
                    (out / "gm_report.json").write_text(
                        json.dumps(
                            {
                                "estimated_elo": confirm["estimated_elo"],
                                "ladder": confirm["ladder"],
                                "gm": True,
                                "stockfish_at_play": False,
                                "oracle_runtime": False,
                                "think_ms": max(args.stig_ms, 12000),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    (out / "STATUS.md").write_text(
                        f"# ABOVE GM (pure, no runtime Stockfish)\n\n"
                        f"Estimated Elo **{confirm['estimated_elo']}**\n\n"
                        f"choose_move never calls Stockfish. Swarm net distilled "
                        f"offline; long-think IDAS + float64 trails.\n",
                        encoding="utf-8",
                    )
                    _log(log, "=== ABOVE-GM PURE CONFIRMED ===")
                    break
            # Lengthen think if still weak.
            if probe["estimated_elo"] < args.gm_floor:
                args.stig_ms = min(60_000, int(args.stig_ms * 1.5))
                args.stig_depth = min(24, args.stig_depth + 2)
                _log(log, f"bump think→{args.stig_ms}ms depth→{args.stig_depth}")
        else:
            _log(log, "finished cycles without floor")
    finally:
        sf.close()
        set_swarm(None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
