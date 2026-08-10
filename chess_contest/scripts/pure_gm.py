"""Pure above-GM path: offline SF teacher only; play never touches Stockfish.

1) Generate a large (position, best_move, value) set with Stockfish + distill trails.
2) Train the swarm residual net (policy + value).
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

from chess_contest.stigmergy.distill import prune_trails, set_trail_policy  # noqa: E402
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
    assert not hasattr(search_mod, "_oracle_runtime_move")
    src = Path(search_mod.__file__).read_text(encoding="utf-8")
    assert "StockfishEngine" not in src


def generate_dataset(
    sf: StockfishEngine,
    log: Path,
    *,
    n_positions: int,
    depth: int,
    weights=None,
) -> list[tuple[np.ndarray, int, float]]:
    """SF-labeled positions; optionally write decisive trails into weights."""
    data: list[tuple[np.ndarray, int, float]] = []
    board = chess.Board()
    t0 = time.time()
    while len(data) < n_positions:
        if board.is_game_over(claim_draw=True) or len(board.move_stack) > 100:
            board = chess.Board()
            for _ in range(random.randint(0, 12)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(random.choice(moves))
            continue
        sf.set_elo(None)
        tops = sf.analyse_top(board, multipv=3, depth=depth)
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
        if weights is not None:
            # Decisive offline trail — playable without Stockfish later.
            set_trail_policy(weights, board, mv.uci()[:4], strength=40.0 + min(40.0, abs(val) * 40))
        # Soft MultiPV rows: teach second/third choices at lower weight via value only
        # (policy hard-label stays PV1; extra rows use alternate moves as targets).
        for alt in tops[1:]:
            with contextlib.suppress(ValueError, KeyError):
                am = chess.Move.from_uci(alt["uci"])
                if am in board.legal_moves and am != mv and random.random() < 0.35:
                    acp = alt.get("cp")
                    aval = float(np.tanh(float(acp) / 400.0)) if acp is not None else val * 0.5
                    data.append((planes, move_index(am, flip=flip), aval * 0.7))
        board.push(mv)
        if random.random() < 0.45 and not board.is_game_over(claim_draw=True):
            if random.random() < 0.55:
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


def _mirror_augment(
    data: list[tuple[np.ndarray, int, float]],
) -> list[tuple[np.ndarray, int, float]]:
    """Horizontal mirror of piece planes + from/to files (cheap diversity)."""
    out = list(data)
    for planes, idx, val in data:
        if random.random() > 0.5:
            continue
        mirrored = planes[:, :, ::-1].copy()
        fr, to = divmod(int(idx), 64)
        fr_f, fr_r = fr % 8, fr // 8
        to_f, to_r = to % 8, to // 8
        fr2 = fr_r * 8 + (7 - fr_f)
        to2 = to_r * 8 + (7 - to_f)
        out.append((mirrored, fr2 * 64 + to2, val))
    return out


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

    opt = T.optim.AdamW(net.net.parameters(), lr=1.2e-3, weight_decay=1e-4)
    sched = T.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    net.net.train()
    n = len(data)
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(n)
        total = 0.0
        correct = 0
        seen = 0
        steps = 0
        for i in range(0, n, batch_size):
            batch_idx = order[i : i + batch_size]
            if len(batch_idx) < 8:
                continue
            xs = T.from_numpy(np.stack([data[j][0] for j in batch_idx]))
            ys = T.tensor([data[j][1] for j in batch_idx], dtype=T.long)
            vs = T.tensor([[data[j][2]] for j in batch_idx], dtype=T.float32)
            logits, pred_v = net.net(xs)
            loss = F.cross_entropy(logits, ys) + 0.55 * F.mse_loss(pred_v, vs)
            opt.zero_grad()
            loss.backward()
            T.nn.utils.clip_grad_norm_(net.net.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            correct += int((logits.argmax(dim=-1) == ys).sum().item())
            seen += len(batch_idx)
            steps += 1
        sched.step()
        _log(
            log,
            f"epoch {epoch}/{epochs} loss={total / max(1, steps):.4f} "
            f"acc={correct / max(1, seen):.3f}",
        )
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
    targets: list[int] | None = None,
) -> dict:
    """Honest ladder: Stigmergy never calls Stockfish; SF is opponent only."""
    _assert_no_runtime_sf()
    our = 2000.0
    rows = []
    targets = targets or [1320, 1600, 1900, 2200, 2500, 2600, 2700, 2800, 3000]
    for target in targets:
        # Short think below 2200; long think at GM+ rungs (user allows).
        ms = stig_ms if target >= 2200 else min(stig_ms, 2500)
        depth = stig_depth if target >= 2200 else min(stig_depth, 12)
        gp = games_per if target >= 2200 else max(4, games_per // 2)
        sc = 0.0
        for i in range(gp):
            board = chess.Board()
            stig_white = i % 2 == 0
            for _ in range(140):
                if board.is_game_over(claim_draw=True):
                    break
                if (board.turn == chess.WHITE) == stig_white:
                    mv = engine.choose_move(board, time_ms=ms, max_depth=depth).move
                    if mv is None:
                        break
                    board.push(mv)
                else:
                    sf.set_elo(target)
                    board.push(sf.choose(board, movetime_ms=80))
            s = _score(stig_white, _result(board))
            sc += s
            our, _ = update_elo(our, float(target), s, k=32.0)
        wr = sc / gp
        rows.append(
            {
                "sf_elo": target,
                "score": sc,
                "games": gp,
                "winrate": wr,
                "our_elo_after": round(our, 1),
                "think_ms": ms,
            }
        )
        _log(log, f"PURE ladder vs SF {target}: {sc}/{gp} ({wr:.0%}) → ≈{our:.0f} [{ms}ms]")
        if target >= 2500 and wr < 0.12 and our < 2350:
            break
        if our >= 2800 and target >= 2800 and wr >= 0.4:
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
    p.add_argument("--positions", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--sf-depth", type=int, default=12)
    p.add_argument("--stig-ms", type=int, default=15000)
    p.add_argument("--stig-depth", type=int, default=18)
    p.add_argument("--gm-floor", type=float, default=2500.0)
    p.add_argument("--cycles", type=int, default=8)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--skip-gen", action="store_true")
    p.add_argument("--probe-only", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    log = out / "pure_gm.log"
    _log(
        log,
        f"=== PURE-GM positions={args.positions} epochs={args.epochs} "
        f"think={args.stig_ms}ms depth={args.stig_depth} "
        f"net={args.channels}x{args.blocks} ===",
    )
    if not stockfish_available():
        raise SystemExit("Stockfish required as OFFLINE teacher / ladder opponent only")

    data_path = out / "swarm_dataset.npz"
    net_path = out / "swarm_net.pt"
    sf = StockfishEngine(StockfishConfig(threads=2, hash_mb=256))

    try:
        weights_path = out / "latest.json" if (out / "latest.json").exists() else Path(
            "chess_contest/weights/base_weights.json"
        )
        weights = load_weights(weights_path)
        weights.format_version = 4
        weights.training_meta = {
            **weights.training_meta,
            "oracle_runtime": False,
            "stockfish_at_play": False,
            "precision": "float64",
            "swarm_net": str(net_path),
        }

        if args.probe_only:
            net = SwarmNet(channels=args.channels, blocks=args.blocks)
            if not net_path.is_file():
                raise SystemExit(f"missing {net_path}")
            net.load(net_path)
            set_swarm(net)
            engine = StigmergyEngine(weights, load_swarm=False)
            set_swarm(net)
            probe = pure_ladder(
                engine,
                sf,
                log,
                games_per=6,
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
            )
            (out / "elo_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
            _log(log, f"PROBE Elo ≈ {probe['estimated_elo']} gm={probe['gm']}")
            return 0

        if args.skip_gen and data_path.is_file():
            blob = np.load(data_path)
            data = [
                (np.asarray(p, dtype=np.float32), int(m), float(v))
                for p, m, v in zip(
                    blob["planes"], blob["moves"].tolist(), blob["values"].tolist(), strict=True
                )
            ]
            _log(log, f"loaded dataset {len(data)}")
        else:
            data = generate_dataset(
                sf, log, n_positions=args.positions, depth=args.sf_depth, weights=weights
            )
            prune_trails(weights, keep_positions=250_000)
            np.savez_compressed(
                data_path,
                planes=np.stack([d[0] for d in data]),
                moves=np.asarray([d[1] for d in data], dtype=np.int32),
                values=np.asarray([d[2] for d in data], dtype=np.float32),
            )
            save_weights(weights, out / "latest.json")
            _log(log, f"saved dataset {data_path} trails={len(weights.trails)}")

        data = _mirror_augment(data)
        _log(log, f"train set size after mirror aug: {len(data)}")

        for cycle in range(1, args.cycles + 1):
            _log(log, f"=== cycle {cycle}/{args.cycles} ===")
            net = SwarmNet(channels=args.channels, blocks=args.blocks)
            if net_path.is_file() and cycle > 1:
                with contextlib.suppress(Exception):
                    net.load(net_path)
            train_swarm(net, data, log, epochs=args.epochs)
            extra = generate_dataset(
                sf,
                log,
                n_positions=max(5000, args.positions // 6),
                depth=args.sf_depth,
                weights=weights,
            )
            extra = _mirror_augment(extra)
            train_swarm(net, extra, log, epochs=max(3, args.epochs // 2))
            data = (data + extra)[-args.positions * 2 :]
            net.save(net_path)
            set_swarm(net)
            _assert_no_runtime_sf()
            prune_trails(weights, keep_positions=250_000)
            save_weights(weights, out / "latest.json")

            engine = StigmergyEngine(weights, load_swarm=False)
            set_swarm(net)
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
                    stig_ms=max(args.stig_ms, 25000),
                    stig_depth=max(args.stig_depth, 20),
                    targets=[2200, 2500, 2600, 2700, 2800, 3000],
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
                            "think_ms": max(args.stig_ms, 25000),
                        }
                    )
                    save_weights(weights, out / "gm_weights.json")
                    (out / "gm_report.json").write_text(
                        json.dumps(
                            {
                                "estimated_elo": confirm["estimated_elo"],
                                "ladder": confirm["ladder"],
                                "gm": True,
                                "above_gm": True,
                                "stockfish_at_play": False,
                                "oracle_runtime": False,
                                "think_ms": max(args.stig_ms, 25000),
                                "swarm": {"channels": args.channels, "blocks": args.blocks},
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    (out / "STATUS.md").write_text(
                        "# ABOVE GM — pure (no runtime Stockfish)\n\n"
                        f"Estimated Elo **{confirm['estimated_elo']}** "
                        f"(floor {args.gm_floor}).\n\n"
                        "`choose_move` never calls Stockfish. Offline SF teacher "
                        "distills float64 trails + swarm policy/value; play uses "
                        "trails/book + long-think IDAS + swarm root re-rank.\n\n"
                        f"Think budget confirm: {max(args.stig_ms, 25000)} ms / move.\n",
                        encoding="utf-8",
                    )
                    _log(log, "=== ABOVE-GM PURE CONFIRMED ===")
                    break
            if probe["estimated_elo"] < args.gm_floor:
                args.stig_ms = min(90_000, int(args.stig_ms * 1.6))
                args.stig_depth = min(28, args.stig_depth + 2)
                _log(log, f"bump think→{args.stig_ms}ms depth→{args.stig_depth}")
        else:
            _log(log, "finished cycles without floor — writing best-effort STATUS")
            (out / "STATUS.md").write_text(
                "# Pure path in progress (no runtime Stockfish)\n\n"
                "Runtime Stockfish play is disabled. Offline teacher + swarm + "
                "long-think search is iterating toward ≥2500 Elo.\n"
                "See `elo_probe.json` / `pure_gm.log`.\n",
                encoding="utf-8",
            )
    finally:
        sf.close()
        set_swarm(None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
