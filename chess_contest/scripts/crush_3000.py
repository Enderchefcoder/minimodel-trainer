"""Crush 3000-Elo neural competitors - offline SF teacher only.

Play never calls Stockfish. Pipeline:
1) Generate a large MultiPV soft-label set with full-strength SF
2) Train SwarmNet v2 (field-aware residual tower)
3) Distill decisive trails
4) Long-think honest ladder vs SF UCI_Elo 2200-3000
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
    IN_CH,
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


def generate_dataset(
    sf: StockfishEngine,
    log: Path,
    *,
    n_positions: int,
    depth: int,
    weights=None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Return (planes, soft_policy_4096, value) rows from full-strength SF."""
    data: list[tuple[np.ndarray, np.ndarray, float]] = []
    board = chess.Board()
    t0 = time.time()
    while len(data) < n_positions:
        if board.is_game_over(claim_draw=True) or len(board.move_stack) > 100:
            board = chess.Board()
            # Diverse starts: random walk + occasional from FEN-like noise.
            for _ in range(random.randint(0, 16)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(random.choice(moves))
            continue
        sf.set_elo(None)  # full strength teacher
        tops = sf.analyse_top(board, multipv=5, depth=depth)
        if not tops or not tops[0].get("uci"):
            board = chess.Board()
            continue
        try:
            best = chess.Move.from_uci(tops[0]["uci"])
        except ValueError:
            board = chess.Board()
            continue
        if best not in board.legal_moves:
            board = chess.Board()
            continue

        flip = board.turn == chess.BLACK
        planes = encode_board(board)
        soft = np.zeros(4096, dtype=np.float32)
        # Soft labels from MultiPV cp gaps (softmax over teacher scores).
        scores: list[tuple[chess.Move, float]] = []
        for row in tops:
            with contextlib.suppress(ValueError, KeyError):
                mv = chess.Move.from_uci(row["uci"])
                if mv not in board.legal_moves:
                    continue
                mate = row.get("mate")
                cp = row.get("cp")
                if mate is not None:
                    sc = 5000.0 if int(mate) > 0 else -5000.0
                elif cp is not None:
                    sc = float(cp)
                else:
                    continue
                scores.append((mv, sc))
        if not scores:
            board = chess.Board()
            continue
        # Softmax over teacher scores (temperature 120cp).
        arr = np.asarray([s for _, s in scores], dtype=np.float64)
        arr = arr - arr.max()
        prob = np.exp(arr / 120.0)
        prob /= prob.sum()
        for (mv, _), p in zip(scores, prob, strict=True):
            soft[move_index(mv, flip=flip)] = float(p)

        mate0 = tops[0].get("mate")
        cp0 = tops[0].get("cp")
        if mate0 is not None:
            val = 1.0 if int(mate0) > 0 else -1.0
        elif cp0 is not None:
            val = float(np.tanh(float(cp0) / 400.0))
        else:
            val = 0.0
        data.append((planes, soft, val))
        if weights is not None:
            set_trail_policy(
                weights, board, best.uci()[:4], strength=50.0 + min(40.0, abs(val) * 40)
            )

        board.push(best)
        # Branch for diversity.
        if random.random() < 0.5 and not board.is_game_over(claim_draw=True):
            if random.random() < 0.55:
                board.push(random.choice(list(board.legal_moves)))
            else:
                alts = sf.analyse_top(board, multipv=4, depth=max(6, depth - 3))
                if alts:
                    with contextlib.suppress(ValueError, KeyError):
                        board.push(chess.Move.from_uci(random.choice(alts)["uci"]))
        if len(data) % 1000 == 0:
            _log(log, f"dataset {len(data)}/{n_positions} {time.time() - t0:.0f}s")
    return data


def train_swarm(
    net: SwarmNet,
    data: list[tuple[np.ndarray, np.ndarray, float]],
    log: Path,
    *,
    epochs: int,
    batch_size: int = 48,
) -> None:
    import torch as T
    import torch.nn.functional as F

    opt = T.optim.AdamW(net.net.parameters(), lr=1.0e-3, weight_decay=1e-4)
    sched = T.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    net.net.train()
    n = len(data)
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(n)
        total = 0.0
        top1 = 0
        seen = 0
        steps = 0
        for i in range(0, n, batch_size):
            batch_idx = order[i : i + batch_size]
            if len(batch_idx) < 8:
                continue
            xs = T.from_numpy(np.stack([data[j][0] for j in batch_idx]))
            soft = T.from_numpy(np.stack([data[j][1] for j in batch_idx]))
            vs = T.tensor([[data[j][2]] for j in batch_idx], dtype=T.float32)
            logits, pred_v = net.net(xs)
            log_probs = F.log_softmax(logits, dim=-1)
            # Soft cross-entropy to MultiPV teacher distribution.
            policy_loss = -(soft * log_probs).sum(dim=-1).mean()
            loss = policy_loss + 0.5 * F.mse_loss(pred_v, vs)
            opt.zero_grad()
            loss.backward()
            T.nn.utils.clip_grad_norm_(net.net.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            hard = soft.argmax(dim=-1)
            top1 += int((logits.argmax(dim=-1) == hard).sum().item())
            seen += len(batch_idx)
            steps += 1
        sched.step()
        _log(
            log,
            f"epoch {epoch}/{epochs} loss={total / max(1, steps):.4f} "
            f"top1={top1 / max(1, seen):.3f}",
        )
    net.net.eval()
    net._logit_cache.clear()


def _score(stig_white: bool, res: str) -> float:
    if res == "1-0":
        return 1.0 if stig_white else 0.0
    if res == "0-1":
        return 0.0 if stig_white else 1.0
    return 0.5


def crush_ladder(
    engine: StigmergyEngine,
    sf: StockfishEngine,
    log: Path,
    *,
    games_per: int,
    stig_ms: int,
    stig_depth: int,
) -> dict:
    our = 2200.0
    rows = []
    for target in [2200, 2500, 2600, 2700, 2800, 2900, 3000, 3100]:
        ms = stig_ms if target >= 2500 else min(stig_ms, 12000)
        depth = stig_depth if target >= 2500 else min(stig_depth, 16)
        sc = 0.0
        for i in range(games_per):
            board = chess.Board()
            stig_white = i % 2 == 0
            for _ in range(160):
                if board.is_game_over(claim_draw=True):
                    break
                if (board.turn == chess.WHITE) == stig_white:
                    mv = engine.choose_move(board, time_ms=ms, max_depth=depth).move
                    if mv is None:
                        break
                    board.push(mv)
                else:
                    sf.set_elo(target)
                    board.push(sf.choose(board, movetime_ms=100))
            res = board.result(claim_draw=True)
            if board.is_checkmate():
                res = "0-1" if board.turn == chess.WHITE else "1-0"
            s = _score(stig_white, res)
            sc += s
            our, _ = update_elo(our, float(target), s, k=24.0)
            _log(log, f"  game vs {target} #{i} → {res} s={s} (running≈{our:.0f})")
        wr = sc / games_per
        rows.append(
            {
                "sf_elo": target,
                "score": sc,
                "games": games_per,
                "winrate": wr,
                "our_elo_after": round(our, 1),
                "think_ms": ms,
            }
        )
        _log(log, f"CRUSH ladder vs SF {target}: {sc}/{games_per} ({wr:.0%}) → ≈{our:.0f} [{ms}ms]")
        if target >= 2800 and wr >= 0.45 and our >= 2800:
            # Keep climbing toward 3000.
            pass
        if target >= 3000 and our >= 3000 and wr >= 0.4:
            break
        if target >= 2700 and wr < 0.1 and our < 2400:
            break
    return {
        "estimated_elo": round(our, 1),
        "ladder": rows,
        "gm": our >= 2500,
        "above_gm": our >= 2500,
        "crush_3000": our >= 3000,
        "stockfish_at_play": False,
        "oracle_runtime": False,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--positions", type=int, default=120000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--sf-depth", type=int, default=12)
    p.add_argument("--stig-ms", type=int, default=20000)
    p.add_argument("--stig-depth", type=int, default=18)
    p.add_argument("--floor", type=float, default=2800.0)
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--channels", type=int, default=192)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--skip-gen", action="store_true")
    p.add_argument("--probe-only", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    log = out / "crush_3000.log"
    _log(
        log,
        f"=== CRUSH-3000 pos={args.positions} ep={args.epochs} "
        f"net={args.channels}x{args.blocks} in={IN_CH} "
        f"think={args.stig_ms}ms ===",
    )
    if not stockfish_available():
        raise SystemExit("Stockfish required as OFFLINE teacher / ladder opponent")

    data_path = out / "crush_dataset.npz"
    net_path = out / "swarm_net.pt"
    sf = StockfishEngine(StockfishConfig(threads=2, hash_mb=512))

    try:
        wp = out / "latest.json" if (out / "latest.json").exists() else Path(
            "chess_contest/weights/base_weights.json"
        )
        weights = load_weights(wp)
        weights.format_version = 4
        weights.training_meta.update(
            {
                "oracle_runtime": False,
                "stockfish_at_play": False,
                "swarm_net": str(net_path),
                "crush_3000": True,
            }
        )

        if args.probe_only:
            net = try_or_load(net_path, args.channels, args.blocks)
            set_swarm(net)
            eng = StigmergyEngine(weights, load_swarm=False)
            set_swarm(net)
            probe = crush_ladder(
                eng, sf, log, games_per=6, stig_ms=args.stig_ms, stig_depth=args.stig_depth
            )
            write_reports(out, probe, args)
            return 0

        if args.skip_gen and data_path.is_file():
            blob = np.load(data_path)
            data = [
                (np.asarray(p, dtype=np.float32), np.asarray(s, dtype=np.float32), float(v))
                for p, s, v in zip(
                    blob["planes"], blob["soft"], blob["values"], strict=True
                )
            ]
            _log(log, f"loaded crush dataset {len(data)}")
        else:
            data = generate_dataset(
                sf, log, n_positions=args.positions, depth=args.sf_depth, weights=weights
            )
            prune_trails(weights, keep_positions=300_000)
            np.savez_compressed(
                data_path,
                planes=np.stack([d[0] for d in data]),
                soft=np.stack([d[1] for d in data]),
                values=np.asarray([d[2] for d in data], dtype=np.float32),
            )
            save_weights(weights, out / "latest.json")
            _log(log, f"saved {data_path} trails={len(weights.trails)}")

        for cycle in range(1, args.cycles + 1):
            _log(log, f"=== crush cycle {cycle}/{args.cycles} ===")
            net = SwarmNet(channels=args.channels, blocks=args.blocks, in_ch=IN_CH)
            if net_path.is_file():
                with contextlib.suppress(Exception):
                    net.load(net_path)
                    _log(log, f"warm-started from {net_path}")
            train_swarm(net, data, log, epochs=args.epochs)
            # Fresh teacher batch each cycle.
            extra = generate_dataset(
                sf,
                log,
                n_positions=max(8000, args.positions // 8),
                depth=args.sf_depth,
                weights=weights,
            )
            train_swarm(net, extra, log, epochs=max(2, args.epochs // 2))
            data = (data + extra)[-args.positions * 2 :]
            net.save(net_path)
            set_swarm(net)
            prune_trails(weights, keep_positions=300_000)
            save_weights(weights, out / "latest.json")

            eng = StigmergyEngine(weights, load_swarm=False)
            set_swarm(net)
            probe = crush_ladder(
                eng,
                sf,
                log,
                games_per=6 if cycle < args.cycles else 8,
                stig_ms=args.stig_ms,
                stig_depth=args.stig_depth,
            )
            write_reports(out, probe, args)
            _log(log, f"CRUSH Elo ≈ {probe['estimated_elo']} crush3000={probe['crush_3000']}")
            if probe["estimated_elo"] >= args.floor:
                # Confirm with longer think.
                confirm = crush_ladder(
                    eng,
                    sf,
                    log,
                    games_per=8,
                    stig_ms=max(args.stig_ms, 45000),
                    stig_depth=max(args.stig_depth, 20),
                )
                write_reports(out, confirm, args)
                _log(log, f"CONFIRM ≈ {confirm['estimated_elo']}")
                if confirm["estimated_elo"] >= args.floor:
                    weights.training_meta["pure_gm"] = confirm
                    weights.training_meta["gm"] = True
                    weights.training_meta["crush_3000"] = confirm["estimated_elo"] >= 3000
                    save_weights(weights, out / "gm_weights.json")
                    _log(log, "=== CRUSH FLOOR REACHED ===")
                    break
            args.stig_ms = min(120_000, int(args.stig_ms * 1.4))
            args.stig_depth = min(28, args.stig_depth + 2)
            _log(log, f"bump think→{args.stig_ms}ms depth→{args.stig_depth}")
        else:
            _log(log, "cycles finished without floor")
    finally:
        sf.close()
        set_swarm(None)
    return 0


def try_or_load(path: Path, channels: int, blocks: int) -> SwarmNet:
    net = SwarmNet(channels=channels, blocks=blocks, in_ch=IN_CH)
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    net.load(path)
    return net


def write_reports(out: Path, probe: dict, args) -> None:
    (out / "elo_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    (out / "gm_report.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    elo = probe["estimated_elo"]
    (out / "STATUS.md").write_text(
        "# Crush path - no runtime Stockfish\n\n"
        f"Estimated Elo **{elo}** "
        f"(floor target {args.floor}; crush-3000={probe.get('crush_3000')}).\n\n"
        "`choose_move` never calls Stockfish. SwarmNet v2 distilled from "
        "full-strength SF offline; policy-first play + long-think IDAS.\n\n"
        f"Think budget: {args.stig_ms} ms / move.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
