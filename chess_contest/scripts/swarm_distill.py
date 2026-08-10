"""Distill SF-MAX into the swarm policy/value net, then probe Elo."""

from __future__ import annotations

import argparse
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


def collect_batch(sf: StockfishEngine, n: int, depth: int) -> list[tuple[np.ndarray, int, float]]:
    """Self-play SF-MAX lines; label each root with top1 + soft value."""
    out: list[tuple[np.ndarray, int, float]] = []
    board = chess.Board()
    while len(out) < n:
        if board.is_game_over(claim_draw=True) or len(board.move_stack) > 80:
            board = chess.Board()
            # Randomize opening with a few legal moves.
            for _ in range(random.randint(0, 6)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(random.choice(moves))
            continue
        sf.set_elo(None)
        tops = sf.analyse_top(board, multipv=1, depth=depth)
        if not tops:
            board.push(random.choice(list(board.legal_moves)))
            continue
        uci = tops[0]["uci"]
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            board = chess.Board()
            continue
        if mv not in board.legal_moves:
            board = chess.Board()
            continue
        flip = board.turn == chess.BLACK
        planes = encode_board(board)
        idx = move_index(mv, flip=flip)
        cp = tops[0].get("cp")
        mate = tops[0].get("mate")
        if mate is not None:
            val = 1.0 if mate > 0 else -1.0
        elif cp is not None:
            val = float(np.tanh(cp / 400.0))
        else:
            val = 0.0
        out.append((planes, idx, val))
        board.push(mv)
        # Occasional offbeat reply to diversify.
        if random.random() < 0.35 and not board.is_game_over(claim_draw=True):
            replies = list(board.legal_moves)
            board.push(random.choice(replies))
    return out


def train_loop(net: SwarmNet, sf: StockfishEngine, log: Path, *, steps: int, depth: int) -> None:
    import torch as T
    import torch.nn.functional as F

    opt = T.optim.Adam(net.net.parameters(), lr=1e-3)
    net.net.train()
    for step in range(1, steps + 1):
        batch = collect_batch(sf, n=32, depth=depth)
        xs = T.from_numpy(np.stack([b[0] for b in batch]))
        ys = T.tensor([b[1] for b in batch], dtype=T.long)
        vs = T.tensor([[b[2]] for b in batch], dtype=T.float32)
        logits, pred_v = net.net(xs)
        loss_p = F.cross_entropy(logits, ys)
        loss_v = F.mse_loss(pred_v, vs)
        loss = loss_p + 0.5 * loss_v
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == 1:
            _log(log, f"train step={step}/{steps} loss={float(loss.detach()):.4f} p={float(loss_p.detach()):.4f}")
    net.net.eval()


def ladder(engine: StigmergyEngine, sf: StockfishEngine, log: Path, games_per: int) -> dict:
    our = 2000.0
    rows = []
    for target in [1320, 1600, 1900, 2200, 2500, 2600, 2800]:
        sc = 0.0
        for i in range(games_per):
            board = chess.Board()
            stig_w = i % 2 == 0
            for _ in range(110):
                if board.is_game_over(claim_draw=True):
                    break
                if (board.turn == chess.WHITE) == stig_w:
                    board.push(engine.choose_move(board, time_ms=400, max_depth=7).move)
                else:
                    sf.set_elo(target)
                    board.push(sf.choose(board, movetime_ms=50))
            res = board.result(claim_draw=True)
            if board.is_checkmate():
                res = "0-1" if board.turn == chess.WHITE else "1-0"
            s = 1.0 if (res == "1-0" and stig_w) or (res == "0-1" and not stig_w) else (
                0.5 if res == "1/2-1/2" else 0.0
            )
            if res not in ("1-0", "0-1", "1/2-1/2"):
                s = 0.5
            sc += s
            our, _ = update_elo(our, float(target), s, k=32.0)
        wr = sc / games_per
        rows.append({"sf_elo": target, "score": sc, "games": games_per, "winrate": wr, "our_elo_after": round(our, 1)})
        _log(log, f"ladder vs SF {target}: {sc}/{games_per} ({wr:.0%}) → ≈{our:.0f}")
        if target >= 2500 and wr < 0.2 and our < 2400:
            break
        if target >= 2500 and our >= 2500:
            break
    return {"estimated_elo": round(our, 1), "ladder": rows, "gm": our >= 2500.0}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--gm-floor", type=float, default=2500.0)
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    log = out / "swarm_train.log"
    _log(log, f"=== SWARM distill steps={args.steps} ===")
    if not stockfish_available():
        raise SystemExit("Stockfish required")

    net = SwarmNet(channels=48)
    sf = StockfishEngine(StockfishConfig(threads=2, hash_mb=128))
    try:
        train_loop(net, sf, log, steps=args.steps, depth=args.depth)
        net_path = out / "swarm_net.pt"
        net.save(net_path)
        _log(log, f"saved {net_path}")

        weights = load_weights(out / "latest.json" if (out / "latest.json").exists() else "chess_contest/weights/base_weights.json")
        weights.training_meta = {**weights.training_meta, "swarm_net": str(net_path), "swarm": True}
        engine = StigmergyEngine(weights)
        engine.swarm = net  # type: ignore[attr-defined]

        # Wire swarm into search module global for choose_move path.
        from chess_contest.stigmergy import search as search_mod

        search_mod._SWARM = net  # type: ignore[attr-defined]

        probe = ladder(engine, sf, log, games_per=6)
        (out / "elo_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
        _log(log, f"Elo probe ≈ {probe['estimated_elo']} gm={probe['gm']}")
        if probe["estimated_elo"] >= args.gm_floor:
            confirm = ladder(engine, sf, log, games_per=8)
            weights.training_meta["confirm"] = confirm
            weights.training_meta["gm"] = True
            save_weights(weights, out / "gm_weights.json")
            save_weights(weights, Path("chess_contest/weights/gm_weights.json"))
            save_weights(weights, Path("chess_contest/weights/base_weights.json"))
            _log(log, "=== GRANDMASTER FLOOR CONFIRMED ===")
        else:
            save_weights(weights, out / "latest.json")
    finally:
        sf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
