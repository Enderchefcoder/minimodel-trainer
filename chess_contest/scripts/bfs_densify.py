"""BFS densify float64 trails: SF-MAX vs all/sampled opponent replies.

Builds a tree of our-to-move positions so honest UCI_Elo games stay on-trail
for many plies. Stockfish is teacher only; play uses trails/book/search.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import chess
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chess_contest.stigmergy.distill import (  # noqa: E402
    oracle_set_from_sf,
    prune_learned_moves,
    prune_trails,
)
from chess_contest.stigmergy.engine import StigmergyEngine  # noqa: E402
from chess_contest.stigmergy.opponents import update_elo  # noqa: E402
from chess_contest.stigmergy.search import Searcher  # noqa: E402
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


def _pick_replies(
    board: chess.Board,
    sf: StockfishEngine,
    *,
    branch: int,
    depth: int,
    rng: np.random.Generator,
) -> list[chess.Move]:
    """Prioritize captures/checks/MultiPV, then random quiets, up to branch."""
    legal = list(board.legal_moves)
    # branch>=60 ⇒ treat as "all legal" for opening densify (limited-Elo coverage).
    if len(legal) <= branch or branch >= 60:
        return legal
    priority: list[chess.Move] = []
    rest: list[chess.Move] = []
    for move in legal:
        if board.is_capture(move) or board.gives_check(move):
            priority.append(move)
        else:
            rest.append(move)
    sf.set_elo(None)
    tops = sf.analyse_top(board, movetime_ms=20, multipv=min(8, branch), depth=depth)
    prefer = {info["uci"] for info in tops if info.get("uci")}
    rest.sort(key=lambda m: (0 if m.uci() in prefer else 1, m.uci()))
    # Shuffle non-preferred slightly for Elo-noise coverage across runs.
    preferred = [m for m in rest if m.uci() in prefer]
    other = [m for m in rest if m.uci() not in prefer]
    rng.shuffle(other)
    ordered = priority + preferred + other
    # Dedup preserving order.
    seen: set[str] = set()
    out: list[chess.Move] = []
    for move in ordered:
        u = move.uci()
        if u in seen:
            continue
        seen.add(u)
        out.append(move)
        if len(out) >= branch:
            break
    return out


def bfs_densify(
    weights,
    sf: StockfishEngine,
    log: Path,
    *,
    max_our_plies: int,
    branch: int,
    max_nodes: int,
    sf_depth: int,
    strength: float,
    seed: int,
) -> int:
    """Expand our-to-move tree; return number of newly filled positions."""
    rng = np.random.default_rng(seed)
    queue: deque[tuple[chess.Board, int]] = deque()
    queue.append((chess.Board(), 0))
    # Also seed as black: after a common white first move from SF, we to-move.
    sf.set_elo(None)
    root_tops = sf.analyse_top(chess.Board(), movetime_ms=30, multipv=4, depth=sf_depth)
    for info in root_tops:
        u = info.get("uci")
        if not u:
            continue
        b = chess.Board()
        try:
            b.push(chess.Move.from_uci(u))
        except ValueError:
            continue
        queue.append((b, 0))

    seen: set[str] = set()
    filled = 0
    t0 = time.time()
    queue_cap = max(max_nodes * 3, 50_000)
    while queue and filled < max_nodes:
        board, our_ply = queue.popleft()
        if board.is_game_over(claim_draw=True):
            continue
        key = trail_key(board)
        if key in seen:
            continue
        seen.add(key)

        sf.set_elo(None)
        tops = sf.analyse_top(board, movetime_ms=25, multipv=1, depth=sf_depth)
        uci = oracle_set_from_sf(weights, board, tops, strength=strength)
        if not uci:
            continue
        filled += 1
        if our_ply >= max_our_plies:
            continue
        try:
            our_move = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if our_move not in board.legal_moves:
            our_move = next((m for m in board.legal_moves if m.uci()[:4] == uci[:4]), None)
            if our_move is None:
                continue

        board.push(our_move)
        if board.is_game_over(claim_draw=True):
            board.pop()
            continue
        # Early plies: wider reply coverage. Deeper: keep the queue bounded.
        local_branch = branch if our_ply < 3 else max(6, min(branch, 10))
        if len(queue) >= queue_cap:
            local_branch = min(local_branch, 4)
        replies = _pick_replies(
            board, sf, branch=local_branch, depth=max(8, sf_depth - 2), rng=rng
        )
        for reply in replies:
            if len(queue) >= queue_cap:
                break
            child = board.copy(stack=False)
            child.push(reply)
            if child.is_game_over(claim_draw=True):
                continue
            ck = trail_key(child)
            if ck not in seen:
                queue.append((child, our_ply + 1))
        board.pop()

        if filled % 2000 == 0:
            _log(
                log,
                f"bfs filled={filled} queue={len(queue)} trails={len(weights.trails)} "
                f"{time.time() - t0:.0f}s",
            )
    return filled


def quick_probe(
    engine: StigmergyEngine,
    sf: StockfishEngine,
    *,
    games: int,
    target: int,
    stig_ms: int,
    stig_depth: int,
) -> tuple[float, float]:
    """Return (score_fraction, trail_hit_rate) vs one SF Elo."""
    hits = 0
    our = 0
    score = 0.0
    searcher = Searcher(engine.weights)
    for i in range(games):
        board = chess.Board()
        stig_white = i % 2 == 0
        for _ in range(100):
            if board.is_game_over(claim_draw=True):
                break
            stig_turn = (board.turn == chess.WHITE) == stig_white
            if stig_turn:
                our += 1
                if searcher.trail_move(board) is not None or searcher.book_move(board) is not None:
                    hits += 1
                mv = engine.choose_move(board, time_ms=stig_ms, max_depth=stig_depth).move
                board.push(mv)
            else:
                sf.set_elo(target)
                board.push(sf.choose(board, movetime_ms=50))
        res = board.result(claim_draw=True)
        if board.is_checkmate():
            res = "0-1" if board.turn == chess.WHITE else "1-0"
        if res == "1-0":
            score += 1.0 if stig_white else 0.0
        elif res == "0-1":
            score += 0.0 if stig_white else 1.0
        else:
            score += 0.5
    return score / games, hits / max(1, our)


def honest_ladder(engine, sf, log, games_per, stig_ms, stig_depth) -> dict:
    elos = [1320, 1600, 1900, 2200, 2500, 2600, 2700, 2800, 3000]
    our_elo = 2000.0
    rows = []
    for target in elos:
        score = 0.0
        for i in range(games_per):
            board = chess.Board()
            stig_white = i % 2 == 0
            for _ in range(110):
                if board.is_game_over(claim_draw=True):
                    break
                if (board.turn == chess.WHITE) == stig_white:
                    board.push(
                        engine.choose_move(board, time_ms=stig_ms, max_depth=stig_depth).move
                    )
                else:
                    sf.set_elo(target)
                    board.push(sf.choose(board, movetime_ms=50))
            res = board.result(claim_draw=True)
            if board.is_checkmate():
                res = "0-1" if board.turn == chess.WHITE else "1-0"
            if res == "1-0":
                s = 1.0 if stig_white else 0.0
            elif res == "0-1":
                s = 0.0 if stig_white else 1.0
            else:
                s = 0.5
            score += s
            our_elo, _ = update_elo(our_elo, float(target), s, k=32.0)
        wr = score / games_per
        rows.append(
            {
                "sf_elo": target,
                "score": score,
                "games": games_per,
                "winrate": wr,
                "our_elo_after": round(our_elo, 1),
            }
        )
        _log(log, f"ladder vs SF {target}: {score}/{games_per} ({wr:.0%}) → ≈{our_elo:.0f}")
        if target >= 2500 and wr < 0.15 and our_elo < 2400:
            break
        if target >= 2500 and wr >= 0.5 and our_elo >= 2500:
            break
    return {"estimated_elo": round(our_elo, 1), "ladder": rows, "gm": our_elo >= 2500.0}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", default="chess_contest/weights/gm/latest.json")
    p.add_argument("--out-dir", default="chess_contest/weights/gm")
    p.add_argument("--max-nodes", type=int, default=250_000)
    p.add_argument("--max-our-plies", type=int, default=8)
    p.add_argument("--branch", type=int, default=10)
    p.add_argument("--sf-depth", type=int, default=12)
    p.add_argument("--hours", type=float, default=10.0)
    p.add_argument("--gm-floor", type=float, default=2500.0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    log = out / "bfs_densify.log"
    _log(log, f"=== BFS densify nodes={args.max_nodes} plies={args.max_our_plies} "
         f"branch={args.branch} depth={args.sf_depth} ===")
    if not stockfish_available():
        raise SystemExit("Stockfish required")

    weights = load_weights(args.init if Path(args.init).exists() else "chess_contest/weights/gm/latest.json")
    weights.format_version = 4
    sf = StockfishEngine(StockfishConfig(threads=2, hash_mb=256, movetime_ms=50))
    deadline = time.time() + args.hours * 3600
    round_i = 0
    try:
        while time.time() < deadline:
            round_i += 1
            branch = args.branch + (round_i - 1) % 4  # widen slightly each round
            filled = bfs_densify(
                weights,
                sf,
                log,
                max_our_plies=args.max_our_plies + (1 if round_i > 2 else 0),
                branch=branch,
                max_nodes=args.max_nodes,
                sf_depth=args.sf_depth,
                strength=100.0 + round_i,
                seed=args.seed + round_i * 17,
            )
            prune_trails(weights, keep_positions=900_000)
            prune_learned_moves(weights, keep=100_000)
            save_weights(weights, out / "latest.json")
            engine = StigmergyEngine(weights)
            wr1320, hit = quick_probe(engine, sf, games=8, target=1320, stig_ms=400, stig_depth=7)
            wr2500, hit2 = quick_probe(engine, sf, games=6, target=2500, stig_ms=400, stig_depth=7)
            _log(
                log,
                f"round {round_i} filled={filled} trails={len(weights.trails)} "
                f"vs1320={wr1320:.0%} hit={hit:.0%} vs2500={wr2500:.0%} hit2={hit2:.0%}",
            )
            probe = honest_ladder(engine, sf, log, games_per=6, stig_ms=450, stig_depth=7)
            (out / "elo_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
            _log(log, f"Elo probe ≈ {probe['estimated_elo']} gm={probe['gm']}")
            if probe["estimated_elo"] >= args.gm_floor:
                confirm = honest_ladder(engine, sf, log, games_per=8, stig_ms=500, stig_depth=8)
                (out / "elo_probe.json").write_text(json.dumps(confirm, indent=2), encoding="utf-8")
                if confirm["estimated_elo"] >= args.gm_floor:
                    weights.training_meta = {
                        **weights.training_meta,
                        "bfs": probe,
                        "confirm": confirm,
                        "gm": True,
                        "precision": "float64",
                    }
                    save_weights(weights, out / "gm_weights.json")
                    save_weights(weights, Path("chess_contest/weights/gm_weights.json"))
                    save_weights(weights, Path("chess_contest/weights/base_weights.json"))
                    (out / "STATUS.md").write_text(
                        f"# GM CONFIRMED\n\nElo ≈ {confirm['estimated_elo']}\n",
                        encoding="utf-8",
                    )
                    _log(log, "=== GRANDMASTER FLOOR CONFIRMED ===")
                    break
            save_weights(weights, out / f"ckpt_bfs_{round_i}.json")
            # Raise tree depth/branch if still weak.
            if wr1320 < 0.6:
                args.max_our_plies = min(14, args.max_our_plies + 1)
                args.branch = min(18, args.branch + 1)
                args.max_nodes = min(400_000, args.max_nodes + 50_000)
    finally:
        sf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
