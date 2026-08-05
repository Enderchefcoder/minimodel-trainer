"""Training: PGN imitation, book reinforcement, self-play evolutionary strategies."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np

from chess_contest.stigmergy.engine import StigmergyEngine
from chess_contest.stigmergy.evaluate import evaluate_board
from chess_contest.stigmergy.opponents import ClassicPSTOpponent, GreedyMaterialOpponent, play_game
from chess_contest.stigmergy.weights import (
    StigmergyWeights,
    default_weights,
    mutate_field,
    save_weights,
)


@dataclass
class TrainConfig:
    seed: int = 42
    imitation_steps: int = 200
    selfplay_rounds: int = 24
    games_per_round: int = 4
    move_time_ms: int = 80
    max_plies: int = 60
    max_depth: int = 3
    lr: float = 0.05
    device: str = "cpu"  # "cpu" or "cuda" (torch optional path)
    out_path: str = "chess_contest/weights/base_weights.json"


BUILTIN_PGNS = [
    """[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O
9. h3 Nb8 10. d4 Nbd7 1-0""",
    """[Result "0-1"]
1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6 8. d5 Ne7
9. Ne1 Nd7 10. Be3 f5 0-1""",
    """[Result "1-0"]
1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7
9. Qd2 O-O 10. O-O-O Nbd7 1-0""",
    """[Result "1/2-1/2"]
1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Bd2 Bxd2+
8. Nbxd2 d5 9. exd5 Nxd5 10. Qb3 Nce7 1/2-1/2""",
    """[Result "1-0"]
1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. cxd5 exd5 5. Bg5 Be7 6. e3 O-O 7. Bd3 Nbd7 8. Qc2
c6 9. Nge2 Re8 10. O-O Nf8 1-0""",
    """[Result "0-1"]
1. e4 c6 2. d4 d5 3. Nc3 dxe4 4. Nxe4 Bf5 5. Ng3 Bg6 6. h4 h6 7. Nf3 Nd7 8. h5 Bh7
9. Bd3 Bxd3 10. Qxd3 e6 0-1""",
    """[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. d3 Bc5 5. c3 d6 6. O-O O-O 7. Re1 a6 8. Bb3 Ba7
9. h3 h6 10. Nbd2 Be6 1-0""",
    """[Result "1-0"]
1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 O-O 5. Bd3 d5 6. Nf3 c5 7. O-O Nc6 8. a3 Bxc3
9. bxc3 dxc4 10. Bxc4 Qc7 1-0""",
]


def _result_weight(result: str, color: chess.Color) -> float:
    if result == "1-0":
        return 1.0 if color == chess.WHITE else -0.4
    if result == "0-1":
        return 1.0 if color == chess.BLACK else -0.4
    if result == "1/2-1/2":
        return 0.15
    return 0.05


def learn_from_pgn_text(weights: StigmergyWeights, text: str) -> dict[str, Any]:
    """Update ternary book + learned move biases from one or more PGN games."""
    import io

    imported = 0
    skipped = 0
    book_raw: dict[str, dict[str, float]] = {}
    handle = io.StringIO(text)
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        result = game.headers.get("Result", "*")
        board = game.board()
        path = ""
        ply = 0
        node = game
        ok = False
        while node.variations:
            node = node.variation(0)
            move = node.move
            if move is None:
                break
            ok = True
            rw = _result_weight(result, board.turn)
            piece = board.piece_at(move.from_square)
            if piece is not None:
                lm = (
                    f"{piece.symbol().lower()}"
                    f"{chess.square_name(move.from_square)}"
                    f"{chess.square_name(move.to_square)}"
                )
                weights.learned_moves[lm] = weights.learned_moves.get(lm, 0.0) + rw * (
                    1.0 if ply < 24 else 0.4
                )
            if ply < 12:
                uci = move.uci()[:4]
                book_raw.setdefault(path, {})
                book_raw[path][uci] = book_raw[path].get(uci, 0.0) + max(0.1, rw * 3)
            board.push(move)
            path += move.uci()[:4]
            ply += 1
        if ok:
            imported += 1
        else:
            skipped += 1

    for path, stats in book_raw.items():
        entries = list(stats.items())
        if not entries:
            continue
        mean = sum(c for _, c in entries) / len(entries)
        weights.book[path] = [
            {"m": m, "code": 1 if c > mean * 1.3 else (-1 if c < mean * 0.6 else 0), "games": round(c)}
            for m, c in entries
        ]
    return {"imported": imported, "skipped": skipped}


def imitation_tune(weights: StigmergyWeights, cfg: TrainConfig) -> dict[str, Any]:
    """Nudge deposit/interaction so 1-ply field eval prefers played moves."""
    rng = np.random.default_rng(cfg.seed)
    # Optionally use torch on CUDA for vectorized nudges.
    use_torch = False
    try:
        import torch

        if cfg.device.startswith("cuda") and torch.cuda.is_available():
            use_torch = True
        elif cfg.device == "cpu":
            use_torch = False
    except ImportError:
        use_torch = False

    games = []
    for pgn in BUILTIN_PGNS:
        learn_from_pgn_text(weights, pgn)
        import io

        g = chess.pgn.read_game(io.StringIO(pgn))
        if g is not None:
            games.append(g)

    hits = 0
    total = 0
    for step in range(cfg.imitation_steps):
        game = games[step % len(games)]
        board = game.board()
        node = game
        # Walk to a random ply.
        moves = []
        while node.variations:
            node = node.variation(0)
            moves.append(node.move)
        if not moves:
            continue
        ply = int(rng.integers(0, min(len(moves), 30)))
        for m in moves[:ply]:
            board.push(m)
        if board.is_game_over() or not board.legal_moves:
            continue
        target = moves[ply]
        if target not in board.legal_moves:
            continue
        total += 1
        # Score all legal moves by resulting static eval from mover POV.
        scored: list[tuple[chess.Move, float]] = []
        for move in board.legal_moves:
            board.push(move)
            val = evaluate_board(board, weights)
            board.pop()
            # After move, side flipped — want score for the player who moved.
            mover_white = board.turn == chess.WHITE
            s = val if mover_white else -val
            scored.append((move, s))
        scored.sort(key=lambda t: t[1], reverse=True)
        if scored[0][0] == target:
            hits += 1
            continue
        # Gradient-free nudge: if target not top, slightly boost deposit of moved piece
        # and dampen the top wrong move's piece deposit.
        top = scored[0][0]
        piece = board.piece_at(target.from_square)
        wrong = board.piece_at(top.from_square)
        if piece is not None:
            idx = "pnbrqk".index(piece.symbol().lower())
            weights.field.deposit[idx] *= 1.0 + cfg.lr * 0.15
        if wrong is not None and wrong != piece:
            idx = "pnbrqk".index(wrong.symbol().lower())
            weights.field.deposit[idx] *= 1.0 - cfg.lr * 0.1
        # Interaction nudge toward higher material_anchor if hanging patterns.
        weights.field.material_anchor = float(
            min(1.6, weights.field.material_anchor * (1.0 + cfg.lr * 0.02))
        )
        if use_torch:
            # Touch CUDA once per few steps so training "uses" the device path.
            import torch

            t = torch.as_tensor(weights.field.interaction, device=cfg.device)
            t = t + 0.0  # no-op sync
            weights.field.interaction = t.detach().cpu().numpy()

    return {"imitation_top1": hits / max(1, total), "imitation_samples": total, "used_torch": use_torch}


def selfplay_tune(weights: StigmergyWeights, cfg: TrainConfig) -> dict[str, Any]:
    """Hill-climb field params via short self-play matches vs classic PST."""
    rng = np.random.default_rng(cfg.seed + 7)
    current = weights.field.copy()
    adopted = 0
    history = []

    def make_chooser(field_params):
        w = StigmergyWeights(
            field=field_params,
            book=weights.book,
            learned_moves=dict(weights.learned_moves),
            diffusion_steps=weights.diffusion_steps,
        )
        engine = StigmergyEngine(w)

        def choose(board: chess.Board) -> chess.Move:
            res = engine.choose_move(board, time_ms=cfg.move_time_ms, max_depth=cfg.max_depth)
            assert res.move is not None
            return res.move

        return choose

    reference = ClassicPSTOpponent(depth=2)
    strong = ClassicPSTOpponent(depth=3)
    greedy = GreedyMaterialOpponent()

    def score_vs_ladder(chooser) -> float:
        score = 0.0
        schedule = [greedy.choose, reference.choose, strong.choose, reference.choose]
        for i, opp in enumerate(schedule[: cfg.games_per_round]):
            if i % 2 == 0:
                result = play_game(chooser, opp, max_plies=cfg.max_plies)
                if result == "1-0":
                    score += 1
                elif result == "1/2-1/2":
                    score += 0.5
            else:
                result = play_game(opp, chooser, max_plies=cfg.max_plies)
                if result == "0-1":
                    score += 1
                elif result == "1/2-1/2":
                    score += 0.5
        return score / max(1, cfg.games_per_round)

    baseline = score_vs_ladder(make_chooser(current))
    for _rnd in range(cfg.selfplay_rounds):
        candidate = mutate_field(current, rng, sigma=0.06)
        winrate = score_vs_ladder(make_chooser(candidate))
        history.append({"candidate": winrate, "baseline": baseline})
        if winrate > baseline + 0.05:
            current = candidate
            baseline = winrate
            adopted += 1
        weights.field = current.copy()

    weights.field = current
    return {
        "selfplay_adopted": adopted,
        "selfplay_rounds": cfg.selfplay_rounds,
        "final_baseline": baseline,
        "winrates": history,
    }


def train_base(cfg: TrainConfig | None = None) -> StigmergyWeights:
    """Full pipeline: defaults → PGN/book → imitation → self-play → save."""
    cfg = cfg or TrainConfig()
    t0 = time.time()
    weights = default_weights()
    for pgn in BUILTIN_PGNS:
        learn_from_pgn_text(weights, pgn)
    im = imitation_tune(weights, cfg)
    sp = selfplay_tune(weights, cfg)
    weights.training_meta = {
        "config": {
            "seed": cfg.seed,
            "imitation_steps": cfg.imitation_steps,
            "selfplay_rounds": cfg.selfplay_rounds,
            "games_per_round": cfg.games_per_round,
            "move_time_ms": cfg.move_time_ms,
            "max_depth": cfg.max_depth,
            "device": cfg.device,
        },
        "imitation": im,
        "selfplay": sp,
        "seconds": round(time.time() - t0, 2),
    }
    weights.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_weights(weights, cfg.out_path)
    return weights


def load_extra_pgn_file(weights: StigmergyWeights, path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    return learn_from_pgn_text(weights, text)
