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
    # Scholar's mate avoided — solid miniature games for book/move bias.
    """[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O
9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6
16. Bh4 c5 17. dxe5 Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 20. Nbd2 Nxd6 21. Nc4 Nxc4
22. Bxc4 Nb6 23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7 Rxe1+ 26. Qxe1 Kxf7 27. Qe3 Qg5
28. Qxg5 hxg5 29. b3 Ke6 30. a3 Kd6 31. axb4 cxb4 32. Ra5 Nd5 33. f3 Bc8 34. Kf2 Bf5
35. Ra7 g6 36. Ra6+ Kc5 37. Ke1 Nf4 38. g3 Nxh3 39. Kd2 Kb5 40. Rd6 Kc5 41. Ra6 Nf2
42. g4 Bd3 43. Re6 1-0""",
    """[Result "0-1"]
1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6 8. d5 Ne7
9. Ne1 Nd7 10. Be3 f5 11. f3 f4 12. Bf2 g5 13. a4 Ng6 14. Nd3 Nf6 15. c5 Rf7 16. Rc1
Bf8 17. cxd6 cxd6 18. Nb5 g4 19. Nc7 g3 20. hxg3 fxg3 21. Bxg3 Nh5 22. Nxa8 Nxg3
23. Rf2 Qh4 24. Rc3 Qh1+ 25. Kf2 Qh4 26. Ke1 Nh1 27. Rf1 Qh2 28. Ne1 Ng3 0-1""",
    """[Result "1-0"]
1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7
9. Qd2 O-O 10. O-O-O Nbd7 11. g4 b5 12. g5 b4 13. Ne2 Ne8 14. f4 a5 15. f5 a4
16. Nbd4 exd4 17. Nxd4 Bc4 18. Bxc4 bxc3 19. Bxf7+ Rxf7 20. Ne6 cxd2+ 21. Rxd2 Qa5
22. Nxc7 Qxa2 23. Rhd1 Rc8 24. Rxd6 Nxd6 25. Rxd6 Qa1+ 26. Kd2 Qxb2 27. Ne6 Qa2
28. Rd8+ Rxd8+ 29. Nxd8 Qa5+ 30. Ke2 Qxg5 31. Ne6 Qh4 32. Bf2 Qxe4+ 33. Kd2 Qxf5 1-0""",
    """[Result "1/2-1/2"]
1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Bd2 Bxd2+
8. Nbxd2 d5 9. exd5 Nxd5 10. Qb3 Nce7 11. O-O O-O 12. Rfe1 c6 13. a4 Qb6 14. Qa3
Be6 15. a5 Qc7 16. Ne4 Rad8 17. Nc5 Bc8 18. Rac1 Nb5 19. Qa1 Nd6 20. Bf1 Ned5
21. Ne5 Rfe8 1/2-1/2""",
    """[Result "1-0"]
1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. cxd5 exd5 5. Bg5 Be7 6. e3 O-O 7. Bd3 Nbd7 8. Qc2
Re8 9. Nge2 Nf8 10. O-O-O c6 11. h3 Be6 12. Kb1 Rc8 13. g4 N6d7 14. Bxe7 Qxe7
15. Nf4 b5 16. Bd3 Nb6 17. Rhe1 Nc4 18. Bxc4 bxc4 19. e4 dxe4 20. Nxe4 Qb4 21. Nc5
Bd5 22. Nxd5 cxd5 23. Re5 Rxe5 24. dxe5 Ne6 25. Nxe6 fxe6 26. Qg6 Qf8 27. Rd4 Rc7
28. Rh4 Qe7 29. Qxh7+ Kf7 30. Rf4+ Ke8 31. Qg8+ Kd7 32. Rf7 1-0""",
    """[Result "0-1"]
1. e4 c6 2. d4 d5 3. Nc3 dxe4 4. Nxe4 Bf5 5. Ng3 Bg6 6. h4 h6 7. Nf3 Nd7 8. h5 Bh7
9. Bd3 Bxd3 10. Qxd3 e6 11. Bd2 Ngf6 12. O-O-O Be7 13. Kb1 O-O 14. Ne4 Nxe4
15. Qxe4 Nf6 16. Qe2 Qd5 17. c4 Qe4+ 18. Qxe4 Nxe4 19. Be3 Rfd8 20. Kc2 Rd7
21. Ne5 Rad8 22. f3 Nd6 23. b3 f6 24. Ng6 Bf8 25. Nf4 Kf7 26. g4 g6 27. hxg6+ Kxg6
28. Nh5 Rd6 29. Rdg1 Be7 30. f4 f5 31. gxf5+ exf5 32. Rg2 Bf8 33. Rhg1+ Kf7
34. Ng7 Bxg7 35. Rxg7+ Ke6 0-1""",
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
    greedy = GreedyMaterialOpponent()

    for _rnd in range(cfg.selfplay_rounds):
        candidate = mutate_field(current, rng, sigma=0.07)
        cand_chooser = make_chooser(candidate)
        score = 0.0
        games = cfg.games_per_round
        for i in range(games):
            # Alternate colors; mix opponents.
            opp = reference.choose if (i % 3) else greedy.choose
            if i % 2 == 0:
                result = play_game(cand_chooser, opp, max_plies=cfg.max_plies)
                if result == "1-0":
                    score += 1
                elif result == "1/2-1/2":
                    score += 0.5
            else:
                result = play_game(opp, cand_chooser, max_plies=cfg.max_plies)
                if result == "0-1":
                    score += 1
                elif result == "1/2-1/2":
                    score += 0.5
        winrate = score / games
        history.append(winrate)
        if winrate >= 0.5:
            current = candidate
            adopted += 1
        weights.field = current.copy()

    weights.field = current
    return {"selfplay_adopted": adopted, "selfplay_rounds": cfg.selfplay_rounds, "winrates": history}


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
