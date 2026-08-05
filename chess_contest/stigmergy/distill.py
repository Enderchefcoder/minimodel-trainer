"""Winner-distillation: learn book + move biases + field nudges from game winners.

Stockfish is only an oracle/opponent. Distilled knowledge lands in Stigmergy's
ternary trails and pheromone field — never an NNUE clone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess
import numpy as np

from chess_contest.stigmergy.evaluate import evaluate_board
from chess_contest.stigmergy.weights import StigmergyWeights, mutate_field


@dataclass
class DistillStats:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    moves_reinforced: int = 0
    book_updates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "moves_reinforced": self.moves_reinforced,
            "book_updates": self.book_updates,
        }


def _lm_key(board: chess.Board, move: chess.Move) -> str | None:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return None
    return (
        f"{piece.symbol().lower()}"
        f"{chess.square_name(move.from_square)}"
        f"{chess.square_name(move.to_square)}"
    )


def distill_game(
    weights: StigmergyWeights,
    moves: list[chess.Move],
    result: str,
    *,
    winner_boost: float = 1.0,
    loser_penalty: float = 0.35,
    book_plies: int = 16,
) -> DistillStats:
    """Reinforce the winner's moves; softly discourage the loser's."""
    stats = DistillStats(games=1)
    if result == "1-0":
        winner = chess.WHITE
        stats.wins = 1  # from white's perspective of result, not stigmergy
    elif result == "0-1":
        winner = chess.BLACK
        stats.losses = 1
    else:
        winner = None
        stats.draws = 1

    board = chess.Board()
    path = ""
    book_raw: dict[str, dict[str, float]] = {}
    for ply, move in enumerate(moves):
        if move not in board.legal_moves:
            break
        key = _lm_key(board, move)
        is_winner_move = winner is not None and board.turn == winner
        is_loser_move = winner is not None and board.turn != winner
        if key is not None:
            decay = 1.0 if ply < 30 else 0.5
            if is_winner_move:
                delta = winner_boost * decay
                weights.learned_moves[key] = weights.learned_moves.get(key, 0.0) + delta
                stats.moves_reinforced += 1
            elif is_loser_move:
                weights.learned_moves[key] = weights.learned_moves.get(key, 0.0) - loser_penalty * decay
                stats.moves_reinforced += 1
            elif winner is None:
                weights.learned_moves[key] = weights.learned_moves.get(key, 0.0) + 0.05 * decay

        if ply < book_plies:
            uci = move.uci()[:4]
            w = 3.0 if is_winner_move else (0.4 if is_loser_move else 1.0)
            book_raw.setdefault(path, {})
            book_raw[path][uci] = book_raw[path].get(uci, 0.0) + w

        board.push(move)
        path += move.uci()[:4]

    for bpath, entries in book_raw.items():
        mean = sum(entries.values()) / max(1, len(entries))
        weights.book[bpath] = [
            {
                "m": m,
                "code": 1 if c > mean * 1.25 else (-1 if c < mean * 0.55 else 0),
                "games": round(c),
            }
            for m, c in entries.items()
        ]
        stats.book_updates += 1
    return stats


def distill_stockfish_pv(
    weights: StigmergyWeights,
    board: chess.Board,
    pv_uci: list[str],
    *,
    boost: float = 1.2,
) -> int:
    """Reinforce an SF principal variation as if it were a strong winner line."""
    n = 0
    b = board.copy(stack=False)
    path = "".join(m.uci()[:4] for m in board.move_stack)
    for i, uci in enumerate(pv_uci[:12]):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in b.legal_moves:
            break
        key = _lm_key(b, move)
        if key is not None:
            weights.learned_moves[key] = weights.learned_moves.get(key, 0.0) + boost * (1.0 if i < 6 else 0.4)
            n += 1
        if i < 10:
            slot = weights.book.setdefault(path, [])
            existing = {e["m"]: e for e in slot}
            if uci[:4] in existing:
                existing[uci[:4]]["code"] = 1
                existing[uci[:4]]["games"] = int(existing[uci[:4]].get("games", 0)) + 2
            else:
                slot.append({"m": uci[:4], "code": 1, "games": 2})
            weights.book[path] = list(existing.values()) if existing else slot
        b.push(move)
        path += uci[:4]
    return n


def imitation_toward_move(
    weights: StigmergyWeights,
    board: chess.Board,
    target: chess.Move,
    rng: np.random.Generator,
    lr: float = 0.04,
) -> bool:
    """Nudge field so 1-ply eval prefers Stockfish's move over our top wrong move."""
    if target not in board.legal_moves:
        return False
    scored: list[tuple[chess.Move, float]] = []
    mover_white = board.turn == chess.WHITE
    for move in board.legal_moves:
        board.push(move)
        val = evaluate_board(board, weights)
        board.pop()
        s = val if mover_white else -val
        scored.append((move, s))
    scored.sort(key=lambda t: t[1], reverse=True)
    if scored[0][0] == target:
        return True
    # Soft ES-style nudge of deposit for the target piece.
    piece = board.piece_at(target.from_square)
    wrong = board.piece_at(scored[0][0].from_square)
    if piece is not None:
        idx = "pnbrqk".index(piece.symbol().lower())
        weights.field.deposit[idx] *= 1.0 + lr
        weights.field.deposit[idx] += rng.normal(0, lr * 0.05, weights.field.deposit[idx].shape)
    if wrong is not None and wrong != piece:
        idx = "pnbrqk".index(wrong.symbol().lower())
        weights.field.deposit[idx] *= 1.0 - lr * 0.5
    weights.field.material_anchor = float(min(2.0, weights.field.material_anchor * (1.0 + lr * 0.01)))
    if hasattr(weights.field, "field_head") and weights.field.field_head is not None:
        weights.field.field_head = weights.field.field_head + rng.normal(0, lr * 0.02, weights.field.field_head.shape)
    return False


def prune_learned_moves(weights: StigmergyWeights, keep: int = 8000) -> None:
    """Cap memory so JSON stays manageable."""
    if len(weights.learned_moves) <= keep:
        return
    top = sorted(weights.learned_moves.items(), key=lambda kv: abs(kv[1]), reverse=True)[:keep]
    weights.learned_moves = dict(top)


def evolve_against_baseline(
    weights: StigmergyWeights,
    rng: np.random.Generator,
    score_fn,
    sigma: float = 0.05,
) -> bool:
    """Adopt mutated field if score_fn(candidate_weights) > score_fn(current)."""
    baseline = score_fn(weights)
    cand_field = mutate_field(weights.field, rng, sigma=sigma)
    trial = StigmergyWeights(
        field=cand_field,
        book=weights.book,
        learned_moves=dict(weights.learned_moves),
        diffusion_steps=weights.diffusion_steps,
        format_version=weights.format_version,
        training_meta=dict(weights.training_meta),
    )
    # Preserve field_head if present on params.
    if getattr(weights.field, "field_head", None) is not None:
        trial.field.field_head = weights.field.field_head + rng.normal(
            0, sigma * 0.3, weights.field.field_head.shape
        )
    cand_score = score_fn(trial)
    if cand_score > baseline + 0.02:
        weights.field = trial.field
        return True
    return False
