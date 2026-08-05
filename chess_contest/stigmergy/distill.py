"""Winner-distillation: learn book + move biases + field nudges from game winners.

Stockfish is only an oracle/opponent. Distilled knowledge lands in Stigmergy's
continuous float trails and pheromone field — never an NNUE clone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess
import numpy as np

from chess_contest.stigmergy.evaluate import evaluate_board
from chess_contest.stigmergy.weights import StigmergyWeights, mutate_field, trail_key


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


def _book_code(c: float, mean: float) -> int:
    return 1 if c > mean * 1.25 else (-1 if c < mean * 0.55 else 0)


def _reinforce_trail(
    weights: StigmergyWeights,
    board: chess.Board,
    uci: str,
    amount: float,
) -> None:
    if amount <= 0:
        return
    key = trail_key(board)
    slot = weights.trails.setdefault(key, {})
    slot[uci[:4]] = slot.get(uci[:4], 0.0) + amount


def set_trail_policy(
    weights: StigmergyWeights,
    board: chess.Board,
    uci: str,
    *,
    strength: float = 50.0,
) -> None:
    """Replace a position's trail with a single decisive float64 policy move."""
    key = trail_key(board)
    move = uci[:4]
    weights.trails[key] = {move: float(strength)}
    path = "".join(m.uci()[:4] for m in board.move_stack)
    _reinforce_book_entry(weights, path, move, strength)
    lm = _lm_key(board, chess.Move.from_uci(uci if len(uci) > 4 else move))
    if lm is not None:
        weights.learned_moves[lm] = float(weights.learned_moves.get(lm, 0.0)) + strength


def _reinforce_book_entry(
    weights: StigmergyWeights,
    path: str,
    uci: str,
    amount: float,
) -> None:
    slot = weights.book.setdefault(path, [])
    existing = {e["m"]: e for e in slot}
    move = uci[:4]
    if move in existing:
        entry = existing[move]
        entry["w"] = float(entry.get("w", 0.0)) + amount
        entry["code"] = _book_code(entry["w"], max(entry["w"], 1.0))
        entry["games"] = int(entry.get("games", 0)) + 1
    else:
        existing[move] = {"m": move, "w": float(amount), "code": 1, "games": 1}
    weights.book[path] = list(existing.values())


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
                _reinforce_trail(weights, board, move.uci(), delta)
                stats.moves_reinforced += 1
            elif is_loser_move:
                delta = loser_penalty * decay
                weights.learned_moves[key] = weights.learned_moves.get(key, 0.0) - delta
                stats.moves_reinforced += 1
            elif winner is None:
                delta = 0.05 * decay
                weights.learned_moves[key] = weights.learned_moves.get(key, 0.0) + delta
                _reinforce_trail(weights, board, move.uci(), delta * 0.5)

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
                "w": float(c),
                "code": _book_code(c, mean),
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
        rank_decay = boost * (1.0 if i < 6 else 0.4)
        key = _lm_key(b, move)
        if key is not None:
            weights.learned_moves[key] = weights.learned_moves.get(key, 0.0) + rank_decay
            n += 1
        _reinforce_trail(weights, b, uci, rank_decay)
        if i < 10:
            _reinforce_book_entry(weights, path, uci, rank_decay)
        b.push(move)
        path += uci[:4]
    return n


def distill_stockfish_top(
    weights: StigmergyWeights,
    board: chess.Board,
    tops: list[dict[str, Any]],
    *,
    boost: float = 1.0,
) -> int:
    """Reinforce multipv Stockfish lines into trails, book, and learned moves.

    Top-1 gets the bulk of float mass so trail_move follows a single clear policy.
    """
    n = 0
    for rank, info in enumerate(tops):
        # Strong top-1 preference for a decisive continuous trail.
        decay = boost * (1.0 if rank == 0 else 0.15 / rank)
        pv = info.get("pv") or []
        uci = info.get("uci") or (pv[0] if pv else None)
        if not uci:
            continue
        b = board.copy(stack=False)
        path = "".join(m.uci()[:4] for m in board.move_stack)
        line = pv if pv else [uci]
        for i, move_uci in enumerate(line[:8]):
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                break
            if move not in b.legal_moves:
                break
            amount = decay * (1.0 if i < 3 else 0.25)
            lm = _lm_key(b, move)
            if lm is not None:
                weights.learned_moves[lm] = weights.learned_moves.get(lm, 0.0) + amount
                n += 1
            _reinforce_trail(weights, b, move_uci, amount)
            if i < 6:
                _reinforce_book_entry(weights, path, move_uci, amount)
            b.push(move)
            path += move_uci[:4]
    return n


def clip_field_params(params) -> None:
    """Keep pheromone parameters in a numerically stable band."""
    params.deposit = np.clip(params.deposit, -15.0, 15.0)
    params.decay = np.clip(params.decay, 0.12, 0.95)
    params.mix = np.clip(params.mix, 0.05, 0.9)
    params.interaction = np.clip(params.interaction, -3.0, 3.0)
    params.self_energy = np.clip(params.self_energy, -2.0, 2.0)
    params.king_resonance = np.clip(params.king_resonance, -3.0, 3.0)
    params.material_anchor = float(np.clip(params.material_anchor, 0.4, 2.5))
    params.tempo_bonus = float(np.clip(params.tempo_bonus, 0.0, 40.0))
    params.passed_pawn_scale = float(np.clip(params.passed_pawn_scale, 0.2, 3.0))
    params.mobility_scale = float(np.clip(params.mobility_scale, 0.2, 3.0))
    params.swarm_scale = float(np.clip(params.swarm_scale, 0.05, 5.0))
    if params.field_head is not None:
        params.field_head = np.clip(params.field_head, -8.0, 8.0)


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
    # Additive, clipped nudges — multiplicative updates explode overnight.
    piece = board.piece_at(target.from_square)
    wrong = board.piece_at(scored[0][0].from_square)
    if piece is not None:
        idx = "pnbrqk".index(piece.symbol().lower())
        weights.field.deposit[idx] += lr * 0.15 + rng.normal(0, lr * 0.02, weights.field.deposit[idx].shape)
    if wrong is not None and wrong != piece:
        idx = "pnbrqk".index(wrong.symbol().lower())
        weights.field.deposit[idx] -= lr * 0.1
    weights.field.material_anchor = float(
        min(2.0, weights.field.material_anchor + lr * 0.01)
    )
    if weights.field.field_head is not None:
        weights.field.field_head = weights.field.field_head + rng.normal(
            0, lr * 0.015, weights.field.field_head.shape
        )
    clip_field_params(weights.field)
    return False


def prune_learned_moves(weights: StigmergyWeights, keep: int = 50000) -> None:
    """Cap memory so JSON stays manageable."""
    if len(weights.learned_moves) <= keep:
        return
    top = sorted(weights.learned_moves.items(), key=lambda kv: abs(kv[1]), reverse=True)[:keep]
    weights.learned_moves = dict(top)


def prune_trails(weights: StigmergyWeights, keep_positions: int = 200000) -> None:
    """Keep the strongest trail positions by peak move intensity."""
    if len(weights.trails) <= keep_positions:
        return
    ranked = sorted(
        weights.trails.items(),
        key=lambda kv: max((abs(v) for v in kv[1].values()), default=0.0),
        reverse=True,
    )[:keep_positions]
    weights.trails = {k: dict(v) for k, v in ranked}


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
        trails={k: dict(v) for k, v in weights.trails.items()},
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
