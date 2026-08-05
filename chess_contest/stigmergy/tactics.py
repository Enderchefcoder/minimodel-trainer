"""Tactical floor for Stigmergy — classical signals fused under the field eval.

This is *not* NNUE or Stockfish. It is a cheap, interpretable floor so the
pheromone field never hangs queens while dreaming about diffusion. Strength
comes from the hybrid: tactical floor + unique field residual.
"""

from __future__ import annotations

import chess

# Material (centipawns).
_MAT = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-square tables (white POV, square 0 = a1). Midgame-oriented.
_PST: dict[int, list[int]] = {
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, -20, -20, 10, 10, 5,
        5, -5, -10, 0, 0, -10, -5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, 5, 10, 25, 25, 10, 5, 5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    chess.KNIGHT: [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ],
    chess.BISHOP: [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    chess.ROOK: [
        0, 0, 0, 5, 5, 0, 0, 0,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 10, 10, 10, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    chess.QUEEN: [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -10, 5, 5, 5, 5, 5, 0, -10,
        0, 0, 5, 5, 5, 5, 0, -5,
        -5, 0, 5, 5, 5, 5, 0, -5,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20,
    ],
    chess.KING: [
        20, 30, 10, 0, 0, 10, 30, 20,
        20, 20, 0, 0, 0, 0, 20, 20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
    ],
}


def material_of(piece_type: int) -> int:
    """Centipawn material for a piece type."""
    return _MAT.get(piece_type, 0)


def pst_value(piece: chess.Piece, square: int) -> int:
    """Piece-square bonus for ``piece`` on ``square`` (white-oriented table)."""
    table = _PST.get(piece.piece_type)
    if table is None:
        return 0
    idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
    return table[idx]


def see(board: chess.Board, move: chess.Move) -> int:
    """Fast static exchange estimate in centipawns (side that moves).

    Uses a 2-ply capture/recapture model (MVV minus cheapest recapture) so
    move ordering and quiescence stay cheap on CPU. Good enough to refuse
    queen-for-pawn disasters without walking the full attacker set.
    """
    to_sq = move.to_square
    if board.is_en_passant(move):
        gain = material_of(chess.PAWN)
    else:
        victim = board.piece_at(to_sq)
        if victim is None and not move.promotion:
            return 0
        gain = material_of(victim.piece_type) if victim else 0
    if move.promotion:
        gain += material_of(move.promotion) - material_of(chess.PAWN)

    attacker = board.piece_at(move.from_square)
    if attacker is None:
        return int(gain)

    board.push(move)
    try:
        if not board.is_attacked_by(board.turn, to_sq):
            return int(gain)
        # Cheapest legal recapture value.
        recapture_cost = material_of(attacker.piece_type)
        for sq in board.attackers(board.turn, to_sq):
            piece = board.piece_at(sq)
            if piece is None:
                continue
            cost = material_of(piece.piece_type)
            if cost < recapture_cost:
                recapture_cost = cost
        return int(gain - recapture_cost)
    finally:
        board.pop()


def hanging_penalty(board: chess.Board) -> float:
    """Penalize undefended attacked pieces (white-positive)."""
    score = 0.0
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        enemy = not piece.color
        if board.is_attacked_by(enemy, sq) and not board.is_attacked_by(piece.color, sq):
            pen = 0.55 * material_of(piece.piece_type)
            score += -pen if piece.color == chess.WHITE else pen
    return score


def tactical_floor(board: chess.Board) -> float:
    """Classical material + PST + structure + hanging — white-positive cp."""
    score = 0.0
    bishops = [0, 0]
    for sq, piece in board.piece_map().items():
        val = material_of(piece.piece_type) + pst_value(piece, sq)
        score += val if piece.color == chess.WHITE else -val
        if piece.piece_type == chess.BISHOP:
            bishops[int(piece.color)] += 1
    if bishops[chess.WHITE] >= 2:
        score += 35
    if bishops[chess.BLACK] >= 2:
        score -= 35

    # Rook on open/semi-open files.
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        our_pawns = board.pieces(chess.PAWN, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        for rook_sq in board.pieces(chess.ROOK, color):
            f = chess.square_file(rook_sq)
            file_mask = chess.BB_FILES[f]
            if not (our_pawns & file_mask):
                score += 12 * sign
                if not (enemy_pawns & file_mask):
                    score += 12 * sign

    # King safety: count enemy attacks near king.
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        king_sq = board.king(color)
        if king_sq is None:
            continue
        danger = 0
        for att in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
            if board.is_attacked_by(not color, att):
                danger += 1
        score -= sign * danger * 8
        if board.has_castling_rights(color):
            score += sign * 15

    score += hanging_penalty(board)
    return float(score)
