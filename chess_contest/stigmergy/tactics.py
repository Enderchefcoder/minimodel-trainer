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

# Endgame king PST — centralize for mating attacks.
_KING_EG: list[int] = [
    -30, -20, -10, -10, -10, -10, -20, -30,
    -20, -10, 0, 5, 5, 0, -10, -20,
    -10, 0, 10, 15, 15, 10, 0, -10,
    -10, 5, 15, 25, 25, 15, 5, -10,
    -10, 5, 15, 25, 25, 15, 5, -10,
    -10, 0, 10, 15, 15, 10, 0, -10,
    -20, -10, 0, 0, 0, 0, -10, -20,
    -30, -20, -10, -10, -10, -10, -20, -30,
]


def material_of(piece_type: int) -> int:
    """Centipawn material for a piece type."""
    return _MAT.get(piece_type, 0)


def pst_value(piece: chess.Piece, square: int, *, phase_taper: float = 1.0) -> int:
    """Piece-square bonus for ``piece`` on ``square`` (white-oriented table)."""
    table = _PST.get(piece.piece_type)
    if table is None:
        return 0
    idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
    if piece.piece_type == chess.KING:
        mg = table[idx]
        eg = _KING_EG[idx]
        return int(mg * phase_taper + eg * (1.0 - phase_taper))
    return table[idx]


def _phase_taper(board: chess.Board) -> float:
    """1.0 = full midgame king PST, 0.0 = full endgame king PST."""
    pawns = len(board.pieces(chess.PAWN, chess.WHITE))
    pawns += len(board.pieces(chess.PAWN, chess.BLACK))
    minors = len(board.pieces(chess.KNIGHT, chess.WHITE))
    minors += len(board.pieces(chess.KNIGHT, chess.BLACK))
    minors += len(board.pieces(chess.BISHOP, chess.WHITE))
    minors += len(board.pieces(chess.BISHOP, chess.BLACK))
    majors = len(board.pieces(chess.ROOK, chess.WHITE))
    majors += len(board.pieces(chess.ROOK, chess.BLACK))
    majors += len(board.pieces(chess.QUEEN, chess.WHITE))
    majors += len(board.pieces(chess.QUEEN, chess.BLACK))
    phase = pawns + minors * 2 + majors * 4
    return min(1.0, max(0.0, phase / 24.0))


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


def threat_bonus(board: chess.Board) -> float:
    """Reward attacks on higher-value undefended enemy pieces (white-positive cp)."""
    score = 0.0
    for victim_sq, victim in board.piece_map().items():
        if victim.piece_type == chess.KING:
            continue
        victim_val = material_of(victim.piece_type)
        if not board.is_attacked_by(not victim.color, victim_sq):
            continue
        if board.is_attacked_by(victim.color, victim_sq):
            continue
        for attacker_color in (chess.WHITE, chess.BLACK):
            if not board.is_attacked_by(attacker_color, victim_sq):
                continue
            cheapest = victim_val
            for att_sq in board.attackers(attacker_color, victim_sq):
                att = board.piece_at(att_sq)
                if att is None:
                    continue
                cheapest = min(cheapest, material_of(att.piece_type))
            if victim_val > cheapest:
                bonus = 0.35 * victim_val
                score += bonus if attacker_color == chess.WHITE else -bonus
                break
    return score


def tactical_floor(board: chess.Board) -> float:
    """Classical material + PST + structure + hanging — white-positive cp."""
    score = 0.0
    taper = _phase_taper(board)
    bishops = [0, 0]
    for sq, piece in board.piece_map().items():
        val = material_of(piece.piece_type) + pst_value(piece, sq, phase_taper=taper)
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
    score += threat_bonus(board)

    # Mobility (legal moves) — strong signal at club/master level.
    # Approximate without flipping turn: count attack-set sizes per side.
    mob = 0.0
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            attacks = chess.SquareSet(chess.BB_KING_ATTACKS[sq])
        elif piece.piece_type == chess.KNIGHT:
            attacks = chess.SquareSet(chess.BB_KNIGHT_ATTACKS[sq])
        elif piece.piece_type == chess.PAWN:
            attacks = chess.SquareSet(chess.BB_PAWN_ATTACKS[piece.color][sq])
        else:
            attacks = chess.SquareSet(board.attacks(sq))
        # Don't count occupied by own pieces.
        own = board.occupied_co[piece.color]
        n = len(attacks & ~chess.SquareSet(own))
        mob += n if piece.color == chess.WHITE else -n
    score += 2.5 * mob

    return float(score)
