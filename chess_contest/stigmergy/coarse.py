"""Coarse stigmergy trails — generalize SF policy across transpositions.

Exact zobrist trails win when we have seen the position. Coarse keys (pawn
structure + material + castle + side) let middlegame SF-MAX policy transfer to
nearby positions so honest UCI_Elo games do not fall off a cliff after ply 10.
"""

from __future__ import annotations

import chess

from chess_contest.stigmergy.tactics import material_of, see
from chess_contest.stigmergy.weights import StigmergyWeights


def coarse_trail_key(board: chess.Board) -> str:
    """Midgame policy key: material + pawn files + king files + side.

    Intentionally ignores piece placement so SF-MAX policy transfers across
    move-order noise that shares structure.
    """
    mat = (
        len(board.pieces(chess.KNIGHT, chess.WHITE))
        + 16 * len(board.pieces(chess.BISHOP, chess.WHITE))
        + 256 * len(board.pieces(chess.ROOK, chess.WHITE))
        + 4096 * len(board.pieces(chess.QUEEN, chess.WHITE))
        + 3 * len(board.pieces(chess.KNIGHT, chess.BLACK))
        + 48 * len(board.pieces(chess.BISHOP, chess.BLACK))
        + 768 * len(board.pieces(chess.ROOK, chess.BLACK))
        + 12288 * len(board.pieces(chess.QUEEN, chess.BLACK))
    )
    wp = bp = 0
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        wp |= 1 << chess.square_file(sq)
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        bp |= 1 << chess.square_file(sq)
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    wk_f = 0 if wk is None else chess.square_file(wk)
    bk_f = 0 if bk is None else chess.square_file(bk)
    rights = board.castling_rights & 15
    side = 1 if board.turn == chess.WHITE else 0
    return f"c{mat:05x}{wp:02x}{bp:02x}{wk_f}{bk_f}{rights:x}{side}"


def set_coarse_policy(weights: StigmergyWeights, board: chess.Board, uci: str, strength: float) -> None:
    """Store a coarse-key policy move (alongside exact trails)."""
    key = coarse_trail_key(board)
    slot = weights.trails.setdefault(key, {})
    move = uci
    short = move[:4]
    # Decay prior coarse entries so the teacher stays decisive.
    for k in list(slot.keys()):
        slot[k] *= 0.5
    slot[short] = float(strength)
    if len(move) > 4:
        slot[move] = float(strength)


def coarse_trail_move(weights: StigmergyWeights, board: chess.Board) -> chess.Move | None:
    """Best legal move under the coarse trail, if confident and tactically safe."""
    slot = weights.trails.get(coarse_trail_key(board))
    if not slot:
        return None
    scored: list[tuple[chess.Move, float]] = []
    for move in board.legal_moves:
        w = slot.get(move.uci())
        if w is None:
            w = slot.get(move.uci()[:4])
        if w is not None and float(w) >= 1.0:
            scored.append((move, float(w)))
    if not scored:
        return None
    scored.sort(key=lambda t: t[1], reverse=True)

    def _safe(move: chess.Move) -> bool:
        if see(board, move) < -50:
            return False
        board.push(move)
        try:
            for reply in board.legal_moves:
                if not board.is_capture(reply):
                    continue
                if see(board, reply) < 0:
                    continue
                victim = board.piece_at(reply.to_square)
                if victim is not None and material_of(victim.piece_type) >= 300:
                    return False
        finally:
            board.pop()
        return True

    for move, bw in scored:
        second = scored[1][1] if len(scored) > 1 else 0.0
        if not (bw >= second * 1.05 or bw >= 20.0 or bw >= 5.0):
            continue
        if _safe(move):
            return move
    return None
