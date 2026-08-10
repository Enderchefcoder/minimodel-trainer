"""Multi-channel pheromone deposition and Jacobi diffusion.

This is the unique core: the board is not scored via PST tables or an NNUE
accumulator. Pieces emit typed scents into a lattice; information spreads by
diffusion; evaluation later reads cross-color interactions of the fields.
"""

from __future__ import annotations

import numpy as np

from chess_contest.stigmergy.weights import N_CHANNELS, PIECE_ORDER, FieldParams

# Classic material for the soft anchor (centipawns).
MATERIAL = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900, "k": 0}

# Fixed harmonic basis on an 8x8 board (channel "harmonic" residual).
_YS, _XS = np.mgrid[0:8, 0:8]
HARMONIC = (
    np.sin(np.pi * (_XS + 1) / 9) * np.sin(np.pi * (_YS + 1) / 9)
    + 0.5 * np.sin(2 * np.pi * (_XS + 1) / 9) * np.cos(np.pi * (_YS + 1) / 9)
).astype(np.float64)


def _sq_rc(sq: int) -> tuple[int, int]:
    """python-chess square index -> (row from white's POV top=0, col)."""
    file_ = sq % 8
    rank = sq // 8
    row = 7 - rank
    return row, file_


def deposit_fields(board, params: FieldParams) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build white/black (C,8,8) seed fields and aux geometry from a Board."""
    fw = np.zeros((N_CHANNELS, 8, 8), dtype=np.float64)
    fb = np.zeros((N_CHANNELS, 8, 8), dtype=np.float64)
    w_king = b_king = None
    material = 0
    w_pawns: list[list[int]] = [[] for _ in range(8)]
    b_pawns: list[list[int]] = [[] for _ in range(8)]

    for sq, piece in board.piece_map().items():
        row, col = _sq_rc(sq)
        idx = PIECE_ORDER.index(piece.symbol().lower())
        seed = params.deposit[idx]
        target = fw if piece.color else fb
        # Flip vertical for black so deposition is side-relative for pawn_chain.
        r_dep = row if piece.color else (7 - row)
        # Material / generic channels use absolute coordinates; pawn_chain uses
        # side-relative rank via a second write for channel 2 only.
        target[:, row, col] += seed
        # Side-relative pawn-chain boost.
        if idx == 0:
            target[2, row, col] += 0.15 * (r_dep / 7.0)
        # Color-complex: bishops on light/dark.
        if idx == 2:
            target[4, row, col] += 0.35 if (row + col) % 2 == 0 else -0.35
        # Harmonic residual seeded by piece presence.
        target[9, row, col] += 0.2 * HARMONIC[row, col] * (1.0 + 0.1 * idx)

        sign = 1 if piece.color else -1
        material += sign * MATERIAL[piece.symbol().lower()]
        if idx == 5:
            if piece.color:
                w_king = (row, col)
            else:
                b_king = (row, col)
        if idx == 0:
            (w_pawns if piece.color else b_pawns)[col].append(row)

    aux = {
        "material": material,
        "w_king": w_king,
        "b_king": b_king,
        "w_pawns": w_pawns,
        "b_pawns": b_pawns,
    }
    return fw, fb, aux


def diffuse(field: np.ndarray, decay: np.ndarray, mix: np.ndarray, steps: int) -> np.ndarray:
    """Jacobi diffusion: each channel blends with 4-neighbors independently."""
    out = field.copy()
    for _ in range(max(0, steps)):
        # Average of N/E/S/W with edge clamp (pheromone reflects at borders).
        up = np.concatenate([out[:, :1, :], out[:, :-1, :]], axis=1)
        down = np.concatenate([out[:, 1:, :], out[:, -1:, :]], axis=1)
        left = np.concatenate([out[:, :, :1], out[:, :, :-1]], axis=2)
        right = np.concatenate([out[:, :, 1:], out[:, :, -1:]], axis=2)
        neigh = 0.25 * (up + down + left + right)
        # Broadcast decay/mix over spatial dims.
        d = decay[:, None, None]
        m = mix[:, None, None]
        out = (1.0 - m) * out + m * neigh
        out *= d
        # Soft renormalize to keep magnitudes stable across depths.
        out *= 1.0 / (0.85 + 0.15 * d.max())
    return out


def passed_pawn_score(w_pawns: list[list[int]], b_pawns: list[list[int]]) -> float:
    """Geometry bonus read off the pawn lists (feeds field scoring)."""
    score = 0.0
    bonus = [0, 120, 90, 60, 40, 25, 12, 0]
    for f in range(8):
        for r in w_pawns[f]:
            passed = True
            for nf in (f - 1, f, f + 1):
                if nf < 0 or nf > 7:
                    continue
                for br in b_pawns[nf]:
                    if br < r:
                        passed = False
            if passed:
                score += bonus[r]
            if len(w_pawns[f]) > 1:
                score -= 12
            if not ((f > 0 and w_pawns[f - 1]) or (f < 7 and w_pawns[f + 1])):
                score -= 15
        for r in b_pawns[f]:
            passed = True
            for nf in (f - 1, f, f + 1):
                if nf < 0 or nf > 7:
                    continue
                for wr in w_pawns[nf]:
                    if wr > r:
                        passed = False
            if passed:
                score -= bonus[7 - r]
            if len(b_pawns[f]) > 1:
                score += 12
            if not ((f > 0 and b_pawns[f - 1]) or (f < 7 and b_pawns[f + 1])):
                score += 15
    return score


def build_diffused(
    board, params: FieldParams, steps: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Deposit + diffuse; returns white/black fields and aux."""
    fw, fb, aux = deposit_fields(board, params)
    fw = diffuse(fw, params.decay, params.mix, steps)
    fb = diffuse(fb, params.decay, params.mix, steps)
    return fw, fb, aux
