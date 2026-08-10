"""Reference opponents for ELO estimation and brackets (all unique-policy free)."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections.abc import Callable

import chess

MATERIAL = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

PST_PAWN = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]


class Opponent(ABC):
    name: str

    @abstractmethod
    def choose(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError


class RandomOpponent(Opponent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, board: chess.Board) -> chess.Move:
        return self.rng.choice(list(board.legal_moves))


class GreedyMaterialOpponent(Opponent):
    """1-ply material + capture preference (weak ~800 Elo class)."""

    name = "greedy_material"

    def choose(self, board: chess.Board) -> chess.Move:
        best = None
        best_s = -1e18
        mover_white = board.turn == chess.WHITE
        for move in board.legal_moves:
            board.push(move)
            val = _material(board)
            board.pop()
            score = val if mover_white else -val
            if board.is_capture(move):
                score += 15
            if move.promotion:
                score += 50
            if score > best_s:
                best_s = score
                best = move
        assert best is not None
        return best


class ClassicPSTOpponent(Opponent):
    """Shallow classic PST minimax — reference ladder rung, not a contest entry."""

    def __init__(self, depth: int = 2, name: str | None = None):
        self.depth = depth
        self.name = name or f"classic_pst_d{depth}"

    def choose(self, board: chess.Board) -> chess.Move:
        best_move = next(iter(board.legal_moves))
        best = -1e18
        for move in board.legal_moves:
            board.push(move)
            try:
                score = -self._negamax(board, self.depth - 1, -1e18, 1e18)
            finally:
                board.pop()
            if score > best:
                best = score
                best_move = move
        return best_move

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        if board.is_checkmate():
            return -100000
        if board.is_game_over():
            return 0.0
        if depth <= 0:
            val = _material(board) + _pst(board)
            return val if board.turn == chess.WHITE else -val
        best = -1e18
        for move in board.legal_moves:
            board.push(move)
            try:
                score = -self._negamax(board, depth - 1, -beta, -alpha)
            finally:
                board.pop()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best


def _material(board: chess.Board) -> float:
    total = 0
    for _sq, piece in board.piece_map().items():
        v = MATERIAL[piece.piece_type]
        total += v if piece.color == chess.WHITE else -v
    return float(total)


def _pst(board: chess.Board) -> float:
    score = 0.0
    for sq, piece in board.piece_map().items():
        if piece.piece_type != chess.PAWN:
            continue
        # python-chess square: file + 16*rank... actually 0-63 file + 8*rank.
        rank = chess.square_rank(sq)
        file_ = chess.square_file(sq)
        idx = (7 - rank) * 8 + file_ if piece.color == chess.WHITE else rank * 8 + file_
        bonus = PST_PAWN[idx]
        score += bonus if piece.color == chess.WHITE else -bonus
    return score


def play_game(
    white: Callable[[chess.Board], chess.Move],
    black: Callable[[chess.Board], chess.Move],
    max_plies: int = 120,
) -> str:
    """Return '1-0', '0-1', or '1/2-1/2'."""
    board = chess.Board()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        mover = white if board.turn == chess.WHITE else black
        move = mover(board)
        if move not in board.legal_moves:
            # Illegal = loss for side to move.
            return "0-1" if board.turn == chess.WHITE else "1-0"
        board.push(move)
        plies += 1
    if board.is_checkmate():
        return "0-1" if board.turn == chess.WHITE else "1-0"
    return "1/2-1/2"


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (elo_b - elo_a) / 400.0))


def update_elo(elo_a: float, elo_b: float, score_a: float, k: float = 32.0) -> tuple[float, float]:
    ea = expected_score(elo_a, elo_b)
    eb = 1.0 - ea
    return elo_a + k * (score_a - ea), elo_b + k * ((1.0 - score_a) - eb)
