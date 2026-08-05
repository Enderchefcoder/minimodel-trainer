"""Swarm policy/value net — SF-distilled move prior that is NOT NNUE.

A tiny conv/MLP over piece planes. Used when exact trails miss so out-of-book
play stays near the teacher. Branding: swarm_vote_policy / field readout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

# Lazy torch — optional heavy dep (AGENTS rule 3).
_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch
        import torch.nn as nn

        _TORCH = (torch, nn)
    return _TORCH


PIECE_PLANES = [
    (chess.PAWN, chess.WHITE),
    (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE),
    (chess.ROOK, chess.WHITE),
    (chess.QUEEN, chess.WHITE),
    (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK),
    (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.BLACK),
    (chess.QUEEN, chess.BLACK),
    (chess.KING, chess.BLACK),
]


def encode_board(board: chess.Board) -> np.ndarray:
    """12x8x8 piece planes, white-to-move oriented (flip if black to move)."""
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    flip = board.turn == chess.BLACK
    for i, (pt, color) in enumerate(PIECE_PLANES):
        for sq in board.pieces(pt, color):
            if flip:
                sq = chess.square_mirror(sq)
            planes[i, chess.square_rank(sq), chess.square_file(sq)] = 1.0
    if flip:
        # Swap white/black plane blocks after mirror.
        planes = np.concatenate([planes[6:], planes[:6]], axis=0)
    return planes


def move_index(move: chess.Move, *, flip: bool) -> int:
    fr, to = move.from_square, move.to_square
    if flip:
        fr = chess.square_mirror(fr)
        to = chess.square_mirror(to)
    return fr * 64 + to


def index_to_move(idx: int, board: chess.Board) -> chess.Move | None:
    fr, to = divmod(idx, 64)
    if board.turn == chess.BLACK:
        fr = chess.square_mirror(fr)
        to = chess.square_mirror(to)
    promo = None
    # Try queen promo if pawn to back rank.
    piece = board.piece_at(fr)
    if piece is not None and piece.piece_type == chess.PAWN:
        tr = chess.square_rank(to)
        if tr in (0, 7):
            promo = chess.QUEEN
    try:
        mv = chess.Move(fr, to, promotion=promo)
    except ValueError:
        return None
    if mv in board.legal_moves:
        return mv
    if promo is not None:
        for p in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            mv = chess.Move(fr, to, promotion=p)
            if mv in board.legal_moves:
                return mv
    return None


class SwarmNet:
    """Small policy+value network."""

    def __init__(self, channels: int = 32):
        torch, nn = _torch()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(12, channels, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.ReLU(),
                )
                self.policy = nn.Linear(channels * 64, 4096)
                self.value = nn.Sequential(
                    nn.Linear(channels * 64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Tanh(),
                )

            def forward(self, x):
                h = self.stem(x).reshape(x.shape[0], -1)
                return self.policy(h), self.value(h)

        self.net = Net()
        self.device = torch.device("cpu")
        self.net.to(self.device)
        self.net.eval()

    def choose(self, board: chess.Board) -> chess.Move | None:
        torch, _ = _torch()
        planes = encode_board(board)
        x = torch.from_numpy(planes).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(x)
            logits = logits[0].numpy()
        best = None
        best_s = -1e18
        flip = board.turn == chess.BLACK
        for move in board.legal_moves:
            idx = move_index(move, flip=flip)
            s = float(logits[idx])
            if s > best_s:
                best_s = s
                best = move
        return best

    def value(self, board: chess.Board) -> float:
        torch, _ = _torch()
        planes = encode_board(board)
        x = torch.from_numpy(planes).unsqueeze(0)
        with torch.no_grad():
            _, v = self.net(x)
        # White-positive-ish: network is side-to-move oriented via encode flip.
        val = float(v[0].item()) * 300.0
        return val if board.turn == chess.WHITE else -val

    def save(self, path: str | Path) -> None:
        torch, _ = _torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path: str | Path) -> None:
        torch, _ = _torch()
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.net.eval()


@dataclass
class SwarmBundle:
    path: Path

    def exists(self) -> bool:
        return self.path.is_file()
