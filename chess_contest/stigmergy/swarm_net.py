"""Swarm policy/value net — offline-SF-distilled, never NNUE, never runtime SF.

Piece-plane conv tower with residual blocks. Value ≈ teacher cp (tanh scale);
policy ranks legal moves. Used only as stigmergy search prior / leaf eval.
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

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
    """12x8x8 piece planes, always side-to-move as 'white' planes."""
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    flip = board.turn == chess.BLACK
    for i, (pt, color) in enumerate(PIECE_PLANES):
        for sq in board.pieces(pt, color):
            sq2 = chess.square_mirror(sq) if flip else sq
            planes[i, chess.square_rank(sq2), chess.square_file(sq2)] = 1.0
    if flip:
        planes = np.concatenate([planes[6:], planes[:6]], axis=0)
    return planes


def move_index(move: chess.Move, *, flip: bool) -> int:
    fr, to = move.from_square, move.to_square
    if flip:
        fr = chess.square_mirror(fr)
        to = chess.square_mirror(to)
    return fr * 64 + to


class SwarmNet:
    """Residual conv policy+value — uniqueness: swarm field readout, not NNUE."""

    def __init__(self, channels: int = 96, blocks: int = 6):
        torch, nn = _torch()
        self.channels = channels
        self.blocks = blocks

        class ResBlock(nn.Module):
            def __init__(self, c: int):
                super().__init__()
                self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(c)
                self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(c)

            def forward(self, x):
                h = torch.relu(self.bn1(self.conv1(x)))
                h = self.bn2(self.conv2(h))
                return torch.relu(x + h)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(12, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                )
                self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
                self.policy_conv = nn.Conv2d(channels, 32, 1)
                self.policy = nn.Linear(32 * 64, 4096)
                self.value_conv = nn.Conv2d(channels, 8, 1)
                self.value = nn.Sequential(
                    nn.Linear(8 * 64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Tanh(),
                )

            def forward(self, x):
                h = self.tower(self.stem(x))
                p = self.policy_conv(h).reshape(x.shape[0], -1)
                v = self.value_conv(h).reshape(x.shape[0], -1)
                return self.policy(p), self.value(v)

        self.net = Net()
        self.device = torch.device("cpu")
        self.net.to(self.device)
        self.net.eval()
        self._logit_cache: dict[str, tuple[np.ndarray, float]] = {}

    def _forward_np(self, board: chess.Board) -> tuple[np.ndarray, float]:
        torch, _ = _torch()
        fen = board.fen()
        cached = self._logit_cache.get(fen)
        if cached is not None:
            return cached
        planes = encode_board(board)
        x = torch.from_numpy(planes).unsqueeze(0)
        with torch.no_grad():
            logits, v = self.net(x)
            logits_np = logits[0].detach().numpy()
            # Side-to-move value in "white-positive after flip" space: multiply by 400cp.
            stm_cp = float(v[0].item()) * 400.0
        out = (logits_np, stm_cp)
        if len(self._logit_cache) > 50_000:
            self._logit_cache.clear()
        self._logit_cache[fen] = out
        return out

    def choose(self, board: chess.Board) -> chess.Move | None:
        logits, _ = self._forward_np(board)
        flip = board.turn == chess.BLACK
        best = None
        best_s = -1e18
        for move in board.legal_moves:
            s = float(logits[move_index(move, flip=flip)])
            if s > best_s:
                best_s = s
                best = move
        return best

    def choose_with_margin(self, board: chess.Board) -> tuple[chess.Move | None, float]:
        """Return (best_move, logit_margin over second-best legal)."""
        scored = self._score_legals(board)
        if not scored:
            return None, 0.0
        best_s, best = scored[0]
        second = scored[1][0] if len(scored) > 1 else best_s - 10.0
        return best, best_s - second

    def top_moves(self, board: chess.Board, k: int = 5) -> list[chess.Move]:
        """Return up to k legal moves sorted by descending policy logit."""
        scored = self._score_legals(board)
        return [m for _, m in scored[: max(1, k)]]

    def _score_legals(self, board: chess.Board) -> list[tuple[float, chess.Move]]:
        logits, _ = self._forward_np(board)
        flip = board.turn == chess.BLACK
        scored: list[tuple[float, chess.Move]] = [
            (float(logits[move_index(move, flip=flip)]), move) for move in board.legal_moves
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored

    def policy_score(self, board: chess.Board, move: chess.Move) -> float:
        logits, _ = self._forward_np(board)
        flip = board.turn == chess.BLACK
        return float(logits[move_index(move, flip=flip)])

    def value_stm(self, board: chess.Board) -> float:
        """Side-to-move centipawn estimate (positive = good for side to move)."""
        _, stm_cp = self._forward_np(board)
        return stm_cp

    def value_white(self, board: chess.Board) -> float:
        """White-positive centipawn estimate for hybrid floor fusion."""
        stm = self.value_stm(board)
        return stm if board.turn == chess.WHITE else -stm

    def save(self, path: str | Path) -> None:
        torch, _ = _torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "channels": self.channels,
                "blocks": self.blocks,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        torch, _ = _torch()
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "state_dict" in blob:
            # Rebuild if arch differs.
            ch = int(blob.get("channels", self.channels))
            bl = int(blob.get("blocks", self.blocks))
            if ch != self.channels or bl != self.blocks:
                self.__init__(channels=ch, blocks=bl)
            self.net.load_state_dict(blob["state_dict"])
        else:
            self.net.load_state_dict(blob)
        self.net.eval()
        self._logit_cache.clear()


def try_load_swarm(path: str | Path = "chess_contest/weights/gm/swarm_net.pt") -> SwarmNet | None:
    """Load residual swarm weights if present and compatible; else None."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        torch, _ = _torch()
        blob = torch.load(path, map_location="cpu", weights_only=False)
        # Legacy flat OrderedDict nets (no residual tower) are rejected.
        if not isinstance(blob, dict) or "state_dict" not in blob:
            return None
        ch = int(blob.get("channels", 96))
        bl = int(blob.get("blocks", 6))
        net = SwarmNet(channels=ch, blocks=bl)
        net.load(path)
        return net
    except Exception:
        return None
