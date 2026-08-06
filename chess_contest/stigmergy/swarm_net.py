"""Swarm policy/value net — offline-SF-distilled, never NNUE, never runtime SF.

v2: richer STM-centric planes (pieces + castling/ep/check + pheromone field
summaries) and a deeper residual tower. Distilled from full-strength Stockfish
offline so play can approach 3000-Elo neural competitors without calling SF.
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

from chess_contest.stigmergy.fields import deposit_fields, diffuse
from chess_contest.stigmergy.weights import default_field_params

_TORCH = None

# v2 input: 12 pieces + 4 castling + 1 ep + 1 check + 4 field summaries = 22
IN_CH = 22


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
    """22x8x8 STM-centric planes including stigmergy field summaries."""
    flip = board.turn == chess.BLACK
    planes = np.zeros((IN_CH, 8, 8), dtype=np.float32)

    for i, (pt, color) in enumerate(PIECE_PLANES):
        for sq in board.pieces(pt, color):
            sq2 = chess.square_mirror(sq) if flip else sq
            planes[i, chess.square_rank(sq2), chess.square_file(sq2)] = 1.0
    if flip:
        planes[:12] = np.concatenate([planes[6:12], planes[:6]], axis=0)

    # Castling rights from STM POV: us-K, us-Q, opp-K, opp-Q
    if flip:
        us_k, us_q = board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(
            chess.BLACK
        )
        opp_k, opp_q = board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(
            chess.WHITE
        )
    else:
        us_k, us_q = board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(
            chess.WHITE
        )
        opp_k, opp_q = board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(
            chess.BLACK
        )
    if us_k:
        planes[12, :, :] = 1.0
    if us_q:
        planes[13, :, :] = 1.0
    if opp_k:
        planes[14, :, :] = 1.0
    if opp_q:
        planes[15, :, :] = 1.0

    if board.ep_square is not None:
        sq2 = chess.square_mirror(board.ep_square) if flip else board.ep_square
        planes[16, chess.square_rank(sq2), chess.square_file(sq2)] = 1.0
    if board.is_check():
        planes[17, :, :] = 1.0

    # Stigmergy uniqueness: 4 summary planes from deposited/diffused fields.
    try:
        params = default_field_params()
        fw, fb, _aux = deposit_fields(board, params)
        fw = diffuse(fw, params.decay, params.mix, steps=1)
        fb = diffuse(fb, params.decay, params.mix, steps=1)
        # Channel means → 2 maps per color, STM-oriented.
        w_sum = fw.mean(axis=0).astype(np.float32)
        b_sum = fb.mean(axis=0).astype(np.float32)
        w_max = fw.max(axis=0).astype(np.float32)
        b_max = fb.max(axis=0).astype(np.float32)
        if flip:
            w_sum, b_sum = np.flipud(b_sum), np.flipud(w_sum)
            w_max, b_max = np.flipud(b_max), np.flipud(w_max)
        else:
            # deposit uses row-from-top; align to rank-file plane layout.
            w_sum, b_sum = np.flipud(w_sum), np.flipud(b_sum)
            w_max, b_max = np.flipud(w_max), np.flipud(b_max)
        # Normalize lightly.
        for arr, idx in ((w_sum, 18), (b_sum, 19), (w_max, 20), (b_max, 21)):
            scale = float(np.max(np.abs(arr))) + 1e-6
            planes[idx] = arr / scale
    except Exception:
        pass
    return planes


def move_index(move: chess.Move, *, flip: bool) -> int:
    fr, to = move.from_square, move.to_square
    if flip:
        fr = chess.square_mirror(fr)
        to = chess.square_mirror(to)
    return fr * 64 + to


class SwarmNet:
    """Deep residual conv policy+value — stigmergy field-aware, not NNUE."""

    def __init__(self, channels: int = 192, blocks: int = 10, in_ch: int = IN_CH):
        torch, nn = _torch()
        self.channels = channels
        self.blocks = blocks
        self.in_ch = in_ch

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
                    nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                )
                self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
                self.policy_conv = nn.Conv2d(channels, 32, 1)
                self.policy = nn.Linear(32 * 64, 4096)
                self.value_conv = nn.Conv2d(channels, 8, 1)
                self.value = nn.Sequential(
                    nn.Linear(8 * 64, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
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
            stm_cp = float(v[0].item()) * 400.0
        out = (logits_np, stm_cp)
        if len(self._logit_cache) > 80_000:
            self._logit_cache.clear()
        self._logit_cache[fen] = out
        return out

    def choose(self, board: chess.Board) -> chess.Move | None:
        scored = self._score_legals(board)
        return scored[0][1] if scored else None

    def choose_with_margin(self, board: chess.Board) -> tuple[chess.Move | None, float]:
        scored = self._score_legals(board)
        if not scored:
            return None, 0.0
        best_s, best = scored[0]
        second = scored[1][0] if len(scored) > 1 else best_s - 10.0
        return best, best_s - second

    def top_moves(self, board: chess.Board, k: int = 5) -> list[chess.Move]:
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
        _, stm_cp = self._forward_np(board)
        return stm_cp

    def value_white(self, board: chess.Board) -> float:
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
                "in_ch": self.in_ch,
                "version": 2,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        torch, _ = _torch()
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "state_dict" not in blob:
            raise ValueError("incompatible swarm checkpoint")
        ch = int(blob.get("channels", self.channels))
        bl = int(blob.get("blocks", self.blocks))
        inch = int(blob.get("in_ch", 12))
        if ch != self.channels or bl != self.blocks or inch != self.in_ch:
            self.__init__(channels=ch, blocks=bl, in_ch=inch)
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval()
        self._logit_cache.clear()


def try_load_swarm(path: str | Path = "chess_contest/weights/gm/swarm_net.pt") -> SwarmNet | None:
    """Load residual swarm weights if present and v2-compatible; else None."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        torch, _ = _torch()
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "state_dict" not in blob:
            return None
        # Reject legacy 12-plane / flat nets — crush path needs v2.
        inch = int(blob.get("in_ch", 12))
        if inch < 20:
            return None
        ch = int(blob.get("channels", 192))
        bl = int(blob.get("blocks", 10))
        net = SwarmNet(channels=ch, blocks=bl, in_ch=inch)
        net.load(path)
        return net
    except Exception:
        return None
