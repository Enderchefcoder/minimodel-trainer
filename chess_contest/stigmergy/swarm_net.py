"""Swarm policy/value net - offline-SF-distilled, never NNUE, never runtime SF.

v3 (crush-big): ~20M+ residual tower (default 256ch x 12 blocks) with wider
policy/value heads and stigmergy field planes. Distilled from full-strength
Stockfish offline so play can climb toward 3000-Elo neural competitors
without calling SF at move time.
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

from chess_contest.stigmergy.fields import deposit_fields, diffuse
from chess_contest.stigmergy.weights import default_field_params

_TORCH = None

# v2/v3 input: 12 pieces + 4 castling + 1 ep + 1 check + 4 field summaries = 22
IN_CH = 22
# Crush-big defaults - ~23M params (10M+ required for the 3000 push).
DEFAULT_CHANNELS = 256
DEFAULT_BLOCKS = 12
DEFAULT_POLICY_PLANES = 64
DEFAULT_VALUE_PLANES = 16
DEFAULT_VALUE_HIDDEN = 512


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
        w_sum = fw.mean(axis=0).astype(np.float32)
        b_sum = fb.mean(axis=0).astype(np.float32)
        w_max = fw.max(axis=0).astype(np.float32)
        b_max = fb.max(axis=0).astype(np.float32)
        if flip:
            w_sum, b_sum = np.flipud(b_sum), np.flipud(w_sum)
            w_max, b_max = np.flipud(b_max), np.flipud(w_max)
        else:
            w_sum, b_sum = np.flipud(w_sum), np.flipud(b_sum)
            w_max, b_max = np.flipud(w_max), np.flipud(b_max)
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
    """Deep residual conv policy+value - stigmergy field-aware, not NNUE."""

    def __init__(
        self,
        channels: int = DEFAULT_CHANNELS,
        blocks: int = DEFAULT_BLOCKS,
        in_ch: int = IN_CH,
        *,
        policy_planes: int = DEFAULT_POLICY_PLANES,
        value_planes: int = DEFAULT_VALUE_PLANES,
        value_hidden: int = DEFAULT_VALUE_HIDDEN,
    ):
        torch, nn = _torch()
        self.channels = channels
        self.blocks = blocks
        self.in_ch = in_ch
        self.policy_planes = policy_planes
        self.value_planes = value_planes
        self.value_hidden = value_hidden

        class ResBlock(nn.Module):
            def __init__(self, c: int):
                super().__init__()
                self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(c)
                self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(c)
                # Lightweight channel attention (stigmergy-ish gating, not NNUE).
                self.se = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(c, max(8, c // 8)),
                    nn.ReLU(),
                    nn.Linear(max(8, c // 8), c),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                h = torch.relu(self.bn1(self.conv1(x)))
                h = self.bn2(self.conv2(h))
                gate = self.se(h).unsqueeze(-1).unsqueeze(-1)
                return torch.relu(x + h * gate)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                )
                self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
                self.policy_conv = nn.Conv2d(channels, policy_planes, 1)
                self.policy = nn.Linear(policy_planes * 64, 4096)
                self.value_conv = nn.Conv2d(channels, value_planes, 1)
                self.value = nn.Sequential(
                    nn.Linear(value_planes * 64, value_hidden),
                    nn.ReLU(),
                    nn.Linear(value_hidden, 1),
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

    def count_params(self) -> int:
        """Trainable parameter count (crush-big target is 10M+)."""
        return int(sum(p.numel() for p in self.net.parameters()))

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
                "policy_planes": self.policy_planes,
                "value_planes": self.value_planes,
                "value_hidden": self.value_hidden,
                "params": self.count_params(),
                "version": 3,
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
        # Infer head widths from legacy v2 checkpoints (32/8/256).
        pp = int(blob.get("policy_planes", 32 if int(blob.get("version", 2)) < 3 else self.policy_planes))
        vp = int(blob.get("value_planes", 8 if int(blob.get("version", 2)) < 3 else self.value_planes))
        vh = int(blob.get("value_hidden", 256 if int(blob.get("version", 2)) < 3 else self.value_hidden))
        # Detect head sizes from state_dict when metadata missing.
        sd = blob["state_dict"]
        if "policy_conv.weight" in sd:
            pp = int(sd["policy_conv.weight"].shape[0])
        if "value_conv.weight" in sd:
            vp = int(sd["value_conv.weight"].shape[0])
        if "value.0.weight" in sd:
            vh = int(sd["value.0.weight"].shape[0])
        need = (
            ch != self.channels
            or bl != self.blocks
            or inch != self.in_ch
            or pp != self.policy_planes
            or vp != self.value_planes
            or vh != self.value_hidden
        )
        if need:
            self.__init__(
                channels=ch,
                blocks=bl,
                in_ch=inch,
                policy_planes=pp,
                value_planes=vp,
                value_hidden=vh,
            )
        # Legacy v2 towers lack SE layers - reject incompatible towers.
        if any(k.startswith("tower.0.se.") for k in self.net.state_dict()) and not any(
            k.startswith("tower.0.se.") for k in sd
        ):
            raise ValueError("legacy residual tower without SE - retrain crush-big")
        self.net.load_state_dict(sd)
        self.net.eval()
        self._logit_cache.clear()


def try_load_swarm(path: str | Path = "chess_contest/weights/gm/swarm_net.pt") -> SwarmNet | None:
    """Load residual swarm weights if present and v2+/field-aware; else None."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        torch, _ = _torch()
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "state_dict" not in blob:
            return None
        # Reject legacy 12-plane / flat nets - crush path needs field-aware input.
        inch = int(blob.get("in_ch", 12))
        if inch < 20:
            return None
        ch = int(blob.get("channels", DEFAULT_CHANNELS))
        bl = int(blob.get("blocks", DEFAULT_BLOCKS))
        net = SwarmNet(channels=ch, blocks=bl, in_ch=inch)
        net.load(path)
        return net
    except Exception:
        return None
