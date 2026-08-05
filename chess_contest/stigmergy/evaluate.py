"""Field-interaction evaluation (centipawn-ish units from white's POV)."""

from __future__ import annotations

import chess
import chess.polyglot
import numpy as np

from chess_contest.stigmergy.fields import build_diffused, passed_pawn_score
from chess_contest.stigmergy.weights import FIELD_HEAD_DIM, FieldParams, StigmergyWeights

# Tiny process-local cache: field eval is the hot path in deep search.
_EVAL_CACHE: dict[tuple[int, int], float] = {}
_EVAL_CACHE_MAX = 80_000


def clear_eval_cache() -> None:
    _EVAL_CACHE.clear()


def evaluate_board(board: chess.Board, weights: StigmergyWeights) -> float:
    """Return a scalar score in roughly-centipawn units, white-positive."""
    if board.is_checkmate():
        return -100000.0 if board.turn == chess.WHITE else 100000.0
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_threefold_repetition()
        or board.is_fifty_moves()
    ):
        return 0.0

    key = (chess.polyglot.zobrist_hash(board), id(weights.field))
    cached = _EVAL_CACHE.get(key)
    if cached is not None:
        return cached

    params = weights.field
    fw, fb, aux = build_diffused(board, params, weights.diffusion_steps)
    score = _score_fields(board, fw, fb, aux, params)
    if len(_EVAL_CACHE) > _EVAL_CACHE_MAX:
        _EVAL_CACHE.clear()
    _EVAL_CACHE[key] = score
    return score


def _swarm_features(fw: np.ndarray, fb: np.ndarray, aux: dict) -> np.ndarray:
    """Pool pheromone fields into a fixed swarm-readout vector (unique, not NNUE)."""
    feats = np.zeros(FIELD_HEAD_DIM, dtype=np.float64)
    c = fw.shape[0]
    # Channel mean differentials (clipped to first 10 slots).
    for i in range(min(c, 10)):
        feats[i] = float(fw[i].mean() - fb[i].mean())
    # King-local enemy pressure (next slots).
    if aux.get("w_king") is not None:
        r, col = aux["w_king"]
        local = 0.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, col + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    local += float(fb[:, rr, cc].sum())
        feats[10] = local
    if aux.get("b_king") is not None:
        r, col = aux["b_king"]
        local = 0.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, col + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    local += float(fw[:, rr, cc].sum())
        feats[11] = -local
    feats[12] = float(aux.get("material", 0)) / 1000.0
    feats[13] = float(fw.sum() - fb.sum()) / 100.0
    feats[14] = float((fw**2).mean() - (fb**2).mean())
    # Harmonic residual energy.
    feats[15] = float(fw[9].sum() - fb[9].sum()) if c > 9 else 0.0
    # Pad / small nonlinear mixes.
    feats[16] = feats[0] * feats[10]
    feats[17] = feats[1] * feats[11]
    feats[18] = np.tanh(feats[12])
    feats[19] = np.tanh(feats[13])
    feats[20] = float(np.sign(feats[0]))
    feats[21] = float(np.sign(feats[1]))
    feats[22] = float(fw[2].sum() - fb[2].sum()) if c > 2 else 0.0
    feats[23] = float(fw[3].sum() - fb[3].sum()) if c > 3 else 0.0
    return feats


def _score_fields(board, fw, fb, aux, params: FieldParams) -> float:
    fw_flat = fw.reshape(fw.shape[0], -1)
    fb_flat = fb.reshape(fb.shape[0], -1)
    gram = fw_flat @ fb_flat.T
    bilinear = float((params.interaction * gram).sum())

    self_w = float((params.self_energy[:, None] * (fw_flat**2)).sum())
    self_b = float((params.self_energy[:, None] * (fb_flat**2)).sum())
    self_term = self_w - self_b

    king_term = 0.0
    if aux["w_king"] is not None:
        r, c = aux["w_king"]
        king_term += float((params.king_resonance * fb[:, r, c]).sum())
    if aux["b_king"] is not None:
        r, c = aux["b_king"]
        king_term -= float((params.king_resonance * fw[:, r, c]).sum())

    material = float(aux["material"]) * params.material_anchor
    passed = passed_pawn_score(aux["w_pawns"], aux["b_pawns"]) * params.passed_pawn_scale

    mobility = len(list(board.legal_moves)) * params.mobility_scale
    if board.turn == chess.BLACK:
        mobility = -mobility

    tempo = params.tempo_bonus if board.turn == chess.WHITE else -params.tempo_bonus

    swarm = 0.0
    if params.field_head is not None and params.field_head.size:
        feats = _swarm_features(fw, fb, aux)
        n = min(feats.size, params.field_head.size)
        swarm = float(np.dot(params.field_head[:n], feats[:n])) * params.swarm_scale

    return (
        material
        + passed
        + mobility
        + tempo
        + 18.0 * bilinear
        + 4.0 * self_term
        + 55.0 * king_term
        + 12.0 * swarm
    )


def relative_eval(board: chess.Board, weights: StigmergyWeights) -> float:
    """Score from the side-to-move's perspective (for negamax)."""
    score = evaluate_board(board, weights)
    return score if board.turn == chess.WHITE else -score
