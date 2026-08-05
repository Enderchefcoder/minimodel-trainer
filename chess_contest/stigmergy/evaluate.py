"""Field-interaction evaluation (centipawn-ish units from white's POV)."""

from __future__ import annotations

import chess
import chess.polyglot

from chess_contest.stigmergy.fields import build_diffused, passed_pawn_score
from chess_contest.stigmergy.weights import FieldParams, StigmergyWeights

# Tiny process-local cache: field eval is the hot path in deep search.
_EVAL_CACHE: dict[tuple[int, int], float] = {}
_EVAL_CACHE_MAX = 50_000


def evaluate_board(board: chess.Board, weights: StigmergyWeights) -> float:
    """Return a scalar score in roughly-centipawn units, white-positive."""
    if board.is_checkmate():
        # Side to move is mated.
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


def _score_fields(board, fw, fb, aux, params: FieldParams) -> float:
    # Bilinear cross-color interactions: sum_cd W[c,d] * <Fw[c], Fb[d]>
    # <Fw[c], Fb[d]> = sum over squares
    # Efficient: for each c,d — (fw[c] * interaction-weighted).
    # Fw flattened (C,64), Fb (C,64): interaction @ (Fb @ Fw.T) style.
    fw_flat = fw.reshape(fw.shape[0], -1)
    fb_flat = fb.reshape(fb.shape[0], -1)
    # Gram-like: G[c,d] = sum_s fw[c,s]*fb[d,s]
    gram = fw_flat @ fb_flat.T  # (C,C)
    bilinear = float((params.interaction * gram).sum())

    # Self-energy: prefer structured own fields.
    self_w = float((params.self_energy[:, None] * (fw_flat**2)).sum())
    self_b = float((params.self_energy[:, None] * (fb_flat**2)).sum())
    self_term = self_w - self_b

    # King resonance: enemy pheromone under our king (danger).
    king_term = 0.0
    if aux["w_king"] is not None:
        r, c = aux["w_king"]
        king_term += float((params.king_resonance * fb[:, r, c]).sum())
    if aux["b_king"] is not None:
        r, c = aux["b_king"]
        king_term -= float((params.king_resonance * fw[:, r, c]).sum())

    material = float(aux["material"]) * params.material_anchor
    passed = passed_pawn_score(aux["w_pawns"], aux["b_pawns"]) * params.passed_pawn_scale

    # Mobility mist: cheap proxy from legal move counts (side to move only is
    # biased; sample both by temporarily flipping — expensive). Use turn-aware
    # one-sided count scaled by mobility_scale.
    mobility = len(list(board.legal_moves)) * params.mobility_scale
    # From white POV: if black to move, negate the mobility contribution.
    if board.turn == chess.BLACK:
        mobility = -mobility

    tempo = params.tempo_bonus if board.turn == chess.WHITE else -params.tempo_bonus

    # Scale bilinear/self/king into centipawn-ish range.
    return (
        material
        + passed
        + mobility
        + tempo
        + 18.0 * bilinear
        + 4.0 * self_term
        + 55.0 * king_term
    )


def relative_eval(board: chess.Board, weights: StigmergyWeights) -> float:
    """Score from the side-to-move's perspective (for negamax)."""
    score = evaluate_board(board, weights)
    return score if board.turn == chess.WHITE else -score
