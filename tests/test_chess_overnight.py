"""Stockfish oracle + overnight distillation smoke tests."""

from __future__ import annotations

import chess
import pytest

from chess_contest.stigmergy.distill import distill_game, distill_stockfish_pv, prune_learned_moves
from chess_contest.stigmergy.evaluate import evaluate_board
from chess_contest.stigmergy.stockfish_uci import (
    StockfishConfig,
    StockfishEngine,
    stockfish_available,
)
from chess_contest.stigmergy.weights import default_weights


@pytest.mark.skipif(not stockfish_available(), reason="Stockfish binary missing")
def test_stockfish_returns_legal_move() -> None:
    with StockfishEngine(StockfishConfig(movetime_ms=50, hash_mb=16)) as sf:
        board = chess.Board()
        move = sf.choose(board)
        assert move in board.legal_moves


@pytest.mark.skipif(not stockfish_available(), reason="Stockfish binary missing")
def test_stockfish_limit_elo() -> None:
    with StockfishEngine(StockfishConfig(movetime_ms=40, hash_mb=16)) as sf:
        sf.set_elo(1400)
        board = chess.Board()
        move = sf.choose(board)
        assert move in board.legal_moves


def test_distill_reinforces_winner_moves() -> None:
    w = default_weights()
    board = chess.Board()
    moves = []
    # Scholar-ish line truncated.
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]:
        m = chess.Move.from_uci(uci)
        moves.append(m)
        board.push(m)
    before = dict(w.learned_moves)
    distill_game(w, moves, "1-0", winner_boost=2.0)
    assert len(w.learned_moves) >= len(before)
    # White moves should get positive mass.
    assert any(v > 0 for k, v in w.learned_moves.items() if k.startswith(("p", "n", "b", "r", "q", "k")))


def test_swarm_head_changes_eval() -> None:
    from chess_contest.stigmergy.evaluate import clear_eval_cache

    w = default_weights()
    board = chess.Board()
    clear_eval_cache()
    base = evaluate_board(board, w)
    assert w.field.field_head is not None
    w.field.field_head = w.field.field_head.copy()
    w.field.field_head[:] = 0.5
    w.field.swarm_scale = 5.0
    clear_eval_cache()
    bumped = evaluate_board(board, w)
    assert bumped != base


def test_prune_learned_moves() -> None:
    w = default_weights()
    for i in range(100):
        w.learned_moves[f"n{i}a2b3"] = float(i)
    prune_learned_moves(w, keep=20)
    assert len(w.learned_moves) == 20


def test_distill_pv() -> None:
    w = default_weights()
    board = chess.Board()
    n = distill_stockfish_pv(w, board, ["e2e4", "e7e5", "g1f3"], boost=1.0)
    assert n >= 1
    assert "" in w.book or "e2e4" in w.book or any(w.book)
