"""Offline tests for the Stigmergy chess contest package."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_contest.stigmergy.bracket import BracketPlayer, round_robin
from chess_contest.stigmergy.engine import StigmergyEngine
from chess_contest.stigmergy.evaluate import evaluate_board
from chess_contest.stigmergy.fields import build_diffused, deposit_fields, diffuse
from chess_contest.stigmergy.opponents import GreedyMaterialOpponent, RandomOpponent, play_game
from chess_contest.stigmergy.train import TrainConfig, learn_from_pgn_text, train_base
from chess_contest.stigmergy.uniqueness import composite_contest_score, score_uniqueness
from chess_contest.stigmergy.weights import (
    default_weights,
    load_weights,
    mutate_field,
    save_weights,
    uniqueness_fingerprint,
)


def test_default_weights_roundtrip(tmp_path: Path) -> None:
    w = default_weights()
    path = tmp_path / "w.json"
    save_weights(w, path)
    w2 = load_weights(path)
    assert w2.diffusion_steps == w.diffusion_steps
    assert w2.field.deposit.shape == (6, 10)
    assert "e2e4" in {e["m"] for e in w2.book[""]}


def test_field_shapes_startpos() -> None:
    board = chess.Board()
    w = default_weights()
    fw, fb, aux = deposit_fields(board, w.field)
    assert fw.shape == (10, 8, 8)
    assert fb.shape == (10, 8, 8)
    assert aux["material"] == 0
    assert aux["w_king"] is not None
    diffused = diffuse(fw, w.field.decay, w.field.mix, 2)
    assert diffused.shape == fw.shape
    fw2, fb2, _ = build_diffused(board, w.field, 2)
    assert np.isfinite(fw2).all() and np.isfinite(fb2).all()


def test_evaluate_symmetric_start() -> None:
    board = chess.Board()
    score = evaluate_board(board, default_weights())
    # Tempo bonus makes it not exactly 0; should be small.
    assert abs(score) < 80


def test_engine_picks_legal_move() -> None:
    engine = StigmergyEngine(default_weights())
    board = chess.Board()
    res = engine.choose_move(board, time_ms=50, max_depth=2)
    assert res.move is not None
    assert res.move in board.legal_moves


def test_engine_beats_random_short() -> None:
    engine = StigmergyEngine(default_weights())
    rnd = RandomOpponent(seed=3)

    def stig(b: chess.Board) -> chess.Move:
        return engine.choose_move(b, time_ms=120, max_depth=3).move  # type: ignore[return-value]

    wins = 0
    for i in range(4):
        if i % 2 == 0:
            result = play_game(stig, rnd.choose, max_plies=50)
            if result == "1-0":
                wins += 1
        else:
            result = play_game(rnd.choose, stig, max_plies=50)
            if result == "0-1":
                wins += 1
    assert wins >= 2


def test_uniqueness_high_for_stigmergy() -> None:
    u = score_uniqueness(uniqueness_fingerprint())
    assert u.score >= 85
    assert u.to_dict()["normalized"] >= 0.85


def test_uniqueness_penalizes_stockfish_family() -> None:
    bad = score_uniqueness({"family": "stockfish-nnue-clone", "features": [], "not": []})
    good = score_uniqueness(uniqueness_fingerprint())
    assert bad.score < good.score


def test_composite_bounds() -> None:
    c = composite_contest_score(elo=1600, bracket_winrate=0.7, uniqueness=90)
    assert 0 < c["composite"] < 1


def test_pgn_learning_updates_book() -> None:
    w = default_weights()
    before = len(w.book)
    stats = learn_from_pgn_text(
        w,
        '[Result "1-0"]\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0',
    )
    assert stats["imported"] == 1
    assert len(w.book) >= before
    assert w.learned_moves


def test_mutate_field_changes_params() -> None:
    rng = np.random.default_rng(0)
    base = default_weights().field
    mut = mutate_field(base, rng, sigma=0.2)
    assert not np.allclose(base.deposit, mut.deposit)


def test_quick_train(tmp_path: Path) -> None:
    out = tmp_path / "base.json"
    cfg = TrainConfig(
        seed=1,
        imitation_steps=8,
        selfplay_rounds=1,
        games_per_round=2,
        move_time_ms=30,
        max_depth=2,
        max_plies=20,
        out_path=str(out),
    )
    w = train_base(cfg)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["engine"] == "stigmergy-dpfe"
    assert "trainingMeta" in data
    assert w.field.deposit.shape == (6, 10)


def test_bracket_runs() -> None:
    engine = StigmergyEngine(default_weights())
    rnd = RandomOpponent(0)
    greedy = GreedyMaterialOpponent()

    def stig(b: chess.Board) -> chess.Move:
        return engine.choose_move(b, time_ms=25, max_depth=1).move  # type: ignore[return-value]

    players = [
        BracketPlayer("stigmergy", stig),
        BracketPlayer("random", rnd.choose),
        BracketPlayer("greedy", greedy.choose),
    ]
    result = round_robin(players, games_per_pair=1, max_plies=24)
    assert "stigmergy" in result["standings"]
    assert len(result["matches"]) == 3


@pytest.mark.slow
def test_mate_in_one_found() -> None:
    engine = StigmergyEngine(default_weights())
    # Scholar's mate pattern: Qxf7# is forced mate in one.
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    res = engine.choose_move(board, time_ms=200, max_depth=2)
    assert res.move is not None
    assert res.move.uci() == "h5f7"
    board.push(res.move)
    assert board.is_checkmate()
