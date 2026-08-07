"""Offline tests for the Stigmergy chess contest package."""

from __future__ import annotations

import json
import time
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
    quantize_learned_moves,
    save_weights,
    trail_key,
    uniqueness_fingerprint,
)


def test_default_weights_roundtrip(tmp_path: Path) -> None:
    w = default_weights()
    w.learned_moves["pe2e4"] = 1.23456789
    w.trails[trail_key(chess.Board())] = {"e2e4": 2.5}
    path = tmp_path / "w.json"
    save_weights(w, path)
    w2 = load_weights(path)
    assert w2.diffusion_steps == w.diffusion_steps
    assert w2.field.deposit.shape == (6, 10)
    assert "e2e4" in {e["m"] for e in w2.book[""]}
    assert w2.format_version == 4
    assert w2.learned_moves["pe2e4"] == 1.23456789
    assert w2.trails[trail_key(chess.Board())]["e2e4"] == 2.5
    data = json.loads(path.read_text())
    assert data["formatVersion"] == 4
    assert data["learnedMoves"]["precision"] == "float64"
    assert data["learnedMoves"]["values"]["pe2e4"] == 1.23456789
    assert "codes" not in data["learnedMoves"]
    assert data["trails"]["precision"] == "float64"


def test_format_v4_float_roundtrip_exact(tmp_path: Path) -> None:
    w = default_weights()
    w.learned_moves = {
        "pe2e4": 0.123456789012345,
        "ng1f3": -2.718281828,
    }
    path = tmp_path / "v4.json"
    save_weights(w, path)
    w2 = load_weights(path)
    assert w2.learned_moves["pe2e4"] == 0.123456789012345
    assert w2.learned_moves["ng1f3"] == -2.718281828


def test_legacy_ternary_learned_moves_load(tmp_path: Path) -> None:
    w = default_weights()
    path = tmp_path / "legacy.json"
    legacy = w.to_dict()
    legacy["formatVersion"] = 3
    legacy["learnedMoves"] = quantize_learned_moves({"pe2e4": 2.0, "pe7e5": -1.5})
    del legacy["trails"]
    path.write_text(json.dumps(legacy), encoding="utf-8")
    w2 = load_weights(path)
    assert w2.learned_moves["pe2e4"] > 0
    assert w2.learned_moves["pe7e5"] < 0


def test_coarse_trail_generalizes_pawn_structure() -> None:
    from chess_contest.stigmergy.coarse import coarse_trail_key, coarse_trail_move
    from chess_contest.stigmergy.distill import set_trail_policy
    from chess_contest.stigmergy.search import Searcher

    w = default_weights()
    # Italian: exact trail + coarse key stored together by set_trail_policy.
    a = chess.Board()
    for u in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]:
        a.push(chess.Move.from_uci(u))
    set_trail_policy(w, a, "d2d3", strength=80.0)
    assert coarse_trail_key(a).startswith("c")
    assert Searcher(w).trail_move(a).uci()[:4] == "d2d3"
    # Drop exact zobrist trail; coarse should still recommend d2d3.
    del w.trails[trail_key(a)]
    assert coarse_trail_move(w, a) is not None
    assert coarse_trail_move(w, a).uci()[:4] == "d2d3"


def test_search_has_no_runtime_stockfish() -> None:
    import chess_contest.stigmergy.search as search_mod
    from chess_contest.stigmergy.search import set_swarm

    assert not hasattr(search_mod, "_oracle_runtime_move")
    src = Path(search_mod.__file__).read_text(encoding="utf-8")
    assert "StockfishEngine" not in src
    assert "oracle_runtime" not in src
    set_swarm(None)


def test_search_inf_is_float_safe() -> None:
    """±1e18 null-windows collapse in float64 and false-cutoff null move."""
    from chess_contest.stigmergy.search import INF, Searcher

    assert (1e18 - 1) == 1e18
    assert (INF - 1) != INF
    w = default_weights()
    w.book.clear()
    w.trails.clear()
    board = chess.Board()
    # Alekhine-like hang trap that previously returned -inf at depth>=3.
    for u in ["b1c3", "d7d5", "g1f3", "d5d4", "c3e4", "e7e5", "e2e3", "d4e3", "f3e5", "d8d5"]:
        board.push(chess.Move.from_uci(u))
    s = Searcher(w)
    result = s.search_with_root_update(board, time_ms=800, max_depth=8)
    assert result.move is not None
    assert abs(result.score) < 1e12
    # Must not play Nc3 hanging the knight on e5.
    assert result.move.uci() != "e4c3"


def test_swarm_big_param_floor() -> None:
    """Crush-big default tower must clear the 10M-parameter floor."""
    from chess_contest.stigmergy.swarm_net import SwarmNet

    net = SwarmNet()  # 256x12 wide heads
    assert net.count_params() >= 10_000_000
    assert net.channels >= 256
    assert net.blocks >= 12


def test_policy_sprint_instant_and_safe() -> None:
    """Policy sprint skips IDAS, still returns a legal non-None move."""
    from chess_contest.stigmergy.engine import StigmergyEngine
    from chess_contest.stigmergy.search import set_policy_sprint, set_swarm
    from chess_contest.stigmergy.swarm_net import SwarmNet

    set_swarm(None)
    set_policy_sprint(False)
    net = SwarmNet(channels=16, blocks=1, policy_planes=4, value_planes=2, value_hidden=16)
    set_swarm(net)
    set_policy_sprint(True)
    try:
        w = default_weights()
        w.book.clear()
        w.trails.clear()
        eng = StigmergyEngine(w, load_swarm=False)
        set_swarm(net)
        board = chess.Board()
        t0 = time.perf_counter()
        res = eng.choose_move(board, time_ms=5000, max_depth=12)
        elapsed = time.perf_counter() - t0
        assert res.move is not None
        assert res.move in board.legal_moves
        # Sprint must not burn the full think budget on IDAS.
        assert elapsed < 2.0
    finally:
        set_policy_sprint(False)
        set_swarm(None)


def test_swarm_top_moves_and_engine_strips_oracle_flag() -> None:
    from chess_contest.stigmergy.engine import StigmergyEngine
    from chess_contest.stigmergy.search import set_swarm
    from chess_contest.stigmergy.swarm_net import SwarmNet

    set_swarm(None)
    net = SwarmNet(channels=16, blocks=1, policy_planes=4, value_planes=2, value_hidden=16)
    board = chess.Board()
    tops = net.top_moves(board, k=3)
    assert len(tops) >= 1
    assert all(m in board.legal_moves for m in tops)
    w = default_weights()
    w.training_meta["oracle_runtime"] = True
    eng = StigmergyEngine(w, load_swarm=False)
    assert eng.weights.training_meta.get("oracle_runtime") is False
    set_swarm(None)


def test_set_trail_policy_decisive() -> None:
    from chess_contest.stigmergy.distill import set_trail_policy
    from chess_contest.stigmergy.search import Searcher

    w = default_weights()
    board = chess.Board()
    set_trail_policy(w, board, "e2e4", strength=80.0)
    key = trail_key(board)
    assert w.trails[key]["e2e4"] == 80.0
    # Replacing policy clears competing moves.
    set_trail_policy(w, board, "d2d4", strength=90.0)
    assert "e2e4" not in w.trails[key] or w.trails[key].get("e2e4", 0) < 1
    assert Searcher(w).trail_move(board).uci()[:4] == "d2d4"


def test_fanout_opponent_replies_offline() -> None:
    from chess_contest.stigmergy.distill import fanout_opponent_replies
    from chess_contest.stigmergy.search import Searcher

    w = default_weights()
    board = chess.Board()

    def fake_analyse(
        b: chess.Board, movetime_ms: int = 40, multipv: int = 1, depth: int | None = None
    ):
        del movetime_ms, depth
        moves = list(b.legal_moves)
        out = []
        for i, m in enumerate(moves[: multipv]):
            out.append({"uci": m.uci(), "pv": [m.uci()], "cp": 20 - i, "mate": None})
        return out

    n = fanout_opponent_replies(
        w, board, fake_analyse, max_replies=6, fill_ms=1, fill_depth=None, strength=50.0
    )
    assert n >= 2
    assert Searcher(w).trail_move(board) is not None
    assert len(w.trails) >= 2


def test_trails_reinforce_and_trail_move() -> None:
    from chess_contest.stigmergy.distill import distill_stockfish_pv
    from chess_contest.stigmergy.search import Searcher

    w = default_weights()
    board = chess.Board()
    distill_stockfish_pv(w, board, ["e2e4"], boost=2.0)
    key = trail_key(board)
    assert key in w.trails
    assert w.trails[key].get("e2e4", 0.0) >= 2.0
    searcher = Searcher(w)
    move = searcher.trail_move(board)
    assert move is not None
    assert move.uci()[:4] == "e2e4"


def test_trail_move_requires_confidence_gap() -> None:
    from chess_contest.stigmergy.search import Searcher

    w = default_weights()
    board = chess.Board()
    key = trail_key(board)
    # Near-tie still picks the leader under the relaxed GM trail policy.
    w.trails[key] = {"e2e4": 1.0, "d2d4": 0.95}
    move = Searcher(w).trail_move(board)
    assert move is not None
    assert move.uci()[:4] == "e2e4"
    # Clear leader preferred.
    w.trails[key]["d2d4"] = 5.0
    move = Searcher(w).trail_move(board)
    assert move is not None
    assert move.uci()[:4] == "d2d4"


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


def test_tactical_floor_material_and_hanging() -> None:
    from chess_contest.stigmergy.tactics import hanging_penalty, see, tactical_floor

    start = chess.Board()
    assert abs(tactical_floor(start)) < 50  # symmetric + castling rights cancel

    # White up a queen.
    up = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert tactical_floor(up) > 800

    # Hanging black queen on d4 attacked by white pawn c3, undefended.
    hang = chess.Board("4k3/8/8/8/3q4/2P5/8/4K3 w - - 0 1")
    assert hanging_penalty(hang) > 400

    # SEE: PxQ should be strongly positive.
    board = chess.Board("4k3/8/8/8/3q4/2P5/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("c3d4")
    assert see(board, move) > 700


def test_hybrid_eval_prefers_material() -> None:
    w = default_weights()
    equal = chess.Board()
    up_queen = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert evaluate_board(up_queen, w) > evaluate_board(equal, w) + 500
