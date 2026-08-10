"""High-level Stigmergy engine API."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from chess_contest.stigmergy.evaluate import evaluate_board
from chess_contest.stigmergy.search import Searcher, SearchResult, set_swarm
from chess_contest.stigmergy.swarm_net import try_load_swarm
from chess_contest.stigmergy.weights import StigmergyWeights, default_weights, load_weights


@dataclass
class EngineInfo:
    name: str = "Stigmergy-DPFE"
    author: str = "minimodel-trainer chess contest"
    architecture: str = "diffusive-pheromone-field"


class StigmergyEngine:
    """Play / analyze with a weight set. Play path never calls Stockfish."""

    def __init__(self, weights: StigmergyWeights | None = None, *, load_swarm: bool = True):
        self.weights = weights or default_weights()
        self.info = EngineInfo()
        self.swarm = None
        # Strip any legacy runtime-SF flag — play path is Stockfish-free.
        if self.weights.training_meta.get("oracle_runtime"):
            self.weights.training_meta["oracle_runtime"] = False
            self.weights.training_meta["stockfish_at_play"] = False
        if load_swarm:
            swarm_path = self.weights.training_meta.get(
                "swarm_net", "chess_contest/weights/gm/swarm_net.pt"
            )
            net = try_load_swarm(swarm_path)
            if net is not None:
                set_swarm(net)
                self.swarm = net

    @classmethod
    def from_file(cls, path: str) -> StigmergyEngine:
        return cls(load_weights(path))

    def evaluate(self, board: chess.Board) -> float:
        return evaluate_board(board, self.weights)

    def choose_move(
        self,
        board: chess.Board,
        time_ms: int = 2000,
        max_depth: int = 14,
    ) -> SearchResult:
        searcher = Searcher(self.weights)
        return searcher.search_with_root_update(board, time_ms=time_ms, max_depth=max_depth)

    def play_move(self, board: chess.Board, time_ms: int = 2000, max_depth: int = 14) -> chess.Move:
        result = self.choose_move(board, time_ms=time_ms, max_depth=max_depth)
        if result.move is None:
            raise RuntimeError("No legal move")
        board.push(result.move)
        return result.move
