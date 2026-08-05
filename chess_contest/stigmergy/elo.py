"""ELO estimation from match results against a reference ladder."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess

from chess_contest.stigmergy.engine import StigmergyEngine
from chess_contest.stigmergy.opponents import (
    ClassicPSTOpponent,
    GreedyMaterialOpponent,
    Opponent,
    RandomOpponent,
    play_game,
    update_elo,
)
from chess_contest.stigmergy.weights import StigmergyWeights, load_weights


@dataclass
class EloReport:
    engine_elo: float
    ratings: dict[str, float]
    matches: list[dict[str, Any]] = field(default_factory=list)
    games: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_elo": round(self.engine_elo, 1),
            "ratings": {k: round(v, 1) for k, v in self.ratings.items()},
            "games": self.games,
            "matches": self.matches,
        }


# Anchor ratings for reference bots (approximate class strengths).
ANCHORS = {
    "random": 200.0,
    "greedy_material": 850.0,
    "classic_pst_d2": 1250.0,
    "classic_pst_d3": 1450.0,
}


def estimate_elo(
    weights: StigmergyWeights,
    games_per_opponent: int = 6,
    move_time_ms: int = 200,
    max_depth: int = 4,
    max_plies: int = 80,
    seed: int = 0,
) -> EloReport:
    """Play a ladder and fit Elo with fixed anchors for reference engines."""
    engine = StigmergyEngine(weights)
    ratings = dict(ANCHORS)
    ratings["stigmergy"] = 1200.0

    opponents: list[Opponent] = [
        RandomOpponent(seed=seed),
        GreedyMaterialOpponent(),
        ClassicPSTOpponent(depth=2),
        ClassicPSTOpponent(depth=3),
    ]

    def stig_choose(board: chess.Board) -> chess.Move:
        res = engine.choose_move(board, time_ms=move_time_ms, max_depth=max_depth)
        assert res.move is not None
        return res.move

    matches: list[dict[str, Any]] = []
    games = 0
    for opp in opponents:
        for i in range(games_per_opponent):
            stig_white = i % 2 == 0
            if stig_white:
                result = play_game(stig_choose, opp.choose, max_plies=max_plies)
                score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
            else:
                result = play_game(opp.choose, stig_choose, max_plies=max_plies)
                score = {"1-0": 0.0, "0-1": 1.0, "1/2-1/2": 0.5}[result]
            # Only update stigmergy; keep anchors fixed after blend.
            new_s, new_o = update_elo(ratings["stigmergy"], ratings[opp.name], score, k=24.0)
            ratings["stigmergy"] = new_s
            # Soft pull opponent toward anchor to stabilize.
            ratings[opp.name] = 0.85 * ANCHORS[opp.name] + 0.15 * new_o
            matches.append(
                {
                    "opponent": opp.name,
                    "stigmergy_color": "white" if stig_white else "black",
                    "result": result,
                    "score": score,
                    "elo_after": round(ratings["stigmergy"], 1),
                }
            )
            games += 1

    return EloReport(engine_elo=ratings["stigmergy"], ratings=ratings, matches=matches, games=games)


def estimate_elo_from_file(path: str | Path, **kwargs: Any) -> EloReport:
    return estimate_elo(load_weights(path), **kwargs)


def save_report(report: EloReport, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
