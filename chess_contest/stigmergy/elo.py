"""ELO estimation from match results against a reference ladder."""

from __future__ import annotations

import json
import math
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
    mle_elo: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_elo": round(self.engine_elo, 1),
            "mle_elo": None if self.mle_elo is None else round(self.mle_elo, 1),
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


def _mle_elo(matches: list[dict[str, Any]], lo: float = 400.0, hi: float = 2400.0) -> float:
    """Binary-search Elo that maximizes Bernoulli log-likelihood vs fixed anchors."""

    def nll(r: float) -> float:
        loss = 0.0
        for m in matches:
            opp = ANCHORS[m["opponent"]]
            p = 1.0 / (1.0 + math.pow(10.0, (opp - r) / 400.0))
            p = min(1 - 1e-9, max(1e-9, p))
            s = float(m["score"])
            loss -= s * math.log(p) + (1.0 - s) * math.log(1.0 - p)
        return loss

    best_r, best_loss = 1200.0, float("inf")
    # Coarse then fine grid (no scipy dependency).
    for r in range(int(lo), int(hi) + 1, 25):
        loss = nll(float(r))
        if loss < best_loss:
            best_loss, best_r = loss, float(r)
    for r in range(int(best_r - 40), int(best_r + 41), 1):
        loss = nll(float(r))
        if loss < best_loss:
            best_loss, best_r = loss, float(r)
    return best_r


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
    ratings["stigmergy"] = 1400.0

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
            # Larger K early so dominant ladders aren't stuck near the prior.
            k = 40.0 if games < 12 else 24.0
            new_s, _new_o = update_elo(ratings["stigmergy"], ANCHORS[opp.name], score, k=k)
            ratings["stigmergy"] = new_s
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

    mle = _mle_elo(matches)
    # Prefer MLE when the sequential update is biased by a low prior.
    engine_elo = max(ratings["stigmergy"], mle)
    return EloReport(
        engine_elo=engine_elo,
        ratings={**ratings, "stigmergy": engine_elo, "stigmergy_sequential": ratings["stigmergy"]},
        matches=matches,
        games=games,
        mle_elo=mle,
    )


def estimate_elo_from_file(path: str | Path, **kwargs: Any) -> EloReport:
    return estimate_elo(load_weights(path), **kwargs)


def save_report(report: EloReport, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
