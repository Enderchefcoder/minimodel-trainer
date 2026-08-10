"""Single-elimination / round-robin bracket helpers for the contest."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import chess

from chess_contest.stigmergy.opponents import play_game

MoveChooser = Callable[[chess.Board], chess.Move]


@dataclass
class BracketPlayer:
    name: str
    choose: MoveChooser


@dataclass
class MatchResult:
    white: str
    black: str
    result: str  # 1-0 / 0-1 / 1/2-1/2

    @property
    def winner(self) -> str | None:
        if self.result == "1-0":
            return self.white
        if self.result == "0-1":
            return self.black
        return None


def round_robin(
    players: list[BracketPlayer],
    games_per_pair: int = 2,
    max_plies: int = 100,
) -> dict[str, Any]:
    """Play a double round-robin (colors alternate within each pair)."""
    standings = {p.name: {"wins": 0, "losses": 0, "draws": 0, "points": 0.0} for p in players}
    matches: list[dict[str, Any]] = []
    by_name = {p.name: p for p in players}

    for a, b in itertools.combinations(players, 2):
        for i in range(games_per_pair):
            if i % 2 == 0:
                white, black = a, b
            else:
                white, black = b, a
            result = play_game(white.choose, black.choose, max_plies=max_plies)
            matches.append({"white": white.name, "black": black.name, "result": result})
            if result == "1-0":
                standings[white.name]["wins"] += 1
                standings[white.name]["points"] += 1
                standings[black.name]["losses"] += 1
            elif result == "0-1":
                standings[black.name]["wins"] += 1
                standings[black.name]["points"] += 1
                standings[white.name]["losses"] += 1
            else:
                standings[white.name]["draws"] += 1
                standings[black.name]["draws"] += 1
                standings[white.name]["points"] += 0.5
                standings[black.name]["points"] += 0.5

    ranking = sorted(standings.items(), key=lambda kv: kv[1]["points"], reverse=True)
    return {
        "standings": standings,
        "ranking": [{"name": n, **s} for n, s in ranking],
        "matches": matches,
        "players": list(by_name),
    }


def winrate_for(name: str, bracket: dict[str, Any]) -> float:
    s = bracket["standings"][name]
    games = s["wins"] + s["losses"] + s["draws"]
    if games == 0:
        return 0.0
    return s["points"] / games
