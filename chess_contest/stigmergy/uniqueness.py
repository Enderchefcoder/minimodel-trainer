"""Uniqueness rubric for contest entries.

Higher uniqueness is better. Known mega-engine families are heavily penalized.
Stigmergy's diffusive-pheromone fingerprint scores near the ceiling by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Features that scream "not unique" if claimed alone.
GENERIC_PENALTIES = {
    "stockfish": 40,
    "nnue": 35,
    "alphazero": 35,
    "lc0": 30,
    "classic-pst-only": 25,
    "pure-minimax-material": 20,
}

# Novel building blocks we reward.
NOVEL_FEATURES = {
    "multi_channel_pheromone_lattice": 12,
    "jacobi_diffusion_kernels": 12,
    "bilinear_cross_color_interaction": 10,
    "king_resonance_coupling": 8,
    "ternary_ant_trail_book": 10,
    "ternary_move_pheromone_bias": 6,
    "harmonic_board_channel": 8,
    "spectral_attack_graph": 8,
    "chemical_reaction_eval": 8,
    "swarm_vote_policy": 6,
    "swarm_field_head_readout": 8,
    "winner_distillation_from_oracles": 6,
    "tactical_floor_field_fusion": 7,
    "see_ordered_idas": 5,
}


@dataclass
class UniquenessScore:
    score: float
    max_score: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "max_score": self.max_score,
            "normalized": round(self.score / self.max_score, 3),
            "details": self.details,
        }


def score_uniqueness(fingerprint: dict[str, Any] | None) -> UniquenessScore:
    """Score an architecture fingerprint dict (from weights.uniquenessFingerprint)."""
    fp = fingerprint or {}
    details: dict[str, Any] = {"bonuses": [], "penalties": []}
    score = 40.0  # base for submitting anything custom
    max_score = 100.0

    family = str(fp.get("family", "")).lower()
    if "pheromone" in family or "stigmergy" in family or "diffusive" in family:
        score += 15
        details["bonuses"].append({"reason": "novel_family", "delta": 15})

    not_list = [str(x).lower() for x in (fp.get("not") or [])]
    for bad, pen in GENERIC_PENALTIES.items():
        if bad in not_list:
            # Explicitly distancing from mega-engines is good (small bonus).
            score += 2
            details["bonuses"].append({"reason": f"rejects_{bad}", "delta": 2})
        if bad.replace("-", "_") in family or bad in family:
            score -= pen
            details["penalties"].append({"reason": bad, "delta": -pen})

    features = [str(x) for x in (fp.get("features") or [])]
    for feat in features:
        bonus = NOVEL_FEATURES.get(feat, 0)
        if bonus:
            score += bonus
            details["bonuses"].append({"reason": feat, "delta": bonus})

    # Cap.
    score = max(0.0, min(max_score, score))
    details["feature_count"] = len(features)
    details["family"] = family
    return UniquenessScore(score=score, max_score=max_score, details=details)


def composite_contest_score(
    elo: float,
    bracket_winrate: float,
    uniqueness: float,
    elo_floor: float = 800.0,
    elo_ceil: float = 2000.0,
) -> dict[str, float]:
    """Combine the three contest axes into one leaderboard number."""
    elo_n = max(0.0, min(1.0, (elo - elo_floor) / (elo_ceil - elo_floor)))
    uniq_n = max(0.0, min(1.0, uniqueness / 100.0))
    bracket_n = max(0.0, min(1.0, bracket_winrate))
    # Weights: strength matters, but uniqueness is first-class.
    total = 0.45 * elo_n + 0.25 * bracket_n + 0.30 * uniq_n
    return {
        "elo_component": round(elo_n, 4),
        "bracket_component": round(bracket_n, 4),
        "uniqueness_component": round(uniq_n, 4),
        "composite": round(total, 4),
    }
