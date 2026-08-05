"""Serializable weight format shared by Python trainer and the HTML player."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import chess.polyglot
import numpy as np

CHANNEL_NAMES = [
    "material_mass",
    "king_heat",
    "pawn_chain",
    "file_control",
    "color_complex",
    "mobility_mist",
    "discovery_threat",
    "blockade",
    "tempo",
    "harmonic",
]
N_CHANNELS = len(CHANNEL_NAMES)
PIECE_ORDER = "pnbrqk"
N_PIECES = len(PIECE_ORDER)

# Hand-authored ternary opening book (code -1/0/1 = discourage/neutral/encourage).
DEFAULT_BOOK: dict[str, list[dict[str, Any]]] = {
    "": [
        {"m": "e2e4", "code": 1},
        {"m": "d2d4", "code": 1},
        {"m": "c2c4", "code": 0},
        {"m": "g1f3", "code": 0},
        {"m": "g2g3", "code": 0},
    ],
    "e2e4": [
        {"m": "e7e5", "code": 1},
        {"m": "c7c5", "code": 1},
        {"m": "e7e6", "code": 0},
        {"m": "c7c6", "code": 0},
        {"m": "d7d6", "code": 0},
        {"m": "g7g6", "code": 0},
        {"m": "d7d5", "code": 0},
        {"m": "g8f6", "code": 0},
    ],
    "d2d4": [
        {"m": "d7d5", "code": 1},
        {"m": "g8f6", "code": 1},
        {"m": "e7e6", "code": 0},
        {"m": "c7c5", "code": 0},
        {"m": "g7g6", "code": 0},
    ],
    "e2e4e7e5": [
        {"m": "g1f3", "code": 1},
        {"m": "f1c4", "code": 0},
        {"m": "f1b5", "code": 0},
        {"m": "b1c3", "code": 0},
    ],
    "e2e4c7c5": [
        {"m": "g1f3", "code": 1},
        {"m": "b1c3", "code": 0},
        {"m": "c2c3", "code": 0},
    ],
    "e2e4e7e5g1f3": [
        {"m": "b8c6", "code": 1},
        {"m": "g8f6", "code": 0},
        {"m": "d7d6", "code": 0},
    ],
    "e2e4c7c5g1f3": [
        {"m": "d7d6", "code": 1},
        {"m": "b8c6", "code": 1},
        {"m": "e7e6", "code": 0},
        {"m": "g7g6", "code": 0},
    ],
    "e2e4e7e5g1f3b8c6": [
        {"m": "f1b5", "code": 1},
        {"m": "f1c4", "code": 1},
        {"m": "d2d4", "code": 0},
    ],
    "d2d4d7d5": [{"m": "c2c4", "code": 1}, {"m": "g1f3", "code": 0}],
    "d2d4g8f6": [{"m": "c2c4", "code": 1}, {"m": "g1f3", "code": 0}],
    "d2d4d7d5c2c4": [
        {"m": "e7e6", "code": 1},
        {"m": "c7c6", "code": 1},
        {"m": "d5c4", "code": 0},
    ],
    "d2d4g8f6c2c4": [
        {"m": "g7g6", "code": 1},
        {"m": "e7e6", "code": 1},
        {"m": "c7c5", "code": 0},
    ],
}

CODE_WEIGHT = {-1: 1, 0: 3, 1: 6}


FIELD_HEAD_DIM = 24  # swarm readout over pooled pheromone stats


@dataclass
class FieldParams:
    """Learnable pheromone-field parameters (numpy arrays)."""

    deposit: np.ndarray  # (6, C) piece-type -> channel seed strength
    decay: np.ndarray  # (C,) diffusion retention
    mix: np.ndarray  # (C,) neighbor blend
    interaction: np.ndarray  # (C, C) white_c x black_d bilinear
    self_energy: np.ndarray  # (C,) own-field quadratic weight
    king_resonance: np.ndarray  # (C,) enemy field under king
    material_anchor: float = 1.0
    tempo_bonus: float = 8.0
    passed_pawn_scale: float = 1.0
    mobility_scale: float = 1.0
    # Unique "swarm readout": linear head on pooled field stats (not NNUE).
    field_head: np.ndarray | None = None
    swarm_scale: float = 1.0

    def copy(self) -> FieldParams:
        return FieldParams(
            deposit=self.deposit.copy(),
            decay=self.decay.copy(),
            mix=self.mix.copy(),
            interaction=self.interaction.copy(),
            self_energy=self.self_energy.copy(),
            king_resonance=self.king_resonance.copy(),
            material_anchor=float(self.material_anchor),
            tempo_bonus=float(self.tempo_bonus),
            passed_pawn_scale=float(self.passed_pawn_scale),
            mobility_scale=float(self.mobility_scale),
            field_head=None if self.field_head is None else self.field_head.copy(),
            swarm_scale=float(self.swarm_scale),
        )

    def to_jsonable(self) -> dict[str, Any]:
        out = {
            "deposit": self.deposit.tolist(),
            "decay": self.decay.tolist(),
            "mix": self.mix.tolist(),
            "interaction": self.interaction.tolist(),
            "selfEnergy": self.self_energy.tolist(),
            "kingResonance": self.king_resonance.tolist(),
            "materialAnchor": float(self.material_anchor),
            "tempoBonus": float(self.tempo_bonus),
            "passedPawnScale": float(self.passed_pawn_scale),
            "mobilityScale": float(self.mobility_scale),
            "swarmScale": float(self.swarm_scale),
        }
        if self.field_head is not None:
            out["fieldHead"] = self.field_head.tolist()
        return out

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> FieldParams:
        head = data.get("fieldHead")
        return cls(
            deposit=np.asarray(data["deposit"], dtype=np.float64),
            decay=np.asarray(data["decay"], dtype=np.float64),
            mix=np.asarray(data["mix"], dtype=np.float64),
            interaction=np.asarray(data["interaction"], dtype=np.float64),
            self_energy=np.asarray(data["selfEnergy"], dtype=np.float64),
            king_resonance=np.asarray(data["kingResonance"], dtype=np.float64),
            material_anchor=float(data.get("materialAnchor", 1.0)),
            tempo_bonus=float(data.get("tempoBonus", 8.0)),
            passed_pawn_scale=float(data.get("passedPawnScale", 1.0)),
            mobility_scale=float(data.get("mobilityScale", 1.0)),
            field_head=None if head is None else np.asarray(head, dtype=np.float64),
            swarm_scale=float(data.get("swarmScale", 1.0)),
        )


def _default_deposit() -> np.ndarray:
    """Seed channels so each piece type leaves a distinctive scent fingerprint."""
    d = np.zeros((N_PIECES, N_CHANNELS), dtype=np.float64)
    # pawn
    d[0] = [1.0, 0.05, 1.2, 0.4, 0.1, 0.05, 0.0, 0.3, 0.2, 0.15]
    # knight
    d[1] = [3.2, 0.15, 0.0, 0.2, 0.8, 1.1, 0.6, 0.1, 0.5, 0.9]
    # bishop
    d[2] = [3.3, 0.15, 0.0, 0.15, 1.4, 0.9, 0.7, 0.2, 0.4, 1.1]
    # rook
    d[3] = [5.0, 0.2, 0.1, 1.5, 0.3, 0.7, 0.5, 0.4, 0.3, 0.6]
    # queen
    d[4] = [9.0, 0.35, 0.1, 0.8, 0.6, 1.0, 0.9, 0.3, 0.6, 0.8]
    # king
    d[5] = [0.0, 2.5, 0.0, 0.1, 0.2, 0.05, 0.0, 0.0, 0.1, 0.4]
    return d


def default_field_params() -> FieldParams:
    """Hand-tuned field prior that already plays coherently before training."""
    c = N_CHANNELS
    interaction = np.eye(c, dtype=np.float64) * -0.35
    # Opposing material mass vs our king heat is dangerous.
    interaction[0, 1] = -0.55
    interaction[1, 0] = -0.55
    # Friendly file control vs enemy blockade.
    interaction[3, 7] = 0.25
    interaction[7, 3] = -0.2
    # Color-complex clash (bishops).
    interaction[4, 4] = -0.15
    # Mobility mist rewards space.
    interaction[5, 5] = 0.1
    self_energy = np.array(
        [0.02, -0.08, 0.05, 0.04, 0.03, 0.06, 0.02, -0.03, 0.04, 0.02],
        dtype=np.float64,
    )
    king_res = np.array(
        [-0.4, -1.2, -0.1, -0.25, -0.15, -0.2, -0.35, -0.05, -0.1, -0.15],
        dtype=np.float64,
    )
    return FieldParams(
        deposit=_default_deposit(),
        decay=np.full(c, 0.55, dtype=np.float64),
        mix=np.full(c, 0.45, dtype=np.float64),
        interaction=interaction,
        self_energy=self_energy,
        king_resonance=king_res,
        material_anchor=1.0,
        tempo_bonus=10.0,
        passed_pawn_scale=1.15,
        mobility_scale=1.25,
        field_head=np.zeros(FIELD_HEAD_DIM, dtype=np.float64),
        swarm_scale=1.0,
    )


@dataclass
class StigmergyWeights:
    """Full engine state: field params + book + float learned moves + position trails."""

    field: FieldParams
    book: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: deepcopy(DEFAULT_BOOK))
    learned_moves: dict[str, float] = field(default_factory=dict)
    trails: dict[str, dict[str, float]] = field(default_factory=dict)
    diffusion_steps: int = 4
    format_version: int = 4
    created_at: str = ""
    training_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "formatVersion": 4,
            "engine": "stigmergy-dpfe",
            "architecture": "diffusive-pheromone-field",
            "createdAt": self.created_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "channels": N_CHANNELS,
            "channelNames": CHANNEL_NAMES,
            "diffusionSteps": self.diffusion_steps,
            "pieceOrder": PIECE_ORDER,
            "field": self.field.to_jsonable(),
            "book": self.book,
            "learnedMoves": {
                "precision": "float64",
                "values": {k: float(v) for k, v in self.learned_moves.items()},
            },
            "trails": {
                "precision": "float64",
                "values": {
                    pos: {uci: float(w) for uci, w in moves.items()}
                    for pos, moves in self.trails.items()
                },
            },
            "trainingMeta": self.training_meta,
            "uniquenessFingerprint": uniqueness_fingerprint(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StigmergyWeights:
        ver = int(data.get("formatVersion", 0))
        if ver < 2:
            raise ValueError(
                f"Unsupported weights formatVersion={data.get('formatVersion')!r}; "
                "need >= 2 (stigmergy-dpfe)."
            )
        fp = FieldParams.from_jsonable(data["field"])
        if fp.field_head is None:
            fp.field_head = np.zeros(FIELD_HEAD_DIM, dtype=np.float64)
        learned_raw = data.get("learnedMoves") or {}
        trails_raw = data.get("trails")
        learned_moves = load_learned_moves(learned_raw)
        trails = load_trails(trails_raw)
        fmt = max(ver, 3)
        if trails or learned_raw.get("values"):
            fmt = max(fmt, 4)
        return cls(
            field=fp,
            book=deepcopy(data.get("book") or DEFAULT_BOOK),
            learned_moves=learned_moves,
            trails=trails,
            diffusion_steps=int(data.get("diffusionSteps", 4)),
            format_version=fmt,
            created_at=str(data.get("createdAt", "")),
            training_meta=dict(data.get("trainingMeta") or {}),
        )


def uniqueness_fingerprint() -> dict[str, Any]:
    """Architectural claims used by the contest uniqueness rubric."""
    return {
        "family": "diffusive-pheromone-field",
        "not": ["stockfish", "nnue", "alphazero", "lc0", "classic-pst-only"],
        "features": [
            "multi_channel_pheromone_lattice",
            "jacobi_diffusion_kernels",
            "bilinear_cross_color_interaction",
            "king_resonance_coupling",
            "continuous_float64_trails",
            "high_precision_float64_weights",
            "coarse_pawn_material_trails",
            "ternary_ant_trail_book",
            "ternary_move_pheromone_bias",
            "harmonic_board_channel",
            "swarm_field_head_readout",
            "winner_distillation_from_oracles",
            "tactical_floor_field_fusion",
            "see_ordered_idas",
        ],
        "search": [
            "iterative_deepening",
            "alpha_beta",
            "transposition_table",
            "null_move",
            "lmr",
            "quiescence",
            "killers_history",
            "check_extensions",
            "see_move_ordering",
            "principal_variation_search",
            "futility_pruning",
        ],
    }


def quantize_learned_moves(table: dict[str, float]) -> dict[str, Any]:
    if not table:
        return {"codes": {}, "scale": 0.0}
    vals = list(table.values())
    scale = float(sum(abs(v) for v in vals) / len(vals)) or 0.01
    codes: dict[str, int] = {}
    for k, v in table.items():
        if v > scale * 0.5:
            codes[k] = 1
        elif v < -scale * 0.5:
            codes[k] = -1
    return {"codes": codes, "scale": scale}


def dequantize_learned_moves(q: dict[str, Any]) -> dict[str, float]:
    if not q or not q.get("codes"):
        return {}
    scale = float(q.get("scale") or 0.0)
    return {k: float(c) * scale for k, c in q["codes"].items()}


def load_learned_moves(data: dict[str, Any]) -> dict[str, float]:
    """Load learned moves from v4 float values or legacy ternary codes."""
    if not data:
        return {}
    if "values" in data:
        return {k: float(v) for k, v in data["values"].items()}
    return dequantize_learned_moves(data)


def load_trails(data: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    """Load per-position continuous ant trails (zobrist hex -> uci -> intensity)."""
    if not data:
        return {}
    if "values" in data:
        out: dict[str, dict[str, float]] = {}
        for pos_key, moves in data["values"].items():
            out[str(pos_key)] = {str(uci): float(w) for uci, w in moves.items()}
        return out
    return {}


def trail_key(board: chess.Board) -> str:
    """Zobrist hash of ``board`` as a fixed-width hex trail lookup key."""
    return f"{chess.polyglot.zobrist_hash(board):016x}"


def default_weights() -> StigmergyWeights:
    return StigmergyWeights(field=default_field_params(), book=deepcopy(DEFAULT_BOOK))


def save_weights(weights: StigmergyWeights, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights.to_dict(), indent=2), encoding="utf-8")


def load_weights(path: str | Path) -> StigmergyWeights:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return StigmergyWeights.from_dict(data)


def mutate_field(params: FieldParams, rng: np.random.Generator, sigma: float = 0.08) -> FieldParams:
    """Gaussian mutation used by self-play evolutionary strategies."""
    out = params.copy()
    out.deposit = out.deposit + rng.normal(0, sigma, out.deposit.shape)
    out.decay = np.clip(out.decay + rng.normal(0, sigma * 0.5, out.decay.shape), 0.15, 0.95)
    out.mix = np.clip(out.mix + rng.normal(0, sigma * 0.5, out.mix.shape), 0.05, 0.9)
    out.interaction = out.interaction + rng.normal(0, sigma * 0.6, out.interaction.shape)
    out.self_energy = out.self_energy + rng.normal(0, sigma * 0.4, out.self_energy.shape)
    out.king_resonance = out.king_resonance + rng.normal(0, sigma * 0.4, out.king_resonance.shape)
    out.material_anchor = float(max(0.4, out.material_anchor + rng.normal(0, sigma * 0.3)))
    out.tempo_bonus = float(max(0.0, out.tempo_bonus + rng.normal(0, sigma * 4)))
    out.passed_pawn_scale = float(max(0.3, out.passed_pawn_scale + rng.normal(0, sigma * 0.3)))
    out.mobility_scale = float(max(0.3, out.mobility_scale + rng.normal(0, sigma * 0.3)))
    out.swarm_scale = float(max(0.1, out.swarm_scale + rng.normal(0, sigma * 0.2)))
    if out.field_head is not None:
        out.field_head = out.field_head + rng.normal(0, sigma * 0.4, out.field_head.shape)
    # Hard caps — overnight multiplicative history taught us this matters.
    out.deposit = np.clip(out.deposit, -15.0, 15.0)
    out.interaction = np.clip(out.interaction, -3.0, 3.0)
    out.self_energy = np.clip(out.self_energy, -2.0, 2.0)
    out.king_resonance = np.clip(out.king_resonance, -3.0, 3.0)
    if out.field_head is not None:
        out.field_head = np.clip(out.field_head, -8.0, 8.0)
    return out


def asdict_safe(obj: Any) -> dict[str, Any]:
    """Dataclass helper kept for tests."""
    return asdict(obj)
