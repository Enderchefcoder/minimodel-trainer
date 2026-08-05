"""UCI wrapper for Stockfish — sparring partner only, never our architecture."""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Any

import chess

DEFAULT_STOCKFISH = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")


@dataclass
class StockfishConfig:
    path: str = DEFAULT_STOCKFISH
    threads: int = 1
    hash_mb: int = 64
    skill_level: int | None = None  # 0-20; None = full strength
    limit_strength: bool = False
    uci_elo: int | None = None  # 1320-3190 when limit_strength
    movetime_ms: int = 200
    multipv: int = 1


class StockfishEngine:
    """Minimal UCI client. Used as an oracle/opponent, not as our eval."""

    def __init__(self, cfg: StockfishConfig | None = None):
        self.cfg = cfg or StockfishConfig()
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._start()

    def _start(self) -> None:
        self._proc = subprocess.Popen(
            [self.cfg.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._cmd("uci")
        self._wait_for("uciok")
        self._cmd(f"setoption name Threads value {self.cfg.threads}")
        self._cmd(f"setoption name Hash value {self.cfg.hash_mb}")
        if self.cfg.skill_level is not None:
            self._cmd(f"setoption name Skill Level value {self.cfg.skill_level}")
        if self.cfg.limit_strength and self.cfg.uci_elo is not None:
            self._cmd("setoption name UCI_LimitStrength value true")
            self._cmd(f"setoption name UCI_Elo value {self.cfg.uci_elo}")
        else:
            self._cmd("setoption name UCI_LimitStrength value false")
        if self.cfg.multipv > 1:
            self._cmd(f"setoption name MultiPV value {self.cfg.multipv}")
        self._cmd("isready")
        self._wait_for("readyok")

    def _cmd(self, line: str) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def _wait_for(self, token: str, timeout_lines: int = 5000) -> list[str]:
        assert self._proc and self._proc.stdout
        lines: list[str] = []
        for _ in range(timeout_lines):
            line = self._proc.stdout.readline()
            if not line:
                break
            line = line.rstrip("\n")
            lines.append(line)
            if token in line:
                break
        return lines

    def new_game(self) -> None:
        with self._lock:
            self._cmd("ucinewgame")
            self._cmd("isready")
            self._wait_for("readyok")

    def set_elo(self, elo: int | None) -> None:
        """Reconfigure limited strength. elo=None → full strength (Skill 20)."""
        with self._lock:
            if elo is None:
                self.cfg.limit_strength = False
                self.cfg.uci_elo = None
                self.cfg.skill_level = 20
                self._cmd("setoption name UCI_LimitStrength value false")
                self._cmd("setoption name Skill Level value 20")
            else:
                elo = int(max(1320, min(3190, elo)))
                self.cfg.limit_strength = True
                self.cfg.uci_elo = elo
                self._cmd("setoption name UCI_LimitStrength value true")
                self._cmd(f"setoption name UCI_Elo value {elo}")
            self._cmd("isready")
            self._wait_for("readyok")

    def choose(self, board: chess.Board, movetime_ms: int | None = None) -> chess.Move:
        with self._lock:
            mt = movetime_ms if movetime_ms is not None else self.cfg.movetime_ms
            fen = board.fen()
            self._cmd(f"position fen {fen}")
            self._cmd(f"go movetime {mt}")
            best: chess.Move | None = None
            lines = self._wait_for("bestmove")
            for line in lines:
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] != "(none)":
                        best = chess.Move.from_uci(parts[1])
            if best is None or best not in board.legal_moves:
                # Fallback: first legal.
                best = next(iter(board.legal_moves))
            return best

    def analyse_top(
        self, board: chess.Board, movetime_ms: int = 300, multipv: int = 3
    ) -> list[dict[str, Any]]:
        """Return top multipv moves with scores (SF POV for side to move)."""
        with self._lock:
            self._cmd(f"setoption name MultiPV value {multipv}")
            self._cmd(f"position fen {board.fen()}")
            self._cmd(f"go movetime {movetime_ms}")
            infos: dict[int, dict[str, Any]] = {}
            lines = self._wait_for("bestmove")
            for line in lines:
                if not line.startswith("info") or " pv " not in line:
                    continue
                if " multipv " not in line and multipv > 1:
                    # Still accept single-pv lines.
                    pass
                mpv = 1
                score_cp = None
                score_mate = None
                pv = []
                toks = line.split()
                i = 0
                while i < len(toks):
                    t = toks[i]
                    if t == "multipv" and i + 1 < len(toks):
                        mpv = int(toks[i + 1])
                        i += 2
                        continue
                    if t == "score" and i + 2 < len(toks):
                        if toks[i + 1] == "cp":
                            score_cp = int(toks[i + 2])
                            i += 3
                            continue
                        if toks[i + 1] == "mate":
                            score_mate = int(toks[i + 2])
                            i += 3
                            continue
                    if t == "pv":
                        pv = toks[i + 1 :]
                        break
                    i += 1
                if pv:
                    infos[mpv] = {
                        "uci": pv[0],
                        "pv": pv,
                        "cp": score_cp,
                        "mate": score_mate,
                    }
            self._cmd("setoption name MultiPV value 1")
            return [infos[k] for k in sorted(infos)]

    def close(self) -> None:
        with self._lock:
            if self._proc is None:
                return
            with contextlib.suppress(Exception):
                self._cmd("quit")
            with contextlib.suppress(Exception):
                self._proc.kill()
            self._proc = None

    def __enter__(self) -> StockfishEngine:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def stockfish_available(path: str = DEFAULT_STOCKFISH) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)
