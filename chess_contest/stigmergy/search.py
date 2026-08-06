"""Deep iterative-deepening alpha-beta search for Stigmergy.

No Stockfish at play time. Strength = float64 trails/book/coarse + swarm
policy/value (offline-distilled) + deep IDAS. Long think times are intentional.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass

import chess
import chess.polyglot

from chess_contest.stigmergy.coarse import coarse_trail_move
from chess_contest.stigmergy.tactics import (
    hanging_penalty,
    material_of,
    see,
    tactical_floor,
    tactical_floor_fast,
)
from chess_contest.stigmergy.weights import CODE_WEIGHT, StigmergyWeights, trail_key

MATE = 100_000
# Finite infinity for alpha-beta. ±1e18 is unsafe in float64: (1e18 - 1) == 1e18,
# so null-window (-beta, -beta+1) collapses and null-move + quiesce false-cutoffs
# every node at depth >= 3 (search returns -inf and hangs pieces).
INF = MATE * 10
DELTA_MARGIN = 200  # cp: skip hopeless captures in quiescence
_SWARM = None  # optional SwarmNet — set by pure_gm / try_load_swarm


def set_swarm(net) -> None:
    """Install the offline-distilled swarm net for search (or None to clear)."""
    global _SWARM
    _SWARM = net


def _search_eval(board: chess.Board, weights: StigmergyWeights) -> float:
    """Side-to-move score for alpha-beta leaves — classical + hanging (fast).

    Swarm value is applied at the root (policy prior / re-rank), never in the
    hot tree, so long-think IDAS keeps workable nps. No Stockfish.
    """
    del weights
    floor = tactical_floor_fast(board) + hanging_penalty(board)
    tempo = 8.0 if board.turn == chess.WHITE else -8.0
    white = floor + tempo
    return white if board.turn == chess.WHITE else -white


@dataclass
class SearchResult:
    move: chess.Move | None
    score: float
    depth: int
    nodes: int
    book: bool = False
    trail: bool = False
    pv: list[str] | None = None


class _TTEntry:
    __slots__ = ("depth", "flag", "key", "move", "score")

    def __init__(self, key: int, depth: int, score: float, flag: int, move: chess.Move | None):
        self.key = key
        self.depth = depth
        self.score = score
        self.flag = flag  # 0 exact, -1 upper, 1 lower
        self.move = move


class Searcher:
    """Negamax searcher bound to a weight set."""

    def __init__(self, weights: StigmergyWeights):
        self.weights = weights
        self.nodes = 0
        self.deadline = 0.0
        self.tt: dict[int, _TTEntry] = {}
        self.killers: dict[int, list[chess.Move | None]] = {}
        self.history: dict[int, int] = {}
        self._rep_stack: list[int] = []

    def _check_time(self) -> None:
        if (self.nodes & 2047) == 0 and time.perf_counter() >= self.deadline:
            raise RuntimeError("TIME_UP")

    def _book_entry_score(self, entry: dict) -> float:
        if "w" in entry:
            return float(entry["w"])
        code = int(entry.get("code", 0))
        return float(CODE_WEIGHT.get(code, 1))

    def book_move(self, board: chess.Board) -> chess.Move | None:
        hist = board.move_stack
        key = "".join(m.uci()[:4] for m in hist)
        entry = self.weights.book.get(key)
        if not entry:
            return None
        best = max(entry, key=self._book_entry_score)
        # Require strong continuous weight - refuse polluted low-mass overnight trails.
        score = self._book_entry_score(best)
        if score < 8.0 and int(best.get("code", 0)) < 1:
            return None
        if score < 3.0:
            return None
        uci = best["m"]
        try:
            move = chess.Move.from_uci(uci if len(uci) >= 4 else uci + "q")
            if len(uci) == 4 and move not in board.legal_moves:
                for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                    m2 = chess.Move.from_uci(uci + chess.piece_symbol(promo))
                    if m2 in board.legal_moves:
                        return m2
                return None
            return move if move in board.legal_moves else None
        except ValueError:
            return None

    def trail_move(self, board: chess.Board) -> chess.Move | None:
        """High-confidence continuous ant-trail move from distilled oracle lines.

        Thresholds rise when a swarm net is loaded so weak overnight trails do not
        short-circuit long-think search that the swarm+IDAS path needs for GM+.
        """
        pos_trails = self.weights.trails.get(trail_key(board))
        if not pos_trails:
            return None
        scored: list[tuple[chess.Move, float]] = []
        for move in board.legal_moves:
            uci_full = move.uci()
            uci = uci_full[:4]
            w = pos_trails.get(uci_full)
            if w is None:
                w = pos_trails.get(uci)
            if w is not None and float(w) > 0.05:
                scored.append((move, float(w)))
        if not scored:
            return None
        scored.sort(key=lambda t: t[1], reverse=True)
        best_move, best_w = scored[0]
        second_w = scored[1][1] if len(scored) > 1 else 0.0
        # With swarm: only auto-play strong offline-distilled trails (still SF-free).
        if _SWARM is not None:
            if (
                best_w >= 8.0
                and best_w >= second_w * 1.25
                and see(board, best_move) >= -30
                and not self._major_hang_quick(board, best_move)
            ):
                return best_move
            return None
        if best_w >= second_w * 1.08 or best_w >= 1.0:
            return best_move
        if len(scored) == 1 and best_w >= 0.3:
            return best_move
        return best_move if best_w >= 0.5 else None

    def _move_score(
        self,
        board: chess.Board,
        move: chess.Move,
        ply: int,
        tt_move: chess.Move | None,
        *,
        check_checks: bool = False,
        pos_trails: dict[str, float] | None = None,
    ) -> float:
        if tt_move is not None and move == tt_move:
            return 1_000_000.0
        score = 0.0
        if board.is_capture(move) or board.is_en_passant(move):
            # MVV-LVA for ordering (full SEE is reserved for qsearch pruning).
            victim = board.piece_at(move.to_square)
            if board.is_en_passant(move):
                v = material_of(chess.PAWN)
            else:
                v = material_of(victim.piece_type) if victim else 0
            attacker = board.piece_at(move.from_square)
            a = material_of(attacker.piece_type) if attacker else 0
            score += 10_000 + 10 * v - a
        if move.promotion:
            score += 8000 + 100 * move.promotion
        if check_checks and board.gives_check(move):
            score += 7000
        killers = self.killers.get(ply)
        if killers:
            if killers[0] == move:
                score += 9000
            elif killers[1] == move:
                score += 8000
        score += min(self.history.get(move.from_square << 6 | move.to_square, 0), 5000)
        uci = move.uci()[:4]
        if pos_trails:
            score += float(pos_trails.get(uci, 0.0)) * 500.0
            score += float(pos_trails.get(move.uci(), 0.0)) * 500.0
        piece = board.piece_at(move.from_square)
        if piece is not None:
            lk = (
                f"{piece.symbol().lower()}"
                f"{chess.square_name(move.from_square)}"
                f"{chess.square_name(move.to_square)}"
            )
            bias = self.weights.learned_moves.get(lk, 0.0)
            score += bias * 400.0
        if _SWARM is not None and ply == 0:
            with contextlib.suppress(Exception):
                score += _SWARM.policy_score(board, move) * 80.0
        return score

    def _ordered_moves(
        self,
        board: chess.Board,
        ply: int,
        tt_move: chess.Move | None,
        *,
        check_checks: bool = False,
    ) -> list[chess.Move]:
        moves = list(board.legal_moves)
        pos_trails = self.weights.trails.get(trail_key(board))
        moves.sort(
            key=lambda m: self._move_score(
                board, m, ply, tt_move, check_checks=check_checks, pos_trails=pos_trails
            ),
            reverse=True,
        )
        return moves

    def quiesce(self, board: chess.Board, alpha: float, beta: float, ply: int) -> float:
        self.nodes += 1
        self._check_time()
        if board.is_checkmate():
            return -MATE + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        # Fast stand-pat (1 diffusion step) - tactics floor still exact.
        stand = _search_eval(board, self.weights)
        if stand >= beta:
            return beta
        if alpha < stand:
            alpha = stand
        if ply > 48:
            return alpha

        in_check = board.is_check()
        if in_check:
            moves = list(board.legal_moves)
        else:
            moves = [
                m
                for m in board.legal_moves
                if board.is_capture(m)
                or m.promotion
                or board.is_en_passant(m)
                or board.gives_check(m)
            ]
        moves.sort(key=lambda m: self._move_score(board, m, ply, None), reverse=True)

        for move in moves[:32]:
            if not in_check and board.is_capture(move):
                # Delta pruning: skip captures that cannot raise alpha.
                victim = board.piece_at(move.to_square)
                vmat = material_of(victim.piece_type) if victim else material_of(chess.PAWN)
                if move.promotion:
                    vmat += material_of(move.promotion) - material_of(chess.PAWN)
                if stand + vmat + DELTA_MARGIN < alpha:
                    continue
                if see(board, move) < -DELTA_MARGIN:
                    continue

            board.push(move)
            try:
                score = -self.quiesce(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        allow_null: bool,
    ) -> float:
        self.nodes += 1
        self._check_time()

        if board.is_checkmate():
            return -MATE + ply
        if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
            return 0.0
        # Cheap repetition: same zobrist already on the search path (not full history).
        key = chess.polyglot.zobrist_hash(board)
        if ply > 0 and key in self._rep_stack:
            return 0.0
        self._rep_stack.append(key)
        try:
            return self._negamax_inner(board, depth, alpha, beta, ply, allow_null, key)
        finally:
            self._rep_stack.pop()

    def _negamax_inner(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        allow_null: bool,
        key: int,
    ) -> float:
        alpha_orig = alpha
        tt = self.tt.get(key)
        tt_move = tt.move if tt and tt.key == key else None
        if tt is not None and tt.key == key and tt.depth >= depth:
            if tt.flag == 0:
                return tt.score
            if tt.flag == 1:
                alpha = max(alpha, tt.score)
            elif tt.flag == -1:
                beta = min(beta, tt.score)
            if alpha >= beta:
                return tt.score

        in_check = board.is_check()
        ext = 1 if in_check else 0
        if depth + ext <= 0:
            return self.quiesce(board, alpha, beta, ply)

        # Reverse futility / static null-move style prune at shallow depth.
        if depth <= 2 and not in_check and abs(beta) < MATE - 1000:
            stand = _search_eval(board, self.weights)
            margin = 120 * depth
            if stand - margin >= beta:
                return stand - margin

        # Null-move pruning (skip when window is degenerate).
        if (
            allow_null
            and depth >= 3
            and not in_check
            and beta < INF - 1
            and (board.occupied_co[board.turn] & ~board.pawns & ~board.kings)
        ):
            R = 3 if depth >= 6 else 2
            board.push(chess.Move.null())
            try:
                score = -self.negamax(
                    board, max(0, depth - 1 - R), -beta, -beta + 1, ply + 1, False
                )
            finally:
                board.pop()
            if score >= beta:
                return beta

        moves = self._ordered_moves(board, ply, tt_move, check_checks=False)
        if not moves:
            return -MATE + ply if in_check else 0.0

        best = -INF
        best_move = moves[0]
        for i, move in enumerate(moves):
            is_cap = board.is_capture(move) or board.is_en_passant(move)
            is_promo = bool(move.promotion)

            # Futility pruning: late quiet moves at depth 1.
            if (
                depth == 1
                and i >= 3
                and not in_check
                and not is_cap
                and not is_promo
                and abs(alpha) < MATE - 1000
            ):
                continue

            reduction = 0
            if (
                depth >= 3
                and i >= 3
                and not in_check
                and not is_cap
                and not is_promo
            ):
                reduction = 1 + (i // 6)
                if depth >= 6 and i >= 8:
                    reduction += 1

            board.push(move)
            try:
                # PVS: first move full window; rest null window then research.
                if i == 0:
                    score = -self.negamax(
                        board, depth - 1 - reduction + ext, -beta, -alpha, ply + 1, True
                    )
                else:
                    score = -self.negamax(
                        board, depth - 1 - reduction + ext, -alpha - 1, -alpha, ply + 1, True
                    )
                    if score > alpha and score < beta:
                        score = -self.negamax(
                            board, depth - 1 + ext, -beta, -alpha, ply + 1, True
                        )
                if reduction and score > alpha:
                    score = -self.negamax(board, depth - 1 + ext, -beta, -alpha, ply + 1, True)
            finally:
                board.pop()

            if score > best:
                best = score
                best_move = move
            if best > alpha:
                alpha = best
            if alpha >= beta:
                if not is_cap:
                    k = self.killers.setdefault(ply, [None, None])
                    k[1] = k[0]
                    k[0] = move
                    hkey = move.from_square << 6 | move.to_square
                    self.history[hkey] = self.history.get(hkey, 0) + depth * depth
                break

        flag = 0
        if best <= alpha_orig:
            flag = -1
        elif best >= beta:
            flag = 1
        self.tt[key] = _TTEntry(key, depth, best, flag, best_move)
        if len(self.tt) > 250_000:
            self.tt.clear()
        return best

    def search(self, board: chess.Board, time_ms: int, max_depth: int = 14) -> SearchResult:
        return self.search_with_root_update(board, time_ms=time_ms, max_depth=max_depth)

    def _root_search(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        pv_move: chess.Move,
    ) -> float:
        moves = self._ordered_moves(board, 0, pv_move, check_checks=True)
        # With swarm: focus root nodes on policy beam + tactics (deeper effective search).
        if _SWARM is not None and len(moves) > 14:
            beam: set[chess.Move] = set()
            with contextlib.suppress(Exception):
                beam.update(_SWARM.top_moves(board, k=10))
            if pv_move is not None:
                beam.add(pv_move)
            filtered = [
                m
                for m in moves
                if m in beam
                or board.is_capture(m)
                or board.is_en_passant(m)
                or m.promotion
                or board.gives_check(m)
            ]
            if len(filtered) >= 8:
                moves = filtered
        best = -INF
        best_move = moves[0]
        for i, move in enumerate(moves):
            board.push(move)
            try:
                if i == 0:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, 1, True)
                else:
                    score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, 1, True)
                    if score > alpha and score < beta:
                        score = -self.negamax(board, depth - 1, -beta, -alpha, 1, True)
            finally:
                board.pop()
            if score > best:
                best = score
                best_move = move
            if best > alpha:
                alpha = best
        self._last_best_move = best_move  # type: ignore[attr-defined]
        return best

    def _mate_in_one(self, board: chess.Board, legal: list[chess.Move]) -> chess.Move | None:
        for move in legal:
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return move
        return None

    def _avoid_hanging_root(
        self, board: chess.Board, move: chess.Move, legal: list[chess.Move]
    ) -> chess.Move:
        """If the chosen root move hangs a major, switch to a *safe* alternative.

        Never replace a hanging move with a different hanging move — that path
        previously preferred developing Nc3 while leaving Ne5 en prise.
        """
        if see(board, move) >= -50 and not self._major_hang_quick(board, move):
            return move

        safe: list[chess.Move] = [
            cand
            for cand in legal
            if see(board, cand) >= -50 and not self._major_hang_quick(board, cand)
        ]
        if not safe:
            return move

        scored: list[tuple[float, chess.Move]] = []
        mover_white = board.turn == chess.WHITE
        for cand in safe:
            board.push(cand)
            try:
                if board.is_checkmate():
                    return cand
                white_floor = tactical_floor(board)
                our_score = white_floor if mover_white else -white_floor
            finally:
                board.pop()
            our_score += 0.02 * see(board, cand)
            scored.append((our_score, cand))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0][1]

    def search_with_root_update(
        self, board: chess.Board, time_ms: int, max_depth: int = 14
    ) -> SearchResult:
        """Iterative deepening that keeps the best completed-depth move."""
        # Offline SF trails: with swarm, only autoplay very strong distilled lines
        # (strength ≥40 from pure_gm teacher). Weaker/polluted trails stay order-bias.
        trail = self.trail_move(board)
        if trail is not None:
            if _SWARM is None:
                return SearchResult(
                    move=trail, score=0.0, depth=0, nodes=0, book=True, trail=True
                )
            # Confirm mass is SF-teacher grade before short-circuiting search.
            pos = self.weights.trails.get(trail_key(board)) or {}
            tw = float(pos.get(trail.uci(), pos.get(trail.uci()[:4], 0.0)))
            if tw >= 40.0 and see(board, trail) >= -20 and not self._major_hang_quick(
                board, trail
            ):
                return SearchResult(
                    move=trail, score=0.0, depth=0, nodes=0, book=True, trail=True
                )

        if _SWARM is None:
            book = self.book_move(board)
            if book is not None:
                return SearchResult(move=book, score=0.0, depth=0, nodes=0, book=True)

            coarse = coarse_trail_move(self.weights, board)
            if coarse is not None:
                return SearchResult(
                    move=coarse, score=0.0, depth=0, nodes=0, book=True, trail=True
                )

        legal = list(board.legal_moves)
        if not legal:
            return SearchResult(move=None, score=0.0, depth=0, nodes=0)
        if len(legal) == 1:
            return SearchResult(move=legal[0], score=0.0, depth=1, nodes=1)

        mate = self._mate_in_one(board, legal)
        if mate is not None:
            return SearchResult(
                move=mate, score=float(MATE - 1), depth=1, nodes=len(legal), book=False
            )

        # Swarm policy priors for ordering / optional instant play (no SF).
        swarm_move = None
        swarm_margin = 0.0
        policy_top: list[chess.Move] = []
        if _SWARM is not None:
            try:
                swarm_move, swarm_margin = _SWARM.choose_with_margin(board)
                policy_top = _SWARM.top_moves(board, k=5)
            except Exception:
                swarm_move, swarm_margin, policy_top = None, 0.0, []
            # Policy-first for crush path: trained swarm plays when clearly ahead
            # (SEE/hang filtered). Long IDAS only when the net is uncertain.
            if (
                swarm_move is not None
                and swarm_margin >= 0.85
                and see(board, swarm_move) >= -20
                and not self._major_hang_quick(board, swarm_move)
            ):
                return SearchResult(
                    move=swarm_move, score=0.0, depth=0, nodes=0, book=False, trail=True
                )
            if policy_top:
                preferred = {m: i for i, m in enumerate(policy_top)}
                legal = sorted(
                    legal,
                    key=lambda m: (
                        preferred.get(m, 99),
                        -(_SWARM.policy_score(board, m) if _SWARM else 0.0),
                    ),
                )

        # Honour long think budgets — above-GM path uses multi-second moves.
        think_ms = max(50, time_ms)
        self.nodes = 0
        self.deadline = time.perf_counter() + max(0.05, think_ms / 1000.0)
        self.tt.clear()
        self.killers.clear()
        self.history.clear()
        self._rep_stack = []

        best_move = legal[0]
        best_score = 0.0
        depth_reached = 0
        try:
            for depth in range(1, max_depth + 1):
                if depth == 1:
                    score = self._root_search(board, depth, -INF, INF, best_move)
                    best_score = score
                    best_move = getattr(self, "_last_best_move", best_move)
                    depth_reached = 1
                    continue
                window = 60.0
                alpha = best_score - window
                beta = best_score + window
                completed = False
                for _ in range(6):
                    score = self._root_search(board, depth, alpha, beta, best_move)
                    if score <= alpha:
                        alpha -= window
                        window *= 2
                        continue
                    if score >= beta:
                        beta += window
                        window *= 2
                        continue
                    best_score = score
                    best_move = getattr(self, "_last_best_move", best_move)
                    completed = True
                    break
                if completed:
                    depth_reached = depth
                if abs(best_score) > MATE - 1000:
                    break
                if time.perf_counter() >= self.deadline:
                    break
        except RuntimeError:
            if hasattr(self, "_last_best_move"):
                best_move = self._last_best_move

        best_move = self._avoid_hanging_root(board, best_move, legal)
        # Soft root blend: keep search PV unless swarm+classical clearly prefer
        # another SEE-safe candidate from the policy beam (never Stockfish).
        if _SWARM is not None and depth_reached >= 1:
            candidates: list[chess.Move] = []
            for m in [best_move, swarm_move, *policy_top, *legal[:6]]:
                if m is not None and m in board.legal_moves and m not in candidates:
                    candidates.append(m)
            scored: list[tuple[float, chess.Move]] = []
            mover_white = board.turn == chess.WHITE
            for cand in candidates:
                if see(board, cand) < -100:
                    continue
                if self._major_hang_quick(board, cand):
                    continue
                board.push(cand)
                try:
                    if board.is_checkmate():
                        scored.append((1e6, cand))
                        continue
                    # tactical_floor_fast is white-positive.
                    floor = tactical_floor_fast(board)
                    classical = floor if mover_white else -floor
                    swarm_v = 0.0
                    with contextlib.suppress(Exception):
                        # After our move, opponent to move: low opp value ⇒ good for us.
                        swarm_v = -float(_SWARM.value_stm(board))
                    # Search PV gets a bonus so we do not casually override IDAS.
                    bonus = 35.0 if cand == best_move else 0.0
                    scored.append((0.55 * classical + 0.45 * swarm_v + bonus, cand))
                finally:
                    board.pop()
            if scored:
                scored.sort(key=lambda t: t[0], reverse=True)
                pick = scored[0][1]
                # Only switch if the alternative clearly beats the search PV.
                pv_score = next((s for s, m in scored if m == best_move), None)
                if pv_score is None or scored[0][0] >= pv_score + 40.0:
                    best_move = pick
                    best_score = scored[0][0]

        return SearchResult(
            move=best_move,
            score=best_score,
            depth=depth_reached,
            nodes=self.nodes,
            book=False,
            pv=[best_move.uci()] if best_move else None,
        )

    def _major_hang_quick(self, board: chess.Board, move: chess.Move) -> bool:
        board.push(move)
        try:
            for reply in board.legal_moves:
                if not board.is_capture(reply):
                    continue
                if see(board, reply) < 0:
                    continue
                victim = board.piece_at(reply.to_square)
                if victim is not None and material_of(victim.piece_type) >= 300:
                    return True
        finally:
            board.pop()
        return False
