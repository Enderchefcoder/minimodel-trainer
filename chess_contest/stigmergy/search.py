"""Deep iterative-deepening alpha-beta search for Stigmergy.

Slower than Stockfish by design — uniqueness is in the field eval, strength
comes from depth: TT, null-move, LMR, killers, history, quiescence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chess
import chess.polyglot

from chess_contest.stigmergy.evaluate import relative_eval
from chess_contest.stigmergy.weights import CODE_WEIGHT, StigmergyWeights

MATE = 100_000
TIME_UP = object()


@dataclass
class SearchResult:
    move: chess.Move | None
    score: float
    depth: int
    nodes: int
    book: bool = False
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

    def _check_time(self) -> None:
        if (self.nodes & 2047) == 0 and time.perf_counter() >= self.deadline:
            raise RuntimeError("TIME_UP")

    def book_move(self, board: chess.Board) -> chess.Move | None:
        hist = board.move_stack
        key = "".join(m.uci()[:4] for m in hist)
        entry = self.weights.book.get(key)
        if not entry:
            return None
        best = max(
            entry, key=lambda e: (int(e.get("code", 0)), CODE_WEIGHT.get(int(e.get("code", 0)), 1))
        )
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

    def _move_score(self, board: chess.Board, move: chess.Move, ply: int, tt_move: chess.Move | None) -> float:
        if tt_move is not None and move == tt_move:
            return 1_000_000.0
        score = 0.0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            v = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900}.get(
                victim.piece_type if victim else chess.PAWN, 100
            )
            attacker = board.piece_at(move.from_square)
            a = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 50}.get(
                attacker.piece_type if attacker else chess.PAWN, 100
            )
            score += 10_000 + 10 * v - a
        if move.promotion:
            score += 8000 + 100 * move.promotion
        if board.gives_check(move):
            score += 7000
        killers = self.killers.get(ply)
        if killers:
            if killers[0] == move:
                score += 9000
            elif killers[1] == move:
                score += 8000
        score += min(self.history.get(move.from_square << 6 | move.to_square, 0), 5000)
        piece = board.piece_at(move.from_square)
        if piece is not None:
            lk = f"{piece.symbol().lower()}{chess.square_name(move.from_square)}{chess.square_name(move.to_square)}"
            bias = self.weights.learned_moves.get(lk, 0.0)
            score += bias * 400.0
        return score

    def _ordered_moves(self, board: chess.Board, ply: int, tt_move: chess.Move | None) -> list[chess.Move]:
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: self._move_score(board, m, ply, tt_move), reverse=True)
        return moves

    def quiesce(self, board: chess.Board, alpha: float, beta: float, ply: int) -> float:
        self.nodes += 1
        self._check_time()
        if board.is_checkmate():
            return -MATE + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        stand = relative_eval(board, self.weights)
        if stand >= beta:
            return beta
        if alpha < stand:
            alpha = stand
        if ply > 64:
            return alpha
        moves = [
            m
            for m in board.legal_moves
            if board.is_capture(m) or m.promotion or board.gives_check(m)
        ]
        moves.sort(key=lambda m: self._move_score(board, m, ply, None), reverse=True)
        for move in moves[:32]:
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
        if board.can_claim_threefold_repetition() and ply > 0:
            return 0.0

        alpha_orig = alpha
        key = chess.polyglot.zobrist_hash(board)
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
        # Check extension: look one ply deeper when in check.
        ext = 1 if in_check else 0
        if depth + ext <= 0:
            return self.quiesce(board, alpha, beta, ply)

        # Null-move pruning (skip when in check or endgame-ish).
        if (
            allow_null
            and depth >= 3
            and not in_check
            and any(
                p.piece_type not in (chess.PAWN, chess.KING)
                for p in board.piece_map().values()
                if p.color == board.turn
            )
        ):
            board.push(chess.Move.null())
            try:
                score = -self.negamax(board, depth - 3, -beta, -beta + 1, ply + 1, False)
            finally:
                board.pop()
            if score >= beta:
                return beta

        moves = self._ordered_moves(board, ply, tt_move)
        if not moves:
            return -MATE + ply if in_check else 0.0

        best = -1e18
        best_move = moves[0]
        for i, move in enumerate(moves):
            # Late move reductions.
            reduction = 0
            if (
                depth >= 3
                and i >= 4
                and not in_check
                and not board.is_capture(move)
                and not move.promotion
            ):
                reduction = 1 + (i // 8)

            board.push(move)
            try:
                score = -self.negamax(
                    board, depth - 1 - reduction + ext, -beta, -alpha, ply + 1, True
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
                if not board.is_capture(move):
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
        # Bound TT size.
        if len(self.tt) > 200_000:
            self.tt.clear()
        return best

    def search(self, board: chess.Board, time_ms: int, max_depth: int = 12) -> SearchResult:
        book = self.book_move(board)
        if book is not None:
            return SearchResult(move=book, score=0.0, depth=0, nodes=0, book=True)

        legal = list(board.legal_moves)
        if not legal:
            return SearchResult(move=None, score=0.0, depth=0, nodes=0)
        if len(legal) == 1:
            return SearchResult(move=legal[0], score=0.0, depth=1, nodes=1)

        self.nodes = 0
        self.deadline = time.perf_counter() + max(0.01, time_ms / 1000.0)
        self.tt.clear()
        self.killers.clear()
        self.history.clear()

        best_move = legal[0]
        best_score = 0.0
        depth_reached = 0
        try:
            for depth in range(1, max_depth + 1):
                # Aspiration window around previous score.
                window = 50.0
                alpha = best_score - window
                beta = best_score + window
                for _attempt in range(4):
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
                    break
                depth_reached = depth
                if abs(best_score) > MATE - 1000:
                    break
                if time.perf_counter() >= self.deadline:
                    break
        except RuntimeError:
            pass

        return SearchResult(
            move=best_move,
            score=best_score,
            depth=depth_reached,
            nodes=self.nodes,
            book=False,
            pv=[best_move.uci()] if best_move else None,
        )

    def _root_search(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        pv_move: chess.Move,
    ) -> float:
        moves = self._ordered_moves(board, 0, pv_move)
        best = -1e18
        best_move = moves[0]
        for move in moves:
            board.push(move)
            try:
                score = -self.negamax(board, depth - 1, -beta, -alpha, 1, True)
            finally:
                board.pop()
            if score > best:
                best = score
                best_move = move
            if best > alpha:
                alpha = best
        # Stash for outer loop.
        self._last_best_move = best_move  # type: ignore[attr-defined]
        return best

    def search_with_root_update(self, board: chess.Board, time_ms: int, max_depth: int = 12) -> SearchResult:
        """Like search(), but updates best move after each completed depth."""
        book = self.book_move(board)
        if book is not None:
            return SearchResult(move=book, score=0.0, depth=0, nodes=0, book=True)

        legal = list(board.legal_moves)
        if not legal:
            return SearchResult(move=None, score=0.0, depth=0, nodes=0)
        if len(legal) == 1:
            return SearchResult(move=legal[0], score=0.0, depth=1, nodes=1)

        # Instant tactical probe: never miss mate-in-one even with tiny budgets.
        for move in legal:
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return SearchResult(move=move, score=float(MATE - 1), depth=1, nodes=len(legal), book=False)

        self.nodes = 0
        # Guarantee a small absolute floor so depth-1 always finishes on CPU.
        self.deadline = time.perf_counter() + max(0.05, time_ms / 1000.0)
        self.tt.clear()
        self.killers.clear()
        self.history.clear()
        self._eval_cache: dict[int, float] = {}

        best_move = legal[0]
        best_score = 0.0
        depth_reached = 0
        try:
            for depth in range(1, max_depth + 1):
                # Depth-1 is untimed-ish: use a wide window and finish.
                if depth == 1:
                    score = self._root_search(board, depth, -1e18, 1e18, best_move)
                    best_score = score
                    best_move = getattr(self, "_last_best_move", best_move)
                    depth_reached = 1
                    continue
                window = 80.0
                alpha = best_score - window
                beta = best_score + window
                completed = False
                for _ in range(5):
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

        return SearchResult(
            move=best_move,
            score=best_score,
            depth=depth_reached,
            nodes=self.nodes,
            book=False,
            pv=[best_move.uci()] if best_move else None,
        )
