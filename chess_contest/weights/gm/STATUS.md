# Until 3000 — fresh depth-12 SF-MAX trails (no Stockfish at play)

Running `fresh_crush_3000.py` (strength-ranked depth-12 refresh).

## Honest status

- Prior peak Elo ≈ **2591** (`crush_3000=false`).
- Fresh book cycle-1 probe: hit≈26%, wr≈3% vs UCI_Elo 3000.
- Ceiling check: SF depth-12 **live** ≈69%+ vs Elo 3000 (enough for 3000 Elo).

## Bugs fixed this session

1. Converge hit inflation (mid-game SF install ≠ scored hit).
2. Fanout leaf gaps / thin opening tree (first miss ~ply 6–10).
3. **strength≥200 lock** preserved losing shallow PVs (`f1e1` vs SF-MAX `c2c4`).
4. Elo-only farm never labeled the SF-MAX spine (agree 5/12) — now plant spine each cycle.

`choose_move` never calls Stockfish. trail_first=true, stockfish_at_play=false.
