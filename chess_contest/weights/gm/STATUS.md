# Pure path — no runtime Stockfish (binding)

Previous **2656 Elo** GM claim used `oracle_runtime=true` (Stockfish at play).
**Revoked.**

## Status (2026-08-06)

Play path is Stockfish-free.

### Fixes landed
- Alpha-beta `INF = MATE*10` (float64 ±1e18 null-window trap)
- Hanging-piece leaf penalty; quiescence checks; safe root hang-avoid
- Swarm policy beam at root; SF-teacher trails (≥40) may autoplay
- Offline teacher: 80k swarm dataset + 250k trails

### Pure Elo so far (honest, no SF at play)
| Opponent | Score | Think |
|----------|-------|-------|
| SF 1320  | 2.5/8 (31%) | 8s/move |

Estimated sequential Elo ~1830 — **below GM**. Long-think probe continues
(`pure_probe.py`). Target ≥2500 (aim 2800+).
