# Pure path — no runtime Stockfish (binding)

Previous **2656 Elo** GM claim used `oracle_runtime=true` (Stockfish at play).
**Revoked.**

## Status (2026-08-06)

Play path is Stockfish-free. Critical search bugs fixed:

- Alpha-beta no longer uses ±1e18 (float64 null-window false cutoffs)
- Leaf eval includes hanging-piece penalty
- Root hang-avoid only switches to safe moves
- Quiescence includes checks

Offline teacher + swarm residual net + long-think IDAS training continues
(`pure_gm.py` → `pure_gm.log` / `elo_probe.json`).

Target: estimated Elo **≥ 2500** (aim 2800+) with `stockfish_at_play: false`.
