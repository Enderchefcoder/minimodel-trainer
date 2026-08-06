# Pure path — no runtime Stockfish (binding constraint)

Previous **2656 Elo** GM claim used `oracle_runtime=true` (Stockfish at play
time). That path is **revoked**. Play must never call Stockfish.

## Current approach

- Offline Stockfish teacher only (dataset + ladder opponent)
- Float64 trails / book distilled offline
- Swarm residual policy+value net (not NNUE)
- Long-think IDAS + classical leaves + swarm root re-rank

See `pure_gm.log` / `elo_probe.json` for live pure-mode Elo.

Target: estimated Elo **≥ 2500** (aim 2800+) with `stockfish_at_play: false`.
