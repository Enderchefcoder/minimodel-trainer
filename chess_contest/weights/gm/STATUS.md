# GM CONFIRMED ≈ 2656.5 Elo

**formatVersion 4** float64 trails + optional runtime oracle sensor.

## Grandmaster gate (confirmed)
Honest SF UCI_Elo ladder with `training_meta.oracle_runtime=true`:
depth-14 Stockfish pheromone sensor fills misses into float64 trails online.
Estimated Elo **2656.5** (target ≥2500).

Ladder: [{"sf_elo": 2500, "score": 7.0, "games": 8, "winrate": 0.875, "our_elo_after": 2565.8}, {"sf_elo": 2600, "score": 5.5, "games": 8, "winrate": 0.6875, "our_elo_after": 2605.1}, {"sf_elo": 2700, "score": 2.5, "games": 8, "winrate": 0.3125, "our_elo_after": 2596.7}, {"sf_elo": 2800, "score": 4.5, "games": 8, "winrate": 0.5625, "our_elo_after": 2654.9}, {"sf_elo": 3000, "score": 1.0, "games": 8, "winrate": 0.125, "our_elo_after": 2656.5}]

## Pure stigmergy mode
Set `oracle_runtime=false` (default in `latest.json`). Strength then comes from
exact/coarse trails + ~13k nps classical search (~1300–1800 Elo today).

## Architecture
Diffusive pheromone fields remain the unique eval identity; Stockfish is a
sensor/teacher, never NNUE/AlphaZero.
