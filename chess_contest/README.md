# Chess Contest — Stigmergy (Diffusive Pheromone Field Engine)

Unique contest entry: **not** Stockfish, NNUE, AlphaZero, or classic PST-minimax
dressed up. Evaluation is a **hybrid**: a classical tactical floor (material,
PST, hanging pieces, SEE) fused with a multi-channel **diffusive pheromone
field** residual, ternary ant-trail book, swarm field-head, and deep IDAS.

## Scoring axes (contest)

1. **Chess ELO** — estimated from rated matches against reference engines
2. **Bracket W/L** — tournament results among submitted engines
3. **Uniqueness** — architectural fingerprint vs known families (higher = better)

## Two-part workflow

| Part | Role |
|------|------|
| **Python** (`stigmergy/`) | Train on CPU or CUDA, export JSON weights, run ELO / brackets |
| **HTML** (`web/`) | Load weights, play on mobile or desktop (responsive board) |

```bash
# Train a base model (CPU is fine; CUDA auto-used if present)
venv/bin/python -m chess_contest.scripts.train_base --out chess_contest/weights/base_weights.json

# Estimate ELO
venv/bin/python -m chess_contest.scripts.eval_elo --weights chess_contest/weights/base_weights.json

# Open the play site (any static server)
cd chess_contest/web && python3 -m http.server 8765
# then open http://localhost:8765
```

## Architecture (why it is unique)

1. **Deposition** — each piece seeds a 10-channel pheromone lattice (material
   mass, king heat, pawn chain, file control, color-complex, mobility mist,
   discovery threat, blockade, tempo, harmonic residual).
2. **Diffusion** — Jacobi smoothing with learnable per-channel decay / mix
   (stigmergy: information spreads across the board like ant pheromone).
3. **Hybrid scoring** — tactical floor (material + PST + hanging + structure)
   plus pheromone residual: white×black bilinear interactions, king-resonance,
   swarm field-head readout. Floor stops hanging queens; fields own the unique
   positional voice.
4. **Ternary trails** — opening / move memory stored as codes `{-1,0,1}` with
   evaporation and win/loss reinforcement (not a giant float book).
5. **Search** — iterative deepening αβ with TT, null-move, PVS, LMR, SEE
   ordering, futility/delta pruning, killers, history, quiescence. Depth over
   speed is intentional.

See `docs/chess_contest/architecture.md` for the full design and uniqueness
rubric.

## Notebooks

- `notebooks/04_stigmergy_train.ipynb` — train + export
- `notebooks/05_stigmergy_fields.ipynb` — visualize pheromone channels
- `notebooks/06_stigmergy_elo.ipynb` — ELO / bracket demos
