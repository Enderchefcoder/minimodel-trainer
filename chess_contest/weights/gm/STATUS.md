# Crush 3000 — no runtime Stockfish

## Ablation (SF 1320)
| Mode | Score |
|------|-------|
| Classical IDAS only | 0% |
| SwarmNet order + IDAS (no trail autoplay) | **25%** |
| Trails + swarm | 8% |
| Half-trained neural beam | 0% |

## Current
- SwarmNet v2: 192×10, 22-plane field-aware, 80k MultiPV soft labels
- Train top1 ≈53%, OOD SF-d10 match ≈42%
- Trail autoplay disabled when swarm loaded
- Continuing training toward ≥60% OOD match / Elo 2800–3000

`choose_move` never calls Stockfish. See `crush_3000.log`.
