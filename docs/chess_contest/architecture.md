# Stigmergy-DPFE architecture

## Contest axes

1. **Elo** — ladder matches vs anchored reference bots  
2. **Bracket** — round-robin / elimination among contest entries  
3. **Uniqueness** — fingerprint rubric (rejects Stockfish/NNUE/AlphaZero clones)

Composite ≈ `0.45·Elo_n + 0.25·bracket + 0.30·uniqueness`.

## Why this architecture is unique

Most hobby engines are “material + PST + αβ”. Mega engines are NNUE or
self-play residual nets. **Stigmergy-DPFE** does neither:

| Idea | What it does |
|------|----------------|
| Multi-channel pheromone lattice | Each piece seeds 10 typed scent channels |
| Jacobi diffusion | Scents spread with learnable decay/mix (stigmergy) |
| Bilinear cross-color scoring | `Σ W[c,d] ⟨F_w[c], F_b[d]⟩` |
| King resonance | Enemy field sampled under the king |
| Soft material / tactical floor | Classical material+PST+hanging+SEE fused under fields |
| Ternary ant-trail book | Openings as `{-1,0,1}` codes with reinforcement |
| Harmonic channel | Fixed board basis mixed into the lattice |
| Swarm field-head | Compact pheromone readout (not NNUE) |

Search is deliberately deep and “slow is fine”: iterative deepening, TT,
null-move (Python), PVS, LMR, SEE ordering, killers, history, quiescence.

## Train → play split

```
Python (CPU/CUDA)          JSON weights           HTML
train_base / notebooks  →  base_weights.json  →  web/ player
```

```bash
venv/bin/python chess_contest/scripts/train_base.py --device cpu
venv/bin/python chess_contest/scripts/eval_elo.py --weights chess_contest/weights/base_weights.json
cd chess_contest/web && python3 -m http.server 8765
```

## Uniqueness rubric (summary)

- Large penalties for claiming Stockfish / NNUE / AlphaZero / LC0 / PST-only.
- Bonuses for pheromone lattice, diffusion, bilinear interactions, ternary
  trails, harmonic channel, king resonance, etc.
- Stigmergy’s own fingerprint is designed to score near the top of this axis
  while still having to *earn* Elo and bracket points.
