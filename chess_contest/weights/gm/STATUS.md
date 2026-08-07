# Crush-big path - no runtime Stockfish

## SwarmNet v3: **31,746,513 params** (256x12 + SE + wide heads)

## Ladder (4s IDAS, depth 12, SF opponent only)

| Opponent | Score | After Elo |
|----------|-------|-----------|
| SF 1320 | **2.5/6 (42%)** | ≈1893 |
| SF 1600 | 0/6 (0%) | ≈1742 |
| SF 2000 | 0/6 (0%) | ≈1709 |
| SF 2200 | 0/6 (0%) | ≈1698 |
| SF **2500** | **1.5/6 (25%)** draws! | ≈1744 |
| SF 2700 | 0/6 (0%) | ≈1743 |

`crush_3000=false` (not yet). `stockfish_at_play=false`. OOD match after early-stop train: **26%**.

### vs prior (v2 ~15M)
- SF1320: 25% → **42%**
- SF2500: 0% → **25%** (three half-points)

### Still needed for 3000+
- OOD policy match ≥55-60% (now 26%)
- More diverse MultiPV labels + longer OOD-gated train (early-stop hit at ep3)
- Longer think (10-20s) once OOD is strong

### Artifacts
- `swarm_net.pt` / `swarm_net_best_ood.pt` (~122MB, 31.7M params)
- `crush_big_3000.py`, `crush_big.log`
- Legacy v2 archived as `swarm_net_v2_15m_legacy.pt`
