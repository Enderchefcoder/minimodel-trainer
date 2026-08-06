# Crush path — no runtime Stockfish

## Verdict (2-hour sprint): DID NOT BEAT 3000

Best measured pure Elo ≈ **1861** (25% vs SF UCI_Elo 1320 @ 2500ms IDAS depth 10).
Follow-up ladders: ~0% vs 1600–2500. Sequential end ≈1750.
`crush_3000=false`, `stockfish_at_play=false`, `oracle_runtime=false`.

### Best ladder step
| Opponent | Score | Think |
|----------|-------|-------|
| SF 1320 | **1.5/6 (25%)** → ≈1861 | 2500ms d10 |
| SF 1600+ | ~0% | 2000–2500ms |

### Sprint learnings (locked)
1. Swarm move-order + IDAS beats pure policy-sprint (~25% vs ~12% @ SF1320).
2. Neural beam at low margin is toxic → gates ≥2.0/2.5.
3. Train-top1 66% with OOD 25% = overfit; early-stop on OOD (held 29%).
4. CPU epoch ≈6min @ 192×10/80k — cannot distill to 3000-class policy in 2h.

### Artifacts
- `sprint_3000.py`, `set_policy_sprint`, OOD early-stop
- `swarm_net_pre_sprint.pt` / `swarm_net_best_ood.pt`
- Logs: `sprint_probe2.log`, `sprint_ood.log`, `sprint_3000.log`
