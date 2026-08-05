# Overnight GM push — live status

Updated automatically-ish by the cloud agent mid-run.

## Job
- tmux session: `stigmergy-overnight`
- command: `overnight_train.py --hours 8` from `ckpt_book.json`
- started (stable restart): 2026-08-05T03:58Z

## First SF UCI_Elo probe (~40 min in)
| SF Elo | Score | Winrate |
|--------|-------|---------|
| 1320 | 1.0/4 | 25% |
| 1600 | 2.0/4 | 50% |
| 1900 | 1.0/4 | 25% |
| 2200 | 0.5/4 | 12% |
| 2500 | 0.0/4 | 0% |

**Estimated Elo ≈ 1676** (vs SF limited-strength ladder). Uniqueness still maxed. Architecture unchanged (pheromone DPFE — not NNUE).

## Stability
Earlier run exploded eval to ~1e62 via multiplicative deposit updates — fixed with additive+clip; restarted from SF book.

## Artifacts
- `latest.json` / `ckpt_*.json` (gitignored, regenerable)
- `overnight.log`
- `elo_probe.json`

Target remains ~3000; overnight continues distilling SF-max winners.
