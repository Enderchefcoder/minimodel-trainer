# GM push status

**formatVersion 4** float64 learned moves + continuous zobrist trails.

Running `probe_fit.py` (tmux `stigmergy-fit`): refreshes SF-MAX trail mass on
every training root, then honest SF UCI_Elo probes until estimated Elo ≥ 2500.

Artifacts: `latest.json` (large, gitignored), `elo_probe.json`, eventual
`gm_weights.json`.
