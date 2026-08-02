# merging/

`slerp.py` implements five state-dict merges — `linear`, `slerp`
(norm-preserving, the pairwise default), `task_arithmetic` (add/subtract
capability deltas), `ties` (trim-elect-merge for 3+ conflicting fine-tunes),
`dare` (random delta dropping) — plus `merge_models` for disk-to-disk use and
the `MERGE_METHODS` registry the CLI consumes.

When each wins: [docs/merging.md](../../docs/merging.md).
