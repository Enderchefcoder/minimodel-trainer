# benchmarking/

- `tasks.py` — task shapes (multiple choice, minimal pairs, generation,
  perplexity) + auto-normalisation of ARC/HellaSwag/PIQA/WinoGrande layouts +
  bundled offline sanity tasks.
- `bench.py` — the harness: likelihood-based scoring wherever possible,
  verifier-checked generation where not, plus prefill/decode throughput.
  `run_suite` → `BenchmarkResult` JSON.
- `compare.py` — Markdown/CSV/JSON tables with direction-aware best-value
  highlighting; run-vs-run comparison from `metrics.jsonl`; Pareto frontier.
- `visualize.py` — grouped bars, log-log scaling curves, throughput charts;
  ASCII fallbacks everywhere.

Docs: [docs/evaluation.md](../../docs/evaluation.md). What the numbers mean at
small scale: same page. Expected scores: [BENCHMARKS.md](BENCHMARKS.md).
