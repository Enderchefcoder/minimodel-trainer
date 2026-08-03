# Expected numbers

Reference points for sanity-checking your own runs. All scores are
length-normalised accuracy from this repo's harness; chance is 0.25 for
ARC/HellaSwag, 0.5 for the rest. "Good data" = fineweb-edu + cosmopedia at
≥300 tokens/param.

| Params | BLiMP | HellaSwag | ARC-E | WinoGrande | WikiText ppl* |
| --- | --- | --- | --- | --- | --- |
| ~1.4M (supra2) | 0.55–0.62 | ~0.25 | ~0.25 | ~0.50 | 300–600 |
| 3–5M | 0.58–0.68 | 0.25–0.26 | 0.25–0.27 | ~0.50 | 150–300 |
| 12–16M | 0.65–0.74 | 0.26–0.28 | 0.27–0.31 | ~0.50 | 80–150 |
| 30M | 0.72–0.80 | 0.28–0.31 | 0.30–0.36 | 0.50–0.52 | 50–90 |
| 60M | 0.75–0.82 | 0.30–0.34 | 0.34–0.42 | 0.51–0.53 | 35–60 |
| 125M | 0.80+ | 0.33–0.38 | 0.40–0.48 | 0.52–0.55 | 25–45 |

*Perplexity depends heavily on the tokenizer; only compare models sharing one.

Reading guidance (why BLiMP is the one to watch under 50M, what ±1 point
means): [docs/evaluation.md](../../docs/evaluation.md).
