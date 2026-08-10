# Report 11 — Crush Glint-2: locked 1.4M recipe

*Date: 2026-08-04 (v2 notebook 2026-08-05). Evidence: prior reports 03/04/09/12,
sandbox bake-off, SmolLM2 / FineWeb-Edu / DCLM literature, live Glint-2
verification (report 10).*

## Verdict (the recipe)

| lever | choice | why |
| --- | --- | --- |
| **Architecture** | **dense GQA** `dense_1_4m` — dim=128, L=5, H=8, kv=2, ffn=352, QK-norm, tied embed, value residual → **1,406,506** params | Report 03: dense >> looped at fixed tokens. Sandbox: GQA ≈ MHA; GQA wins T4 throughput. Report 12 Mambas win *at ~1.1M TinyStories*; for 1.4M GPU encyclopedic crush stay dense |
| **Tokenizer** | **4096-vocab byte-level BPE** trained on the mix | Matches Glint-2 for fair WikiText; ~4 bytes/token |
| **Data mix (v2)** | **FineWeb-Edu 55% / DCLM-100BT 28% / TinyStories 12% / soft-QA 5%** | Edu↑ for ARC/WikiText; soft-QA↑ so answers stick; DCLM kept for HellaSwag; TinyStories still bootstraps fluency |
| **Optimizer** | **AdamW** lr=**3e-3**, wd=0.1, β=(0.9, 0.95), clip=1.0 | Sandbox LR sweep: 3e-3 best on dense |
| **Schedule (v2)** | **Wall-clock WSD** warmup 3%, decay in last 20% of the 4h, sqrt | Guarantees LR anneals before Colab preempts; extendable plateau |
| **Context (v2)** | **seq prefer 1024** (OOM-fallback 512), ~65k tokens/step | Sandbox: longer windows beat short at equal tokens; T4 probe picks the fit |
| **Systems (v2)** | **fp16 AMP + GradScaler**, `torch.compile`, **EMA 0.999**, packed non-overlap windows, FineWeb score≥2.5 | More unique tokens / hour; cleaner eval export |
| **Tokens** | **400–800M** (~300–570 tok/param); Colab caps at **4h** | Chinchilla 28M is a floor. ARC-E 36+ needs hundreds of M edu tokens |
| **Eval (v2)** | End-of-run harness vs Glint targets (ARC-E 36.8, WikiText byte-ppl 3.18, BLiMP ≥66) | Same code path as report 10 |

Glint-2 targets to beat (honest metrics, report 10): ARC-Easy **36.8** (real),
WikiText **byte-ppl ~3.18** (their "3.09" is 2^BPB, not token-ppl), BLiMP
**~66–74** (protocol-dependent), shipped size **1.71M** not 1.06M.

## Approaches considered

1. **Looped supra2 / Glint-shape** — loses per-token at fixed budget (report 03).
2. **Dense wide-shallow (160×3)** — slightly better sandbox val but **1.47M** (over budget).
3. **~1.1M Mamba / Griffin (report 12)** — win TinyStories val loss at 2.46M tokens;
   not chosen for the *encyclopedic* 1.4M Colab crush (throughput + FineWeb scale).
4. **Dense GQA 128×5 (chosen)** — on-budget, fast, matches MobileLLM/SmolLM GQA bias.

## Soft-label JSON

`research/data/corpus/slm_next_token_dataset.json` (510 entries, 6.5k soft
steps). Plain CE docs in `slm_next_token_qa.jsonl`. Training uses CE on
`prompt+completion` every step (5% mixture weight) plus `soft_kl_loss` every
10 steps at weight 0.30 (`minimodel.datasets.soft_labels`).

## Colab artefact

`notebooks/03_crush_glint2_colab.ipynb` — **one cell (v2)**: mount Drive,
install, stream the mix, train ≤4h on T4 with AMP/compile/EMA, checkpoint to
Drive, smoke-generate, run Glint-matched eval.

## Projected vs Glint-2

| benchmark | Glint-2 (shipped 1.71M) | this recipe @ ~600M toks (projected) |
| --- | --- | --- |
| params (honest) | 1,710,049 | **1,406,506** |
| ARC-Easy | 36.8 | ≥37 with FineWeb-Edu-heavy mix + 4h tokens |
| ARC-Challenge | (unreported) | above chance via edu mix |
| BLiMP | 66–74 | ≥65 with TinyStories + scale |
| WikiText | byte-ppl ~3.18 | competitive via FineWeb-Edu |
| HellaSwag | (unreported) | DCLM 28% is the HellaSwag lever (SmolLM2) |

Absolute numbers need the GPU run; the notebook writes `run_summary.json` with
harness metrics when eval caches are present.
