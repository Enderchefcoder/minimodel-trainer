# Report 11 — Crush Glint-2: locked 1.4M recipe

*Date: 2026-08-04. Evidence: prior reports 03/04/09, sandbox bake-off
`/tmp/crush_sandbox2/arch_compare_v4096.json`, SmolLM2 / FineWeb-Edu / DCLM
literature, live Glint-2 verification (report 10).*

## Verdict (the recipe)

| lever | choice | why |
| --- | --- | --- |
| **Architecture** | **dense GQA** `dense_1_4m` — dim=128, L=5, H=8, kv=2, ffn=352, QK-norm, tied embed, value residual → **1,406,506** params | Report 03: dense >> looped at fixed tokens. Sandbox: GQA ≈ MHA ≈ deep-112×6; thin-96×7 worse; GQA wins T4 throughput |
| **Tokenizer** | **4096-vocab byte-level BPE** trained on the mix | Matches Glint-2 for fair WikiText; ~4 bytes/token; proven at this scale |
| **Data mix** | **FineWeb-Edu 50% / DCLM-100BT 32% / TinyStories 15% / soft-QA 3%** | SmolLM2: FineWeb-Edu ↑ ARC/MMLU, DCLM ↑ HellaSwag; TinyStories bootstraps fluency/BLiMP; JSON oversampled so answers stick |
| **Optimizer** | **AdamW** lr=**3e-3**, wd=0.1, β=(0.9, 0.95), clip=1.0 | Sandbox LR sweep: 3e-3 best; Muon helps *looped* (report 04), AdamW wins on dense here |
| **Schedule** | **WSD** warmup 3%, decay_ratio 0.2, sqrt decay | Extendable plateau; branchable mid-run (MiniCPM / SmolLM) |
| **Context** | **seq_len=512**, ~65k tokens/step | Sandbox: longer windows beat short at equal tokens; T4 fits easily |
| **Tokens** | **400–800M** (~300–570 tok/param); Colab caps at **4h** | Chinchilla 28M is a floor, not "smart". Contender@49M: BLiMP 57 / ARC-E ~chance. ARC-E 36+ needs encyclopedic coverage × hundreds of M tokens |

Glint-2 targets to beat (honest metrics, report 10): ARC-Easy **36.8** (real),
WikiText **byte-ppl ~3.18** (their "3.09" is 2^BPB, not token-ppl), BLiMP
**~66–74** (protocol-dependent), shipped size **1.71M** not 1.06M.

## Approaches considered

1. **Looped supra2 / Glint-shape** — loses per-token at fixed budget (report 03);
   only wins if you need test-time loop scaling *and* train with Muon + stabilisers.
2. **Dense wide-shallow (160×3)** — slightly better sandbox val but **1.47M** (over budget).
3. **Dense GQA 128×5 (chosen)** — on-budget, fast, matches MobileLLM/SmolLM GQA bias.

## Soft-label JSON

`research/data/corpus/slm_next_token_dataset.json` (510 entries, 6.5k soft
steps). Plain CE docs in `slm_next_token_qa.jsonl`. Training uses CE on
`prompt+completion` every step (3% mixture weight) plus periodic
`soft_kl_loss` on re-encoded 3-way candidates (`minimodel.datasets.soft_labels`).

## Colab artefact

`notebooks/03_crush_glint2_colab.ipynb` — **one cell**: mount Drive, install,
stream the mix, train ≤4h on T4, checkpoint to Drive, smoke-generate JSON
answers.

## Projected vs Glint-2

| benchmark | Glint-2 (shipped 1.71M) | this recipe @ ~600M toks (projected) |
| --- | --- | --- |
| params (honest) | 1,710,049 | **1,406,506** |
| ARC-Easy | 36.8 | ≥37 with FineWeb-Edu-heavy mix |
| ARC-Challenge | (unreported) | above chance via edu mix |
| BLiMP | 66–74 | ≥65 with TinyStories + scale |
| WikiText | byte-ppl ~3.18 | competitive via FineWeb-Edu (contender failed only from TinyStories-only data) |
| HellaSwag | (unreported) | DCLM 32% is the HellaSwag lever (SmolLM2) |

Absolute numbers need the GPU run; the *recipe* is what this report locks.
