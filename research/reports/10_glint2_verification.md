# Report 10 — Fresh verification: Glint-2 is 1.71M, not 1.06M

*Source of truth: live Hugging Face release
[`Glint-Research/Glint-2`](https://huggingface.co/Glint-Research/Glint-2)
downloaded 2026-08-03. Reproducible script:
`research/experiments/prove_glint2_claims.py`. Evidence JSON:
`research/data/results/glint2_proof.json`.*

## Verdict

| claim | result |
| --- | --- |
| Released checkpoint is **1,710,049** params, not advertised **1.06M** | **VERIFIED** |
| Release is **loop + coda**, not "pure-loop / zero unique layers" | **VERIFIED** |
| Repo `generate.py` cannot load its own `glint-2.pt` (`strict=True`) | **VERIFIED** |
| Leaderboard lists params as **1.06M** while the evaluated weights are **1.71M** | **VERIFIED** (misleading) |
| README "wikitext-2 ppl" **3.09** is **2^BPB**, not token perplexity | **VERIFIED** (mislabeled) |
| ARC-Easy **36.80** is reproducible | **VERIFIED** (real score) |
| BLiMP **73.96** is fabricated | **NOT VERIFIED** — unreproduced (−7.6 in our harness); protocol unknown |

Bottom line: the **size and architecture claims are false** against the shipped
artifact. The benchmark table is **misleading on params and on the WikiText
metric name**. ARC is real. BLiMP is unreproduced but not proven fake.

## 1. Param count: advertised 1.06M, shipped 1.71M

README (live):

> "1,065k parameters total… zero unique layers before or after the loop."
> Leaderboard row: **glint-2 | 1.06M | …**

Checkpoint `checkpoints/glint-2.pt` `model_config`:

```
prelude_layers: 0
coda_layers:    1
shared_loops:   16   # table capacity; inference uses 8
dim: 96, ffn_hidden: 2112 (22×), vocab: 4096
```

State-dict parameter tally (fresh download, `numel()` sum):

| prefix | params |
| --- | ---: |
| embed (tied head) | 393,216 |
| shared (looped block) | 645,312 |
| **coda.0 (unique block)** | **645,312** |
| loop_lora + loop_embed | 26,112 |
| final_norm + indexer.gate | 97 |
| **TOTAL** | **1,710,049** |

Without `coda.*`: **1,064,737** ≈ their "1,065k". They published the
*pure-loop recipe's* count and shipped a *loop+coda* checkpoint **61% larger**.

## 2. Their own `generate.py` rejects their own weights

`generate.py` builds a pure-loop `Glint2` (no `coda` modules) then calls
`load_state_dict(ck["model"])` with PyTorch's default `strict=True`.

Live run:

```
RuntimeError: Error(s) in loading state_dict for Glint2:
  Unexpected key(s) in state_dict: "coda.0.attn_norm.weight",
  "coda.0.attn.qkv.weight", "coda.0.attn.out.weight",
  "coda.0.ffn_norm.weight", "coda.0.ffn.gate_up.weight",
  "coda.0.ffn.down.weight".
```

So the public inference script describes a different model than the file in
`checkpoints/`. Our faithful loader
(`research/baselines/glint2_model.py`) includes the coda and loads
`strict=True` at **1,710,049** params.

## 3. What is and is not "fake" about the benchmarks

| metric | README | our harness on the **1.71M** weights | reading |
| --- | --- | --- | --- |
| params column | **1.06M** | **1,710,049** | false |
| ARC-Easy | 36.80 | **36.78** | real; matches |
| WikiText-2 "ppl" | 3.09 | token-ppl **33.69**; **2^BPB = 3.18** | score ≈ real, **label wrong** |
| BLiMP | 73.96 | **66.36** | unreproduced (−7.6) |

### WikiText labeling

Token perplexity on a 4096-BPE is ~33.7. The advertised **3.09** matches
**byte-normalised perplexity** \(2^{\mathrm{BPB}}\) within 0.09 of our full
WikiText-2 measurement (BPB = 1.668 → 3.18). Byte-ppl is a good metric; calling
it plain "wikitext-2 ppl" next to a **1.06M** param column is not.

### BLiMP

We cannot reproduce 73.96 with a standard full-BLiMP likelihood comparison
(67 paradigms → 66.36). ARC + byte-ppl matching to the decimal argues the
loader/harness are sound, so the gap is almost certainly an unreleased
"tiny-ml leaderboard" protocol difference — **not proven fabrication**, but
**not independently verifiable** either. Absolute BLiMP numbers from that
README are not portable.

### The leaderboard params lie compounds everything

Even if every score were measured honestly on the shipped weights, listing
those scores under **1.06M** places a **1.71M** model on a 1M-parameter
comparison table. That is the core integrity failure.

## 4. How to re-run the proof

```bash
venv/bin/pip install huggingface_hub torch tokenizers
venv/bin/python research/experiments/prove_glint2_claims.py
# or, if already downloaded:
venv/bin/python research/experiments/prove_glint2_claims.py --local-dir /tmp/glint2-fresh
```

Exit code 0 means the three architectural tripwires still hold against the
live Hub revision.
