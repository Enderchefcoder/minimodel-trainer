# Data

Everything about getting text and images into a trainable form.

## The registry

Datasets are *described*, not scripted, in
[`src/datasets/config/datasets.yaml`](../src/datasets/config/datasets.yaml)
(text) and
[`image_datasets.yaml`](../src/datasets/config/image_datasets.yaml) (images).
An entry records where the data lives, which field is the text, what stage it
belongs to, its rough size and its license:

```yaml
cosmopedia-v2:
  source: huggingface
  repo: HuggingFaceTB/smollm-corpus
  config: cosmopedia-v2
  split: train
  text_field: text
  stage: pretrain          # pretrain | sft | cot | preference | rlvr | eval
  format: text             # text | chat | instruction | preference | verifiable
  tokens: 28B
  license: apache-2.0
  description: >-
    Synthetic textbooks... (why you would use it)
```

Browse and filter:

```bash
minimodel data list                    # everything
minimodel data list --stage sft        # instruction-tuning sets
minimodel data list --json             # machine-readable
minimodel vision datasets              # the image catalogue
```

## The catalogue, by intent

**Pretraining (web/edu):** `fineweb-edu-dedup` (the SmolLM staple),
`fineweb-edu-10bt` / `-350bt` samples, `finemath`, `open-web-math`,
`python-edu`, `starcoder-python`, `wikipedia-en`, `gutenberg`.

**Pretraining (synthetic/simple):** `cosmopedia-v2` and `cosmopedia-full`
(synthetic textbooks - the highest signal per token available for small
models), `tinystories` and `simple-wikipedia` (for models under ~10M),
`tiny-textbooks`, `tiny-codes`, `finepersonas` (seed set for generating your
own synthetic data).

**SFT:** `smoltalk` (the best single choice - built for small models),
`alpaca-cleaned`, `casual-conversation` (a small slice keeps "hey" from
triggering an essay), `ultrachat-200k`, `openhermes-2.5`, `dolly-15k`,
`no-robots`, `slim-orca`, `open-platypus`.

**Chain-of-thought:** `openthoughts3` (1.2M traces, the largest open set),
`openr1-math`, `metamathqa`, `gsm8k-cot`, `limo` and `s1k` (tiny, curated,
surprisingly strong).

**Preference:** `ultrafeedback-binarized` (the default), `orca-dpo-pairs`,
`hh-rlhf`, `helpsteer2`.

**RLVR:** `gsm8k-rlvr` (numeric verifier), `math-competition` (boxed-answer
verifier), `countdown` (expression verifier).

**Eval:** `wikitext-103`, `blimp` (the most informative benchmark under 50M -
it measures syntax, not knowledge), `arc-easy`/`arc-challenge`, `hellaswag`,
`piqa`, `winogrande`, `lambada`.

**Images:** `pixelgpt-24x24` (the 20K-sprite set the bundled PixelGPT is
sized for), `pixel-art-sprites`, `diffusiondb-pixelart`, `cifar10`, `mnist`,
`butterflies`, `pokemon-blip` (captioned; fastest text-to-image sanity check),
`instructpix2pix` + `magicbrush` (editing), and more.

## Mixtures

A mixture is a weighted blend, sampled per-item at load time (no physical
interleaving):

```yaml
mixtures:
  smollm-corpus:
    components:
      - { dataset: fineweb-edu-dedup, weight: 0.70 }
      - { dataset: cosmopedia-v2,     weight: 0.30 }
```

Bundled: `smollm-corpus`, `smollm-corpus-code`, `synthetic-only`,
`tiny-curriculum` (for <10M models), `reasoning-mix`, `sft-general`,
`sft-small`, `cot-mix`, `preference-mix`, `eval-suite`.

In a training recipe, a mixture is a list of tokenized corpora with weights:

```yaml
data:
  mixture:
    - { path: data/tokenized/fineweb-edu, weight: 0.7 }
    - { path: data/tokenized/cosmopedia,  weight: 0.3 }
```

## The three steps

### 1. Pull

```bash
minimodel data pull cosmopedia-v2 --limit 200000       # one dataset
minimodel data pull smollm-corpus --limit 500000       # a whole mixture, split by weight
```

Hugging Face sources stream (`datasets` package required: `pip install
'minimodel-trainer[hf]'`); `--limit` caps records. Output is always
`data/raw/<name>.jsonl` plus a `.meta.json` provenance file. Local files
(`.jsonl`, `.json`, `.txt`) and plain URLs work through the same interface.

### 2. Tokenize

```bash
minimodel data tokenize cosmopedia-v2 -t artifacts/tokenizer.json
```

The registry entry's `format` decides the packing:

- **text** -> documents joined with `<|endoftext|>`, stored as one flat
  memmapped `uint16` array. No padding is ever stored; every token is a
  training token. 2 bytes/token on disk.
- **chat / instruction** -> rendered through the chat template into tokens
  *plus a parallel label array* with `-100` on prompt positions, so the "only
  train on assistant turns" decision is baked into the data, not re-derived.
- **preference** -> `pairs.jsonl` with chosen/rejected token ids, ready for
  DPO.

### 3. Load (inside the trainers)

- `PackedTextDataset` - random fixed-length windows over the stream. Windows
  deliberately cross document boundaries; the EOS token is how the model
  learns documents end.
- `SupervisedDataset` - windows plus the label mask.
- `MixtureDataset` - weighted sampling across datasets.
- `JsonlPairDataset` - preference pairs.

All are deterministic per `(seed, index)`, so resumed runs see the same data.

## Adding your own data

**Fastest path** - a JSONL file with a `text` field:

```bash
minimodel data tokenize --input my_corpus.jsonl -t artifacts/tokenizer.json -o data/tokenized/mine
```

**Registry path** - add an entry to `datasets.yaml` (source `local` with a
`path`, or `huggingface` with a `repo`), then use it like any bundled set.
Record the license; the model card generator copies it forward.

**Instruction data** - any of these shapes normalise automatically:
`{"instruction", "input", "output"}` (Alpaca), `{"prompt"/"question",
"response"/"answer"}`, `{"messages": [{role, content}]}`, ShareGPT
`{"conversations": [{from, value}]}`, with optional `reasoning`/`system`
fields.

## How much data do you need?

Chinchilla-optimal (~20 tokens/param) is the *compute*-optimal point, not the
quality ceiling - small models keep improving far past it, and since you pay
inference forever, over-training is correct. The bundled templates carry a
`recommended_tokens` range (roughly 300-1000 tokens/param); treat the low end
as a floor for a model you intend to use.

Data quality dominates at this scale. A 30M model cannot absorb the noise in
raw web text the way a 7B model can; using the `-edu` filtered corpora and a
20-40% synthetic (Cosmopedia) slice is worth more than any architecture change
in this repository.
