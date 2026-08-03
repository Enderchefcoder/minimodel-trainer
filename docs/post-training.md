# Post-training: SFT and chain-of-thought

Turning a base model into an assistant, and optionally a reasoner. Preference
methods (DPO/SPIN/RLVR) have [their own page](rl.md).

## Supervised fine-tuning

```bash
minimodel data pull smoltalk --limit 200000
minimodel data tokenize smoltalk -t artifacts/tokenizer.json
minimodel posttrain --config configs/sft/instruct.yaml
```

The recipe (`stage: sft`) starts from a checkpoint and trains on a supervised
corpus. Three decisions matter far more than any other knob, and the defaults
encode all three:

**1. Only assistant tokens are supervised.** The chat template writes a label
array with `-100` on every prompt/system position at *tokenization* time.
Training on prompts teaches a small model to generate user turns - visible in
samples as the model asking itself questions. `supervised_frac` in the logs
shows what fraction of positions carry loss (typically 0.3-0.6).

**2. The learning rate is ~10x below pretraining.** SFT sets are thousands of
times smaller than the pretraining corpus; at pretraining rates the model
becomes fluent in the SFT *format* in a few hundred steps while forgetting
what it knew. `2e-5` is the starting point for a 30M model.

**3. Replay limits forgetting.** `replay_fraction: 0.1` mixes 10% pretraining
batches back in. Cheap, and it reliably preserves perplexity on the original
distribution. Point `data.replay` at any tokenized pretraining corpus.

Also available: `label_smoothing` (0.05 helps small models stop being
overconfident on the answer templates), `track_accuracy` (per-token accuracy
on supervised positions - more interpretable than loss for SFT),
`early_stopping_patience` (SFT overfits fast; 2-3 epochs is usually the
ceiling, watch `val_loss`).

### The chat format

```
<|endoftext|><|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|><|endoftext|>
```

The role markers are reserved token ids in every tokenizer this repo trains
(even for base models), so SFT never resizes the embedding. At inference the
same template renders prompts and `<|end|>` stops generation - `minimodel
chat` and `complete(..., chat=True)` handle this automatically.

## Chain-of-thought distillation

`stage: cot` trains on data where the assistant turn contains a reasoning
trace, rendered as:

```
<|assistant|><|think|> ...trace... <|/think|> ...answer... <|end|>
```

Small models need three controls that big-model recipes do not mention:

**`reasoning_loss_weight`** (default 1.0). A 900-token trace before a 20-token
answer means 98% of gradient goes to imitating the trace. Below ~50M
parameters, that budget is better spent on the answer: `0.5` still teaches the
*shape* of reasoning while anchoring correctness; `0.0` trains answers only,
using traces purely as context (the right call under ~20M -
`configs/cot/answer_only.yaml`).

**`max_length` at tokenization.** Truncate traces to something the model can
actually emit (`minimodel data tokenize openthoughts3 --max-length 1024 ...`).
A model trained on traces longer than its context learns to think forever and
never answer.

**`enforce_think_close`** (0.1 is enough). An auxiliary term on the
`<|/think|>` logit that keeps the trace-to-answer transition sharp. This is
what makes *budget forcing* work at inference: generation can inject
`<|/think|>` when the thinking budget runs out and the model will still
produce a clean answer (see `generate_with_reasoning` in
[inference.md](inference.md)).

The logs add `reasoning_frac` (fraction of tokens inside think spans),
`answer_accuracy` (token accuracy outside them - the number to watch), and
`think_close_loss`.

### Which CoT data

From the registry: `openthoughts3` (1.2M traces; truncate hard),
`openr1-math` (verified math), `metamathqa` (short, cheap), `limo` / `s1k`
(hundreds of curated samples that punch far above their size - the right
starting point for small models). The bundled `cot-mix` blends them.

## `stage: continue`

Plain continued pretraining from a checkpoint - same trainer as pretraining,
useful for domain adaptation before SFT (e.g. a pass over `finemath` before a
math-flavoured SFT).

## Order of operations

The pipeline that works: **pretrain -> (optional domain continue) -> SFT ->
CoT -> DPO/RLVR**. SFT before CoT matters: a model that already follows the
chat format learns reasoning faster than one learning both at once. RLVR
last - it sharpens abilities that exist; it does not create them.
