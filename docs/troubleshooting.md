# Troubleshooting

Ordered by how often each actually happens.

## Loss is flat from step 0

The model is learning nothing, and it is almost never the architecture.

1. **Data is garbage or misaligned.** Decode a batch and *read it*:
   ```python
   from minimodel.datasets import PackedTextDataset
   from minimodel.tokenization import BPETokenizer
   ds = PackedTextDataset("data/tokenized/x", seq_len=128)
   print(BPETokenizer.load("artifacts/tokenizer.json").decode(ds[0][0].tolist()))
   ```
   If that prints noise, the corpus or tokenizer is the problem.
2. **LR far too low** (e.g. an SFT recipe used for pretraining). Small models
   pretrain at 1e-3-3e-3, not 2e-5.
3. **Wrong vocab size** - a model built with a different vocab than the
   corpus was tokenized with trains, badly. The recipe runner injects the
   tokenizer's vocab automatically; if you built the model by hand, check
   `model.vocab_size == tokenizer.vocab_size`.

Sanity anchor: initial loss should be ~`ln(vocab)` (8.3 for 4096, 9.7 for
16384). Meaningfully higher means an init bug; exactly flat afterwards means
no gradient signal.

## Loss spikes / NaN

The `GradientMonitor` callback halts the run on NaN or a 4x explosion, so the
last good checkpoint survives. Then, in order of likelihood:

1. **Warmup too short at high LR** - use 1-2% of steps.
2. **fp16 without meaning to** - use `dtype: auto` (bf16 or fp32; fp16 is
   never auto-selected).
3. **Attention logit blow-up** - confirm `qk_norm: true` (default for dense/
   MoE/hybrid templates).
4. **A pathological batch** (one giant document of junk) - spikes at the
   same step on re-run points to data; find it, filter it.
5. Muon at an AdamW-sized budget interacts badly with tiny matrices - the
   built-in grouping already routes embeddings/norms to AdamW; don't fight
   it.

Resuming after a divergence: rerun the same command (it restores the last
kept checkpoint), with a lower `lr` or later `warmup`.

## Samples are gibberish but loss looks fine

- Base model + chat prompt: a pretrained-only model does not know
  `<|user|>`; drop `--chat` or run SFT first.
- Repetition loops: raise `--repetition-penalty 1.15`, use `--min-p 0.05`,
  lower temperature. Sub-10M models loop; it is what they do.
- Tokenizer mismatch: generation through a *different* tokenizer than
  training decodes to plausible-looking noise. The exported `model/` dir
  bundles the right one; prefer loading that directory.

## Out of memory

In order of cheapness: halve `batch_size` and double `grad_accum_steps`
(identical math, half the activations) -> shorten `seq_len` (activation
memory is linear in it; attention quadratic) -> `dtype: bf16` if on fp32 ->
smaller template / `n_kv_heads: 1` -> for decode-time OOM on long context,
the `hybrid` family exists for exactly this.

Rough training footprint: AdamW holds 3 fp32 copies (weights+m+v) ~ 12
bytes/param, before activations - a 125M model is ~1.5GB of state.

## SFT made the base model worse

Catastrophic forgetting. `replay_fraction: 0.1`, lower the LR (2e-5 -> 1e-5),
fewer steps (watch `val_loss`, stop at the elbow), `label_smoothing: 0.05`.
Verify with `minimodel bench` against the base on the *pretraining* eval
tasks, not just the SFT set.

## DPO/SPIN "trains" but nothing improves

`reward_accuracy` stuck at ~0.5 means the pairs carry no consistent signal
for the model - check the data, or generate fresher SPIN negatives (they go
stale after the policy moves). `reward_margin` up while samples get worse
means the policy is walking away from the reference: raise `beta`, or merge
back toward the SFT parent (`slerp --t 0.6`).

## RLVR solve rate frozen

Zero *or* perfect group rewards give zero advantage (that is the GRPO math).
Fix the task difficulty relative to the model - easier tasks, a stronger SFT
starting point, `format_bonus: 0.01` for the first gradient, larger
`group_size` for rarer successes.

## Diffusion samples are noise / grey mush

Noise after real training usually means sampling the raw weights instead of
the EMA (the exporter writes EMA by default - use the exported dir), or too
few steps (`--steps 50`). Grey mush at high guidance is guidance saturation -
drop `--guidance` to 2-3. Confirm inputs were in [-1, 1] (the bundled
datasets do this; custom loaders are the usual culprit).

## PixelGPT sprites are noise

Check the palette first: `minimodel vision data info` - a 64-colour budget on
1000-colour art quantises to mush before training starts (use `--palette-size
128` or `--palette-method kmeans`). Then sample colder (`--temperature 0.7
--top-p 0.9`). `val_pixel_accuracy` above ~0.5 with bad-looking samples is a
sampling-parameter problem, not a training problem.

## `ModuleNotFoundError: minimodel`

Install the package (`pip install -e .`); the sources live in `src/` but
import as `minimodel`. Inside the repo without installing, `PYTHONPATH` hacks
won't work by design - the editable install takes seconds.

## Hugging Face pulls fail

`pip install 'minimodel-trainer[hf]'` for the `datasets` dependency; gated
sets need `huggingface-cli login`; flaky streams: retry with `--limit`, the
pull is resumable via `--overwrite` semantics (existing files are skipped).

## Something else

Every module has a focused test file under `tests/` - find the behaviour you
expected, see what the test asserts, and run it:

```bash
venv/bin/pytest tests/test_training.py -k resume -x -q
```
