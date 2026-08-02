# Inference

## CLI

```bash
minimodel generate -m runs/my-run/model -p "The river runs" --max-new-tokens 128
minimodel generate -m ... -p "Explain BPE" --chat            # wrap in the chat template
minimodel generate -m looped-model -p "..." --loops 8        # looped models: pick a depth
minimodel chat -m runs/sft/model --system "Be concise."      # streaming REPL
```

A "model" is any directory with `config.json` + `model.pt`; the tokenizer is
found next to it or one directory up (every export path in this repo places
it there).

## Sampling controls

Applied in this order:

1. `--repetition-penalty` / presence penalty - small models loop far more
   than large ones; `1.1` is the practical default here, not a nicety.
2. `--temperature` - `0` = greedy.
3. `--top-k`
4. `--top-p` (nucleus)
5. `--min-p` - keep tokens with `p >= min_p * p_max`. It adapts to the
   model's confidence per step, which behaves better than a fixed `top-p` at
   small scale where confidence swings wildly. `--min-p 0.05 --temperature 1`
   is a strong starting point.

`--seed` makes sampling reproducible. Decoding uses the incremental KV cache
of every architecture (linear, not quadratic, in generated length); the
looped model's `--loops` must stay fixed within one generation (cache slots
are per call site).

## In code

```python
from minimodel.inference import load_for_inference, complete, stream_completion

lm = load_for_inference("runs/sft/model")               # model + tokenizer + template
print(complete(lm, "Say hi", chat=True, max_new_tokens=64))

for piece in stream_completion(lm, "Tell me a story", chat=True):
    print(piece, end="", flush=True)
```

Streaming buffers byte-level tokens until they form valid UTF-8 (capped at 4
tokens so an invalid byte cannot stall the stream) - multi-byte characters
never arrive half-finished.

Batch generation left-pads prompts to a shared length so one decoding loop
serves ragged inputs:

```python
from minimodel.inference import complete_batch
complete_batch(lm, ["prompt one", "a much longer second prompt"])
```

## Reasoning mode

For CoT-trained models, `generate_with_reasoning` decodes the trace and the
answer as **separate phases with separate budgets**:

```python
from minimodel.inference import generate_with_reasoning
out = generate_with_reasoning(lm, "What is 12*7?",
                              max_reasoning_tokens=256, max_answer_tokens=64)
out["reasoning"], out["answer"]
```

The budget is load-bearing: a small model left to decide when to stop
thinking frequently never does. When the trace budget expires, `<|/think|>`
is injected and the answer phase starts from a well-formed prefix (this is
why CoT training has `enforce_think_close`). Reasoning samples slightly hot,
answers slightly cold - exploration while thinking, determinism when
committing.

## Practical numbers

CPU decode for a 30M model runs in the tens of tokens/second - fine for
chat. `minimodel bench` reports your machine's actual prefill/decode rates;
`ms_per_token` x expected response length = perceived latency. For the looped
family, halving `--loops` roughly halves decode time and costs surprisingly
little quality - that trade is the architecture's party trick.
