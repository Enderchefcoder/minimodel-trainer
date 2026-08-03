# Model cards

A card written from memory after the fact is wrong somewhere. `minimodel
card` builds one from the artifacts the run already produced, and marks
anything it cannot derive with an explicit `[TODO: ...]` instead of omitting
it silently.

## Generating

```bash
minimodel card \
    --run runs/dense-30m \                 # run_metadata.json, metrics.jsonl, result.json
    --model runs/dense-30m/model \         # config.json, weights size
    --benchmark bench.json \               # task table + throughput
    --dataset fineweb-edu-dedup cosmopedia-v2 \   # pulled from the registry: tokens, license
    --name mm-30m --org you --license apache-2.0 \
    --stage base \
    -o cards/mm-30m/README.md
```

What lands where:

| Card section | Source |
| --- | --- |
| YAML frontmatter (license, tags, datasets, base_model) | flags + registry |
| Model details (arch, params, vocab, context) | `config.json` / run metadata |
| Benchmarks table | benchmark JSONs (headline metric per task, with n) |
| Training data table (tokens, license per dataset) | the dataset registry |
| Hardware, wall-clock, throughput | run metadata + metrics + bench |
| Hyperparameters (optimizer, lr, schedule, batch, precision, seed) | run metadata |
| Results (final/best loss, perplexity) | `metrics.jsonl` summary |
| Limitations | stated defaults (hallucination, context, language, safety) |
| Reproduction commands | the recipe paths |

The frontmatter is Hugging Face-compatible, so the file works as a hub README
as-is.

## In code

```python
from minimodel.cardgen import collect_card_data, render_card, generate_model_card

data = collect_card_data(run_dir="runs/x", benchmark_paths=["bench.json"], name="x")
data.description = "One-line description worth writing by hand."
print(render_card(data))
```

`ModelCardData` is a plain dataclass - fill any field manually before
rendering when you know better than the artifacts.

## Habits worth keeping

- Generate the card **when the run finishes**, while `runs/<name>` still
  exists; it costs one command.
- Grep for `[TODO:` before publishing - each one is a fact the artifacts
  could not supply.
- Keep the license column honest: the registry carries upstream licenses
  (`cc-by-nc` marks non-commercial sets like original Alpaca), and the card
  copies them forward so downstream users can actually check.
