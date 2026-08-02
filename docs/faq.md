# FAQ

**Why train a small model at all when good large ones exist?**
Ownership and iteration speed. You control every token of data, every
hyperparameter, and the whole artifact fits in version control; a full
experiment loop is minutes, not days. Small models are also the honest way to
*learn* how this all works - and for narrow domains with tight latency/memory
budgets they are genuinely the right deployment choice.

**Can I use this without a GPU?**
Yes. Everything runs on CPU: the quickstarts and test suite are CPU-only, and
models up to ~15M pretrain overnight on a laptop. GPUs just move the ceiling.

**Which architecture should I start with?**
`dense`. Switch to `looped` when parameter count is the constraint, `moe`
when training FLOPs are, `hybrid` when context length is. Details in
[architecture.md](architecture.md).

**How much data do I need?**
Floor: ~100 tokens per parameter of *good* data. Comfortable: 300-1000
tokens/param (the templates' `recommended_tokens`). Quality dominates - the
`-edu` corpora plus a Cosmopedia slice beat a larger raw-web budget at these
scales.

**Do I have to train my own tokenizer?**
You should - it is minutes of work and a tokenizer fitted to your corpus buys
real quality (watch `bytes_per_token`). But `BPETokenizer.load` also reads HF
byte-level-BPE `tokenizer.json` files if you want to reuse one.

**Can I load a Hugging Face model into this trainer?**
Not directly - architectures here are this repo's own implementations. State
dicts with matching shapes can be adapted by hand, but the intended flow is
training from scratch (that is the point of the toolkit).

**Is multi-GPU supported?** Single-node DDP via `torchrun`, yes
([training.md](training.md#multi-gpu)). FSDP/pipeline parallelism are out of
scope - by the time you need them you have outgrown "mini".

**Why is my `--set training.lr=3e-4` a string?** It isn't - bare scientific
notation is parsed as a float here precisely because YAML 1.1 gets it wrong.
If you hit this in a *YAML file*, write `3.0e-4`.

**Where did `runs/.../checkpoints/step_000020` go?**
Retention pruned it (`keep_last: 3`, `keep_best: 1` by default). Raise them or
set `keep_last: 0` to keep everything.

**How do I export just the weights for sharing?**
`trainer.export()` / the `export: true` recipe default writes
`runs/<name>/model/` with `model.pt`, `config.json`, `tokenizer.json` - that
directory is the shippable artifact, loadable by `load_for_inference` and
uploadable as-is with the generated card.

**Why does the looped model give different outputs for the same prompt?**
Check `loops`: during training it samples U{min..max}. At inference pass
`--loops N` (the CLI does for you) - and keep it constant while a KV cache is
alive.

**Can PixelGPT do resolutions other than 24x24?**
Yes - `image_size` is config; templates ship for 16/24/32. Cost grows with
the square of the side (it is a sequence length), so 48x48 = 2304 tokens is
about the practical ceiling.

**Why rectified flow instead of DDPM for the image models?**
Straight probability paths integrate in 20-50 Euler steps instead of
hundreds, and there is no noise-schedule weighting to tune. `objective: ddpm`
exists for comparison.

**What Python/PyTorch versions?**
Python >= 3.10, PyTorch >= 2.1 (needs `scaled_dot_product_attention` and
`torch.amp`). CPU wheels are fine.

**How do I add X?** Dataset -> registry YAML ([data.md](data.md)).
Architecture/size -> [architecture.md](architecture.md#adding-an-architecture).
Objective -> subclass `Trainer`, override `compute_loss`
([training.md](training.md#custom-objectives)). Verifier ->
`@VERIFIERS.register` ([rl.md](rl.md)). Conventions -> [AGENTS.md](../AGENTS.md).
