"""Train the best-effort contender and measure it head-to-head + inference lift.

The architecture bake-off (report 03) showed a dense transformer learns fastest
per training token at a fixed small budget — the right choice when compute, not
memory, is the constraint (which the CPU sandbox is). This trains a dense
contender for a larger token budget, runs the full harness against Glint-2, then
quantifies the two architecture-agnostic inference wins on the *same* model:

1. the effort ladder (best-of-N + chunked beam), measured by 4-gram repetition
   rate and quality-probe P(real) at low vs high vs xhigh;
2. the quality probe itself (real vs generated separation).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from eval_harness import ModelAdapter, eval_arc_easy, eval_blimp, eval_wikitext  # noqa: E402
from run_experiment import ExpConfig, build_model_from, train, val_loss  # noqa: E402

from minimodel.inference.quality_probe import train_quality_probe  # noqa: E402
from minimodel.inference.search import EFFORT_LEVELS, effort_generate, score_continuation  # noqa: E402
from minimodel.tokenization.tokenize import BPETokenizer  # noqa: E402

ART = Path("research/artifacts")
RESULTS = Path("research/data/results")


def contender_config(steps: int) -> ExpConfig:
    return ExpConfig(
        name="contender_dense",
        family="dense_transformer",
        arch=dict(dim=160, n_layers=3, n_heads=5, head_dim=32, n_kv_heads=5, ffn_hidden=512,
                  window=256, qk_norm=True, tie_embeddings=True, max_seq_len=512, vocab_size=4096),
        seq_len=256, batch_size=48, max_steps=steps, lr=3e-3,
        optimizer="adamw", schedule="wsd", warmup=0.03,
        schedule_kwargs={"decay_ratio": 0.2, "decay_shape": "sqrt"},
        blimp_per_paradigm=None, arc_limit=None, wikitext_max_tokens=40000,
    )


def real_texts(n: int = 96) -> list[str]:
    """Held-out TinyStories passages for probe training + repetition prompts."""
    text = Path("research/data/train/tinystories_val.txt").read_text(encoding="utf-8")
    docs = [d.strip() for d in text.split("<eos>") if len(d.strip()) > 120]
    return docs[:n]


def repetition_rate(ids: list[int]) -> float:
    grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
    return round(1.0 - len(set(grams)) / len(grams), 4) if grams else 0.0


def measure_inference_lift(model, tok, probe, prompts: list[str]) -> dict:
    """Repetition + P(real) at increasing effort, showing search pays off."""
    out: dict = {}
    for level in ("low", "high", "xhigh"):
        reps, p_reals = [], []
        for prompt in prompts:
            text = effort_generate(model, tok, prompt, level=level, max_new_tokens=48,
                                   probe=probe, seed=0)
            ids = tok.encode(text, allow_special=False)
            reps.append(repetition_rate(ids))
            prompt_len = len(tok.encode(prompt, add_bos=False))
            p_reals.append(probe.p_real(model, ids, min(prompt_len, len(ids) - 1)))
        out[level] = {
            "mean_repetition": round(sum(reps) / len(reps), 4),
            "mean_p_real": round(sum(p_reals) / len(p_reals), 4),
        }
    return out


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    torch.manual_seed(1234)
    device = torch.device("cpu")
    tok = BPETokenizer.load(ART / "tokenizer_v4096.json")
    cfg = contender_config(steps)
    model = build_model_from(cfg).to(device)
    print(f"=== contender_dense ({model.num_parameters():,} params, {steps} steps) ===", flush=True)

    t0 = time.perf_counter()
    stats = train(model, cfg, device)
    print(f"trained in {time.perf_counter()-t0:.0f}s: {stats}", flush=True)
    model.eval()

    encode = lambda t: tok.encode(t, allow_special=False)  # noqa: E731
    n_bytes = lambda ids: len(tok.decode(ids).encode("utf-8"))  # noqa: E731
    adapter = ModelAdapter(name="contender_dense", encode=encode,
                           forward=lambda x: model(x), n_bytes=n_bytes,
                           max_len=256, batch_size=32, params=model.num_parameters())

    print("evaluating (full BLiMP + ARC + WikiText)...", flush=True)
    result = {"name": "contender_dense", "params": model.num_parameters(),
              "train_tokens": stats["train_tokens"], **stats,
              "val_loss": val_loss(model, cfg, device)}
    result.update(eval_blimp(adapter, per_paradigm=None))
    result.pop("blimp_per_paradigm", None)
    result.update(eval_arc_easy(adapter, limit=None))
    result.update(eval_wikitext(adapter, max_tokens=40000))
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2), flush=True)

    print("training quality probe + measuring effort lift...", flush=True)
    texts = real_texts(96)
    probe = train_quality_probe(model, tok, texts, n_prompts=80, max_new_tokens=48, epochs=250)
    probe_path = Path("research/artifacts/contender_probe.pt")
    probe.save(probe_path)
    prompts = [t.split(".")[0][:40] for t in texts[:24] if t.split(".")[0]]
    result["inference_lift"] = measure_inference_lift(model, tok, probe, prompts)
    result["probe_bytes"] = probe_path.stat().st_size
    print(json.dumps(result["inference_lift"], indent=2), flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "contender_dense.json").write_text(json.dumps(result, indent=2))
    model_dir = Path("research/artifacts/contender_dense")
    model.save_pretrained(model_dir)
    tok.save(model_dir / "tokenizer.json")
    print(f"saved model -> {model_dir}", flush=True)


if __name__ == "__main__":
    main()
