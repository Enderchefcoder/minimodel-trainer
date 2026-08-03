"""The end-to-end pipeline on the bundled corpus.

``minimodel quickstart`` exists to answer one question in under a minute and
with no network access: *does this whole thing work on my machine?* It runs
every stage - tokenizer, corpus, pretraining, SFT, evaluation, generation, model
card - on the small corpus shipped inside the package.

The resulting model is far too small to be useful. That is the point: it makes
the pipeline debuggable end to end before any real compute is spent.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from minimodel.architectures.builder import build_model
from minimodel.benchmarking.bench import run_suite
from minimodel.cardgen.modelcard_autogen import generate_model_card
from minimodel.core.io_utils import ensure_dir, human_duration
from minimodel.core.logging_utils import get_logger
from minimodel.datasets.builtin import builtin_records
from minimodel.datasets.loader import PackedTextDataset, SupervisedDataset
from minimodel.datasets.tokenize_datasets import tokenize_chat_records, tokenize_text_records
from minimodel.inference.sampling import generate_text
from minimodel.tokenization.tokenize import BPETokenizer
from minimodel.training.instruct_posttrainer import InstructTrainer, InstructTrainerConfig
from minimodel.training.trainer import Trainer, TrainerConfig

__all__ = ["run_quickstart"]

logger = get_logger(__name__)


def run_quickstart(
    output_dir: str | Path = "runs/quickstart",
    *,
    steps: int = 200,
    template: str = "dense_3m",
    vocab_size: int = 1024,
    seq_len: int = 128,
    batch_size: int = 8,
    do_sft: bool = True,
    seed: int = 1337,
) -> dict[str, Any]:
    """Run tokenizer -> data -> pretrain -> SFT -> eval -> sample -> card.

    Returns a summary dictionary with the artifacts that were produced.
    """
    started = time.perf_counter()
    root = ensure_dir(output_dir)
    summary: dict[str, Any] = {"output_dir": str(root)}

    # 1. Tokenizer ------------------------------------------------------
    logger.info("[1/6] training the tokenizer")
    texts = [record["text"] for record in builtin_records("pretrain", repeat=8)]
    tokenizer = BPETokenizer.train(texts, vocab_size=vocab_size, min_frequency=2)
    tokenizer_path = tokenizer.save(root / "tokenizer.json")
    summary["tokenizer"] = {
        "path": str(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
        "bytes_per_token": round(tokenizer.compression_ratio(texts[:32]), 3),
    }

    # 2. Corpus ---------------------------------------------------------
    logger.info("[2/6] tokenizing the bundled corpus")
    corpus_dir = root / "data" / "pretrain"
    stats = tokenize_text_records(({"text": t} for t in texts), tokenizer, corpus_dir)
    summary["corpus"] = {"tokens": stats["n_tokens"], "documents": stats["n_documents"]}

    train_dataset = PackedTextDataset(corpus_dir, seq_len=seq_len, seed=seed)
    eval_dataset = PackedTextDataset(corpus_dir, seq_len=seq_len, seed=seed + 991)

    # 3. Pretraining ----------------------------------------------------
    logger.info("[3/6] pretraining for %d steps", steps)
    model = build_model(
        template,
        overrides={
            "vocab_size": tokenizer.vocab_size,
            "max_seq_len": max(seq_len, 256),
            "window": seq_len,
        },
        verify_budget=False,
    )
    pretrain_config = TrainerConfig(
        run_name="pretrain",
        output_dir=str(root),
        max_steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=3e-3,
        warmup=0.1,
        log_every=max(1, steps // 10),
        eval_every=max(1, steps // 4),
        eval_batches=4,
        save_every=0,
        seed=seed,
        resume=False,
    )
    pretrain_result = Trainer(
        model,
        pretrain_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    ).fit()
    summary["pretrain"] = {
        "steps": pretrain_result.steps,
        "final_loss": round(pretrain_result.final_loss, 4),
        "perplexity": round(pretrain_result.final_perplexity, 2),
        "tokens": pretrain_result.total_tokens,
    }

    # 4. Instruction tuning ---------------------------------------------
    if do_sft:
        logger.info("[4/6] instruction tuning")
        sft_dir = root / "data" / "sft"
        tokenize_chat_records(builtin_records("sft", repeat=8), tokenizer, sft_dir)
        sft_dataset = SupervisedDataset(sft_dir, seq_len=seq_len, seed=seed)
        sft_config = InstructTrainerConfig(
            run_name="sft",
            output_dir=str(root),
            max_steps=max(20, steps // 4),
            batch_size=batch_size,
            seq_len=seq_len,
            lr=5e-4,
            log_every=max(1, steps // 20),
            eval_every=0,
            save_every=0,
            seed=seed,
            resume=False,
        )
        sft_result = InstructTrainer(
            model, sft_config, train_dataset=sft_dataset, tokenizer=tokenizer
        ).fit()
        summary["sft"] = {
            "steps": sft_result.steps,
            "final_loss": round(sft_result.final_loss, 4),
        }
    else:
        logger.info("[4/6] skipping instruction tuning")

    # 5. Evaluation and a sample ----------------------------------------
    logger.info("[5/6] evaluating")
    benchmark = run_suite(
        model,
        tokenizer,
        perplexity_corpus=corpus_dir,
        include_throughput=True,
        model_name=template,
    )
    benchmark_path = benchmark.save(root / "benchmark.json")
    summary["benchmark"] = {"path": str(benchmark_path), **benchmark.headline()}

    sample = generate_text(
        model, tokenizer, "The river runs", max_new_tokens=40, temperature=0.8, seed=seed
    )
    summary["sample"] = sample
    logger.info("sample: %s", sample.replace("\n", " ")[:160])

    # 6. Export and card -------------------------------------------------
    logger.info("[6/6] exporting the model and writing a card")
    model_dir = ensure_dir(root / "model")
    model.save_pretrained(model_dir)
    tokenizer.save(model_dir / "tokenizer.json")
    card = generate_model_card(
        model_dir / "README.md",
        run_dir=root / "pretrain",
        model_dir=model_dir,
        benchmark_paths=[benchmark_path],
        name="minimodel-quickstart",
        organization="local",
        datasets=[{"name": "builtin-demo", "stage": "pretrain", "tokens": "20K", "license": "mit"}],
    )
    summary["model_dir"] = str(model_dir)
    summary["model_card"] = str(card)
    summary["elapsed"] = human_duration(time.perf_counter() - started)

    logger.info(
        "quickstart finished in %s. Try:  minimodel chat --model %s",
        summary["elapsed"],
        model_dir,
    )
    return summary
