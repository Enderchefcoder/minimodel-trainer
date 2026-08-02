"""The ``minimodel`` command-line interface.

Design rules, so that both people and coding agents can drive it reliably:

* every command is non-interactive by default and exits non-zero on failure;
* ``--set key=value`` overrides any config key from the command line;
* ``--json`` makes commands print machine-readable output;
* ``minimodel <group> --help`` lists the subcommands of a group.

Command groups::

    tokenizer   train / inspect / encode / decode
    data        pull / tokenize / info / list
    train       pretrain from a recipe
    posttrain   sft / cot / dpo / spin / rlvr from a recipe
    generate    one-shot completion
    chat        interactive chat
    bench       evaluate a model
    compare     compare benchmark results or runs
    merge       merge checkpoints
    card        generate a model card
    plot        loss curves and charts
    models      list architectures and templates
    vision      image-model commands (see minimodel vision --help)
    quickstart  run the whole pipeline on the bundled corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from minimodel import __version__
from minimodel.core.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_json(payload: Any) -> None:
    """Print a JSON payload to stdout."""
    print(json.dumps(payload, indent=2, default=str))


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Add flags shared by every command."""
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key, e.g. --set training.lr=3e-4 (repeatable)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="console log level (default: INFO)",
    )


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
def cmd_tokenizer_train(args: argparse.Namespace) -> int:
    """Train a byte-level BPE tokenizer."""
    from minimodel.datasets.builtin import builtin_records
    from minimodel.datasets.pull_datasets import iter_records
    from minimodel.datasets.registry import get_dataset
    from minimodel.tokenization.tokenize import train_tokenizer

    if args.input:
        texts: list[str] = []
        for path in args.input:
            path = Path(path)
            if path.suffix in {".jsonl", ".ndjson"}:
                from minimodel.core.io_utils import read_jsonl

                texts.extend(
                    str(record.get("text", "")) for record in read_jsonl(path) if record.get("text")
                )
            else:
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
    elif args.dataset:
        spec = get_dataset(args.dataset)
        texts = [
            str(record.get(spec.text_field) or record.get("text", ""))
            for record in iter_records(spec, limit=args.limit)
        ]
    else:
        logger.info("no input given, training on the bundled demo corpus")
        texts = [record["text"] for record in builtin_records("pretrain", repeat=8)]

    texts = [t for t in texts if t.strip()]
    if not texts:
        logger.error("no training text found")
        return 1

    tokenizer = train_tokenizer(
        texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        backend=args.backend,
        verbose=True,
    )
    path = tokenizer.save(args.output)
    summary = {
        "path": str(path),
        "vocab_size": tokenizer.vocab_size,
        "merges": len(tokenizer.merges),
        "documents": len(texts),
        "bytes_per_token": round(tokenizer.compression_ratio(texts[:200]), 3),
    }
    if args.json:
        _print_json(summary)
    else:
        logger.info(
            "trained tokenizer: %d tokens, %.2f bytes/token -> %s",
            summary["vocab_size"],
            summary["bytes_per_token"],
            path,
        )
    return 0


def cmd_tokenizer_inspect(args: argparse.Namespace) -> int:
    """Show tokenizer statistics, or encode/decode a string."""
    from minimodel.tokenization.tokenize import BPETokenizer

    tokenizer = BPETokenizer.load(args.tokenizer)
    if args.text is not None:
        ids = tokenizer.encode(args.text)
        payload = {
            "text": args.text,
            "ids": ids,
            "tokens": [tokenizer.id_to_token(i) for i in ids],
            "n_tokens": len(ids),
            "roundtrip_ok": tokenizer.decode(ids) == args.text,
        }
    elif args.ids:
        payload = {"ids": args.ids, "text": tokenizer.decode(args.ids)}
    else:
        payload = {
            "vocab_size": tokenizer.vocab_size,
            "merges": len(tokenizer.merges),
            "special_tokens": tokenizer.special_tokens,
        }
    _print_json(payload)
    return 0


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def cmd_data_list(args: argparse.Namespace) -> int:
    """List registered datasets and mixtures."""
    from minimodel.datasets.registry import list_datasets, list_mixtures

    datasets = list_datasets(stage=args.stage)
    if args.json:
        _print_json(
            {
                "datasets": [d.to_dict() for d in datasets],
                "mixtures": [
                    {"name": m.name, "stage": m.stage, "components": m.components}
                    for m in list_mixtures()
                ],
            }
        )
        return 0

    print(f"{'NAME':<26} {'STAGE':<11} {'FORMAT':<15} {'TOKENS':<8} SOURCE")
    for spec in datasets:
        print(
            f"{spec.name:<26} {spec.stage:<11} {spec.format:<15} "
            f"{spec.tokens or '-'!s:<8} {spec.display}"
        )
    print("\nMixtures:")
    for mixture in list_mixtures():
        if args.stage and mixture.stage != args.stage:
            continue
        parts = ", ".join(f"{n} {w:.0%}" for n, w in mixture.normalized_weights())
        print(f"  {mixture.name:<24} [{mixture.stage}] {parts}")
    return 0


def cmd_data_pull(args: argparse.Namespace) -> int:
    """Download a dataset or mixture into JSONL."""
    from minimodel.datasets.pull_datasets import pull_dataset, pull_mixture
    from minimodel.datasets.registry import load_registry

    registry = load_registry()
    if args.name in registry["mixtures"]:
        outputs = pull_mixture(
            args.name, args.output, total_records=args.limit, overwrite=args.overwrite
        )
        payload = {name: str(path) for name, path in outputs.items()}
    else:
        path = pull_dataset(args.name, args.output, limit=args.limit, overwrite=args.overwrite)
        payload = {args.name: str(path)}
    if args.json:
        _print_json(payload)
    return 0


def cmd_data_tokenize(args: argparse.Namespace) -> int:
    """Tokenize raw JSONL into packed token shards."""
    from minimodel.datasets.registry import get_dataset
    from minimodel.datasets.tokenize_datasets import tokenize_jsonl
    from minimodel.tokenization.tokenize import BPETokenizer

    tokenizer = BPETokenizer.load(args.tokenizer)
    source = Path(args.input) if args.input else Path(args.raw_dir) / f"{args.name}.jsonl"
    data_format = args.format
    if data_format == "auto":
        try:
            data_format = get_dataset(args.name).format if args.name else "text"
        except Exception:
            data_format = "text"

    output = (
        Path(args.output) if args.output else Path("data/tokenized") / (args.name or source.stem)
    )
    stats = tokenize_jsonl(
        source,
        tokenizer,
        output,
        format=data_format,
        limit=args.limit,
        max_length=args.max_length,
        source_name=str(source),
    )
    if args.json:
        _print_json(stats)
    else:
        logger.info("tokenized -> %s (%s)", output, stats.get("n_tokens", stats.get("pairs")))
    return 0


def cmd_data_info(args: argparse.Namespace) -> int:
    """Print statistics for a tokenized corpus."""
    from minimodel.datasets.shards import TokenizedCorpus

    corpus = TokenizedCorpus(args.path)
    _print_json(corpus.stats())
    return 0


# ---------------------------------------------------------------------------
# train / posttrain
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    """Pretrain a model from a recipe."""
    from minimodel.training.recipe import run_pretrain_recipe

    result = run_pretrain_recipe(args.config, cli_overrides=args.overrides)
    if args.json:
        _print_json(result.to_dict())
    return 0


def cmd_posttrain(args: argparse.Namespace) -> int:
    """Run any post-training stage from a recipe."""
    from minimodel.training.post_train import post_train

    result = post_train(args.config, cli_overrides=args.overrides)
    payload = result.to_dict() if hasattr(result, "to_dict") else result
    if args.json:
        _print_json(payload)
    return 0


# ---------------------------------------------------------------------------
# generate / chat
# ---------------------------------------------------------------------------
def cmd_generate(args: argparse.Namespace) -> int:
    """Generate a completion from a model directory."""
    from minimodel.inference.run import complete, load_for_inference

    loaded = load_for_inference(args.model, tokenizer_path=args.tokenizer, device=args.device)
    model_kwargs = {"loops": args.loops} if args.loops else {}
    text = complete(
        loaded,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        chat=args.chat,
        include_prompt=not args.completion_only,
        **model_kwargs,
    )
    if args.json:
        _print_json({"prompt": args.prompt, "completion": text})
    else:
        print(text)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """Start an interactive chat session."""
    from minimodel.inference.run import chat_loop, load_for_inference

    loaded = load_for_inference(args.model, tokenizer_path=args.tokenizer, device=args.device)
    chat_loop(
        loaded,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=args.system,
        stream=not args.no_stream,
    )
    return 0


# ---------------------------------------------------------------------------
# bench / compare / plot
# ---------------------------------------------------------------------------
def cmd_bench(args: argparse.Namespace) -> int:
    """Evaluate a model."""
    from minimodel.benchmarking.bench import run_suite
    from minimodel.benchmarking.tasks import BUILTIN_TASKS, load_task
    from minimodel.inference.run import load_for_inference

    loaded = load_for_inference(args.model, tokenizer_path=args.tokenizer, device=args.device)
    tasks = list(BUILTIN_TASKS.values())
    if args.task:
        tasks = [load_task(Path(p).stem, p, args.task_kind, limit=args.limit) for p in args.task]

    result = run_suite(
        loaded.model,
        loaded.tokenizer,
        tasks=tasks,
        perplexity_corpus=args.perplexity_corpus,
        device=loaded.device,
        limit=args.limit,
        include_throughput=not args.no_throughput,
        model_name=args.name or Path(args.model).name,
    )
    if args.output:
        result.save(args.output)
    if args.json or not args.output:
        _print_json(result.to_dict())
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare benchmark results or training runs."""
    from minimodel.benchmarking.compare import compare_results, compare_runs, write_comparison

    table = compare_runs(args.paths) if args.runs else compare_results(args.paths)
    if args.output:
        write_comparison(table, args.output, title=args.title)
    if args.json:
        _print_json(table.to_dict())
    else:
        print(table.to_markdown())
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    """Plot loss curves from a run."""
    from minimodel.checkpointing.loss_visualization import plot_loss_curve, summarize_run

    output = plot_loss_curve(args.run, args.output, keys=tuple(args.keys), smoothing=args.smoothing)
    if args.json:
        _print_json({"output": str(output), "summary": summarize_run(args.run)})
    else:
        print(output)
    return 0


# ---------------------------------------------------------------------------
# merge / card / models
# ---------------------------------------------------------------------------
def cmd_merge(args: argparse.Namespace) -> int:
    """Merge several checkpoints into one."""
    from minimodel.merging.slerp import merge_models

    merge_models(
        args.models,
        method=args.method,
        weights=args.weights,
        base=args.base,
        output=args.output,
        density=args.density,
        drop_rate=args.drop_rate,
        t=args.t,
    )
    if args.json:
        _print_json({"output": str(args.output), "method": args.method})
    return 0


def cmd_card(args: argparse.Namespace) -> int:
    """Generate a model card."""
    from minimodel.cardgen.modelcard_autogen import generate_model_card
    from minimodel.datasets.registry import get_dataset

    datasets = []
    for name in args.dataset or []:
        try:
            datasets.append(get_dataset(name).to_dict())
        except Exception:
            datasets.append({"name": name})

    path = generate_model_card(
        args.output,
        run_dir=args.run,
        model_dir=args.model,
        benchmark_paths=args.benchmark or [],
        name=args.name or "model",
        organization=args.org,
        license=args.license,
        datasets=datasets,
        stage=args.stage,
        base_model=args.base_model,
    )
    if args.json:
        _print_json({"path": str(path)})
    else:
        print(path)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """List available architectures and templates."""
    from minimodel.architectures.builder import build_model, list_templates, load_template
    from minimodel.architectures.registry import list_architectures

    rows = []
    for name in list_templates():
        template = load_template(name)
        row = {
            "template": name,
            "family": template.get("family"),
            "params": template.get("params"),
            "active_params": template.get("active_params"),
            "description": template.get("description", ""),
        }
        if args.verify:
            model = build_model(name, verify_budget=False)
            row["built_params"] = model.num_parameters()
        rows.append(row)

    if args.json:
        _print_json({"architectures": list_architectures(), "templates": rows})
        return 0

    print("Architectures: " + ", ".join(list_architectures()))
    print(f"\n{'TEMPLATE':<18} {'FAMILY':<20} {'PARAMS':>13}  DESCRIPTION")
    for row in rows:
        params = f"{row['params']:,}" if row["params"] else "-"
        print(
            f"{row['template']:<18} {row['family']!s:<20} {params:>13}  {row['description'][:60]}"
        )
    return 0


def cmd_quickstart(args: argparse.Namespace) -> int:
    """Run the whole pipeline on the bundled corpus."""
    from minimodel.quickstart import run_quickstart

    summary = run_quickstart(
        output_dir=args.output,
        steps=args.steps,
        template=args.template,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        do_sft=not args.no_sft,
    )
    if args.json:
        _print_json(summary)
    return 0


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser."""
    parser = argparse.ArgumentParser(
        prog="minimodel",
        description="Train, evaluate and run small language and image models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  minimodel quickstart                       # end-to-end on the bundled corpus\n"
            "  minimodel models                           # list architectures and sizes\n"
            "  minimodel data list --stage sft            # browse the dataset registry\n"
            "  minimodel train --config configs/pretrain/demo_tiny.yaml\n"
            "  minimodel chat --model runs/demo/model\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"minimodel-trainer {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # -- tokenizer ---------------------------------------------------------
    tokenizer_parser = subparsers.add_parser("tokenizer", help="train and inspect tokenizers")
    tokenizer_sub = tokenizer_parser.add_subparsers(dest="subcommand", metavar="<subcommand>")

    train_tok = tokenizer_sub.add_parser("train", help="train a byte-level BPE tokenizer")
    train_tok.add_argument("--input", nargs="*", help="text or JSONL files to train on")
    train_tok.add_argument("--dataset", help="registered dataset name to train on")
    train_tok.add_argument("--limit", type=int, default=50000, help="max records from --dataset")
    train_tok.add_argument("--vocab-size", type=int, default=4096)
    train_tok.add_argument("--min-frequency", type=int, default=2)
    train_tok.add_argument(
        "--backend", default="auto", choices=["auto", "python", "fast"], help="training backend"
    )
    train_tok.add_argument("-o", "--output", default="artifacts/tokenizer.json")
    _add_common(train_tok)
    train_tok.set_defaults(func=cmd_tokenizer_train)

    inspect_tok = tokenizer_sub.add_parser("inspect", help="inspect, encode or decode")
    inspect_tok.add_argument("tokenizer")
    inspect_tok.add_argument("--text", help="encode this string")
    inspect_tok.add_argument("--ids", type=int, nargs="*", help="decode these ids")
    _add_common(inspect_tok)
    inspect_tok.set_defaults(func=cmd_tokenizer_inspect)

    # -- data --------------------------------------------------------------
    data_parser = subparsers.add_parser("data", help="pull, tokenize and inspect datasets")
    data_sub = data_parser.add_subparsers(dest="subcommand", metavar="<subcommand>")

    data_list = data_sub.add_parser("list", help="list registered datasets and mixtures")
    data_list.add_argument(
        "--stage", choices=["pretrain", "sft", "cot", "preference", "rlvr", "eval"]
    )
    _add_common(data_list)
    data_list.set_defaults(func=cmd_data_list)

    data_pull = data_sub.add_parser("pull", help="download a dataset or mixture")
    data_pull.add_argument("name")
    data_pull.add_argument("-o", "--output", default="data/raw")
    data_pull.add_argument("--limit", type=int, help="max records to fetch")
    data_pull.add_argument("--overwrite", action="store_true")
    _add_common(data_pull)
    data_pull.set_defaults(func=cmd_data_pull)

    data_tok = data_sub.add_parser("tokenize", help="tokenize JSONL into token shards")
    data_tok.add_argument("name", nargs="?", default="", help="registered dataset name")
    data_tok.add_argument("--input", help="explicit JSONL path (overrides name lookup)")
    data_tok.add_argument("--raw-dir", default="data/raw")
    data_tok.add_argument("-t", "--tokenizer", default="artifacts/tokenizer.json")
    data_tok.add_argument("-o", "--output")
    data_tok.add_argument(
        "--format",
        default="auto",
        choices=["auto", "text", "chat", "instruction", "preference"],
    )
    data_tok.add_argument("--limit", type=int)
    data_tok.add_argument("--max-length", type=int)
    _add_common(data_tok)
    data_tok.set_defaults(func=cmd_data_tokenize)

    data_info = data_sub.add_parser("info", help="show tokenized corpus statistics")
    data_info.add_argument("path")
    _add_common(data_info)
    data_info.set_defaults(func=cmd_data_info)

    # -- train -------------------------------------------------------------
    train_parser = subparsers.add_parser("train", help="pretrain from a recipe")
    train_parser.add_argument("-c", "--config", required=True)
    _add_common(train_parser)
    train_parser.set_defaults(func=cmd_train)

    post_parser = subparsers.add_parser("posttrain", help="sft / cot / dpo / spin / rlvr")
    post_parser.add_argument("-c", "--config", required=True)
    _add_common(post_parser)
    post_parser.set_defaults(func=cmd_posttrain)

    # -- generate / chat ---------------------------------------------------
    generate_parser = subparsers.add_parser("generate", help="generate a completion")
    generate_parser.add_argument("-m", "--model", required=True)
    generate_parser.add_argument("-p", "--prompt", required=True)
    generate_parser.add_argument("-t", "--tokenizer")
    generate_parser.add_argument("--device", default="auto")
    generate_parser.add_argument("--max-new-tokens", type=int, default=128)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--top-k", type=int, default=0)
    generate_parser.add_argument("--top-p", type=float, default=0.9)
    generate_parser.add_argument("--min-p", type=float, default=0.0)
    generate_parser.add_argument("--repetition-penalty", type=float, default=1.1)
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument("--loops", type=int, help="looped models: iterations to run")
    generate_parser.add_argument("--chat", action="store_true", help="apply the chat template")
    generate_parser.add_argument("--completion-only", action="store_true")
    _add_common(generate_parser)
    generate_parser.set_defaults(func=cmd_generate)

    chat_parser = subparsers.add_parser("chat", help="interactive chat")
    chat_parser.add_argument("-m", "--model", required=True)
    chat_parser.add_argument("-t", "--tokenizer")
    chat_parser.add_argument("--device", default="auto")
    chat_parser.add_argument("--max-new-tokens", type=int, default=200)
    chat_parser.add_argument("--temperature", type=float, default=0.8)
    chat_parser.add_argument("--top-p", type=float, default=0.9)
    chat_parser.add_argument("--system", help="system prompt")
    chat_parser.add_argument("--no-stream", action="store_true")
    _add_common(chat_parser)
    chat_parser.set_defaults(func=cmd_chat)

    # -- bench / compare / plot -------------------------------------------
    bench_parser = subparsers.add_parser("bench", help="evaluate a model")
    bench_parser.add_argument("-m", "--model", required=True)
    bench_parser.add_argument("-t", "--tokenizer")
    bench_parser.add_argument("--device", default="auto")
    bench_parser.add_argument("--task", nargs="*", help="task JSONL files")
    bench_parser.add_argument(
        "--task-kind",
        default="multiple_choice",
        choices=["multiple_choice", "minimal_pairs", "generation", "perplexity"],
    )
    bench_parser.add_argument("--perplexity-corpus", help="tokenized corpus for perplexity")
    bench_parser.add_argument("--limit", type=int)
    bench_parser.add_argument("--no-throughput", action="store_true")
    bench_parser.add_argument("--name", help="label for the result")
    bench_parser.add_argument("-o", "--output", help="write results JSON here")
    _add_common(bench_parser)
    bench_parser.set_defaults(func=cmd_bench)

    compare_parser = subparsers.add_parser("compare", help="compare results or runs")
    compare_parser.add_argument("paths", nargs="+")
    compare_parser.add_argument("--runs", action="store_true", help="compare runs, not benchmarks")
    compare_parser.add_argument("--title", default="Comparison")
    compare_parser.add_argument("-o", "--output")
    _add_common(compare_parser)
    compare_parser.set_defaults(func=cmd_compare)

    plot_parser = subparsers.add_parser("plot", help="plot a run's loss curve")
    plot_parser.add_argument("run", help="run directory or metrics.jsonl")
    plot_parser.add_argument("-o", "--output")
    plot_parser.add_argument("--keys", nargs="*", default=["loss", "val_loss"])
    plot_parser.add_argument("--smoothing", type=float, default=0.9)
    _add_common(plot_parser)
    plot_parser.set_defaults(func=cmd_plot)

    # -- merge / card / models --------------------------------------------
    merge_parser = subparsers.add_parser("merge", help="merge checkpoints")
    merge_parser.add_argument("models", nargs="+")
    merge_parser.add_argument(
        "--method",
        default="linear",
        choices=["linear", "slerp", "task_arithmetic", "ties", "dare"],
    )
    merge_parser.add_argument("--weights", type=float, nargs="*")
    merge_parser.add_argument("--base", help="base model (task_arithmetic / ties / dare)")
    merge_parser.add_argument("--density", type=float, default=0.2)
    merge_parser.add_argument("--drop-rate", type=float, default=0.9)
    merge_parser.add_argument("--t", type=float, default=0.5, help="slerp interpolation factor")
    merge_parser.add_argument("-o", "--output", required=True)
    _add_common(merge_parser)
    merge_parser.set_defaults(func=cmd_merge)

    card_parser = subparsers.add_parser("card", help="generate a model card")
    card_parser.add_argument("-o", "--output", default="README.md")
    card_parser.add_argument("--run", help="run directory")
    card_parser.add_argument("--model", help="exported model directory")
    card_parser.add_argument("--benchmark", nargs="*", help="benchmark result JSON files")
    card_parser.add_argument("--name")
    card_parser.add_argument("--org", default="")
    card_parser.add_argument("--license", default="apache-2.0")
    card_parser.add_argument("--dataset", nargs="*", help="dataset names used")
    card_parser.add_argument("--stage", default="base", choices=["base", "instruct", "reasoning"])
    card_parser.add_argument("--base-model", default="")
    _add_common(card_parser)
    card_parser.set_defaults(func=cmd_card)

    models_parser = subparsers.add_parser("models", help="list architectures and templates")
    models_parser.add_argument("--verify", action="store_true", help="build each to check counts")
    _add_common(models_parser)
    models_parser.set_defaults(func=cmd_models)

    quickstart_parser = subparsers.add_parser(
        "quickstart", help="run the whole pipeline on the bundled corpus"
    )
    quickstart_parser.add_argument("-o", "--output", default="runs/quickstart")
    quickstart_parser.add_argument("--steps", type=int, default=200)
    quickstart_parser.add_argument("--template", default="dense_3m")
    quickstart_parser.add_argument("--vocab-size", type=int, default=1024)
    quickstart_parser.add_argument("--seq-len", type=int, default=128)
    quickstart_parser.add_argument("--no-sft", action="store_true")
    _add_common(quickstart_parser)
    quickstart_parser.set_defaults(func=cmd_quickstart)

    # -- vision ------------------------------------------------------------
    from minimodel.vision.cli import add_vision_parser

    add_vision_parser(subparsers, _add_common)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        # A group was named without a subcommand, or nothing was.
        if getattr(args, "command", None):
            for action in parser._subparsers._actions:
                if isinstance(action, argparse._SubParsersAction):
                    subparser = action.choices.get(args.command)
                    if subparser is not None:
                        subparser.print_help()
                        return 1
        parser.print_help()
        return 1

    setup_logging(level=getattr(args, "log_level", "INFO"), force=True)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        logger.warning("interrupted")
        return 130
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
