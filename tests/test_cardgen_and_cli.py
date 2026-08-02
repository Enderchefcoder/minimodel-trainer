"""Tests for model-card generation, the recipe runners and the CLI."""

from __future__ import annotations

import json

import pytest
import yaml

from minimodel.cardgen.modelcard_autogen import (
    ModelCardData,
    collect_card_data,
    generate_model_card,
    render_card,
)
from minimodel.cli import build_parser, main
from minimodel.core.io_utils import append_jsonl, write_json


class TestModelCard:
    """Card data collection and rendering."""

    @pytest.fixture
    def run_dir(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        write_json(
            run / "run_metadata.json",
            {
                "model": {
                    "architecture": "dense-transformer",
                    "parameters": 12345,
                    "config": {"vocab_size": 400, "max_seq_len": 128},
                },
                "config": {"optimizer": "adamw", "lr": 1e-3, "batch_size": 8, "seq_len": 128},
                "device": {"name": "TestCPU", "device_count": 1},
                "dtype": "torch.float32",
                "planned_tokens": 100000,
                "distributed": {"world_size": 1},
            },
        )
        for step in range(1, 6):
            append_jsonl(
                run / "metrics.jsonl",
                {"step": step, "loss": 2.0 / step, "elapsed_s": step, "total_tokens": step * 100},
            )
        write_json(run / "result.json", {"final_loss": 0.4, "total_tokens": 500})
        return run

    @pytest.fixture
    def benchmark_path(self, tmp_path):
        path = tmp_path / "bench.json"
        write_json(
            path,
            {
                "model": "test",
                "parameters": 12345,
                "tasks": {
                    "syntax": {"accuracy": 0.75, "n": 20},
                    "ppl": {"perplexity": 32.5},
                },
                "throughput": {"decode_tokens_per_second": 1234.0},
            },
        )
        return path

    def test_collect_from_run_and_benchmarks(self, run_dir, benchmark_path):
        data = collect_card_data(
            run_dir,
            benchmark_paths=[benchmark_path],
            name="my-model",
            organization="me",
            datasets=[{"name": "demo", "stage": "pretrain", "tokens": "1B", "license": "mit"}],
        )
        assert data.architecture == "dense-transformer"
        assert data.parameters == 12345
        assert data.benchmarks["syntax"]["accuracy"] == 0.75
        assert data.metrics["total_tokens"] == 500
        assert data.throughput["decode_tokens_per_second"] == 1234.0
        assert "minimodel" in data.tags

    def test_render_contains_required_sections(self, run_dir, benchmark_path):
        data = collect_card_data(
            run_dir, benchmark_paths=[benchmark_path], name="my-model", organization="me"
        )
        card = render_card(data)
        for section in (
            "# my-model",
            "## Model details",
            "## Benchmarks",
            "## Usage",
            "## Training",
            "### Hyperparameters",
            "## Limitations",
            "license:",
        ):
            assert section in card
        assert "0.7500" in card
        assert "adamw" in card

    def test_missing_information_becomes_todos(self):
        card = render_card(ModelCardData(name="bare"))
        assert "[TODO:" in card

    def test_generate_writes_readme(self, run_dir, tmp_path):
        path = generate_model_card(
            tmp_path / "card",
            run_dir=run_dir,
            name="gen",
            description="A test model.",
        )
        assert path.name == "README.md"
        assert "A test model." in path.read_text()

    def test_collect_from_model_dir(self, tiny_model, tmp_path):
        model_dir = tmp_path / "model"
        tiny_model.save_pretrained(model_dir)
        data = collect_card_data(model_dir=model_dir, name="m")
        assert data.architecture == "dense-transformer"
        assert data.vocab_size == tiny_model.vocab_size
        assert "weights_size" in data.metrics

    def test_tolerates_malformed_json(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        (run / "run_metadata.json").write_text("{broken", encoding="utf-8")
        data = collect_card_data(run, name="x")
        assert data.name == "x"


class TestRecipeRunners:
    """YAML-driven training entry points."""

    def test_pretrain_recipe_runs(self, tokenizer, corpus_dir, tmp_path):
        from conftest import TINY_MODEL

        from minimodel.training.recipe import run_pretrain_recipe

        tokenizer_path = tokenizer.save(tmp_path / "tok")
        recipe = {
            "run_name": "recipe-test",
            "tokenizer": str(tokenizer_path),
            "model": {"template": "dense_3m", "overrides": dict(TINY_MODEL)},
            "data": {"train": str(corpus_dir)},
            "training": {
                "output_dir": str(tmp_path / "runs"),
                "max_steps": 3,
                "batch_size": 2,
                "seq_len": 16,
                "log_every": 3,
                "eval_every": 0,
                "save_every": 0,
                "resume": False,
            },
        }
        result = run_pretrain_recipe(recipe)
        assert result.steps == 3
        assert (tmp_path / "runs" / "recipe-test" / "model" / "tokenizer.json").exists()

    def test_pretrain_recipe_with_mixture_and_overrides(
        self, tokenizer, corpus_dir, sft_dir, tmp_path
    ):
        from conftest import TINY_MODEL

        from minimodel.training.recipe import run_pretrain_recipe

        recipe = {
            "run_name": "mix",
            "tokenizer": str(tokenizer.save(tmp_path / "tok")),
            "model": {"template": "dense_3m", "overrides": dict(TINY_MODEL)},
            "data": {
                "mixture": [
                    {"path": str(corpus_dir), "weight": 0.7},
                    {"path": str(sft_dir), "weight": 0.3},
                ]
            },
            "training": {
                "output_dir": str(tmp_path / "runs"),
                "max_steps": 2,
                "batch_size": 2,
                "seq_len": 16,
                "eval_every": 0,
                "save_every": 0,
                "resume": False,
            },
            "export": False,
        }
        result = run_pretrain_recipe(recipe, cli_overrides=["training.max_steps=4"])
        assert result.steps == 4

    def test_recipe_validation(self, tmp_path, tokenizer):
        from minimodel.core.config import ConfigError
        from minimodel.training.recipe import run_pretrain_recipe

        with pytest.raises(ConfigError, match="template"):
            run_pretrain_recipe({"model": {}, "data": {"train": "x"}, "training": {}})
        with pytest.raises(ConfigError, match="train"):
            run_pretrain_recipe({"model": {"template": "dense_3m"}, "data": {}, "training": {}})

    def test_seq_len_exceeding_context_rejected(self, tokenizer, corpus_dir, tmp_path):
        from minimodel.core.config import ConfigError
        from minimodel.training.recipe import run_pretrain_recipe

        recipe = {
            "model": {"template": "dense_3m", "overrides": {"max_seq_len": 8}},
            "data": {"train": str(corpus_dir)},
            "training": {"max_steps": 1, "seq_len": 512, "output_dir": str(tmp_path)},
        }
        with pytest.raises(ConfigError, match="max_seq_len"):
            run_pretrain_recipe(recipe)

    def test_post_train_dispatch(self, tokenizer, sft_dir, tmp_path, tiny_model):
        from minimodel.training.post_train import post_train

        model_dir = tmp_path / "base"
        tiny_model.save_pretrained(model_dir)
        recipe = {
            "stage": "sft",
            "tokenizer": str(tokenizer.save(tmp_path / "tok")),
            "model": {"checkpoint": str(model_dir)},
            "data": {"train": str(sft_dir)},
            "training": {
                "output_dir": str(tmp_path / "runs"),
                "max_steps": 2,
                "batch_size": 2,
                "seq_len": 16,
                "eval_every": 0,
                "save_every": 0,
                "resume": False,
                "unknown_key_is_warned": True,
            },
        }
        result = post_train(recipe)
        assert result.steps == 2

    def test_post_train_validation(self, tmp_path):
        from minimodel.core.config import ConfigError
        from minimodel.training.post_train import post_train

        with pytest.raises(ConfigError, match="unknown post-training stage"):
            post_train({"stage": "alchemy"})
        with pytest.raises(ConfigError, match="checkpoint"):
            post_train({"stage": "sft", "model": {"checkpoint": tmp_path / "nope"}})


class TestCLI:
    """Argument parsing and a few full command paths."""

    def test_help_and_version(self, capsys):
        assert main([]) == 1
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--version"])

    def test_group_without_subcommand_prints_help(self):
        assert main(["data"]) == 1

    def test_models_command(self, capsys):
        assert main(["models"]) == 0
        out = capsys.readouterr().out
        assert "supra2_1406240" in out
        assert main(["models", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "templates" in payload

    def test_data_list_command(self, capsys):
        assert main(["data", "list", "--stage", "sft"]) == 0
        assert "alpaca-cleaned" in capsys.readouterr().out
        assert main(["data", "list", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["datasets"]

    def test_tokenizer_train_and_inspect(self, tmp_path, capsys):
        output = tmp_path / "tok.json"
        assert main(["tokenizer", "train", "--vocab-size", "300", "-o", str(output), "--json"]) == 0
        assert output.exists()
        capsys.readouterr()  # discard the training summary
        assert main(["tokenizer", "inspect", str(output), "--text", "hello"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["roundtrip_ok"]

    def test_data_pull_and_tokenize_and_info(self, tmp_path, capsys):
        tokenizer_path = tmp_path / "tok.json"
        main(["tokenizer", "train", "--vocab-size", "300", "-o", str(tokenizer_path)])
        assert main(["data", "pull", "builtin-demo", "-o", str(tmp_path / "raw")]) == 0
        assert (
            main(
                [
                    "data",
                    "tokenize",
                    "builtin-demo",
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "-t",
                    str(tokenizer_path),
                    "-o",
                    str(tmp_path / "tok"),
                ]
            )
            == 0
        )
        assert main(["data", "info", str(tmp_path / "tok")]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["n_tokens"] > 0

    def test_generate_command(self, tiny_model, tokenizer, tmp_path, capsys):
        model_dir = tmp_path / "model"
        tiny_model.save_pretrained(model_dir)
        tokenizer.save(model_dir / "tokenizer.json")
        assert (
            main(
                [
                    "generate",
                    "-m",
                    str(model_dir),
                    "-p",
                    "The",
                    "--max-new-tokens",
                    "4",
                    "--temperature",
                    "0",
                    "--device",
                    "cpu",
                ]
            )
            == 0
        )
        assert capsys.readouterr().out.strip()

    def test_bench_and_compare_and_plot(self, tiny_model, tokenizer, tmp_path, capsys):
        model_dir = tmp_path / "model"
        tiny_model.save_pretrained(model_dir)
        tokenizer.save(model_dir / "tokenizer.json")
        bench_out = tmp_path / "bench.json"
        assert (
            main(
                [
                    "bench",
                    "-m",
                    str(model_dir),
                    "--device",
                    "cpu",
                    "--limit",
                    "2",
                    "--no-throughput",
                    "-o",
                    str(bench_out),
                ]
            )
            == 0
        )
        assert bench_out.exists()
        assert main(["compare", str(bench_out)]) == 0
        assert "|" in capsys.readouterr().out

        run = tmp_path / "run"
        run.mkdir()
        for step in range(1, 4):
            append_jsonl(run / "metrics.jsonl", {"step": step, "loss": 1.0 / step})
        assert main(["plot", str(run), "-o", str(tmp_path / "curve.png")]) == 0

    def test_merge_command(self, tiny_model, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        tiny_model.save_pretrained(a)
        tiny_model.save_pretrained(b)
        assert (
            main(["merge", str(a), str(b), "--method", "linear", "-o", str(tmp_path / "out")]) == 0
        )
        assert (tmp_path / "out" / "model.pt").exists()

    def test_card_command(self, tmp_path, capsys):
        assert (
            main(["card", "-o", str(tmp_path / "card"), "--name", "x", "--dataset", "builtin-demo"])
            == 0
        )
        assert (tmp_path / "card" / "README.md").exists()

    def test_train_command_via_config(self, tokenizer, corpus_dir, tmp_path):
        from conftest import TINY_MODEL

        tokenizer_path = tokenizer.save(tmp_path / "tok")
        config = {
            "run_name": "cli-train",
            "tokenizer": str(tokenizer_path),
            "model": {"template": "dense_3m", "overrides": dict(TINY_MODEL)},
            "data": {"train": str(corpus_dir)},
            "training": {
                "output_dir": str(tmp_path / "runs"),
                "max_steps": 2,
                "batch_size": 2,
                "seq_len": 16,
                "eval_every": 0,
                "save_every": 0,
                "resume": False,
            },
            "export": False,
        }
        path = tmp_path / "recipe.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        assert main(["train", "-c", str(path), "--set", "training.max_steps=3"]) == 0

    def test_error_paths_return_nonzero(self, tmp_path):
        assert main(["data", "info", str(tmp_path / "nope")]) == 1
        assert main(["generate", "-m", str(tmp_path / "missing"), "-p", "x"]) == 1

    def test_vision_models_command(self, capsys):
        assert main(["vision", "models"]) == 0
        assert "pixelgpt_24x24_10m" in capsys.readouterr().out
        assert main(["vision", "datasets"]) == 0
        assert "pixelgpt-24x24" in capsys.readouterr().out
