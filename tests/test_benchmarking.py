"""Tests for the evaluation harness, comparison tables and merging."""

from __future__ import annotations

import json

import pytest
import torch

from minimodel.benchmarking.bench import (
    BenchmarkResult,
    evaluate_generation,
    evaluate_minimal_pairs,
    evaluate_multiple_choice,
    evaluate_perplexity,
    measure_throughput,
    run_suite,
    sequence_logprob,
)
from minimodel.benchmarking.compare import (
    ComparisonTable,
    compare_results,
    compare_runs,
    load_results,
    markdown_table,
    pareto_frontier,
    write_comparison,
)
from minimodel.benchmarking.tasks import (
    BUILTIN_TASKS,
    Task,
    load_task,
    normalize_multiple_choice,
)
from minimodel.benchmarking.visualize import (
    ascii_bar_chart,
    plot_scaling_curve,
    plot_task_comparison,
    plot_throughput,
    render_report,
)
from minimodel.merging.slerp import (
    MERGE_METHODS,
    dare_merge,
    linear_merge,
    merge_models,
    slerp,
    slerp_merge,
    task_arithmetic_merge,
    ties_merge,
)


class TestTaskLoading:
    """Task normalisation across dataset layouts."""

    def test_arc_layout(self):
        item = normalize_multiple_choice(
            {
                "question": "Which?",
                "choices": {"text": ["a", "b"], "label": ["A", "B"]},
                "answerKey": "B",
            }
        )
        assert item.label == 1

    def test_hellaswag_layout(self):
        item = normalize_multiple_choice({"ctx": "c", "endings": ["x", "y"], "label": "1"})
        assert item.choices == ["x", "y"] and item.label == 1

    def test_piqa_and_winogrande_layouts(self):
        assert normalize_multiple_choice({"goal": "g", "sol1": "a", "sol2": "b", "label": 0}).label == 0
        item = normalize_multiple_choice(
            {"sentence": "s", "option1": "a", "option2": "b", "answer": "2"}
        )
        assert item.label == 1

    def test_already_normalised_and_invalid(self):
        assert normalize_multiple_choice({"context": "c", "choices": ["a"], "label": 0}).label == 0
        assert normalize_multiple_choice({"unknown": 1}) is None
        assert normalize_multiple_choice(
            {"question": "q", "choices": {"text": ["a"], "label": ["A"]}, "answerKey": "Z"}
        ) is None

    def test_load_task_from_jsonl(self, tmp_path):
        path = tmp_path / "task.jsonl"
        rows = [
            {"sentence_good": "The cat sleeps.", "sentence_bad": "The cat sleep."},
            {"bad_row": 1},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        task = load_task("pairs", path, "minimal_pairs")
        assert len(task) == 1
        assert "Task" in repr(task)

    def test_load_task_kinds_and_errors(self, tmp_path):
        path = tmp_path / "gen.jsonl"
        path.write_text(json.dumps({"prompt": "p", "answer": "1"}), encoding="utf-8")
        assert load_task("g", path, "generation").items[0].answer == "1"
        text_path = tmp_path / "text.jsonl"
        text_path.write_text(json.dumps({"text": "hello"}), encoding="utf-8")
        assert load_task("t", text_path, "perplexity").items == ["hello"]
        with pytest.raises(ValueError, match="unknown task kind"):
            load_task("x", [{"a": 1}], "interpretive_dance")
        with pytest.raises(FileNotFoundError):
            load_task("x", tmp_path / "missing.jsonl")

    def test_builtin_tasks_present(self):
        assert {"demo-syntax", "demo-choice", "demo-math"} <= set(BUILTIN_TASKS)


class TestHarness:
    """Scoring pipelines."""

    def test_sequence_logprob_additivity(self, tiny_model, tokenizer):
        tiny_model.eval()
        total, count = sequence_logprob(tiny_model, tokenizer, "The river", " runs east")
        assert count > 0
        assert total < 0  # log-probabilities

    def test_multiple_choice_scores(self, tiny_model, tokenizer):
        metrics = evaluate_multiple_choice(tiny_model, tokenizer, BUILTIN_TASKS["demo-choice"])
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["n"] == 4
        assert metrics["chance"] == 0.5
        empty = evaluate_multiple_choice(tiny_model, tokenizer, Task("e", "multiple_choice"))
        assert empty["n"] == 0

    def test_minimal_pairs_scores(self, tiny_model, tokenizer):
        metrics = evaluate_minimal_pairs(tiny_model, tokenizer, BUILTIN_TASKS["demo-syntax"])
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "mean_margin" in metrics
        assert evaluate_minimal_pairs(tiny_model, tokenizer, Task("e", "minimal_pairs"))["n"] == 0

    def test_perplexity_over_corpus(self, tiny_model, corpus_dir):
        metrics = evaluate_perplexity(tiny_model, corpus_dir, seq_len=32, max_batches=3, batch_size=2)
        assert metrics["perplexity"] > 1.0
        assert metrics["n_tokens"] == 3 * 2 * 32
        assert metrics["bits_per_token"] > 0

    def test_generation_task_scoring(self, tiny_model, tokenizer):
        metrics = evaluate_generation(
            tiny_model, tokenizer, BUILTIN_TASKS["demo-math"], max_new_tokens=4, limit=3
        )
        assert 0.0 <= metrics["solve_rate"] <= 1.0
        assert len(metrics["samples"]) <= 3

    def test_throughput_measurement(self, tiny_model):
        metrics = measure_throughput(
            tiny_model, prompt_len=16, generate_tokens=4, warmup=1, repeats=1
        )
        assert metrics["prefill_tokens_per_second"] > 0
        assert metrics["decode_tokens_per_second"] > 0
        assert metrics["ms_per_token"] > 0

    def test_run_suite_and_result_io(self, tiny_model, tokenizer, corpus_dir, tmp_path):
        result = run_suite(
            tiny_model,
            tokenizer,
            perplexity_corpus=corpus_dir,
            include_throughput=True,
            model_name="tiny",
            limit=3,
        )
        assert result.model == "tiny"
        assert "demo-syntax" in result.tasks
        assert "perplexity" in result.tasks
        assert result.throughput
        headline = result.headline()
        assert "demo-choice" in headline
        path = result.save(tmp_path / "result.json")
        assert json.loads(path.read_text())["model"] == "tiny"
        assert "tiny" in repr(result)


class TestCompare:
    """Tables and reports."""

    @pytest.fixture
    def results(self):
        return [
            {
                "model": "a",
                "parameters": 1000,
                "tasks": {"syntax": {"accuracy": 0.7, "chance": 0.5}, "ppl": {"perplexity": 30.0}},
                "throughput": {"decode_tokens_per_second": 100.0, "prefill_tokens_per_second": 500.0},
            },
            {
                "model": "b",
                "parameters": 2000,
                "tasks": {"syntax": {"accuracy": 0.8}, "ppl": {"perplexity": 20.0}},
                "throughput": {"decode_tokens_per_second": 50.0, "prefill_tokens_per_second": 400.0},
            },
        ]

    def test_markdown_highlights_best(self, results):
        table = compare_results(results)
        markdown = table.to_markdown()
        assert "**0.8**" in markdown  # higher accuracy wins
        assert "**20**" in markdown  # lower perplexity wins
        assert "ppl_ppl" in table.columns  # perplexity columns are suffixed
        assert "**2000**" not in markdown  # params are never highlighted
        assert "model" in table.columns
        assert len(table) == 2

    def test_markdown_table_empty(self):
        assert markdown_table([]) == "_no results_"

    def test_csv_and_dict(self, results):
        table = compare_results(results)
        assert "model," in table.to_csv()
        assert table.to_dict()[0]["model"] == "a"
        assert "ComparisonTable" in repr(table)

    def test_write_comparison_formats(self, results, tmp_path):
        table = compare_results(results)
        assert write_comparison(table, tmp_path / "c.md").read_text().startswith("## ")
        assert json.loads((write_comparison(table, tmp_path / "c.json")).read_text())
        assert "model" in write_comparison(table, tmp_path / "c.csv").read_text()

    def test_load_results_from_files_and_dirs(self, results, tmp_path):
        (tmp_path / "one.json").write_text(json.dumps(results[0]), encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "two.json").write_text(json.dumps(results[1]), encoding="utf-8")
        loaded = load_results([tmp_path / "one.json", sub])
        assert len(loaded) == 2
        table = compare_results([tmp_path / "one.json"])
        assert len(table) == 1

    def test_compare_runs_reads_metrics(self, tmp_path):
        from minimodel.core.io_utils import append_jsonl

        run = tmp_path / "run1"
        run.mkdir()
        for step in range(1, 6):
            append_jsonl(run / "metrics.jsonl", {"step": step, "loss": 1.0 / step})
        table = compare_runs([run], names=["first"])
        assert table.rows[0]["run"] == "first"
        assert table.rows[0]["steps"] == 5
        empty = compare_runs([tmp_path / "missing"])
        assert len(empty) == 0

    def test_pareto_frontier(self):
        points = [
            {"model": "small", "parameters": 10, "accuracy": 0.5},
            {"model": "wasteful", "parameters": 20, "accuracy": 0.4},
            {"model": "big", "parameters": 30, "accuracy": 0.9},
        ]
        frontier = pareto_frontier(points)
        names = {p["model"] for p in frontier}
        assert names == {"small", "big"}


class TestVisualize:
    """Charts and their ASCII fallbacks."""

    def test_ascii_bar_chart(self):
        chart = ascii_bar_chart({"a": 1.0, "b": 0.5}, title="scores")
        assert "scores" in chart and "#" in chart
        assert "no data" in ascii_bar_chart({})

    def test_plot_functions_write_files(self, tmp_path):
        results = [
            {"model": "a", "parameters": 100, "tasks": {"t": {"accuracy": 0.6, "chance": 0.5}},
             "throughput": {"prefill_tokens_per_second": 10, "decode_tokens_per_second": 5}},
            {"model": "b", "parameters": 200, "tasks": {"t": {"accuracy": 0.7}},
             "throughput": {"prefill_tokens_per_second": 20, "decode_tokens_per_second": 8}},
        ]
        assert str(plot_task_comparison(results, tmp_path / "tasks.png")).endswith(".png")
        assert str(plot_throughput(results, tmp_path / "tp.png")).endswith(".png")
        points = [{"model": "a", "parameters": 100, "perplexity": 30},
                  {"model": "b", "parameters": 200, "perplexity": 20}]
        assert str(plot_scaling_curve(points, tmp_path / "s.png")).endswith(".png")
        assert plot_scaling_curve([], tmp_path / "e.png") == "no data"
        assert plot_throughput([{}], tmp_path / "x.png") == "no throughput data"
        report = render_report(results, tmp_path / "report")
        assert report.read_text().startswith("## ")


class TestMerging:
    """Weight-space merges."""

    @pytest.fixture
    def states(self, tokenizer):
        from minimodel.architectures.builder import build_model
        from conftest import TINY_MODEL

        torch.manual_seed(0)
        first = build_model(
            "dense_3m", overrides={**TINY_MODEL, "vocab_size": tokenizer.vocab_size},
            verify_budget=False,
        ).state_dict()
        torch.manual_seed(1)
        second = build_model(
            "dense_3m", overrides={**TINY_MODEL, "vocab_size": tokenizer.vocab_size},
            verify_budget=False,
        ).state_dict()
        return first, second

    def test_slerp_preserves_norm(self):
        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        mid_slerp = slerp(a, b, 0.5)
        mid_linear = torch.lerp(a, b, 0.5)
        target = (a.norm() + b.norm()) / 2
        assert abs(mid_slerp.norm() - target) < abs(mid_linear.norm() - target)
        # Near-parallel inputs fall back to lerp without NaNs.
        near = slerp(a, a * 1.0001, 0.5)
        assert torch.isfinite(near).all()

    def test_slerp_endpoints(self):
        a, b = torch.randn(8, 8), torch.randn(8, 8)
        assert torch.allclose(slerp(a, b, 0.0), a, atol=1e-4)
        assert torch.allclose(slerp(a, b, 1.0), b, atol=1e-4)

    def test_linear_merge_average(self, states):
        first, second = states
        merged = linear_merge([first, second])
        key = next(iter(merged))
        expected = (first[key].float() + second[key].float()) / 2
        assert torch.allclose(merged[key].float(), expected, atol=1e-6)
        with pytest.raises(ValueError, match="positive"):
            linear_merge([first, second], [0.0, 0.0])

    def test_slerp_merge_requires_two(self, states):
        first, second = states
        merged = slerp_merge([first, second], t=0.5)
        assert set(merged) == set(first)
        with pytest.raises(ValueError, match="exactly two"):
            slerp_merge([first])

    def test_task_arithmetic_identity_and_negation(self, states):
        base, tuned = states
        same = task_arithmetic_merge(base, [tuned], [0.0])
        key = next(iter(base))
        assert torch.allclose(same[key], base[key])
        doubled = task_arithmetic_merge(base, [tuned], [1.0])
        assert torch.allclose(doubled[key].float(), tuned[key].float(), atol=1e-5)

    def test_ties_and_dare(self, states):
        base, tuned = states
        ties = ties_merge(base, [tuned], density=0.5)
        assert set(ties) == set(base)
        with pytest.raises(ValueError, match="density"):
            ties_merge(base, [tuned], density=0.0)
        dare = dare_merge(base, [tuned], drop_rate=0.5, seed=1)
        assert set(dare) == set(base)
        with pytest.raises(ValueError, match="drop_rate"):
            dare_merge(base, [tuned], drop_rate=1.0)

    def test_merge_models_io(self, tiny_model, tokenizer, tmp_path):
        first_dir = tmp_path / "a"
        second_dir = tmp_path / "b"
        tiny_model.save_pretrained(first_dir)
        tiny_model.save_pretrained(second_dir)
        merge_models(
            [first_dir, second_dir], method="slerp", output=tmp_path / "merged", t=0.3
        )
        assert (tmp_path / "merged" / "model.pt").exists()
        assert (tmp_path / "merged" / "config.json").exists()

        from minimodel.architectures.builder import load_model

        loaded = load_model(tmp_path / "merged")
        assert loaded.num_parameters() == tiny_model.num_parameters()

    def test_merge_models_validation(self, tmp_path):
        with pytest.raises(ValueError, match="unknown merge method"):
            merge_models([], method="blender")
        with pytest.raises(FileNotFoundError):
            merge_models([tmp_path / "missing"], method="linear")

    def test_base_required_for_delta_methods(self, tiny_model, tmp_path):
        model_dir = tmp_path / "m"
        tiny_model.save_pretrained(model_dir)
        with pytest.raises(ValueError, match="requires a `base`"):
            merge_models([model_dir], method="ties", output=tmp_path / "out")

    def test_merge_methods_registry(self):
        assert set(MERGE_METHODS) == {"linear", "slerp", "task_arithmetic", "ties", "dare"}
