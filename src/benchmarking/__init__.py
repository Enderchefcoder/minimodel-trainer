"""Evaluation harness, comparison tables and charts."""

from __future__ import annotations

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
    GenerationItem,
    MinimalPairItem,
    MultipleChoiceItem,
    Task,
    load_task,
)
from minimodel.benchmarking.visualize import (
    ascii_bar_chart,
    plot_scaling_curve,
    plot_task_comparison,
    plot_throughput,
    render_report,
)

__all__ = [
    "BUILTIN_TASKS",
    "BenchmarkResult",
    "ComparisonTable",
    "GenerationItem",
    "MinimalPairItem",
    "MultipleChoiceItem",
    "Task",
    "ascii_bar_chart",
    "compare_results",
    "compare_runs",
    "evaluate_generation",
    "evaluate_minimal_pairs",
    "evaluate_multiple_choice",
    "evaluate_perplexity",
    "load_results",
    "load_task",
    "markdown_table",
    "measure_throughput",
    "pareto_frontier",
    "plot_scaling_curve",
    "plot_task_comparison",
    "plot_throughput",
    "render_report",
    "run_suite",
    "sequence_logprob",
    "write_comparison",
]
