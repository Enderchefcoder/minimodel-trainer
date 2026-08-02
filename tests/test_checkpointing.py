"""Tests for checkpoint management, ETA estimation and loss plots."""

from __future__ import annotations

import time

import pytest
import torch

from minimodel.checkpointing.checkpointing import (
    CheckpointManager,
    find_latest_checkpoint,
    load_checkpoint_metrics,
    save_model_bundle,
)
from minimodel.checkpointing.etr import ETREstimator, ThroughputMeter, estimate_training_time
from minimodel.checkpointing.loss_visualization import (
    ascii_plot,
    load_metrics,
    plot_learning_rate,
    plot_loss_curve,
    smooth,
    summarize_run,
)
from minimodel.core.io_utils import append_jsonl


class TestCheckpointManager:
    """Saving, loading, retention and export."""

    def test_save_load_roundtrip(self, tiny_model, tmp_path):
        manager = CheckpointManager(tmp_path)
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)

        tokens = torch.randint(0, tiny_model.vocab_size, (2, 8))
        loss = tiny_model.forward_with_loss(tokens, tokens).loss
        loss.backward()
        optimizer.step()

        checkpoint = manager.save(
            10, model=tiny_model, optimizer=optimizer, metrics={"val_loss": 1.5}
        )
        assert checkpoint.model_path.exists()
        assert checkpoint.trainer_path.exists()
        assert checkpoint.size_bytes() > 0
        assert "step=10" in repr(checkpoint)

        with torch.no_grad():
            for param in tiny_model.parameters():
                param.zero_()
        state = manager.load(model=tiny_model, optimizer=optimizer)
        assert state["step"] == 10
        assert any(param.abs().sum() > 0 for param in tiny_model.parameters())

    def test_retention_keeps_last_and_best(self, tiny_model, tmp_path):
        manager = CheckpointManager(tmp_path, keep_last=2, keep_best=1, monitor="val_loss")
        losses = {10: 3.0, 20: 1.0, 30: 2.5, 40: 2.0}
        for step, loss in losses.items():
            manager.save(step, model=tiny_model, metrics={"val_loss": loss})
        kept = {checkpoint.step for checkpoint in manager.history}
        # Last two (30, 40) plus best (20).
        assert kept == {20, 30, 40}
        assert manager.best().step == 20
        assert manager.latest().step == 40

    def test_keep_last_zero_keeps_everything(self, tiny_model, tmp_path):
        manager = CheckpointManager(tmp_path, keep_last=0)
        for step in (1, 2, 3):
            manager.save(step, model=tiny_model)
        assert len(manager.history) == 3

    def test_rescan_reads_existing_directory(self, tiny_model, tmp_path):
        CheckpointManager(tmp_path).save(5, model=tiny_model, metrics={"val_loss": 2.0})
        rescanned = CheckpointManager(tmp_path)
        assert rescanned.latest().step == 5
        assert rescanned.summary()[0]["step"] == 5

    def test_load_without_checkpoints_raises(self, tmp_path):
        manager = CheckpointManager(tmp_path)
        with pytest.raises(FileNotFoundError):
            manager.load()
        assert manager.latest() is None
        assert manager.best() is None

    def test_mode_validation(self, tmp_path):
        with pytest.raises(ValueError, match="min.*max"):
            CheckpointManager(tmp_path, mode="sideways")

    def test_export_model_copies_weights_only(self, tiny_model, tmp_path):
        manager = CheckpointManager(tmp_path / "ckpts")
        manager.save(7, model=tiny_model, metrics={"val_loss": 1.0})
        exported = manager.export_model(tmp_path / "release")
        assert (exported / "model.pt").exists()
        assert (exported / "config.json").exists()
        assert not (exported / "trainer.pt").exists()

    def test_find_latest_checkpoint(self, tiny_model, tmp_path):
        assert find_latest_checkpoint(tmp_path) is None
        manager = CheckpointManager(tmp_path)
        manager.save(3, model=tiny_model)
        manager.save(9, model=tiny_model)
        assert find_latest_checkpoint(tmp_path).step == 9

    def test_load_checkpoint_metrics(self, tiny_model, tmp_path):
        manager = CheckpointManager(tmp_path, keep_last=0)
        manager.save(1, model=tiny_model, metrics={"val_loss": 2.0})
        manager.save(2, model=tiny_model, metrics={"val_loss": 1.0})
        rows = load_checkpoint_metrics(tmp_path)
        assert [row["step"] for row in rows] == [1, 2]

    def test_save_model_bundle(self, tiny_model, tokenizer, tmp_path):
        tokenizer_path = tokenizer.save(tmp_path / "tok")
        bundle = save_model_bundle(
            tiny_model,
            tmp_path / "bundle",
            tokenizer_path=tokenizer_path,
            metadata={"tokens": 123},
        )
        assert (bundle / "model.pt").exists()
        assert (bundle / "tokenizer.json").exists()
        assert (bundle / "training_metadata.json").exists()


class TestETR:
    """Time-remaining and throughput estimation."""

    def test_rate_and_remaining(self):
        estimator = ETREstimator(total_steps=100, warmup_steps=0)
        now = time.time()
        for step in range(11):
            estimator.update(step, now=now + step * 0.1)
        assert estimator.steps_per_second == pytest.approx(10.0, rel=0.05)
        assert estimator.remaining_steps(10) == 90
        assert estimator.remaining_seconds(10) == pytest.approx(9.0, rel=0.1)
        assert estimator.eta_timestamp(10) is not None
        assert "it/s" in estimator.format(10)
        stats = estimator.stats(10)
        assert stats["remaining_steps"] == 90

    def test_unknown_before_samples(self):
        estimator = ETREstimator(total_steps=10)
        assert estimator.steps_per_second == 0.0
        assert estimator.seconds_per_step == float("inf")
        assert estimator.remaining_seconds() == float("inf")
        assert estimator.format() == "estimating..."
        assert estimator.eta_timestamp() is None

    def test_warmup_steps_ignored(self):
        estimator = ETREstimator(total_steps=100, warmup_steps=5)
        now = time.time()
        estimator.update(0, now=now)  # ignored
        estimator.update(1, now=now + 100)  # ignored
        assert estimator.steps_per_second == 0.0
        estimator.set_total(200)
        assert estimator.total_steps == 200

    def test_slow_rate_formats_as_seconds_per_step(self):
        estimator = ETREstimator(total_steps=10, warmup_steps=0)
        now = time.time()
        estimator.update(0, now=now)
        estimator.update(1, now=now + 5)
        assert "s/it" in estimator.format(1)

    def test_throughput_meter(self):
        meter = ThroughputMeter()
        now = time.time()
        meter.update(100, now=now)
        meter.update(100, now=now + 1)
        assert meter.tokens_per_second == pytest.approx(100.0, rel=0.05)
        assert meter.total_tokens == 200
        assert "tok/s" in meter.format()
        assert meter.stats()["total_tokens"] == 200
        assert meter.average_tokens_per_second > 0

    def test_estimate_training_time(self):
        estimate = estimate_training_time(1000, 8192, 40_000)
        assert estimate["total_tokens"] == 8_192_000
        assert estimate["seconds"] == pytest.approx(204.8)
        unknown = estimate_training_time(10, 10, 0)
        assert unknown["formatted"] == "unknown"


class TestLossVisualization:
    """Metric loading, smoothing and plotting."""

    @pytest.fixture
    def metrics_path(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        for step in range(1, 41):
            append_jsonl(
                path,
                {
                    "step": step,
                    "loss": 5.0 / step,
                    "lr": 1e-3,
                    "elapsed_s": step * 0.5,
                    "total_tokens": step * 100,
                },
            )
        append_jsonl(path, {"step": 40, "val_loss": 0.2})
        return path

    def test_load_metrics(self, metrics_path):
        rows = load_metrics(metrics_path)
        assert len(rows) == 41
        assert load_metrics(metrics_path.parent)  # directory form
        with pytest.raises(FileNotFoundError):
            load_metrics(metrics_path.parent / "absent.jsonl")

    def test_smooth(self):
        assert smooth([]) == []
        assert smooth([1.0, 0.0], weight=0.5) == [1.0, 0.5]
        assert smooth([2.0], weight=2.0) == [2.0]  # weight clamped

    def test_ascii_plot_variants(self):
        assert "no data" in ascii_plot([])
        assert "single point" in ascii_plot([1.0])
        text = ascii_plot([5.0 / (i + 1) for i in range(50)], width=20, height=6, label="loss")
        assert "loss" in text and "min=" in text
        sparkline = ascii_plot([1, 2, 3], height=1)
        assert "[1.0000 .. 3.0000]" in sparkline

    def test_plot_writes_png_when_matplotlib_present(self, metrics_path, tmp_path):
        result = plot_loss_curve(metrics_path, tmp_path / "curve.png")
        assert str(result).endswith(".png")
        lr = plot_learning_rate(metrics_path, tmp_path / "lr.png")
        assert str(lr).endswith(".png")

    def test_plot_without_data(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        append_jsonl(path, {"step": 1})
        assert plot_loss_curve(path, tmp_path / "x.png") == "no data"

    def test_summarize_run(self, metrics_path):
        summary = summarize_run(metrics_path)
        assert summary["steps"] == 40
        assert summary["final_loss"] == pytest.approx(0.125)
        assert summary["best_loss"] == pytest.approx(0.125)
        assert summary["final_perplexity"] > 1.0
        assert summary["total_tokens"] == 4000
        assert summary["elapsed_seconds"] == pytest.approx(20.0)
        assert summarize_run([]) == {"steps": 0}
