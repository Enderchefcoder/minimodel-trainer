"""Tests for optimizers, schedules, the trainer and post-training."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from minimodel.architectures.builder import build_model
from minimodel.datasets.loader import PackedTextDataset, SupervisedDataset
from minimodel.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    GradientMonitor,
    SampleGenerator,
)
from minimodel.training.instruct_cot_posttrainer import CoTTrainer, CoTTrainerConfig
from minimodel.training.instruct_posttrainer import (
    InstructTrainer,
    InstructTrainerConfig,
    build_sft_mixture,
)
from minimodel.training.optim import (
    OPTIMIZERS,
    CombinedOptimizer,
    Lion,
    Muon,
    build_optimizer,
    param_groups,
    zeropower_via_newtonschulz,
)
from minimodel.training.schedules import SCHEDULES, build_scheduler, resolve_warmup
from minimodel.training.trainer import Trainer, TrainerConfig, count_batch_tokens

from conftest import TINY_MODEL


class TestOptimizers:
    """Parameter grouping and the custom optimizers."""

    def test_param_groups_split_by_dim(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        groups = param_groups(model, weight_decay=0.1)
        assert groups[0]["weight_decay"] == 0.1
        assert all(param.dim() >= 2 for param in groups[0]["params"])
        assert all(param.dim() < 2 for param in groups[1]["params"])

    def test_newton_schulz_flattens_spectrum(self):
        torch.manual_seed(0)
        matrix = torch.randn(16, 8)
        matrix[:, 0] *= 50  # give the input a very skewed spectrum
        before = torch.linalg.svdvals(matrix)
        after = torch.linalg.svdvals(zeropower_via_newtonschulz(matrix, steps=10).float())
        # The Muon coefficients do not drive singular values exactly to 1; they
        # compress them into a band around it, which is what the update needs.
        assert before.max() / before.min() > 25
        assert after.max() / after.min() < 5
        assert 0.2 < after.min() < after.max() < 2.0
        tall = zeropower_via_newtonschulz(torch.randn(4, 12))
        assert tall.shape == (4, 12)

    def test_muon_steps_reduce_loss(self):
        torch.manual_seed(0)
        weight = nn.Parameter(torch.randn(8, 8))
        target = torch.randn(32, 8)
        inputs = torch.randn(32, 8)
        optimizer = Muon([weight], lr=0.05)
        initial = ((inputs @ weight.T) - target).pow(2).mean().item()
        for _ in range(30):
            optimizer.zero_grad()
            loss = ((inputs @ weight.T) - target).pow(2).mean()
            loss.backward()
            optimizer.step()
        assert loss.item() < initial

    def test_muon_validation(self):
        with pytest.raises(ValueError, match="2D"):
            Muon([nn.Parameter(torch.zeros(3))])
        with pytest.raises(ValueError, match="lr"):
            Muon([nn.Parameter(torch.zeros(2, 2))], lr=-1)
        with pytest.raises(ValueError, match="momentum"):
            Muon([nn.Parameter(torch.zeros(2, 2))], momentum=1.5)

    def test_lion_updates(self):
        weight = nn.Parameter(torch.ones(4, 4))
        optimizer = Lion([weight], lr=0.1, weight_decay=0.1)
        (weight.sum()).backward()
        optimizer.step()
        assert not torch.allclose(weight.detach(), torch.ones(4, 4))
        with pytest.raises(ValueError, match="lr"):
            Lion([weight], lr=0)
        with pytest.raises(ValueError, match="betas"):
            Lion([weight], betas=(1.5, 0.9))

    def test_build_optimizer_registry(self, tiny_model):
        for name in ("adamw", "lion", "sgd"):
            optimizer = build_optimizer(tiny_model, name, lr=1e-3)
            assert len(optimizer.param_groups) >= 1
        assert "adamw" in OPTIMIZERS

    def test_muon_hybrid_and_state_roundtrip(self, tiny_model):
        optimizer = build_optimizer(tiny_model, "muon", lr=0.02, adamw_lr=1e-3)
        assert isinstance(optimizer, CombinedOptimizer)
        assert "CombinedOptimizer" in repr(optimizer)

        tokens = torch.randint(0, tiny_model.vocab_size, (2, 8))
        tiny_model.forward_with_loss(tokens, tokens).loss.backward()
        optimizer.step()
        state = optimizer.state_dict()
        optimizer.zero_grad(set_to_none=True)
        optimizer.load_state_dict(state)
        # Schedulers mutate group lr in place; the merged view must expose it.
        for group in optimizer.param_groups:
            group["lr"] = 123.0
        optimizer.param_groups = []  # setter is a no-op by design

    def test_combined_optimizer_requires_children(self):
        with pytest.raises(ValueError, match="at least one"):
            CombinedOptimizer([])


class TestSchedules:
    """Learning-rate schedule shapes."""

    def test_cosine_shape(self):
        schedule = SCHEDULES.get("cosine")(100, 10, min_lr_ratio=0.1)
        assert schedule(0) < 1.0
        assert schedule(9) == pytest.approx(1.0)
        assert schedule(100) == pytest.approx(0.1, abs=1e-6)

    def test_wsd_plateau_and_decay(self):
        schedule = SCHEDULES.get("wsd")(100, 10, decay_ratio=0.2)
        assert schedule(50) == 1.0
        assert schedule(79) == 1.0
        assert schedule(90) < 1.0
        assert schedule(100) == pytest.approx(0.0, abs=1e-6)
        sqrt_shape = SCHEDULES.get("wsd")(100, 0, decay_ratio=0.2, decay_shape="sqrt")
        cosine_shape = SCHEDULES.get("wsd")(100, 0, decay_ratio=0.2, decay_shape="cosine")
        assert sqrt_shape(90) < cosine_shape(90)

    def test_linear_and_constant_and_isqrt(self):
        linear = SCHEDULES.get("linear")(100, 0)
        assert linear(50) == pytest.approx(0.5)
        constant = SCHEDULES.get("constant")(100, 10)
        assert constant(50) == 1.0
        isqrt = SCHEDULES.get("inverse_sqrt")(100, 4)
        assert isqrt(3) == 1.0
        assert isqrt(99) == pytest.approx(math.sqrt(4 / 100))

    def test_build_scheduler_applies_to_optimizer(self):
        weight = nn.Parameter(torch.zeros(2, 2))
        optimizer = torch.optim.AdamW([weight], lr=1.0)
        scheduler = build_scheduler(optimizer, "cosine", total_steps=10, warmup_steps=0.2)
        rates = []
        for _ in range(10):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        assert rates[1] == pytest.approx(1.0)
        assert rates[-1] < rates[1]

    def test_resolve_warmup(self):
        assert resolve_warmup(0.02, 10_000) == 200
        assert resolve_warmup(500, 10_000) == 500
        assert resolve_warmup(0, 10_000) == 0


def _trainer_config(tmp_path, **overrides) -> TrainerConfig:
    defaults = dict(
        run_name="t",
        output_dir=str(tmp_path),
        max_steps=6,
        batch_size=2,
        seq_len=16,
        lr=1e-3,
        log_every=3,
        eval_every=3,
        eval_batches=2,
        save_every=3,
        warmup=0.2,
        resume=False,
    )
    defaults.update(overrides)
    return TrainerConfig(**defaults)


class TestTrainer:
    """The core loop: fit, evaluate, resume."""

    def test_fit_reduces_loss_and_writes_artifacts(self, tiny_model, corpus_dir, tmp_path):
        dataset = PackedTextDataset(corpus_dir, seq_len=16)
        trainer = Trainer(
            tiny_model,
            _trainer_config(tmp_path, max_steps=25, eval_every=0, save_every=0),
            train_dataset=dataset,
        )
        result = trainer.fit()
        assert result.steps == 25
        assert result.final_loss < 6.5
        assert result.final_perplexity > 0
        assert (trainer.run_dir / "metrics.jsonl").exists()
        assert (trainer.run_dir / "run_metadata.json").exists()
        assert (trainer.run_dir / "result.json").exists()
        assert result.total_tokens == 25 * 2 * 16

    def test_evaluation_and_checkpointing(self, tiny_model, corpus_dir, tmp_path):
        dataset = PackedTextDataset(corpus_dir, seq_len=16)
        trainer = Trainer(
            tiny_model,
            _trainer_config(tmp_path),
            train_dataset=dataset,
            eval_dataset=PackedTextDataset(corpus_dir, seq_len=16, seed=9),
        )
        result = trainer.fit()
        assert result.best_val_loss is not None
        assert trainer.checkpoints.latest() is not None
        exported = trainer.export(tmp_path / "release")
        assert (exported / "model.pt").exists()

    def test_resume_continues_from_checkpoint(self, corpus_dir, tokenizer, tmp_path):
        dataset = PackedTextDataset(corpus_dir, seq_len=16)
        overrides = {**TINY_MODEL, "vocab_size": tokenizer.vocab_size}

        model = build_model("dense_3m", overrides=overrides, verify_budget=False)
        first = Trainer(
            model, _trainer_config(tmp_path, max_steps=4, save_every=2, eval_every=0), train_dataset=dataset
        )
        first.fit()

        model2 = build_model("dense_3m", overrides=overrides, verify_budget=False)
        second = Trainer(
            model2,
            _trainer_config(tmp_path, max_steps=8, save_every=4, eval_every=0, resume=True),
            train_dataset=dataset,
        )
        assert second.maybe_resume()
        assert second.step == 4
        result = second.fit()
        assert result.steps == 8

    def test_grad_accumulation_counts_tokens(self, tiny_model, corpus_dir, tmp_path):
        dataset = PackedTextDataset(corpus_dir, seq_len=16)
        trainer = Trainer(
            tiny_model,
            _trainer_config(
                tmp_path, max_steps=2, grad_accum_steps=3, eval_every=0, save_every=0
            ),
            train_dataset=dataset,
        )
        result = trainer.fit()
        assert result.total_tokens == 2 * 3 * 2 * 16

    def test_missing_dataset_raises(self, tiny_model, tmp_path):
        trainer = Trainer(tiny_model, _trainer_config(tmp_path, eval_every=0, save_every=0))
        with pytest.raises(ValueError, match="train_dataset"):
            trainer.fit()

    def test_config_validation(self):
        with pytest.raises(ValueError, match="max_steps"):
            TrainerConfig(max_steps=0)
        with pytest.raises(ValueError, match="grad_accum"):
            TrainerConfig(grad_accum_steps=0)
        config = TrainerConfig(batch_size=4, seq_len=8, grad_accum_steps=2)
        assert config.tokens_per_step == 64

    def test_count_batch_tokens(self):
        batch = {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "chosen_ids": torch.zeros(2, 3, dtype=torch.long),
            "labels": torch.zeros(2, 4, dtype=torch.long),
        }
        assert count_batch_tokens(batch) == 8 + 6

    def test_evaluate_without_loader_is_empty(self, tiny_model, corpus_dir, tmp_path):
        trainer = Trainer(
            tiny_model,
            _trainer_config(tmp_path, eval_every=0, save_every=0),
            train_dataset=PackedTextDataset(corpus_dir, seq_len=16),
        )
        assert trainer.evaluate() == {}


class TestCallbacks:
    """Early stopping, divergence detection and fan-out."""

    class _Recorder(Callback):
        def __init__(self):
            self.events = []

        def on_train_begin(self, trainer):
            self.events.append("begin")

        def on_step_end(self, trainer, step, metrics):
            self.events.append(("step", step))

        def on_train_end(self, trainer):
            self.events.append("end")

    def test_callback_hooks_fire_in_order(self, tiny_model, corpus_dir, tmp_path):
        recorder = self._Recorder()
        Trainer(
            tiny_model,
            _trainer_config(tmp_path, max_steps=2, eval_every=0, save_every=0),
            train_dataset=PackedTextDataset(corpus_dir, seq_len=16),
            callbacks=[recorder],
        ).fit()
        assert recorder.events[0] == "begin"
        assert recorder.events[-1] == "end"
        assert ("step", 1) in recorder.events

    def test_callback_list_fanout(self):
        a, b = self._Recorder(), self._Recorder()
        fan = CallbackList([a])
        fan.append(b)
        assert len(fan) == 2
        fan.on_train_begin(None)
        assert a.events == ["begin"] and b.events == ["begin"]
        assert list(iter(fan)) == [a, b]

    def test_early_stopping_requests_stop(self, tiny_model, corpus_dir, tmp_path):
        trainer = Trainer(
            tiny_model,
            _trainer_config(tmp_path, max_steps=2, eval_every=0, save_every=0),
            train_dataset=PackedTextDataset(corpus_dir, seq_len=16),
        )
        stopper = EarlyStopping("val_loss", patience=2)
        stopper.on_evaluate(trainer, 1, {"val_loss": 1.0})
        stopper.on_evaluate(trainer, 2, {"val_loss": 1.1})
        assert not trainer.should_stop
        stopper.on_evaluate(trainer, 3, {"val_loss": 1.2})
        assert trainer.should_stop
        stopper.on_evaluate(trainer, 4, {"other": 1})  # ignored
        with pytest.raises(ValueError, match="min.*max"):
            EarlyStopping(mode="diagonal")

    def test_early_stopping_max_mode(self, tiny_model, corpus_dir, tmp_path):
        trainer = Trainer(
            tiny_model,
            _trainer_config(tmp_path, max_steps=2, eval_every=0, save_every=0),
            train_dataset=PackedTextDataset(corpus_dir, seq_len=16),
        )
        stopper = EarlyStopping("accuracy", patience=1, mode="max")
        stopper.on_evaluate(trainer, 1, {"accuracy": 0.5})
        stopper.on_evaluate(trainer, 2, {"accuracy": 0.4})
        assert trainer.should_stop

    def test_gradient_monitor_stops_on_nan_and_divergence(self, tiny_model, corpus_dir, tmp_path):
        trainer = Trainer(
            tiny_model,
            _trainer_config(tmp_path, max_steps=2, eval_every=0, save_every=0),
            train_dataset=PackedTextDataset(corpus_dir, seq_len=16),
        )
        monitor = GradientMonitor(max_loss_ratio=2.0)
        monitor.on_step_end(trainer, 1, {"loss": 1.0})
        assert not trainer.should_stop
        monitor.on_step_end(trainer, 2, {"loss": 5.0})
        assert trainer.should_stop

        trainer.should_stop = False
        nan_monitor = GradientMonitor()
        nan_monitor.on_step_end(trainer, 3, {"loss": float("nan")})
        assert trainer.should_stop
        nan_monitor.on_step_end(trainer, 4, {})  # missing loss is fine

    def test_early_stopping_config_wires_in(self, tiny_model, corpus_dir, tmp_path):
        trainer = Trainer(
            tiny_model,
            _trainer_config(
                tmp_path, max_steps=2, eval_every=0, save_every=0, early_stopping_patience=2
            ),
            train_dataset=PackedTextDataset(corpus_dir, seq_len=16),
        )
        assert any(isinstance(cb, EarlyStopping) for cb in trainer.callbacks)

    def test_sample_generator_logs_text(self, tiny_model, corpus_dir, tokenizer, tmp_path, caplog):
        trainer = Trainer(
            tiny_model,
            _trainer_config(tmp_path, max_steps=2, eval_every=0, save_every=0),
            train_dataset=PackedTextDataset(corpus_dir, seq_len=16),
            tokenizer=tokenizer,
        )
        generator = SampleGenerator("The", max_new_tokens=4)
        generator.on_evaluate(trainer, 1, {})
        without_tokenizer = SampleGenerator("The")
        trainer.tokenizer = None
        without_tokenizer.on_evaluate(trainer, 1, {})  # silently does nothing


class TestPostTraining:
    """SFT and chain-of-thought trainers."""

    def test_instruct_trainer_masks_prompt(self, tiny_model, sft_dir, tmp_path):
        config = InstructTrainerConfig(
            run_name="sft",
            output_dir=str(tmp_path),
            max_steps=4,
            batch_size=2,
            seq_len=16,
            lr=1e-3,
            log_every=2,
            eval_every=2,
            eval_batches=2,
            save_every=0,
            resume=False,
        )
        trainer = InstructTrainer(
            tiny_model,
            config,
            train_dataset=SupervisedDataset(sft_dir, seq_len=16),
            eval_dataset=SupervisedDataset(sft_dir, seq_len=16, seed=5),
        )
        result = trainer.fit()
        assert result.steps == 4
        evaluation = trainer.evaluate()
        assert "val_token_accuracy" in evaluation
        assert 0.0 <= evaluation["val_token_accuracy"] <= 1.0

    def test_replay_mixture(self, sft_dir, corpus_dir):
        sft = SupervisedDataset(sft_dir, seq_len=16)
        replay = PackedTextDataset(corpus_dir, seq_len=16)
        mixed = build_sft_mixture(sft, replay, 0.3)
        assert type(mixed).__name__ == "MixtureDataset"
        assert build_sft_mixture(sft, None, 0.3) is sft
        assert build_sft_mixture(sft, replay, 0.0) is sft
        with pytest.raises(ValueError, match="< 1"):
            build_sft_mixture(sft, replay, 1.0)

    def test_label_smoothing_applies(self, tiny_model, sft_dir, tmp_path):
        config = InstructTrainerConfig(
            run_name="ls",
            output_dir=str(tmp_path),
            max_steps=2,
            batch_size=2,
            seq_len=16,
            label_smoothing=0.1,
            eval_every=0,
            save_every=0,
            resume=False,
        )
        trainer = InstructTrainer(
            tiny_model, config, train_dataset=SupervisedDataset(sft_dir, seq_len=16)
        )
        assert trainer.fit().steps == 2

    def test_cot_trainer_reasoning_mask(self, tiny_model, cot_dir, tokenizer, tmp_path):
        config = CoTTrainerConfig(
            run_name="cot",
            output_dir=str(tmp_path),
            max_steps=3,
            batch_size=2,
            seq_len=24,
            lr=1e-3,
            reasoning_loss_weight=0.5,
            enforce_think_close=0.1,
            eval_every=0,
            save_every=0,
            resume=False,
        )
        trainer = CoTTrainer(
            tiny_model,
            config,
            tokenizer=tokenizer,
            train_dataset=SupervisedDataset(cot_dir, seq_len=24),
        )
        assert config.think_open_id == tokenizer.token_to_id("<|think|>")
        result = trainer.fit()
        assert result.steps == 3

        open_id = tokenizer.token_to_id("<|think|>")
        close_id = tokenizer.token_to_id("<|/think|>")
        sequence = torch.tensor([[1, open_id, 5, 6, close_id, 7]])
        mask = trainer.reasoning_mask(sequence)
        assert mask.tolist() == [[False, True, True, True, True, False]]

    def test_cot_without_markers_warns(self, tiny_model, cot_dir, tmp_path, caplog):
        config = CoTTrainerConfig(
            run_name="nomark",
            output_dir=str(tmp_path),
            max_steps=1,
            batch_size=2,
            seq_len=16,
            eval_every=0,
            save_every=0,
            resume=False,
        )
        trainer = CoTTrainer(
            tiny_model, config, train_dataset=SupervisedDataset(cot_dir, seq_len=16)
        )
        mask = trainer.reasoning_mask(torch.zeros(1, 4, dtype=torch.long))
        assert not mask.any()
