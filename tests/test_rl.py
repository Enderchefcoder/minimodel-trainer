"""Tests for DPO, RLVR and SPIN."""

from __future__ import annotations

import json

import pytest
import torch

from minimodel.training.rl.dpo import DPOConfig, DPOTrainer, _PairIterator, dpo_loss
from minimodel.training.rl.rlvr import (
    VERIFIERS,
    RLVRConfig,
    RLVRTrainer,
    extract_final_number,
    group_advantages,
    verify_exact_match,
    verify_expression,
    verify_latex_answer,
    verify_length,
    verify_numeric,
)
from minimodel.training.rl.spin import SPINConfig, SPINTrainer, generate_self_play_pairs


class TestDPOLoss:
    """The preference loss itself."""

    def test_correct_preference_gives_low_loss(self):
        good = torch.tensor([2.0])
        bad = torch.tensor([-2.0])
        zero = torch.zeros(1)
        low, chosen, rejected = dpo_loss(good, bad, zero, zero, beta=1.0)
        high, _, _ = dpo_loss(bad, good, zero, zero, beta=1.0)
        assert low.item() < high.item()
        assert chosen.item() > rejected.item()

    def test_reference_anchors_the_ratio(self):
        policy_chosen = torch.tensor([1.0])
        policy_rejected = torch.tensor([0.0])
        # If the reference already prefers chosen by the same margin, the
        # implicit reward difference is zero and the loss is log(2).
        loss, _, _ = dpo_loss(
            policy_chosen, policy_rejected, policy_chosen, policy_rejected, beta=1.0
        )
        assert loss.item() == pytest.approx(0.6931, abs=1e-3)

    @pytest.mark.parametrize("loss_type", ["sigmoid", "ipo", "hinge", "cpo"])
    def test_all_variants_are_finite(self, loss_type):
        loss, _, _ = dpo_loss(
            torch.tensor([1.0]),
            torch.tensor([0.5]),
            torch.tensor([0.2]),
            torch.tensor([0.1]),
            loss_type=loss_type,
        )
        assert torch.isfinite(loss)

    def test_label_smoothing_softens(self):
        args = (torch.tensor([3.0]), torch.tensor([-3.0]), torch.zeros(1), torch.zeros(1))
        sharp, _, _ = dpo_loss(*args, beta=1.0, label_smoothing=0.0)
        smoothed, _, _ = dpo_loss(*args, beta=1.0, label_smoothing=0.2)
        assert smoothed.item() > sharp.item()

    def test_unknown_variant_rejected(self):
        with pytest.raises(ValueError, match="loss_type"):
            dpo_loss(torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), loss_type="x")


class TestDPOTrainer:
    """End-to-end DPO on the bundled pairs."""

    def _config(self, tmp_path, **overrides) -> DPOConfig:
        defaults = dict(
            run_name="dpo",
            output_dir=str(tmp_path),
            max_steps=4,
            batch_size=2,
            lr=1e-5,
            log_every=2,
            eval_every=0,
            save_every=0,
            warmup=0,
            resume=False,
        )
        defaults.update(overrides)
        return DPOConfig(**defaults)

    def test_fit_and_diagnostics(self, tiny_model, pairs_path, tokenizer, tmp_path):
        trainer = DPOTrainer(
            tiny_model, self._config(tmp_path), pairs_path=pairs_path, tokenizer=tokenizer
        )
        assert trainer.reference is not None
        for param in trainer.reference.parameters():
            assert not param.requires_grad
        result = trainer.fit()
        assert result.steps == 4

        batch = trainer._next_batch()
        loss, extras = trainer.compute_loss(batch)
        assert torch.isfinite(loss)
        assert 0.0 <= extras["reward_accuracy"] <= 1.0
        assert "reward_margin" in extras

    def test_cpo_needs_no_reference(self, tiny_model, pairs_path, tokenizer, tmp_path):
        trainer = DPOTrainer(
            tiny_model,
            self._config(tmp_path, loss_type="cpo", sft_weight=0.5),
            pairs_path=pairs_path,
            tokenizer=tokenizer,
        )
        assert trainer.reference is None
        loss, extras = trainer.compute_loss(trainer._next_batch())
        assert "sft_loss" in extras
        assert torch.isfinite(loss)

    def test_evaluate_reports_averages(self, tiny_model, pairs_path, tokenizer, tmp_path):
        trainer = DPOTrainer(
            tiny_model, self._config(tmp_path), pairs_path=pairs_path, tokenizer=tokenizer
        )
        metrics = trainer.evaluate(max_batches=2)
        assert "val_loss" in metrics
        assert "val_reward_accuracy" in metrics

    def test_pair_iterator_validation(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="no preference pairs"):
            _PairIterator(empty, 2, 64)
        with pytest.raises(FileNotFoundError):
            _PairIterator(tmp_path / "missing.jsonl", 2, 64)

    def test_pair_iterator_accepts_directory(self, pairs_path):
        iterator = _PairIterator(pairs_path.parent, 2, 64)
        batch = next(iterator.batches())
        assert batch["chosen_ids"].shape[0] == 2


class TestVerifiers:
    """The reward functions."""

    def test_extract_final_number(self):
        assert extract_final_number("12 then 30, so 42.") == "42"
        assert extract_final_number("answer: 1,234") == "1234"
        assert extract_final_number("no numbers") is None

    def test_numeric(self):
        assert verify_numeric("the answer is 42", "42") == 1.0
        assert verify_numeric("the answer is 42.0", "42") == 1.0
        assert verify_numeric("41", "42") == 0.0
        assert verify_numeric("nothing", "42") == 0.0
        assert verify_numeric("x", "x") == 0.0  # non-numeric reference, no number

    def test_exact_match(self):
        assert verify_exact_match("The capital is Paris.", "paris") == 1.0
        assert verify_exact_match("Lyon", "paris") == 0.0

    def test_latex_answer(self):
        assert verify_latex_answer(r"so \boxed{42}", r"\boxed{42}") == 1.0
        assert verify_latex_answer(r"so \boxed{41}", r"\boxed{42}") == 0.0
        assert verify_latex_answer("plain 42", "42") == 1.0

    def test_expression(self):
        assert verify_expression("(6 + 6) * 2 = 24", "24") == 1.0
        assert verify_expression("6 * 3", "24") == 0.0
        assert verify_expression("no math", "24") == 0.0
        assert verify_expression("2+2", "abc") == 0.0

    def test_length(self):
        assert verify_length("one two three", "3") == 1.0
        assert verify_length("one two", "4") == 0.5
        assert verify_length("words", "0") == 0.0
        assert verify_length("words", "abc") == 0.0

    def test_registry_contains_all(self):
        for name in ("numeric", "exact_match", "latex_answer", "expression", "length"):
            assert name in VERIFIERS


class TestGroupAdvantages:
    """GRPO's group-relative baseline."""

    def test_normalises_within_group(self):
        rewards = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
        advantages = group_advantages(rewards)
        assert advantages[0, 0] < 0 < advantages[0, 1]
        # A group with identical rewards carries no signal.
        assert torch.allclose(advantages[1], torch.zeros(2), atol=1e-3)


class TestRLVRTrainer:
    """Rollouts, scoring and the policy-gradient loss."""

    def _config(self, tmp_path, **overrides) -> RLVRConfig:
        defaults = dict(
            run_name="rlvr",
            output_dir=str(tmp_path),
            max_steps=2,
            batch_size=2,
            group_size=4,
            max_new_tokens=6,
            lr=1e-5,
            log_every=1,
            eval_every=0,
            save_every=0,
            warmup=0,
            resume=False,
        )
        defaults.update(overrides)
        return RLVRConfig(**defaults)

    def test_rollout_shapes_and_fit(self, tiny_model, tasks_path, tokenizer, tmp_path):
        trainer = RLVRTrainer(
            tiny_model, self._config(tmp_path), tasks_path=tasks_path, tokenizer=tokenizer
        )
        rollout = trainer.rollout()
        assert rollout["input_ids"].shape[0] == 2 * 4
        assert rollout["advantages"].shape == (8,)
        assert rollout["rewards"].shape == (8,)
        loss, extras = trainer.compute_loss(rollout)
        assert torch.isfinite(loss)
        assert extras["kl"] >= 0
        assert 0.0 <= extras["solve_rate"] <= 1.0
        result = trainer.fit()
        assert result.steps == 2

    def test_score_uses_task_verifier(self, tiny_model, tasks_path, tokenizer, tmp_path):
        trainer = RLVRTrainer(
            tiny_model,
            self._config(tmp_path, format_bonus=0.1),
            tasks_path=tasks_path,
            tokenizer=tokenizer,
        )
        assert trainer.score("the answer is 13", {"answer": "13"}) == pytest.approx(1.1)
        assert trainer.score("wrong", {"answer": "13"}) == pytest.approx(0.1)
        assert trainer.score("Paris", {"answer": "paris", "verifier": "exact_match"}) == pytest.approx(1.1)

    def test_system_prompt_prepended(self, tiny_model, tasks_path, tokenizer, tmp_path):
        trainer = RLVRTrainer(
            tiny_model,
            self._config(tmp_path, system_prompt="Answer with a number."),
            tasks_path=tasks_path,
            tokenizer=tokenizer,
        )
        assert trainer._build_prompt({"prompt": "2+2?"}).startswith("Answer with a number.")

    def test_evaluate_solve_rate(self, tiny_model, tasks_path, tokenizer, tmp_path):
        trainer = RLVRTrainer(
            tiny_model, self._config(tmp_path), tasks_path=tasks_path, tokenizer=tokenizer
        )
        metrics = trainer.evaluate(max_batches=1)
        assert 0.0 <= metrics["val_solve_rate"] <= 1.0

    def test_task_loading_validation(self, tiny_model, tokenizer, tmp_path):
        with pytest.raises(FileNotFoundError):
            RLVRTrainer(
                tiny_model,
                self._config(tmp_path),
                tasks_path=tmp_path / "missing.jsonl",
                tokenizer=tokenizer,
            )
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="at least one task"):
            RLVRTrainer(
                tiny_model, self._config(tmp_path), tasks_path=empty, tokenizer=tokenizer
            )

    def test_in_memory_tasks(self, tiny_model, tokenizer, tmp_path):
        trainer = RLVRTrainer(
            tiny_model,
            self._config(tmp_path),
            tasks_path=[{"prompt": "1+1?", "answer": "2"}],
            tokenizer=tokenizer,
        )
        assert len(trainer.tasks) == 1
        # The cursor wraps around a short task list.
        assert len(trainer._next_tasks(3)) == 3


class TestSPIN:
    """Self-play pair generation and iteration."""

    def test_generate_self_play_pairs(self, tiny_model, tokenizer, tmp_path):
        records = [{"instruction": "Say hello.", "output": "Hello!"}] * 3 + [{"bad": 1}]
        stats = generate_self_play_pairs(
            tiny_model, tokenizer, records, tmp_path / "pairs.jsonl", max_new_tokens=6, seed=0
        )
        assert stats["pairs"] >= 1
        assert stats["skipped"] >= 1
        row = json.loads((tmp_path / "pairs.jsonl").read_text().splitlines()[0])
        assert set(row) >= {"chosen_ids", "chosen_labels", "rejected_ids", "rejected_labels"}

    def test_spin_trainer_iterates(self, tiny_model, sft_jsonl, tokenizer, tmp_path):
        config = SPINConfig(
            run_name="spin",
            output_dir=str(tmp_path),
            max_steps=4,
            iterations=2,
            batch_size=2,
            max_new_tokens=6,
            lr=1e-5,
            log_every=2,
            eval_every=0,
            save_every=0,
            warmup=0,
            resume=False,
        )
        trainer = SPINTrainer(
            tiny_model, config, dataset_path=sft_jsonl, tokenizer=tokenizer
        )
        result = trainer.fit()
        assert result.steps == 4
        # Two iterations means the second pair file was generated.
        assert (trainer.pair_cache_dir / "iteration_000.jsonl").exists()
        assert trainer.iteration >= 1
        loss, extras = trainer.compute_loss(trainer._next_batch())
        assert extras["spin_iteration"] == float(trainer.iteration)

    def test_spin_missing_dataset(self, tiny_model, tokenizer, tmp_path):
        with pytest.raises(FileNotFoundError):
            SPINTrainer(
                tiny_model,
                SPINConfig(run_name="s", output_dir=str(tmp_path), resume=False),
                dataset_path=tmp_path / "missing.jsonl",
                tokenizer=tokenizer,
            )
