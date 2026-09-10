"""Tests for the ENDER architecture, trainer and leaderboard suite loader."""

from __future__ import annotations

import pytest
import torch

from minimodel.architectures.ender import (
    EnderAdapter,
    EnderLoss,
    EnderRecurrenceModule,
    EnderTransformer,
    build_ender_optimizer,
)
from minimodel.benchmarking.leaderboard import (
    LEADERBOARD_DATASETS,
    load_leaderboard_tasks,
)

TINY = {
    "vocab_size": 48,
    "dim": 32,
    "n_heads": 2,
    "head_dim": 16,
    "ffn_hidden": 64,
    "r_latent": 16,
    "num_steps": 2,
    "min_steps": 1,
}


def _tiny(**overrides):
    """Build a small ENDER model."""
    return EnderTransformer({**TINY, **overrides})


class TestEnderRecurrence:
    """The latent recurrence module in isolation."""

    def test_shapes_and_aux(self):
        module = EnderRecurrenceModule(32, r_latent=16, num_steps=3, num_heads=2, head_dim=8)
        h = torch.randn(2, 7, 32)
        h_prime, aux = module(h, steps=3)
        assert h_prime.shape == h.shape
        assert aux["z_final"].shape == (2, 7, 16)
        assert aux["innovation"].shape == (2, 7, 16)

    def test_fresh_module_is_noop(self):
        module = EnderRecurrenceModule(32, r_latent=16, num_steps=4, num_heads=2, head_dim=8)
        h = torch.randn(1, 5, 32)
        h_prime, _ = module(h)
        assert torch.allclose(h_prime, h, atol=1e-7)

    def test_more_steps_than_embed_table_is_safe(self):
        module = EnderRecurrenceModule(32, r_latent=16, num_steps=2, num_heads=2, head_dim=8)
        h = torch.randn(1, 4, 32)
        h_prime, _ = module(h, steps=6)
        assert h_prime.shape == h.shape

    def test_rejects_invalid_steps(self):
        module = EnderRecurrenceModule(32, r_latent=16, num_steps=2, num_heads=2, head_dim=8)
        with pytest.raises(ValueError, match="steps"):
            module(torch.randn(1, 4, 32), steps=0)

    def test_head_geometry_validated(self):
        with pytest.raises(ValueError, match="r_latent"):
            EnderRecurrenceModule(32, r_latent=16, num_heads=3, head_dim=8)


class TestEnderTransformer:
    """The first-class ENDER language model."""

    def test_forward_shape(self):
        model = _tiny().eval()
        tokens = torch.randint(0, TINY["vocab_size"], (2, 10))
        logits = model(tokens)
        assert logits.shape == (2, 10, TINY["vocab_size"])
        assert model.ender_aux["z_final"].shape[-1] == TINY["r_latent"]

    def test_steps_change_output(self):
        model = _tiny()
        with torch.no_grad():
            model.ender.alpha.normal_(0.0, 0.5)  # make the recurrence matter
        tokens = torch.randint(0, TINY["vocab_size"], (1, 8))
        one = model(tokens, steps=1)
        four = model(tokens, steps=4)
        assert not torch.allclose(one, four)

    def test_variable_steps_training(self):
        model = _tiny(variable_steps=True)
        model.train()
        resolved = {model.resolve_steps(None) for _ in range(20)}
        assert resolved <= {1, 2} and model.resolve_steps(None) >= 1

    def test_cache_equivalence(self):
        model = _tiny().eval()
        tokens = torch.randint(0, TINY["vocab_size"], (1, 8))
        full = model(tokens, steps=2)
        cache = model.new_cache()
        incremental = torch.cat(
            [model(tokens[:, i : i + 1], steps=2, cache=cache) for i in range(8)], dim=1
        )
        assert torch.allclose(full, incremental, atol=1e-4)

    def test_save_and_load_roundtrip(self, tmp_path):
        model = _tiny()
        model.save_pretrained(tmp_path / "model")
        from minimodel.architectures.builder import load_model

        restored = load_model(tmp_path / "model")
        tokens = torch.randint(0, TINY["vocab_size"], (1, 6))
        assert torch.allclose(model(tokens), restored(tokens), atol=1e-6)


class TestEnderAdapter:
    """Grafting ENDER onto an existing backbone."""

    def test_adapter_is_noop_at_init(self, tiny_model):
        tiny_model.eval()
        tokens = torch.randint(0, tiny_model.vocab_size, (1, 6))
        with torch.no_grad():
            reference = tiny_model(tokens)
        adapter = EnderAdapter(tiny_model, {"dim": tiny_model.dim, "r_latent": 16, "num_steps": 2})
        adapter.eval()
        with torch.no_grad():
            wrapped = adapter(tokens)
        assert torch.allclose(wrapped, reference, atol=1e-6)

    def test_set_recurrence_steps_dial(self, tiny_model):
        adapter = EnderAdapter(tiny_model, {"dim": tiny_model.dim, "r_latent": 16, "num_steps": 2})
        with torch.no_grad():
            adapter.ender.alpha.normal_(0.0, 0.5)
        tokens = torch.randint(0, tiny_model.vocab_size, (1, 6))
        adapter.set_recurrence_steps(1)
        one = adapter(tokens)
        adapter.set_recurrence_steps(4)
        four = adapter(tokens)
        assert not torch.allclose(one, four)

    def test_staged_optimizer_freezes_backbone(self, tiny_model):
        adapter = EnderAdapter(tiny_model, {"dim": tiny_model.dim, "r_latent": 16, "num_steps": 2})
        optimizer = build_ender_optimizer(adapter, base_lr=1e-3, unfreeze_top_blocks=1)
        trainable = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
        assert all(p.requires_grad for p in adapter.ender.parameters())
        assert any(not p.requires_grad for p in adapter.backbone.parameters())
        assert len(trainable) > 0

    def test_final_norm_resolution_failure(self, tiny_model):
        with pytest.raises(ValueError, match="no module named"):
            EnderAdapter(tiny_model, {"dim": tiny_model.dim}, final_norm_name="does.not.exist")


class TestEnderLoss:
    """The composite objective."""

    def _logits(self, vocab=48):
        return torch.randn(2, 6, vocab, requires_grad=True)

    def test_ce_only(self):
        loss_engine = EnderLoss(lambda_delta=0.0, lambda_kd=0.0)
        loss, extras = loss_engine(self._logits(), torch.randint(0, 48, (2, 6)))
        assert loss.ndim == 0 and "loss_ce" in extras and "loss_delta" not in extras

    def test_delta_and_kd_terms(self):
        loss_engine = EnderLoss(lambda_delta=0.1, lambda_kd=0.5)
        aux = {"z_first": torch.randn(2, 6, 16), "z_final": torch.randn(2, 6, 16)}
        loss, extras = loss_engine(
            self._logits(), torch.randint(0, 48, (2, 6)), aux=aux, teacher_logits=torch.randn(2, 6, 48)
        )
        assert "loss_delta" in extras and "loss_kd" in extras
        assert torch.isfinite(loss)


class TestLeaderboardSuite:
    """The Open_SLM_Leaderboard mapping."""

    def test_required_tasks_are_mapped(self):
        assert "blimp" in LEADERBOARD_DATASETS and "arc_easy" in LEADERBOARD_DATASETS

    def test_unknown_task_fails_loudly(self):
        with pytest.raises(ValueError, match="unknown leaderboard"):
            load_leaderboard_tasks(["not_a_benchmark"])

    def test_missing_data_is_skipped(self):
        tasks = load_leaderboard_tasks(["blimp", "arc_easy"], data_dir="/nonexistent")
        assert tasks == {}


class TestEnderTrainer:
    """The trainer subclass, via the standard loop."""

    def test_fit_reduces_loss(self, tokenizer, corpus_dir, tmp_path):
        from minimodel.architectures.builder import build_model
        from minimodel.datasets.loader import PackedTextDataset
        from minimodel.training.ender import EnderTrainer, EnderTrainerConfig

        model = build_model(
            "ender_12m",
            overrides={
                "vocab_size": tokenizer.vocab_size,
                "dim": 32,
                "n_layers": 2,
                "n_heads": 2,
                "head_dim": 16,
                "ffn_hidden": 64,
                "r_latent": 16,
                "num_steps": 2,
                "min_steps": 1,
                "max_seq_len": 128,
                "window": 64,
            },
            verify_budget=False,
        )
        trainer = EnderTrainer(
            model,
            EnderTrainerConfig(
                run_name="ender",
                output_dir=str(tmp_path),
                max_steps=6,
                batch_size=2,
                seq_len=16,
                lr=1e-3,
                log_every=3,
                eval_every=0,
                save_every=0,
                lambda_delta=0.05,
                resume=False,
            ),
            train_dataset=PackedTextDataset(corpus_dir, seq_len=16),
        )
        result = trainer.fit()
        assert result.steps == 6
        assert result.final_loss < 8.0
        metrics = (tmp_path / "ender" / "metrics.jsonl").read_text().strip().splitlines()
        assert "loss_delta" in metrics[0]
