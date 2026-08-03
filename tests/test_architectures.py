"""Tests for layers, models and the template builder."""

from __future__ import annotations

import json

import pytest
import torch

from minimodel.architectures.base import ModelOutput
from minimodel.architectures.builder import (
    build_model,
    describe_model,
    list_templates,
    load_model,
    load_template,
    template_to_model_config,
)
from minimodel.architectures.dense import DenseTransformer
from minimodel.architectures.hybrid import HybridRecurrentTransformer
from minimodel.architectures.layers import (
    CausalLocalAttention,
    FactorizedEmbedding,
    GatedRecurrentUnit,
    KVCache,
    MoEFeedForward,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUFeedForward,
    apply_rope,
    build_attention_mask,
    repeat_kv,
)
from minimodel.architectures.looped import LoopedTransformer
from minimodel.architectures.moe import MoETransformer
from minimodel.architectures.registry import (
    ARCHITECTURES,
    list_architectures,
    register_architecture,
)
from minimodel.core.config import ConfigError

TINY = {
    "vocab_size": 48,
    "dim": 32,
    "n_heads": 2,
    "head_dim": 16,
    "ffn_hidden": 64,
    "n_kv_heads": 1,
}


def _tiny(cls, **overrides):
    """Build a small instance of ``cls``."""
    return cls({**TINY, **overrides})


class TestLayers:
    """Individual building blocks."""

    def test_rmsnorm_normalises(self):
        norm = RMSNorm(8, eps=1e-6)
        x = torch.randn(2, 4, 8) * 10
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)
        assert "eps=1e-06" in norm.extra_repr()

    def test_rope_is_rotation(self):
        rope = RotaryEmbedding(8)
        cos, sin = rope(6, device=torch.device("cpu"))
        assert cos.shape == (1, 1, 6, 4)
        x = torch.randn(2, 2, 6, 8)
        rotated = apply_rope(x, cos, sin)
        # A rotation preserves the norm of each (even, odd) channel pair.
        assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)

    def test_rope_rejects_odd_head_dim(self):
        with pytest.raises(ValueError, match="even"):
            RotaryEmbedding(7)

    def test_rope_cache_reuse_and_growth(self):
        rope = RotaryEmbedding(8)
        first, _ = rope(4, device=torch.device("cpu"))
        again, _ = rope(4, device=torch.device("cpu"))
        assert torch.equal(first, again)
        longer, _ = rope(16, device=torch.device("cpu"))
        assert longer.shape[2] == 16

    def test_attention_mask_causal_and_windowed(self):
        mask = build_attention_mask(4, 4, window=None, device=torch.device("cpu"))
        assert mask[0].tolist() == [True, False, False, False]
        assert mask[3].tolist() == [True] * 4
        windowed = build_attention_mask(4, 4, window=2, device=torch.device("cpu"))
        assert windowed[3].tolist() == [False, False, True, True]

    def test_attention_mask_with_offset(self):
        mask = build_attention_mask(1, 5, window=None, device=torch.device("cpu"), q_offset=4)
        assert mask[0].tolist() == [True] * 5

    def test_swiglu_shapes(self):
        ffn = SwiGLUFeedForward(8, 16)
        assert ffn(torch.randn(2, 3, 8)).shape == (2, 3, 8)
        assert "hidden=16" in ffn.extra_repr()

    def test_factorized_embedding_paths_agree(self):
        embedding = FactorizedEmbedding(20, 4, 8)
        hidden = torch.randn(2, 3, 8)
        lazy = embedding.logits(hidden, materialize=False)
        eager = embedding.logits(hidden, materialize=True)
        assert torch.allclose(lazy, eager, atol=1e-5)
        assert embedding(torch.zeros(2, 3, dtype=torch.long)).shape == (2, 3, 8)

    def test_attention_rejects_mismatched_dims(self):
        with pytest.raises(ValueError, match="must equal dim"):
            CausalLocalAttention(32, 3, 16)
        with pytest.raises(ValueError, match="divisible"):
            CausalLocalAttention(32, 4, 8, n_kv_heads=3)

    def test_repeat_kv(self):
        x = torch.randn(1, 2, 3, 4)
        assert repeat_kv(x, 1) is x
        assert repeat_kv(x, 3).shape == (1, 6, 3, 4)

    def test_value_residual_mixes(self):
        attention = CausalLocalAttention(32, 2, 16, value_residual=True)
        rope = RotaryEmbedding(16)
        cos, sin = rope(4, device=torch.device("cpu"))
        x = torch.randn(1, 4, 32)
        out, v = attention(x, cos, sin)
        out2, mixed = attention(x, cos, sin, v_prev=torch.randn_like(v))
        assert v.shape == (1, 2, 4, 16)
        assert not torch.allclose(out, out2)
        # sigmoid(0) = 0.5, so the mix is the midpoint of the two value tensors.
        assert not torch.allclose(mixed, v)

    def test_kv_cache_slots_and_reset(self):
        cache = KVCache(max_length=2)
        cache.begin_forward()
        k = torch.randn(1, 1, 3, 4)
        out_k, _ = cache.update(k, k)
        assert out_k.shape[2] == 2  # trimmed to max_length
        assert cache.n_slots == 1
        cache.reset()
        assert cache.n_slots == 0

    def test_moe_routes_and_balances(self):
        moe = MoEFeedForward(8, 16, n_routed=4, n_shared=1, top_k=2)
        moe.train()
        out = moe(torch.randn(4, 6, 8))
        assert out.shape == (4, 6, 8)
        stats = moe.load_balance_stats()
        assert stats["max_over_mean"] >= 1.0
        assert "n_routed=4" in moe.extra_repr()

    def test_moe_rejects_bad_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            MoEFeedForward(8, 16, n_routed=2, top_k=5)

    def test_moe_stats_without_traffic(self):
        moe = MoEFeedForward(8, 16, n_routed=2, top_k=1)
        assert moe.load_balance_stats()["max_over_mean"] == 1.0

    def test_gated_recurrent_unit_streams(self):
        unit = GatedRecurrentUnit(8)
        x = torch.randn(1, 6, 8)
        full, _ = unit(x)
        first, state = unit(x[:, :3])
        second, _ = unit(x[:, 3:], state)
        assert torch.allclose(full, torch.cat([first, second], dim=1), atol=1e-4)
        assert "inner=8" in unit.extra_repr()


class TestModels:
    """Behaviour shared by every language model."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: _tiny(DenseTransformer, n_layers=2),
            lambda: _tiny(MoETransformer, n_layers=2, n_routed_experts=4, top_k=2),
            lambda: _tiny(HybridRecurrentTransformer, n_layers=3),
            lambda: _tiny(
                LoopedTransformer, embedding_rank=16, max_loops_table=4, train_loops=2, min_loops=1
            ),
        ],
    )
    def test_forward_shape_and_cache_equivalence(self, factory):
        model = factory().eval()
        tokens = torch.randint(0, TINY["vocab_size"], (2, 10))
        full = model(tokens, loops=2)
        assert full.shape == (2, 10, TINY["vocab_size"])

        cache = model.new_cache()
        incremental = torch.cat(
            [model(tokens[:, i : i + 1], loops=2, cache=cache) for i in range(10)], dim=1
        )
        assert torch.allclose(full, incremental, atol=1e-4)

    def test_rejects_wrong_input_rank(self, tiny_model):
        with pytest.raises(ValueError, match=r"\[B, T\]"):
            tiny_model(torch.zeros(4, dtype=torch.long))

    def test_forward_with_loss_and_logprobs(self, tiny_model):
        tokens = torch.randint(0, tiny_model.vocab_size, (2, 8))
        output = tiny_model.forward_with_loss(tokens, tokens)
        assert isinstance(output, ModelOutput)
        assert output.loss is not None and output.loss.item() > 0
        logits, _loss = output
        assert logits.shape[-1] == tiny_model.vocab_size

        labels = tokens.clone()
        labels[:, :4] = -100
        per_token = tiny_model.token_log_probs(tokens, labels)
        assert torch.all(per_token[:, :4] == 0)
        total = tiny_model.sequence_log_prob(tokens, labels)
        assert torch.allclose(total, per_token.sum(-1), atol=1e-5)
        averaged = tiny_model.sequence_log_prob(tokens, labels, average=True)
        assert torch.all(averaged >= total)

    def test_save_and_load_roundtrip(self, tiny_model, tmp_path):
        tiny_model.save_pretrained(tmp_path / "model", extra={"note": "hi"})
        config = json.loads((tmp_path / "model" / "config.json").read_text())
        assert config["architecture"] == "dense-transformer"
        assert config["note"] == "hi"

        restored = load_model(tmp_path / "model")
        tokens = torch.randint(0, tiny_model.vocab_size, (1, 6))
        tiny_model.eval()
        restored.eval()
        assert torch.allclose(tiny_model(tokens), restored(tokens), atol=1e-6)

    def test_parameter_breakdown_and_describe(self, tiny_model):
        breakdown = tiny_model.parameter_breakdown()
        assert breakdown["total"] == tiny_model.num_parameters()
        info = describe_model(tiny_model)
        assert info["architecture"] == "dense-transformer"
        assert info["parameters"] > 0
        assert tiny_model.device.type == "cpu"
        assert tiny_model.dtype is torch.float32

    def test_return_hidden(self, tiny_model):
        hidden = tiny_model(torch.zeros(1, 4, dtype=torch.long), return_hidden=True)
        assert hidden.shape == (1, 4, tiny_model.dim)


class TestLoopedTransformer:
    """The looped architecture's specific behaviours."""

    def test_matches_declared_parameter_budget(self):
        model = build_model("supra2_1406240")
        assert model.num_parameters() == 1_406_240
        assert model.expected_parameter_count() == 1_406_240

    def test_loop_count_resolution(self):
        model = _tiny(
            LoopedTransformer, embedding_rank=16, max_loops_table=8, train_loops=6, min_loops=2
        )
        model.eval()
        assert model.resolve_loops(None) == 6
        assert model.resolve_loops(3) == 3
        model.train()
        assert 2 <= model.resolve_loops(None) <= 6
        with pytest.raises(ValueError, match="loops must be"):
            model.resolve_loops(0)

    def test_more_loops_change_output(self):
        model = _tiny(
            LoopedTransformer, embedding_rank=16, max_loops_table=8, train_loops=4, min_loops=1
        ).eval()
        tokens = torch.randint(0, TINY["vocab_size"], (1, 6))
        assert not torch.allclose(model(tokens, loops=2), model(tokens, loops=4))

    def test_loops_beyond_table_are_clamped(self):
        model = _tiny(
            LoopedTransformer, embedding_rank=16, max_loops_table=2, train_loops=2, min_loops=1
        ).eval()
        assert model(torch.zeros(1, 4, dtype=torch.long), loops=6).shape[1] == 4

    def test_invalid_configs_rejected(self):
        with pytest.raises(ValueError, match="must equal dim"):
            LoopedTransformer({**TINY, "n_heads": 3})
        with pytest.raises(ValueError, match="max_loops_table"):
            LoopedTransformer({**TINY, "train_loops": 99, "max_loops_table": 4})
        with pytest.raises(ValueError, match="min_loops"):
            LoopedTransformer({**TINY, "train_loops": 2, "min_loops": 5, "max_loops_table": 8})
        with pytest.raises(ValueError, match="n_shared_blocks"):
            LoopedTransformer({**TINY, "n_shared_blocks": 0})

    def test_configurable_prelude_coda_and_tied_embedding(self):
        # Glint-2 shape: tied embed, prelude 0, coda 1, no stabilisers.
        model = LoopedTransformer(
            {**TINY, "embedding_type": "tied", "prelude_layers": 0, "coda_layers": 1,
             "n_shared_blocks": 1, "train_loops": 2, "min_loops": 1, "max_loops_table": 4,
             "use_timestep_scale": False, "use_outer_residual": False}
        )
        assert len(model.prelude) == 0 and len(model.coda) == 1
        assert model.timestep_scale is None and model.outer_gate is None
        assert model.num_parameters() == model.expected_parameter_count()
        assert model(torch.zeros(1, 5, dtype=torch.long), loops=2).shape == (1, 5, TINY["vocab_size"])

    def test_pure_loop_no_unique_layers(self):
        model = LoopedTransformer(
            {**TINY, "embedding_type": "tied", "prelude_layers": 0, "coda_layers": 0,
             "n_shared_blocks": 1, "train_loops": 2, "min_loops": 1, "max_loops_table": 4}
        )
        assert len(model.prelude) == 0 and len(model.coda) == 0
        assert model.num_parameters() == model.expected_parameter_count()

    def test_poisson_loop_sampling_in_range(self):
        model = _tiny(LoopedTransformer, embedding_rank=16, max_loops_table=8, train_loops=6,
                      min_loops=2, variable_loops=True, loop_sampling="poisson")
        model.train()
        counts = {model.resolve_loops(None) for _ in range(200)}
        assert all(2 <= c <= 8 for c in counts)
        assert len(counts) > 1

    def test_truncated_backprop_runs(self):
        # Single shared block so the last (non-detached) loop uses it.
        model = _tiny(LoopedTransformer, embedding_rank=16, max_loops_table=8, train_loops=4,
                      min_loops=4, variable_loops=False, n_shared_blocks=1, backprop_loops=1)
        model.train()
        tokens = torch.randint(0, TINY["vocab_size"], (1, 6))
        model.forward_with_loss(tokens, tokens).loss.backward()
        grad = model.shared[0].ffn.down.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        # Full backprop through the same model also runs and gives finite grads.
        full = _tiny(LoopedTransformer, embedding_rank=16, max_loops_table=8, train_loops=4,
                     min_loops=4, variable_loops=False, n_shared_blocks=1, backprop_loops=None)
        full.train()
        full.forward_with_loss(tokens, tokens).loss.backward()
        assert torch.isfinite(full.shared[0].ffn.down.weight.grad).all()

    def test_lora_up_starts_at_zero(self):
        model = _tiny(
            LoopedTransformer, embedding_rank=16, max_loops_table=4, train_loops=2, min_loops=1
        )
        assert torch.all(model.loop_lora_up == 0)
        assert torch.all(model.timestep_scale == 1)


class TestOtherArchitectures:
    """Architecture-specific checks."""

    def test_dense_window_pattern(self):
        model = DenseTransformer({**TINY, "n_layers": 4, "window": 8, "window_pattern": 2})
        windows = [block.attention.window for block in model.blocks]
        assert windows == [8, None, 8, None]

    def test_dense_untied_head_and_counts(self):
        tied = DenseTransformer({**TINY, "n_layers": 2, "tie_embeddings": True})
        untied = DenseTransformer({**TINY, "n_layers": 2, "tie_embeddings": False})
        assert untied.num_parameters() > tied.num_parameters()
        assert tied.num_parameters() == tied.expected_parameter_count()
        assert untied.num_parameters() == untied.expected_parameter_count()

    def test_dense_rejects_zero_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            DenseTransformer({**TINY, "n_layers": 0})

    def test_moe_active_below_total(self):
        model = _tiny(MoETransformer, n_layers=2, n_routed_experts=4, top_k=1, first_moe_layer=1)
        assert model.active_parameters() < model.num_parameters()
        model.train()
        model(torch.randint(0, TINY["vocab_size"], (2, 6)))
        stats = model.routing_stats()
        assert "max_over_mean" in stats

    def test_hybrid_layer_pattern(self):
        model = _tiny(
            HybridRecurrentTransformer, n_layers=4, layer_pattern=["recurrent", "attention"]
        )
        assert model.layer_types == ["recurrent", "attention", "recurrent", "attention"]
        assert len(model.new_states()) == 4
        with pytest.raises(ValueError, match="unknown layer types"):
            HybridRecurrentTransformer({**TINY, "n_layers": 2, "layer_pattern": ["nope"]})
        with pytest.raises(ValueError, match="must not be empty"):
            HybridRecurrentTransformer({**TINY, "n_layers": 2, "layer_pattern": []})

    def test_hybrid_states_thread_across_chunks(self):
        model = _tiny(HybridRecurrentTransformer, n_layers=3).eval()
        tokens = torch.randint(0, TINY["vocab_size"], (1, 8))
        full = model(tokens)
        states = model.new_states()
        first = model(tokens[:, :4], states=states)
        second = model(tokens[:, 4:], states=states)
        assert first.shape[1] == 4 and second.shape[1] == 4
        assert full.shape[1] == 8


class TestBuilder:
    """Template loading and model construction."""

    def test_every_bundled_template_builds(self):
        for name in list_templates():
            model = build_model(name, verify_budget=True)
            assert model.num_parameters() > 0

    def test_declared_params_match_built(self):
        for name in list_templates():
            template = load_template(name)
            declared = template.get("params")
            if declared:
                assert build_model(name).num_parameters() == int(declared)

    def test_overrides_applied(self, tokenizer):
        model = build_model("dense_3m", overrides={"vocab_size": 123}, verify_budget=False)
        assert model.vocab_size == 123

    def test_annotated_template_extraction(self):
        template = load_template("supra2_1406240").to_dict()
        family, flat = template_to_model_config(template)
        assert family == "looped-transformer"
        assert flat["dim"] == 128
        assert flat["window"] == 256
        assert flat["embedding_rank"] == 64
        assert flat["train_loops"] == 8
        assert flat["outer_gate_init"] == 0.1
        assert flat["init_std"] == 0.02

    def test_missing_family_rejected(self):
        with pytest.raises(ConfigError, match="family"):
            template_to_model_config({"arch": {}})

    def test_bad_arch_section_rejected(self):
        with pytest.raises(ConfigError, match="mapping"):
            template_to_model_config({"family": "dense-transformer", "arch": [1, 2]})

    def test_unknown_template_lists_options(self):
        with pytest.raises(ConfigError, match="not found"):
            load_template("no_such_template")

    def test_budget_mismatch_warns(self, caplog):
        build_model(
            {
                "family": "dense-transformer",
                "params": 1,
                "arch": {
                    "dim": 32,
                    "n_heads": 2,
                    "head_dim": 16,
                    "n_layers": 1,
                    "ffn_hidden": 32,
                    "vocab_size": 16,
                },
            },
        )
        assert "declares" in caplog.text

    def test_build_from_saved_config(self, tiny_model, tmp_path):
        tiny_model.save_pretrained(tmp_path / "m")
        config = json.loads((tmp_path / "m" / "config.json").read_text())
        rebuilt = build_model(config, verify_budget=False)
        assert rebuilt.num_parameters() == tiny_model.num_parameters()

    def test_load_model_missing_config(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"config\.json"):
            load_model(tmp_path)

    def test_device_and_dtype_placement(self):
        model = build_model("dense_3m", device="cpu", dtype="fp32", verify_budget=False)
        assert next(model.parameters()).dtype is torch.float32

    def test_registry_registration(self):
        register_architecture("test-arch", DenseTransformer)
        assert "test-arch" in ARCHITECTURES
        assert "dense_transformer" in list_architectures()
