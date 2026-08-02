"""Tests for the effort ladder and the quality probe."""

from __future__ import annotations

import pytest
import torch

from minimodel.inference.quality_probe import QualityProbe, train_quality_probe
from minimodel.inference.search import (
    EFFORT_LEVELS,
    EffortConfig,
    effort_generate,
    score_continuation,
)


class TestEffortLadder:
    """Search-based generation."""

    def test_levels_present(self):
        assert set(EFFORT_LEVELS) == {"low", "medium", "high", "xhigh", "max", "ultra"}
        assert EFFORT_LEVELS["ultra"].runs == 10
        assert EFFORT_LEVELS["high"].instances == 6

    @pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh"])
    def test_generates_text(self, tiny_model, tokenizer, level):
        text = effort_generate(tiny_model, tokenizer, "The river", level=level,
                               max_new_tokens=12, seed=0)
        assert isinstance(text, str)
        assert text.startswith("The river")

    def test_unknown_level_rejected(self, tiny_model, tokenizer):
        with pytest.raises(ValueError, match="unknown effort level"):
            effort_generate(tiny_model, tokenizer, "hi", level="turbo")

    def test_return_score(self, tiny_model, tokenizer):
        text, score = effort_generate(tiny_model, tokenizer, "The", level="high",
                                      max_new_tokens=8, seed=1, return_score=True)
        assert isinstance(text, str)
        assert isinstance(score, float)

    def test_score_continuation_penalises_repetition(self, tiny_model, tokenizer):
        cfg = EffortConfig()
        prompt = tokenizer.encode("The river", add_bos=False)
        varied = prompt + tokenizer.encode(" runs east through the valley", add_bos=False)
        repeated = prompt + tokenizer.encode(" the the the the the the", add_bos=False)
        s_varied = score_continuation(tiny_model, varied, len(prompt), cfg=cfg, target_len=8)
        s_repeated = score_continuation(tiny_model, repeated, len(prompt), cfg=cfg, target_len=8)
        # The 4-gram repetition penalty should push the repeated text down.
        assert s_repeated < s_varied

    def test_empty_continuation_scores_negative_inf(self, tiny_model, tokenizer):
        ids = tokenizer.encode("hello", add_bos=False)
        assert score_continuation(tiny_model, ids, len(ids), cfg=EffortConfig()) == float("-inf")


class TestQualityProbe:
    """The learned real-vs-generated reranking probe."""

    def test_train_and_predict(self, tiny_model, tokenizer, texts):
        probe = train_quality_probe(tiny_model, tokenizer, texts[:24], n_prompts=24,
                                    max_new_tokens=16, epochs=120)
        assert isinstance(probe, QualityProbe)
        assert probe.dim == tiny_model.dim
        ids = tokenizer.encode(texts[0], add_bos=False)
        p = probe.p_real(tiny_model, ids, prompt_len=2)
        assert 0.0 <= p <= 1.0

    def test_save_load_roundtrip(self, tiny_model, tokenizer, texts, tmp_path):
        probe = train_quality_probe(tiny_model, tokenizer, texts[:16], n_prompts=16,
                                    max_new_tokens=12, epochs=60)
        path = probe.save(tmp_path / "probe.pt")
        assert path.exists()
        # A few-KB artifact, like Glint-2's 3.5 KB probe.
        assert path.stat().st_size < 20_000
        reloaded = QualityProbe.load(path)
        ids = tokenizer.encode(texts[1], add_bos=False)
        assert reloaded.p_real(tiny_model, ids, 2) == pytest.approx(
            probe.p_real(tiny_model, ids, 2), abs=1e-5
        )

    def test_rejects_too_few_texts(self, tiny_model, tokenizer):
        with pytest.raises(ValueError, match="not enough"):
            train_quality_probe(tiny_model, tokenizer, ["hi"], n_prompts=1)

    def test_probe_used_in_effort_score(self, tiny_model, tokenizer, texts):
        probe = train_quality_probe(tiny_model, tokenizer, texts[:16], n_prompts=16,
                                    max_new_tokens=12, epochs=60)
        cfg = EffortConfig(probe_weight=2.0)
        ids = tokenizer.encode("The river runs east", add_bos=False)
        with_probe = score_continuation(tiny_model, ids, 2, cfg=cfg, probe=probe)
        without = score_continuation(tiny_model, ids, 2, cfg=cfg, probe=None)
        # The probe term shifts the score (blended in with weight 2.0).
        assert with_probe != without
