"""Tests for the 20 ~1M Glint-2 architecture candidates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from minimodel.architectures import (
    ExperimentalTransformer,
    MambaLM,
    build_model,
    list_architectures,
    list_glint2_candidates,
    list_templates,
)
from minimodel.architectures.ssm import SelectiveSSM

RESULTS = Path(__file__).resolve().parents[1] / "research" / "data" / "results"
RANKING = RESULTS / "arch_1m_candidates.json"
MERGED = RESULTS / "arch_bakeoff_merged.json"


class TestMM1MCandidates:
    """The ordered ~1M candidate ladder."""

    def test_twenty_candidates_ranked_1_to_20(self):
        rows = list_glint2_candidates()
        assert len(rows) == 20
        assert [r["rank"] for r in rows] == list(range(1, 21))
        classes = {r["candidate_class"] for r in rows}
        assert classes == {"researched", "novel-transformer", "novel-mamba"}
        assert sum(1 for r in rows if r["candidate_class"] == "researched") == 10
        assert sum(1 for r in rows if r["candidate_class"] == "novel-transformer") == 5
        assert sum(1 for r in rows if r["candidate_class"] == "novel-mamba") == 5

    def test_every_candidate_builds_near_one_million(self):
        for row in list_glint2_candidates():
            model = build_model(row["name"])
            assert model.num_parameters() == row["params"]
            assert 900_000 <= row["params"] <= 1_250_000, row["name"]

    def test_template_names_sort_by_rank(self):
        names = [n for n in list_templates() if n.startswith("mm1m_r")]
        assert names == sorted(names)
        assert names[0].startswith("mm1m_r01_")
        assert names[-1].startswith("mm1m_r20_")


class TestNovelFamilies:
    """Experimental transformer + Mamba families."""

    @pytest.mark.parametrize(
        "variant",
        ["resimix", "kv_inherit", "braid", "echo_ffn", "dual_rope"],
    )
    def test_experimental_cache_equivalence(self, variant):
        model = ExperimentalTransformer(
            {
                "vocab_size": 48,
                "dim": 32,
                "n_layers": 2,
                "n_heads": 2,
                "head_dim": 16,
                "n_kv_heads": 1,
                "ffn_hidden": 64,
                "max_seq_len": 64,
                "window": 32,
                "variant": variant,
            }
        ).eval()
        tokens = torch.randint(0, 48, (2, 8))
        full = model(tokens)
        cache = model.new_cache()
        incremental = torch.cat(
            [model(tokens[:, i : i + 1], cache=cache) for i in range(8)], dim=1
        )
        assert torch.allclose(full, incremental, atol=1e-4)

    @pytest.mark.parametrize(
        "variant",
        ["pure", "attn_tail", "multihead", "conv_gate", "braid"],
    )
    def test_mamba_cache_equivalence(self, variant):
        model = MambaLM(
            {
                "vocab_size": 48,
                "dim": 32,
                "n_layers": 2,
                "n_heads": 2,
                "head_dim": 16,
                "n_kv_heads": 1,
                "ffn_hidden": 64,
                "max_seq_len": 64,
                "window": 32,
                "variant": variant,
                "state_dim": 4,
                "expand": 2,
                "conv_kernel": 4,
                "attn_tail_layers": 1,
            }
        ).eval()
        tokens = torch.randint(0, 48, (2, 6))
        full = model(tokens)
        cache = model.new_cache()
        incremental = torch.cat(
            [model(tokens[:, i : i + 1], cache=cache) for i in range(6)], dim=1
        )
        assert torch.allclose(full, incremental, atol=1e-4)

    def test_selective_ssm_streams(self):
        unit = SelectiveSSM(16, state_dim=4, expand=2, conv_kernel=4, n_heads=1)
        x = torch.randn(1, 6, 16)
        full, _ = unit(x)
        first, state = unit(x[:, :3])
        second, _ = unit(x[:, 3:], state)
        assert torch.allclose(full, torch.cat([first, second], dim=1), atol=1e-4)

    def test_families_registered(self):
        names = set(list_architectures())
        assert "experimental_transformer" in names
        assert "mamba_lm" in names

    def test_unknown_variants_rejected(self):
        with pytest.raises(ValueError, match="variant"):
            ExperimentalTransformer({"vocab_size": 16, "dim": 32, "n_heads": 2, "head_dim": 16,
                                     "ffn_hidden": 32, "n_layers": 1, "variant": "nope"})
        with pytest.raises(ValueError, match="variant"):
            MambaLM({"vocab_size": 16, "dim": 32, "n_heads": 2, "head_dim": 16,
                     "ffn_hidden": 32, "n_layers": 1, "variant": "nope"})


class TestRankingArtifact:
    """Committed bake-off ranking JSON stays coherent with templates."""

    def test_smoke_ranking_json_lists_all_twenty(self):
        assert RANKING.exists(), "run bakeoff_1m_candidates.py to create the ranking"
        data = json.loads(RANKING.read_text(encoding="utf-8"))
        prior = data["ordered_by_prior"]
        measured = data["ordered_by_measured_loss"]
        assert len(prior) == 20
        assert len(measured) == 20
        names = {r["name"] for r in list_glint2_candidates()}
        assert {r["name"] for r in prior} == names
        assert [r["measured_rank"] for r in measured] == list(range(1, 21))
        losses = [r["final_loss"] for r in measured]
        assert losses == sorted(losses)

    def test_merged_report03_bakeoff_when_present(self):
        if not MERGED.exists():
            pytest.skip("full report-03 bake-off not run yet")
        data = json.loads(MERGED.read_text(encoding="utf-8"))
        assert data["n_mm1m"] == 20
        assert data["n_original"] == 5
        assert data["n_merged"] == 25
        mm1m = data["mm1m_candidates"]
        assert len(mm1m) == 20
        assert [r["mm1m_measured_rank"] for r in mm1m] == list(range(1, 21))
        # Protocol matches report 03.
        assert data["protocol"]["max_steps"] == 300
        assert data["protocol"]["seq_len"] == 256
        assert data["protocol"]["batch_size"] == 32
        for row in mm1m:
            assert "val_loss" in row
            assert "wikitext_byte_ppl" in row
            assert "blimp_acc" in row
            assert "arc_easy_acc" in row
            assert row["train_tokens"] == 2_457_600

