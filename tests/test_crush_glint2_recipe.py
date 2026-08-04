"""Tests for the crush-glint2 1.4M recipe (template, mix, soft labels)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from minimodel.architectures import build_model, list_templates
from minimodel.datasets.registry import get_dataset, get_mixture, resolve_mixture
from minimodel.datasets.soft_labels import (
    DEFAULT_CORPUS_PATH,
    align_steps_to_tokenizer,
    entries_to_plain_texts,
    load_soft_label_dataset,
    soft_kl_loss,
    write_plain_jsonl,
)
from minimodel.tokenization import BPETokenizer

CORPUS_JSONL = DEFAULT_CORPUS_PATH.with_name("slm_next_token_qa.jsonl")


@pytest.mark.skipif(not DEFAULT_CORPUS_PATH.exists(), reason="soft-label corpus not checked in")
class TestSoftLabels:
    """Bundled soft-label JSON → CE docs + KL targets."""

    def test_load_and_plain_texts(self):
        payload = load_soft_label_dataset()
        assert payload["stats"]["entries"] == 510
        docs = entries_to_plain_texts(payload)
        assert len(docs) == 510
        assert docs[0]["text"].startswith(payload["entries"][0]["prompt"])
        assert "two" in docs[0]["text"].lower() or "2" in docs[0]["text"]

    def test_jsonl_matches_entries(self, tmp_path: Path):
        out = tmp_path / "qa.jsonl"
        n = write_plain_jsonl(out)
        assert n == 510
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 510
        assert "text" in json.loads(lines[0])

    def test_align_and_soft_kl(self, tokenizer: BPETokenizer):
        payload = load_soft_label_dataset()
        entry = payload["entries"][0]
        steps = align_steps_to_tokenizer(entry, lambda s: tokenizer.encode(s, add_bos=False))
        assert steps
        step = steps[0]
        logits = torch.randn(tokenizer.vocab_size)
        loss = soft_kl_loss(logits, step.token_ids, step.probs)
        assert torch.isfinite(loss)
        assert loss.ndim == 0


class TestCrushRecipe:
    """Architecture template + registry mixture stay coherent."""

    def test_dense_1_4m_template(self):
        assert "dense_1_4m" in list_templates()
        model = build_model("dense_1_4m")
        assert model.num_parameters() == 1_406_506
        # Smoke forward
        x = torch.randint(0, 4096, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 4096)

    def test_crush_glint2_mixture(self):
        mix = get_mixture("crush-glint2")
        weights = dict(mix.normalized_weights())
        assert pytest.approx(sum(weights.values())) == 1.0
        assert pytest.approx(weights["fineweb-edu-10bt"], abs=0.01) == 0.50
        assert pytest.approx(weights["dclm-100bt"], abs=0.01) == 0.32
        assert pytest.approx(weights["tinystories"], abs=0.01) == 0.15
        assert pytest.approx(weights["slm-next-token-qa"], abs=0.01) == 0.03
        resolved = resolve_mixture("crush-glint2")
        assert len(resolved) == 4
        assert get_dataset("dclm-100bt").repo == "HuggingFaceFW/dclm_100BT"
        assert get_dataset("slm-next-token-qa").source == "local"

    def test_recipe_config_exists(self):
        path = Path("configs/pretrain/crush_glint2_1.4m.yaml")
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "dense_1_4m" in text
        assert "3.0e-3" in text or "0.003" in text
        assert "wsd" in text

    def test_colab_notebook_is_single_run_cell(self):
        path = Path("notebooks/03_crush_glint2_colab.ipynb")
        assert path.exists()
        nb = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) == 1
        src = "".join(code_cells[0]["source"])
        assert "dense_1_4m" in src
        assert "DRIVE_ROOT" in src
        assert "HOURS = 4" in src
        assert "slm_next_token_dataset" in src
        assert "soft_kl_loss" in src
