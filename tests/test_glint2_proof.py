"""Offline regression: committed Glint-2 proof artifact stays consistent."""

from __future__ import annotations

import json
from pathlib import Path

PROOF = Path(__file__).resolve().parents[1] / "research" / "data" / "results" / "glint2_proof.json"
BASELINE = Path(__file__).resolve().parents[1] / "research" / "data" / "results" / "glint2.json"


def test_glint2_proof_artifact_exists_and_convicts():
    """Committed proof must show 1.71M loop+coda and a broken generate.py load."""
    assert PROOF.is_file(), f"missing {PROOF}"
    data = json.loads(PROOF.read_text(encoding="utf-8"))

    assert data["actual_params"] == 1_710_049
    assert data["advertised_params"] == 1_065_000
    assert data["params_coda_only"] == 645_312
    assert data["model_config"]["coda_layers"] == 1
    assert data["model_config"]["prelude_layers"] == 0
    assert data["by_prefix"]["shared"] == data["by_prefix"]["coda"] == 645_312

    v = data["verdicts"]
    assert v["param_count_misreported"] is True
    assert v["not_pure_loop"] is True
    assert v["generate_py_cannot_strict_load"] is True
    assert v["wikitext_ppl_is_byte_normalised"] is True
    assert v["arc_reproducible"] is True

    gen = data["generate_py"]
    assert gen["strict_load_ok"] is False
    assert gen["constructed_params"] == data["params_excluding_coda"] == 1_064_737
    assert any(k.startswith("coda.") for k in gen["non_strict_unexpected_keys"])


def test_glint2_baseline_eval_agrees_with_proof_params():
    """Harness baseline and proof audit must describe the same artifact size."""
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert baseline["params"] == proof["actual_params"] == 1_710_049
    # WikiText: advertised 3.09 tracks byte-ppl, not token-ppl.
    assert abs(baseline["wikitext_byte_ppl"] - 3.09) < 0.15
    assert baseline["wikitext_ppl"] > 20
    assert abs(baseline["arc_easy_acc"] - 36.80) < 0.1
