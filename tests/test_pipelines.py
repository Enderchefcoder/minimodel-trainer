"""Integration tests: quickstarts, vision recipes and the image registry."""

from __future__ import annotations

import pytest
import yaml

from minimodel.vision.registry import (
    get_image_dataset,
    list_image_datasets,
    load_image_registry,
)


class TestImageRegistry:
    """The image dataset catalogue."""

    def test_registry_loads(self):
        registry = load_image_registry()
        assert len(registry["datasets"]) >= 12

    def test_expected_entries(self):
        spec = get_image_dataset("pixelgpt-24x24")
        assert spec.repo == "unstonio/pixelgpt-24x24-20k"
        assert spec.kind == "pixelart"
        assert spec.image_size == 24
        assert "unstonio" in spec.display
        assert spec.to_dict()["kind"] == "pixelart"

    def test_kinds_present(self):
        kinds = {spec.kind for spec in list_image_datasets()}
        assert {"pixelart", "generation", "edit"} <= kinds
        assert all(s.kind == "edit" for s in list_image_datasets(kind="edit"))

    def test_unknown_raises(self):
        from minimodel.core.config import ConfigError

        with pytest.raises(ConfigError, match="unknown image dataset"):
            get_image_dataset("no-such-set")


@pytest.mark.slow
class TestQuickstarts:
    """The end-to-end quickstart pipelines."""

    def test_text_quickstart(self, tmp_path):
        from minimodel.quickstart import run_quickstart

        summary = run_quickstart(
            tmp_path / "qs", steps=12, vocab_size=300, seq_len=32, batch_size=4
        )
        assert summary["pretrain"]["steps"] == 12
        assert "sample" in summary
        assert (tmp_path / "qs" / "model" / "model.pt").exists()
        assert (tmp_path / "qs" / "model" / "README.md").exists()
        assert (tmp_path / "qs" / "benchmark.json").exists()

    def test_vision_quickstart(self, tmp_path):
        from minimodel.vision.quickstart import run_vision_quickstart

        summary = run_vision_quickstart(
            tmp_path / "vq", steps=6, n_sprites=24, size=12, palette_size=8, batch_size=4
        )
        assert summary["pixelgpt"]["steps"] == 6
        assert summary["dit"]["steps"] == 6
        assert (tmp_path / "vq" / "samples_pixelgpt.png").exists()
        assert (tmp_path / "vq" / "model_pixelgpt" / "palette.json").exists()


@pytest.mark.slow
class TestVisionRecipe:
    """The vision YAML recipe runner."""

    def _write(self, tmp_path, payload) -> str:
        path = tmp_path / "recipe.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return str(path)

    def test_pixelgpt_recipe(self, sprites, tmp_path):
        from minimodel.vision.data.datasets import prepare_pixel_corpus
        from minimodel.vision.recipe import run_vision_recipe

        prepare_pixel_corpus(sprites, tmp_path / "corpus", size=16, palette_size=8)
        recipe = {
            "kind": "pixelgpt",
            "run_name": "recipe-pixel",
            "model": {
                "template": "pixelgpt_16x16_3m",
                "overrides": {
                    "dim": 32,
                    "n_layers": 1,
                    "n_heads": 2,
                    "head_dim": 16,
                    "n_kv_heads": 1,
                    "ffn_hidden": 64,
                },
            },
            "data": {"train": str(tmp_path / "corpus")},
            "training": {
                "output_dir": str(tmp_path / "runs"),
                "max_steps": 2,
                "batch_size": 4,
                "seq_len": 256,
                "eval_every": 0,
                "save_every": 0,
                "resume": False,
            },
        }
        summary = run_vision_recipe(self._write(tmp_path, recipe))
        assert summary["steps"] == 2

    def test_diffusion_recipe(self, sprites, tmp_path):
        from minimodel.vision.data.datasets import prepare_image_corpus
        from minimodel.vision.recipe import run_vision_recipe

        prepare_image_corpus(sprites, tmp_path / "corpus", size=16)
        recipe = {
            "kind": "diffusion",
            "run_name": "recipe-dit",
            "model": {
                "template": "dit_tiny_32",
                "overrides": {
                    "image_size": 16,
                    "patch_size": 4,
                    "dim": 32,
                    "depth": 1,
                    "n_heads": 2,
                    "condition": "none",
                    "num_classes": 0,
                },
            },
            "data": {"train": str(tmp_path / "corpus")},
            "training": {
                "output_dir": str(tmp_path / "runs"),
                "max_steps": 2,
                "batch_size": 4,
                "save_every": 0,
                "resume": False,
            },
        }
        summary = run_vision_recipe(self._write(tmp_path, recipe))
        assert summary["steps"] == 2

    def test_recipe_validation(self, tmp_path):
        from minimodel.core.config import ConfigError
        from minimodel.vision.recipe import run_vision_recipe

        with pytest.raises(ConfigError, match="unknown vision recipe kind"):
            run_vision_recipe({"kind": "hologram"})
        with pytest.raises(ConfigError, match=r"data\.train"):
            run_vision_recipe({"kind": "diffusion", "data": {}})


class TestBundledConfigs:
    """Every YAML recipe in configs/ must at least parse and resolve extends."""

    @pytest.mark.parametrize(
        "recipe",
        sorted(
            str(p.relative_to(p.parents[2]))
            for p in __import__("pathlib")
            .Path(__file__)
            .parent.parent.joinpath("configs")
            .rglob("*.yaml")
        ),
    )
    def test_recipe_parses(self, recipe):
        from pathlib import Path

        from minimodel.core.config import load_config

        cfg = load_config(Path(__file__).parent.parent / recipe.replace("configs/", "configs/"))
        assert isinstance(cfg.to_dict(), dict)
