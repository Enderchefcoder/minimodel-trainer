"""Integration tests for post_train stage dispatch and remaining CLI paths."""

from __future__ import annotations

import json

import pytest

from minimodel.cli import main
from minimodel.training.post_train import build_trainer_config, load_pretrained, post_train
from minimodel.training.trainer import TrainerConfig


@pytest.fixture
def base_model_dir(tiny_model, tmp_path):
    """A saved model to start post-training from."""
    directory = tmp_path / "base"
    tiny_model.save_pretrained(directory)
    return directory


class TestPostTrainDispatch:
    """Every stage reachable from a recipe."""

    def _recipe(self, stage, tokenizer, base_model_dir, tmp_path, data, **training):
        defaults = dict(
            output_dir=str(tmp_path / "runs"),
            max_steps=2,
            batch_size=2,
            seq_len=16,
            eval_every=0,
            save_every=0,
            log_every=2,
            warmup=0,
            resume=False,
        )
        defaults.update(training)
        return {
            "stage": stage,
            "tokenizer": str(tokenizer.save(tmp_path / "tok")),
            "model": {"checkpoint": str(base_model_dir)},
            "data": data,
            "training": defaults,
        }

    def test_cot_stage(self, tokenizer, base_model_dir, cot_dir, tmp_path):
        recipe = self._recipe(
            "cot", tokenizer, base_model_dir, tmp_path, {"train": str(cot_dir)},
            reasoning_loss_weight=0.5,
        )
        result = post_train(recipe)
        assert result.steps == 2

    def test_continue_stage(self, tokenizer, base_model_dir, corpus_dir, tmp_path):
        recipe = self._recipe(
            "continue", tokenizer, base_model_dir, tmp_path, {"train": str(corpus_dir)}
        )
        assert post_train(recipe).steps == 2

    def test_dpo_stage(self, tokenizer, base_model_dir, pairs_path, tmp_path):
        recipe = self._recipe(
            "dpo", tokenizer, base_model_dir, tmp_path, {"pairs": str(pairs_path)}, lr=1e-6
        )
        assert post_train(recipe).steps == 2

    def test_spin_stage(self, tokenizer, base_model_dir, sft_jsonl, tmp_path):
        recipe = self._recipe(
            "spin", tokenizer, base_model_dir, tmp_path, {"pairs": str(sft_jsonl)},
            iterations=1, max_new_tokens=6, lr=1e-6,
        )
        assert post_train(recipe).steps == 2

    def test_rlvr_stage(self, tokenizer, base_model_dir, tasks_path, tmp_path):
        recipe = self._recipe(
            "rlvr", tokenizer, base_model_dir, tmp_path, {"tasks": str(tasks_path)},
            group_size=2, max_new_tokens=6, lr=1e-6,
        )
        summary = post_train(recipe)
        assert summary["steps"] == 2

    def test_missing_pairs_and_tasks_rejected(self, tokenizer, base_model_dir, tmp_path):
        from minimodel.core.config import ConfigError

        with pytest.raises(ConfigError, match="data.pairs"):
            post_train(self._recipe("dpo", tokenizer, base_model_dir, tmp_path, {}))
        with pytest.raises(ConfigError, match="data.tasks"):
            post_train(self._recipe("rlvr", tokenizer, base_model_dir, tmp_path, {}))

    def test_rlvr_requires_tokenizer(self, base_model_dir, tasks_path, tmp_path):
        from minimodel.core.config import ConfigError

        recipe = {
            "stage": "rlvr",
            "model": {"checkpoint": str(base_model_dir)},
            "data": {"tasks": str(tasks_path)},
            "training": {"output_dir": str(tmp_path), "resume": False},
        }
        with pytest.raises(ConfigError, match="tokenizer"):
            post_train(recipe)

    def test_sft_requires_train_data(self, tokenizer, base_model_dir, tmp_path):
        from minimodel.core.config import ConfigError

        with pytest.raises(ConfigError, match="data.train"):
            post_train(self._recipe("sft", tokenizer, base_model_dir, tmp_path, {}))

    def test_replay_dataset_wiring(self, tokenizer, base_model_dir, sft_dir, corpus_dir, tmp_path):
        recipe = self._recipe(
            "sft", tokenizer, base_model_dir, tmp_path,
            {"train": str(sft_dir), "eval": str(sft_dir), "replay": str(corpus_dir)},
            replay_fraction=0.2,
        )
        assert post_train(recipe).steps == 2

    def test_build_trainer_config_warns_on_unknown(self, caplog):
        config = build_trainer_config(TrainerConfig, {"max_steps": 5, "not_a_field": 1})
        assert config.max_steps == 5
        assert "not_a_field" in caplog.text

    def test_load_pretrained_from_template(self):
        from conftest import TINY_MODEL

        model = load_pretrained(
            {"template": "dense_3m", "overrides": dict(TINY_MODEL)}, vocab_size=99
        )
        assert model.vocab_size == 99

    def test_load_pretrained_validation(self, tmp_path):
        from minimodel.core.config import ConfigError

        with pytest.raises(ConfigError, match="checkpoint.*template|template.*checkpoint"):
            load_pretrained({})
        with pytest.raises(ConfigError, match="not found"):
            load_pretrained({"checkpoint": tmp_path / "missing"})


class TestPosttrainCLI:
    """`minimodel posttrain` runs a recipe file."""

    def test_posttrain_command(self, tokenizer, base_model_dir, sft_dir, tmp_path):
        import yaml

        recipe = {
            "stage": "sft",
            "tokenizer": str(tokenizer.save(tmp_path / "tok")),
            "model": {"checkpoint": str(base_model_dir)},
            "data": {"train": str(sft_dir)},
            "training": {
                "output_dir": str(tmp_path / "runs"),
                "max_steps": 1,
                "batch_size": 2,
                "seq_len": 16,
                "eval_every": 0,
                "save_every": 0,
                "resume": False,
            },
        }
        path = tmp_path / "recipe.yaml"
        path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
        assert main(["posttrain", "-c", str(path), "--json"]) == 0


class TestVisionCLI:
    """The vision command group's data and sampling paths."""

    def test_prepare_synthetic_and_info(self, tmp_path, capsys):
        assert (
            main(
                [
                    "vision", "data", "prepare", "--synthetic", "--limit", "12",
                    "--size", "12", "--mode", "palette", "--palette-size", "8",
                    "-o", str(tmp_path / "corpus"), "--json",
                ]
            )
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["n_images"] == 12
        assert main(["vision", "data", "info", str(tmp_path / "corpus")]) == 0
        assert json.loads(capsys.readouterr().out)["mode"] == "palette"

    def test_prepare_rgb_from_directory(self, sprites, tmp_path, capsys):
        from PIL import Image

        source = tmp_path / "imgs" / "classA"
        source.mkdir(parents=True)
        for index, (image, _) in enumerate(sprites[:4]):
            Image.fromarray(image).save(source / f"{index}.png")
        assert (
            main(
                [
                    "vision", "data", "prepare", "--input", str(tmp_path / "imgs"),
                    "--size", "16", "-o", str(tmp_path / "rgb"), "--json",
                ]
            )
            == 0
        )
        assert json.loads(capsys.readouterr().out)["n_images"] == 4

    def test_prepare_requires_a_source(self, tmp_path):
        assert main(["vision", "data", "prepare", "-o", str(tmp_path / "x")]) == 1

    def test_sample_from_pixel_model(self, tmp_path, capsys):
        from minimodel.vision.architectures.pixelgpt import PixelGPT
        from minimodel.vision.data.palette import Palette
        import numpy as np

        model = PixelGPT(
            {
                "image_size": 6, "palette_size": 8, "dim": 32, "n_layers": 1,
                "n_heads": 2, "head_dim": 16, "n_kv_heads": 1, "ffn_hidden": 64,
            }
        )
        model_dir = tmp_path / "model"
        model.save_pretrained(model_dir)
        Palette(np.zeros((8, 3), dtype=np.uint8)).save(model_dir / "palette.json")
        output = tmp_path / "out.png"
        assert (
            main(
                [
                    "vision", "sample", "-m", str(model_dir), "-n", "2",
                    "-o", str(output), "--device", "cpu", "--seed", "0",
                ]
            )
            == 0
        )
        assert output.exists()

    def test_sample_from_diffusion_model(self, tmp_path):
        from minimodel.vision.architectures.dit import DiT

        model = DiT({"image_size": 8, "patch_size": 2, "dim": 32, "depth": 1, "n_heads": 2})
        model_dir = tmp_path / "dit"
        model.save_pretrained(model_dir)
        output = tmp_path / "dit.png"
        assert (
            main(
                [
                    "vision", "sample", "-m", str(model_dir), "-n", "2",
                    "--steps", "2", "-o", str(output), "--device", "cpu",
                ]
            )
            == 0
        )
        assert output.exists()

    def test_vision_group_help(self):
        assert main(["vision"]) == 1
