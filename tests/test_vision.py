"""Tests for the image models, data pipeline, training and sampling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from minimodel.vision.architectures.dit import DiT
from minimodel.vision.architectures.edit import ImageEditModel
from minimodel.vision.architectures.layers import (
    LabelEmbedding,
    PatchEmbed,
    TextConditioner,
    TimestepEmbedding,
    sincos_position_embedding,
    unpatchify,
)
from minimodel.vision.architectures.pixelgpt import N_SPECIAL_TOKENS, PixelGPT
from minimodel.vision.architectures.registry import (
    IMAGE_ARCHITECTURES,
    build_image_model,
    describe_image_model,
    list_image_architectures,
    list_image_templates,
    load_image_model,
    load_vision_template,
)
from minimodel.vision.architectures.unet import UNet
from minimodel.vision.architectures.vae import VAE
from minimodel.vision.data.datasets import (
    ImageDataset,
    PairedImageDataset,
    PixelSequenceDataset,
    build_image_dataloader,
    prepare_image_corpus,
    prepare_pixel_corpus,
    synthetic_sprites,
)
from minimodel.vision.data.palette import (
    Palette,
    build_palette,
    exact_palette,
    kmeans_palette,
    median_cut_palette,
)
from minimodel.vision.data.shards import ImageCorpus, ImageShardWriter
from minimodel.vision.sampling.generate import (
    make_grid,
    sample_pixel_art,
    save_image_grid,
    tensor_to_uint8,
)
from minimodel.vision.sampling.samplers import (
    ddim_sample,
    euler_sample,
    heun_sample,
    sample_images,
    timestep_schedule,
)
from minimodel.vision.training.diffusion import (
    EMA,
    DiffusionConfig,
    DiffusionTrainer,
    cosine_alpha_bar,
    flow_matching_targets,
    sample_timesteps,
)
from minimodel.vision.training.pixel_trainer import (
    PixelGPTConfig,
    PixelGPTTrainer,
    VAETrainer,
    VAETrainerConfig,
)

TINY_DIT = {"image_size": 8, "patch_size": 2, "dim": 32, "depth": 2, "n_heads": 2}
TINY_PIXEL = {
    "image_size": 6,
    "palette_size": 8,
    "dim": 32,
    "n_layers": 2,
    "n_heads": 2,
    "head_dim": 16,
    "n_kv_heads": 1,
    "ffn_hidden": 64,
}


class TestVisionLayers:
    """Conditioning and patching primitives."""

    def test_timestep_embedding_shape_and_distinctness(self):
        embedding = TimestepEmbedding(32)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = embedding(t)
        assert out.shape == (3, 32)
        assert not torch.allclose(out[0], out[2])

    def test_label_embedding_dropout_and_null(self):
        embedding = LabelEmbedding(4, 16, dropout_prob=1.0)
        labels = torch.tensor([0, 1, 2])
        dropped = embedding(labels, train=True)
        null = embedding(torch.tensor([embedding.null_index] * 3), train=False)
        assert torch.allclose(dropped, null)

    def test_text_conditioner_pools_and_nulls(self):
        conditioner = TextConditioner(50, 16, n_layers=1, n_heads=2, max_len=8)
        tokens = torch.randint(1, 50, (2, 6))
        out = conditioner(tokens)
        assert out.shape == (2, 16)
        null = conditioner(None, batch_size=3)
        assert null.shape == (3, 16)

    def test_patch_roundtrip(self):
        patch = PatchEmbed(8, 2, 3, 16)
        images = torch.randn(2, 3, 8, 8)
        tokens = patch(images)
        assert tokens.shape == (2, 16, 16)
        with pytest.raises(ValueError, match="divisible"):
            PatchEmbed(9, 2, 3, 16)
        restored = unpatchify(torch.randn(2, 16, 2 * 2 * 3), 2, 3, 4)
        assert restored.shape == (2, 3, 8, 8)

    def test_sincos_embedding(self):
        table = sincos_position_embedding(32, 4)
        assert table.shape == (1, 16, 32)
        with pytest.raises(ValueError, match="divisible by 4"):
            sincos_position_embedding(30, 4)


class TestImageModels:
    """Forward shapes, conditioning and guidance."""

    def test_dit_forward_and_guidance(self):
        model = DiT({**TINY_DIT, "condition": "class", "num_classes": 4})
        x = torch.randn(2, 3, 8, 8)
        t = torch.rand(2)
        labels = torch.tensor([0, 1])
        out = model(x, t, labels=labels)
        assert out.shape == x.shape
        guided = model.forward_with_guidance(x, t, labels=labels, guidance_scale=3.0)
        assert guided.shape == x.shape
        plain = DiT(TINY_DIT)
        assert plain.forward_with_guidance(x, t, guidance_scale=3.0).shape == x.shape

    def test_dit_text_conditioning(self):
        model = DiT({**TINY_DIT, "condition": "text", "text_vocab_size": 32})
        tokens = torch.randint(1, 32, (2, 5))
        out = model(torch.randn(2, 3, 8, 8), torch.rand(2), text_tokens=tokens)
        assert out.shape == (2, 3, 8, 8)
        guided = model.forward_with_guidance(
            torch.randn(2, 3, 8, 8), torch.rand(2), text_tokens=tokens, guidance_scale=2.0
        )
        assert guided.shape == (2, 3, 8, 8)

    def test_dit_validation(self):
        with pytest.raises(ValueError, match="num_classes"):
            DiT({**TINY_DIT, "condition": "class"})
        with pytest.raises(ValueError, match="text_vocab_size"):
            DiT({**TINY_DIT, "condition": "text"})
        with pytest.raises(ValueError, match="condition"):
            DiT({**TINY_DIT, "condition": "vibes"})

    def test_unet_forward(self):
        model = UNet(
            {
                "image_size": 16,
                "base_channels": 16,
                "channel_multipliers": [1, 2],
                "blocks_per_level": 1,
                "attention_resolutions": [8],
                "condition": "class",
                "num_classes": 3,
            }
        )
        out = model(torch.randn(2, 3, 16, 16), torch.rand(2), labels=torch.tensor([0, 2]))
        assert out.shape == (2, 3, 16, 16)
        guided = model.forward_with_guidance(
            torch.randn(1, 3, 16, 16), torch.rand(1), labels=torch.tensor([1]), guidance_scale=2.0
        )
        assert guided.shape == (1, 3, 16, 16)

    def test_edit_model_dual_guidance(self):
        model = ImageEditModel({**TINY_DIT, "text_vocab_size": 32, "extra_in_channels": 3})
        source = torch.randn(2, 3, 8, 8)
        noisy = torch.randn(2, 3, 8, 8)
        tokens = torch.randint(1, 32, (2, 6))
        out = model(noisy, torch.rand(2), reference=source, text_tokens=tokens)
        assert out.shape == noisy.shape
        guided = model.forward_with_guidance(
            noisy,
            torch.rand(2),
            reference=source,
            text_tokens=tokens,
            guidance_scale=5.0,
            image_guidance=1.5,
        )
        assert guided.shape == noisy.shape

    def test_vae_roundtrip_and_loss(self):
        vae = VAE(
            {
                "image_size": 16,
                "base_channels": 16,
                "channel_multipliers": [1, 2],
                "blocks_per_level": 1,
            }
        )
        images = torch.rand(2, 3, 16, 16) * 2 - 1
        output = vae(images)
        assert output.reconstruction.shape == images.shape
        assert output.kl_divergence() >= 0
        loss, extras = vae.loss(images)
        assert torch.isfinite(loss)
        assert "kl" in extras
        latent = vae.encode_for_diffusion(images)
        assert latent.shape[1] == vae.latent_channels
        decoded = vae.decode_from_diffusion(latent)
        assert decoded.shape == images.shape
        assert vae.latent_shape() == (4, 8, 8)

    def test_pixelgpt_forward_loss_and_generation(self):
        model = PixelGPT(TINY_PIXEL)
        pixels = torch.randint(0, 8, (2, 36))
        logits = model(pixels)
        assert logits.shape == (2, 37, 8 + N_SPECIAL_TOKENS)
        loss = model.loss(pixels)
        assert torch.isfinite(loss)

        images = model.generate(2, temperature=0.8, seed=0)
        assert images.shape == (2, 6, 6)
        assert images.max() < 8
        greedy = model.generate(1, temperature=0.0, seed=1)
        assert greedy.shape == (1, 6, 6)

    def test_pixelgpt_prompt_completion_and_classes(self):
        model = PixelGPT({**TINY_PIXEL, "num_classes": 3})
        prompt = torch.randint(0, 8, (2, 12))
        completed = model.generate(2, prompt=prompt, labels=torch.tensor([0, 1]), seed=0)
        assert completed.shape == (2, 6, 6)
        assert torch.equal(completed.view(2, -1)[:, :12], prompt)
        with pytest.raises(ValueError, match=r"\[B, N\]"):
            model(torch.zeros(4, dtype=torch.long))

    def test_pixelgpt_cache_matches_full(self):
        model = PixelGPT(TINY_PIXEL).eval()
        pixels = torch.randint(0, 8, (1, 36))
        full = model(pixels)
        cache = model.new_cache() if hasattr(model, "new_cache") else None
        from minimodel.architectures.layers import KVCache

        cache = KVCache()
        incremental = [model(pixels[:, :1], cache=cache)]
        for i in range(1, 36):
            incremental.append(model(pixels[:, i : i + 1], cache=cache))
        stitched = torch.cat(incremental, dim=1)
        assert torch.allclose(full[:, -1], stitched[:, -1], atol=1e-4)


class TestVisionRegistry:
    """Template loading and persistence."""

    def test_all_templates_build_with_declared_params(self):
        for name in list_image_templates():
            template = load_vision_template(name)
            model = build_image_model(name)
            assert model.num_parameters() == int(template["params"])

    def test_registry_contents(self):
        assert {"dit", "unet", "pixelgpt", "image_edit", "vae"} <= set(list_image_architectures())
        assert "pixelgpt" in IMAGE_ARCHITECTURES

    def test_save_and_load_roundtrip(self, tmp_path):
        model = PixelGPT(TINY_PIXEL)
        model.save_pretrained(tmp_path / "m")
        loaded = load_image_model(tmp_path / "m")
        assert loaded.num_parameters() == model.num_parameters()
        info = describe_image_model(loaded)
        assert info["architecture"] == "pixelgpt"
        with pytest.raises(FileNotFoundError):
            load_image_model(tmp_path / "missing")

    def test_overrides_and_unknown_template(self):
        from minimodel.core.config import ConfigError

        model = build_image_model(
            "pixelgpt_16x16_3m", overrides={"palette_size": 12}, verify_budget=False
        )
        assert model.palette_size == 12
        with pytest.raises(ConfigError, match="not found"):
            build_image_model("no_such_template")
        with pytest.raises(ConfigError, match="family"):
            build_image_model({"arch": {}})


class TestPalette:
    """Colour quantization."""

    def test_exact_palette_detection(self, sprites):
        images = [image for image, _ in sprites]
        palette = exact_palette(images, max_colors=64)
        assert palette is not None
        assert len(palette) <= 64
        # Exact palette means lossless quantization.
        indices = palette.quantize(images[0])
        assert np.array_equal(palette.dequantize(indices), images[0])

    def test_exact_palette_gives_up_over_budget(self):
        noise = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        assert exact_palette([noise], max_colors=16) is None

    def test_median_cut_and_kmeans(self):
        rng = np.random.default_rng(0)
        pixels = rng.integers(0, 255, (2000, 3), dtype=np.uint8)
        for palette in (median_cut_palette(pixels, 16), kmeans_palette(pixels, 16, iterations=3)):
            assert len(palette) == 16
            indices = palette.quantize(pixels.reshape(40, 50, 3))
            assert indices.max() < 16

    def test_build_palette_dispatch(self, sprites):
        images = [image for image, _ in sprites]
        assert len(build_palette(images, 64, method="auto")) <= 64
        assert len(build_palette(images, 4, method="median_cut")) == 4
        with pytest.raises(ValueError, match="unknown palette method"):
            build_palette(images, 8, method="feelings")
        with pytest.raises(ValueError, match="at least one"):
            build_palette([], 8)

    def test_palette_save_load(self, sprites, tmp_path):
        palette = build_palette([image for image, _ in sprites], 16)
        palette.save(tmp_path / "palette.json")
        restored = Palette.load(tmp_path / "palette.json")
        assert np.array_equal(palette.colors, restored.colors)
        assert restored.to_dict()["size"] == len(palette)


class TestImageData:
    """Shards, corpora and datasets."""

    def test_prepare_rgb_corpus_and_dataset(self, sprites, tmp_path):
        stats = prepare_image_corpus(sprites, tmp_path / "rgb", size=16)
        assert stats["n_images"] == len(sprites)
        assert stats["n_classes"] == 4

        corpus = ImageCorpus(tmp_path / "rgb")
        assert len(corpus) == len(sprites)
        assert corpus.image(0).shape == (16, 16, 3)
        assert corpus.label(0) >= 0
        assert "ImageCorpus" in repr(corpus)
        assert corpus.stats()["n_images"] == len(sprites)

        dataset = ImageDataset(corpus, horizontal_flip=True)
        item = dataset[0]
        assert item["image"].shape == (3, 16, 16)
        assert item["image"].min() >= -1.0 and item["image"].max() <= 1.0
        assert "label" in item

        loader = build_image_dataloader(dataset, batch_size=4)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 3, 16, 16)

    def test_prepare_pixel_corpus_and_sequences(self, sprites, tmp_path):
        stats = prepare_pixel_corpus(sprites, tmp_path / "pal", size=16, palette_size=16)
        assert stats["palette_size"] <= 16
        corpus = ImageCorpus(tmp_path / "pal")
        assert corpus.index.mode == "palette"
        dataset = PixelSequenceDataset(corpus)
        item = dataset[0]
        assert item["pixels"].shape == (256,)
        assert item["pixels"].max() < stats["palette_size"]
        assert "PixelSequenceDataset" in repr(dataset)

    def test_pixel_dataset_rejects_rgb(self, sprites, tmp_path):
        prepare_image_corpus(sprites, tmp_path / "rgb", size=16)
        with pytest.raises(ValueError, match="not a palette corpus"):
            PixelSequenceDataset(tmp_path / "rgb")

    def test_paired_corpus(self, sprites, tmp_path):
        images = [image for image, _ in sprites][:8]
        sources = [np.roll(image, 2, axis=0) for image in images]
        prepare_image_corpus(
            images,
            tmp_path / "pairs",
            size=16,
            captions=[f"edit {i}" for i in range(8)],
            source_images=sources,
        )
        corpus = ImageCorpus(tmp_path / "pairs")
        assert corpus.index.has_pairs
        assert corpus.caption(0) == "edit 0"
        assert corpus.source_image(0) is not None

        class FakeTokenizer:
            vocab_size = 64

            def encode(self, text, allow_special=False):
                return [min(ord(c), 63) for c in text][:8]

        dataset = PairedImageDataset(corpus, tokenizer=FakeTokenizer(), caption_max_len=8)
        item = dataset[0]
        assert item["source"].shape == (3, 16, 16)
        assert item["text_tokens"].shape == (8,)

    def test_paired_dataset_requires_pairs(self, sprites, tmp_path):
        prepare_image_corpus(sprites, tmp_path / "plain", size=16)
        dataset = PairedImageDataset(tmp_path / "plain")
        with pytest.raises(ValueError, match="no paired source"):
            dataset[0]

    def test_shard_writer_validation(self, tmp_path):
        writer = ImageShardWriter(tmp_path, height=8, width=8)
        with pytest.raises(ValueError, match="does not match"):
            writer.write(np.zeros((4, 4, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="source_image"):
            ImageShardWriter(tmp_path / "p", height=8, width=8, with_pairs=True).write(
                np.zeros((8, 8, 3), dtype=np.uint8)
            )

    def test_missing_corpus_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="index.json"):
            ImageCorpus(tmp_path)

    def test_synthetic_sprites_are_symmetric(self):
        sprites = synthetic_sprites(4, size=12, seed=1)
        for image, name in sprites:
            assert image.shape == (12, 12, 3)
            assert np.array_equal(image[:, :6], image[:, 6:][:, ::-1])
            assert name.startswith("class_")


class TestDiffusionTraining:
    """Objectives, EMA and the trainer loop."""

    def test_timestep_distributions(self):
        for distribution in ("uniform", "logit_normal", "cosine"):
            t = sample_timesteps(64, device=torch.device("cpu"), distribution=distribution)
            assert t.shape == (64,)
            assert (t >= 0).all() and (t <= 1).all()
        with pytest.raises(ValueError, match="unknown timestep"):
            sample_timesteps(4, device=torch.device("cpu"), distribution="banana")

    def test_flow_matching_targets(self):
        x0 = torch.randn(4, 3, 8, 8)
        noise = torch.randn_like(x0)
        x_t, velocity = flow_matching_targets(x0, noise, torch.zeros(4))
        assert torch.allclose(x_t, x0)
        assert torch.allclose(velocity, noise - x0)
        x_t1, _ = flow_matching_targets(x0, noise, torch.ones(4))
        assert torch.allclose(x_t1, noise)

    def test_cosine_alpha_bar_monotone(self):
        t = torch.linspace(0, 1, 10)
        alpha = cosine_alpha_bar(t)
        assert (alpha[:-1] >= alpha[1:]).all()
        assert alpha[0] > 0.99

    def test_ema_tracks_and_restores(self):
        model = torch.nn.Linear(4, 4)
        ema = EMA(model, decay=0.5)
        original = model.weight.detach().clone()
        with torch.no_grad():
            model.weight.add_(1.0)
        ema.update(model)
        backup = ema.copy_to(model)
        assert not torch.allclose(model.weight, original + 1.0)
        ema.restore(model, backup)
        assert torch.allclose(model.weight, original + 1.0)
        state = ema.state_dict()
        ema.load_state_dict(state)

    def test_diffusion_trainer_fit_and_export(self, sprites, tmp_path):
        prepare_image_corpus(sprites, tmp_path / "rgb", size=16)
        model = DiT({"image_size": 16, "patch_size": 4, "dim": 32, "depth": 1, "n_heads": 2})
        trainer = DiffusionTrainer(
            model,
            DiffusionConfig(
                run_name="d",
                output_dir=str(tmp_path / "runs"),
                max_steps=3,
                batch_size=4,
                lr=1e-3,
                log_every=3,
                save_every=0,
                eval_every=2,
                resume=False,
            ),
            dataset=ImageDataset(tmp_path / "rgb"),
            eval_dataset=ImageDataset(tmp_path / "rgb"),
        )
        summary = trainer.fit()
        assert summary["steps"] == 3
        exported = trainer.export(tmp_path / "model")
        assert (exported / "model.pt").exists()

    def test_diffusion_trainer_requires_data(self, tmp_path):
        model = DiT(TINY_DIT)
        with pytest.raises(ValueError, match="dataset"):
            DiffusionTrainer(model, DiffusionConfig(output_dir=str(tmp_path), resume=False))

    def test_ddpm_objective(self, sprites, tmp_path):
        prepare_image_corpus(sprites, tmp_path / "rgb", size=16)
        model = DiT({"image_size": 16, "patch_size": 4, "dim": 32, "depth": 1, "n_heads": 2})
        trainer = DiffusionTrainer(
            model,
            DiffusionConfig(
                run_name="ddpm",
                output_dir=str(tmp_path / "runs"),
                max_steps=1,
                batch_size=4,
                objective="ddpm",
                log_every=1,
                save_every=0,
                resume=False,
            ),
            dataset=ImageDataset(tmp_path / "rgb"),
        )
        assert trainer.fit()["steps"] == 1


class TestPixelAndVAETrainers:
    """The autoregressive pixel trainer and the autoencoder trainer."""

    def test_pixelgpt_trainer(self, sprites, tmp_path):
        prepare_pixel_corpus(sprites, tmp_path / "pal", size=16, palette_size=8)
        stats = ImageCorpus(tmp_path / "pal")
        model = PixelGPT(
            {**TINY_PIXEL, "image_size": 16, "palette_size": stats.index.palette_size}
        )
        trainer = PixelGPTTrainer(
            model,
            PixelGPTConfig(
                run_name="p",
                output_dir=str(tmp_path / "runs"),
                max_steps=3,
                batch_size=4,
                seq_len=256,
                lr=1e-3,
                log_every=3,
                eval_every=2,
                eval_batches=1,
                save_every=0,
                resume=False,
            ),
            train_dataset=PixelSequenceDataset(tmp_path / "pal"),
            eval_dataset=PixelSequenceDataset(tmp_path / "pal"),
        )
        result = trainer.fit()
        assert result.steps == 3
        metrics = trainer.evaluate()
        assert "val_pixel_accuracy" in metrics

    def test_vae_trainer(self, sprites, tmp_path):
        prepare_image_corpus(sprites, tmp_path / "rgb", size=16)
        vae = VAE(
            {
                "image_size": 16,
                "base_channels": 8,
                "channel_multipliers": [1, 2],
                "blocks_per_level": 1,
                "attention_at_bottleneck": False,
            }
        )
        trainer = VAETrainer(
            vae,
            VAETrainerConfig(
                run_name="v",
                output_dir=str(tmp_path / "runs"),
                max_steps=2,
                batch_size=4,
                lr=1e-3,
                log_every=2,
                eval_every=2,
                eval_batches=1,
                save_every=0,
                resume=False,
            ),
            train_dataset=ImageDataset(tmp_path / "rgb"),
            eval_dataset=ImageDataset(tmp_path / "rgb"),
        )
        result = trainer.fit()
        assert result.steps == 2
        exported = trainer.export(tmp_path / "vae")
        assert (exported / "model.pt").exists()


class TestSamplers:
    """ODE/SDE integration and image output."""

    def test_timestep_schedule_descends(self):
        t = timestep_schedule(10)
        assert t[0] == 1.0 and t[-1] == 0.0
        assert (t[:-1] > t[1:]).all()
        shifted = timestep_schedule(10, shift=3.0)
        assert shifted[5] > t[5]

    def test_all_samplers_produce_images(self):
        model = DiT(TINY_DIT).eval()
        shape = (2, 3, 8, 8)
        for sampler in (euler_sample, heun_sample, ddim_sample):
            images = sampler(model, shape, n_steps=3)
            assert images.shape == shape
            assert torch.isfinite(images).all()

    def test_sample_images_wrapper(self):
        model = DiT({**TINY_DIT, "condition": "class", "num_classes": 3})
        images = sample_images(
            model, n_samples=2, sampler="euler", n_steps=2,
            labels=torch.tensor([0, 1]), guidance_scale=2.0, seed=0,
        )
        assert images.shape == (2, 3, 8, 8)
        with pytest.raises(ValueError, match="unknown sampler"):
            sample_images(model, sampler="teleport")

    def test_grid_and_saving(self, tmp_path):
        images = torch.rand(5, 3, 8, 8) * 2 - 1
        array = tensor_to_uint8(images)
        assert array.shape == (5, 8, 8, 3)
        assert array.dtype == np.uint8
        grid = make_grid(array, columns=3, padding=1)
        assert grid.shape[0] == 2 * 8 + 3 * 1
        path = save_image_grid(images, tmp_path / "grid.png", scale=2)
        assert path.exists()

    def test_save_palette_indices(self, sprites, tmp_path):
        palette = build_palette([image for image, _ in sprites], 16)
        indices = torch.randint(0, len(palette), (4, 16, 16))
        path = save_image_grid(indices, tmp_path / "pix.png", palette=palette, scale=2)
        assert path.exists()

    def test_sample_pixel_art_end_to_end(self, tmp_path):
        model = PixelGPT(TINY_PIXEL)
        indices = sample_pixel_art(model, n_samples=2, seed=0)
        assert indices.shape == (2, 6, 6)
        palette = Palette(np.zeros((8, 3), dtype=np.uint8))
        path = sample_pixel_art(
            model, n_samples=2, palette=palette, seed=0, output=tmp_path / "s.png"
        )
        assert path.exists()
        with pytest.raises(ValueError, match="palette"):
            sample_pixel_art(model, n_samples=1, output=tmp_path / "fail.png")
