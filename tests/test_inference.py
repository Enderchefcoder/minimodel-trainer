"""Tests for sampling, generation and the inference runner."""

from __future__ import annotations

import pytest
import torch

from minimodel.inference.run import (
    complete,
    complete_batch,
    generate_with_reasoning,
    load_for_inference,
    stream_completion,
)
from minimodel.inference.sampling import (
    SamplingConfig,
    apply_penalties,
    filter_logits,
    generate,
    generate_batch,
    generate_text,
    stream_generate,
)


class TestSamplingConfig:
    """Validation and derived behaviour."""

    def test_zero_temperature_forces_greedy(self):
        config = SamplingConfig(temperature=0.0)
        assert not config.do_sample

    def test_validation(self):
        with pytest.raises(ValueError, match="temperature"):
            SamplingConfig(temperature=-1)
        with pytest.raises(ValueError, match="top_p"):
            SamplingConfig(top_p=2.0)
        with pytest.raises(ValueError, match="min_p"):
            SamplingConfig(min_p=-0.1)


class TestLogitFilters:
    """Top-k, top-p, min-p and penalties."""

    def test_top_k_keeps_k(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        filtered = filter_logits(logits, top_k=2)
        assert torch.isinf(filtered[0, 0]) and torch.isinf(filtered[0, 1])
        assert filtered[0, 3] == 4.0

    def test_top_p_keeps_nucleus(self):
        logits = torch.tensor([[10.0, 1.0, 0.5, 0.1]])
        filtered = filter_logits(logits, top_p=0.5)
        assert filtered[0, 0] == 10.0
        assert torch.isinf(filtered[0, 3])

    def test_top_p_always_keeps_best(self):
        logits = torch.tensor([[5.0, 4.9]])
        filtered = filter_logits(logits, top_p=0.01)
        assert not torch.isinf(filtered[0, 0])

    def test_min_p_adapts_to_confidence(self):
        confident = torch.tensor([[10.0, 0.0, 0.0]])
        assert torch.isinf(filter_logits(confident, min_p=0.5)[0, 1])
        flat = torch.tensor([[1.0, 1.0, 1.0]])
        assert not torch.isinf(filter_logits(flat, min_p=0.5)).any()

    def test_penalties_downweight_seen_tokens(self):
        logits = torch.tensor([[2.0, -1.0, 3.0]])
        generated = torch.tensor([[0, 1]])
        penalised = apply_penalties(logits, generated, repetition_penalty=2.0, presence_penalty=0.5)
        assert penalised[0, 0] == pytest.approx(2.0 / 2.0 - 0.5)
        assert penalised[0, 1] == pytest.approx(-1.0 * 2.0 - 0.5)
        assert penalised[0, 2] == 3.0
        untouched = apply_penalties(logits, generated)
        assert torch.equal(untouched, logits)


class TestGenerate:
    """Batched and streaming generation."""

    def test_greedy_is_deterministic_and_cached(self, tiny_model):
        tiny_model.eval()
        prompt = torch.randint(0, tiny_model.vocab_size, (2, 5))
        config = SamplingConfig(max_new_tokens=8, do_sample=False)
        first = generate(tiny_model, prompt, config)
        second = generate(tiny_model, prompt, config)
        assert torch.equal(first, second)
        assert first.shape == (2, 13)

    def test_seeded_sampling_reproduces(self, tiny_model):
        prompt = torch.randint(0, tiny_model.vocab_size, (1, 4))
        config = SamplingConfig(max_new_tokens=6, temperature=1.0, seed=7)
        assert torch.equal(
            generate(tiny_model, prompt, config), generate(tiny_model, prompt, config)
        )

    def test_stop_tokens_halt_generation(self, tiny_model, tokenizer):
        prompt = torch.randint(0, tiny_model.vocab_size, (1, 4))
        # Every token is a stop token, so generation should stop immediately.
        config = SamplingConfig(
            max_new_tokens=50,
            do_sample=False,
            stop_token_ids=list(range(tiny_model.vocab_size)),
        )
        output = generate(tiny_model, prompt, config)
        assert output.shape[1] == 5

    def test_1d_prompt_promoted(self, tiny_model):
        output = generate(
            tiny_model, torch.zeros(3, dtype=torch.long), SamplingConfig(max_new_tokens=2)
        )
        assert output.dim() == 2

    def test_stream_matches_generate(self, tiny_model):
        tiny_model.eval()
        prompt = torch.randint(0, tiny_model.vocab_size, (1, 4))
        config = SamplingConfig(max_new_tokens=6, do_sample=False)
        batch = generate(tiny_model, prompt, config)[0, 4:].tolist()
        streamed = list(stream_generate(tiny_model, prompt, config))
        assert streamed == batch

    def test_stream_rejects_batches(self, tiny_model):
        with pytest.raises(ValueError, match="single sequence"):
            list(stream_generate(tiny_model, torch.zeros(2, 3, dtype=torch.long)))

    def test_generate_text_and_batch(self, tiny_model, tokenizer):
        text = generate_text(tiny_model, tokenizer, "The river", max_new_tokens=6, temperature=0.0)
        assert text.startswith("The river")
        completion_only = generate_text(
            tiny_model,
            tokenizer,
            "The river",
            max_new_tokens=6,
            temperature=0.0,
            include_prompt=False,
        )
        assert not completion_only.startswith("The river")

        results = generate_batch(
            tiny_model,
            tokenizer,
            ["The river", "A very much longer prompt here"],
            SamplingConfig(max_new_tokens=4, do_sample=False),
        )
        assert len(results) == 2
        assert results[0].startswith("The river")


class TestInferenceRunner:
    """Loading a model directory and the high-level helpers."""

    @pytest.fixture
    def model_dir(self, tiny_model, tokenizer, tmp_path):
        directory = tmp_path / "model"
        tiny_model.save_pretrained(directory)
        tokenizer.save(directory / "tokenizer.json")
        return directory

    def test_load_for_inference(self, model_dir):
        loaded = load_for_inference(model_dir, device="cpu")
        assert loaded.parameters > 0
        assert "LoadedModel" in repr(loaded)
        assert not loaded.model.training

    def test_tokenizer_search_falls_back_to_parent(self, tiny_model, tokenizer, tmp_path):
        directory = tmp_path / "run" / "model"
        tiny_model.save_pretrained(directory)
        tokenizer.save(tmp_path / "run" / "tokenizer.json")
        loaded = load_for_inference(directory, device="cpu")
        assert loaded.tokenizer.vocab_size == tokenizer.vocab_size

    def test_missing_tokenizer_raises(self, tiny_model, tmp_path):
        directory = tmp_path / "bare"
        tiny_model.save_pretrained(directory)
        with pytest.raises(FileNotFoundError, match="tokenizer"):
            load_for_inference(directory, device="cpu")

    def test_complete_plain_and_chat(self, model_dir):
        loaded = load_for_inference(model_dir, device="cpu")
        plain = complete(loaded, "The river", max_new_tokens=6, temperature=0.0)
        assert isinstance(plain, str)
        chat = complete(loaded, "hi", max_new_tokens=6, temperature=0.0, chat=True)
        assert isinstance(chat, str)
        batch = complete_batch(loaded, ["a", "b"], config=SamplingConfig(max_new_tokens=3))
        assert len(batch) == 2

    def test_generate_with_reasoning_phases(self, model_dir):
        loaded = load_for_inference(model_dir, device="cpu")
        result = generate_with_reasoning(
            loaded, "What is 2 plus 2?", max_reasoning_tokens=8, max_answer_tokens=8
        )
        assert set(result) == {"reasoning", "answer", "full"}

    def test_stream_completion_matches_full_decode(self, model_dir):
        loaded = load_for_inference(model_dir, device="cpu")
        prompt_ids = loaded.tokenizer.encode("The", add_bos=True)
        config = SamplingConfig(
            max_new_tokens=6, do_sample=False, stop_token_ids=[loaded.tokenizer.eos_id]
        )
        expected_ids = generate(
            loaded.model, torch.tensor([prompt_ids]), config, device=loaded.device
        )[0, len(prompt_ids) :].tolist()
        streamed = "".join(
            stream_completion(loaded, "The", max_new_tokens=6, temperature=0.0, chat=False)
        )
        # Streamed text equals decoding the same ids in one shot, and no valid
        # multi-byte character is ever split across chunks.
        assert streamed == loaded.tokenizer.decode(expected_ids)
