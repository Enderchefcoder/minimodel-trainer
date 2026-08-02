"""Tests for the tokenizer and chat templating."""

from __future__ import annotations

import json

import pytest

from minimodel.tokenization.chat import (
    IGNORE_INDEX,
    ChatTemplate,
    RenderedChat,
    normalize_messages,
)
from minimodel.tokenization.tokenize import (
    DEFAULT_SPECIAL_TOKENS,
    GPT_SPLIT_PATTERN,
    BPETokenizer,
    bytes_to_unicode,
    train_tokenizer,
)

TRICKY = [
    "hello world",
    "hello world 你好 🌍",
    "a_b__c",
    "def f(x):\n    return x**2  # ok",
    "1234567890",
    "   ",
    "ÀÉîöü — “quoted”",
    "\t\ttabs\r\n",
    "",
]


class TestSplitPattern:
    """The pre-tokenization regex must be total."""

    @pytest.mark.parametrize("text", TRICKY)
    def test_split_is_lossless(self, text):
        assert "".join(GPT_SPLIT_PATTERN.findall(text)) == text

    def test_bytes_to_unicode_is_a_bijection(self):
        mapping = bytes_to_unicode()
        assert len(mapping) == 256
        assert len(set(mapping.values())) == 256


class TestBPETokenizer:
    """Training, encoding, decoding and persistence."""

    @pytest.mark.parametrize("text", TRICKY)
    def test_roundtrip(self, tokenizer, text):
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_vocab_and_special_ids(self, tokenizer):
        assert tokenizer.vocab_size == len(tokenizer)
        assert tokenizer.vocab_size >= 256 + len(DEFAULT_SPECIAL_TOKENS)
        assert tokenizer.eos_id == tokenizer.special_tokens["<|endoftext|>"]
        assert tokenizer.pad_id == tokenizer.special_tokens["<|pad|>"]
        assert tokenizer.bos_id == tokenizer.eos_id
        assert tokenizer.token_to_id("<|user|>") is not None
        assert tokenizer.token_to_id("definitely-not-a-token") is None
        assert tokenizer.id_to_token(10**9).startswith("<|unk:")
        assert "BPETokenizer" in repr(tokenizer)

    def test_bos_and_eos_wrapping(self, tokenizer):
        ids = tokenizer.encode("hi", add_bos=True, add_eos=True)
        assert ids[0] == tokenizer.bos_id
        assert ids[-1] == tokenizer.eos_id

    def test_special_tokens_are_atomic(self, tokenizer):
        ids = tokenizer.encode("<|user|>hi<|end|>")
        assert tokenizer.special_tokens["<|user|>"] in ids
        assert tokenizer.special_tokens["<|end|>"] in ids

    def test_special_tokens_can_be_disabled(self, tokenizer):
        ids = tokenizer.encode("<|user|>hi", allow_special=False)
        assert tokenizer.special_tokens["<|user|>"] not in ids
        assert tokenizer.decode(ids) == "<|user|>hi"

    def test_decode_can_keep_specials(self, tokenizer):
        ids = tokenizer.encode("hi", add_bos=True)
        assert "<|endoftext|>" in tokenizer.decode(ids, skip_special=False)
        assert "<|endoftext|>" not in tokenizer.decode(ids)

    def test_encode_batch(self, tokenizer):
        assert len(tokenizer.encode_batch(["a", "b"])) == 2

    def test_compression_ratio_is_positive(self, tokenizer, texts):
        assert tokenizer.compression_ratio(texts[:8]) > 1.0
        assert tokenizer.compression_ratio([]) == 0.0

    def test_save_and_load(self, tokenizer, tmp_path):
        path = tokenizer.save(tmp_path)
        reloaded = BPETokenizer.load(tmp_path)
        assert reloaded.encode("hello world") == tokenizer.encode("hello world")
        assert path.name == "tokenizer.json"
        assert BPETokenizer.load(path).vocab_size == tokenizer.vocab_size

    def test_load_rejects_unknown_format(self, tmp_path):
        path = tmp_path / "tokenizer.json"
        path.write_text(json.dumps({"type": "sentencepiece"}), encoding="utf-8")
        with pytest.raises(ValueError, match="not a recognised"):
            BPETokenizer.load(path)

    def test_load_hf_style_file(self, tokenizer, tmp_path):
        # Emulate a Hugging Face `tokenizers` export.
        payload = {
            "model": {
                "type": "BPE",
                "vocab": dict(tokenizer.vocab),
                "merges": [" ".join(pair) for pair in tokenizer.merges],
            },
            "added_tokens": [
                {"content": token, "id": index} for token, index in tokenizer.special_tokens.items()
            ],
        }
        path = tmp_path / "tokenizer.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = BPETokenizer.load(path)
        assert loaded.encode("hello world") == tokenizer.encode("hello world")

    def test_vocab_too_small_rejected(self):
        with pytest.raises(ValueError, match="too small"):
            BPETokenizer.train(["hello"], vocab_size=10)

    def test_min_frequency_stops_merging(self):
        few = BPETokenizer.train(["abcdef"], vocab_size=400, min_frequency=100)
        assert len(few.merges) == 0

    def test_max_token_length_caps_merges(self):
        tokenizer = BPETokenizer.train(
            ["hello world " * 40], vocab_size=320, min_frequency=2, max_token_length=3
        )
        assert all(len(a + b) <= 3 for a, b in tokenizer.merges)

    def test_train_tokenizer_backends(self, texts):
        python_backend = train_tokenizer(texts, vocab_size=300, backend="python")
        assert python_backend.vocab_size == 300 or python_backend.vocab_size >= 266
        with pytest.raises(ValueError, match="unknown tokenizer backend"):
            train_tokenizer(texts, backend="magic")

    def test_verbose_training_logs(self, texts, caplog):
        BPETokenizer.train(texts, vocab_size=800, verbose=True)


class TestNormalizeMessages:
    """Dataset-shape normalisation."""

    def test_alpaca_shape(self):
        messages = normalize_messages({"instruction": "Hi", "output": "Hello"})
        assert messages == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

    def test_alpaca_with_input_field(self):
        messages = normalize_messages(
            {"instruction": "Summarise", "input": "long text", "output": "short"}
        )
        assert "long text" in messages[0]["content"]

    def test_system_prompt_preserved(self):
        messages = normalize_messages({"system": "be nice", "prompt": "hi", "response": "hello"})
        assert messages[0]["role"] == "system"

    def test_sharegpt_role_aliases(self):
        messages = normalize_messages(
            [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]
        )
        assert [m["role"] for m in messages] == ["user", "assistant"]

    def test_nested_messages_key(self):
        messages = normalize_messages(
            {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
        )
        assert len(messages) == 2

    def test_reasoning_preserved(self):
        messages = normalize_messages(
            [{"role": "assistant", "content": "a", "reasoning": "because"}]
        )
        assert messages[0]["reasoning"] == "because"

    def test_missing_fields_raise(self):
        with pytest.raises(ValueError, match="prompt field"):
            normalize_messages({"answer": "only"})
        with pytest.raises(ValueError, match="answer field"):
            normalize_messages({"instruction": "only"})
        with pytest.raises(TypeError):
            normalize_messages(42)


class TestChatTemplate:
    """Rendering and loss masking."""

    def test_only_assistant_tokens_are_supervised(self, tokenizer):
        template = ChatTemplate(tokenizer)
        rendered = template.render({"instruction": "Hi", "output": "Hello!"})
        assert isinstance(rendered, RenderedChat)
        assert 0 < rendered.n_supervised < len(rendered)
        assert len(rendered.input_ids) == len(rendered.labels)
        text = tokenizer.decode(rendered.input_ids, skip_special=False)
        assert "<|user|>" in text and "<|assistant|>" in text

    def test_train_on_prompt_supervises_everything_but_markers(self, tokenizer):
        default = ChatTemplate(tokenizer).render({"instruction": "Hi", "output": "Hello"})
        full = ChatTemplate(tokenizer, train_on_prompt=True).render(
            {"instruction": "Hi", "output": "Hello"}
        )
        assert full.n_supervised > default.n_supervised

    def test_reasoning_span_rendered_and_weightable(self, tokenizer):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "reasoning": "because of things"},
        ]
        supervised = ChatTemplate(tokenizer).render(messages)
        unsupervised = ChatTemplate(tokenizer, supervise_reasoning=False).render(messages)
        assert supervised.n_supervised > unsupervised.n_supervised
        text = tokenizer.decode(supervised.input_ids, skip_special=False)
        assert "<|think|>" in text and "<|/think|>" in text

    def test_generation_prompt_ends_with_assistant(self, tokenizer):
        template = ChatTemplate(tokenizer)
        ids = template.render_prompt([{"role": "user", "content": "hi"}])
        assert ids[-1] == tokenizer.token_to_id("<|assistant|>")
        assert all(
            label == IGNORE_INDEX
            for label in template.render(
                [{"role": "user", "content": "hi"}], add_generation_prompt=True
            ).labels
        )

    def test_default_system_prompt_injected(self, tokenizer):
        template = ChatTemplate(tokenizer, default_system="be nice")
        text = template.render({"instruction": "hi", "output": "hello"}).text
        assert "be nice" in text

    def test_stop_token_ids(self, tokenizer):
        stops = ChatTemplate(tokenizer).stop_token_ids()
        assert tokenizer.token_to_id("<|end|>") in stops
        assert tokenizer.eos_id in stops

    def test_works_without_chat_special_tokens(self):
        plain = BPETokenizer.train(["hello world " * 30], vocab_size=300, special_tokens=())
        rendered = ChatTemplate(plain).render({"instruction": "hi", "output": "there"})
        assert len(rendered.input_ids) > 0
