"""Byte-level BPE tokenizer: training, encoding, decoding and serialisation.

The implementation is self-contained - no ``tokenizers``, ``transformers`` or
``sentencepiece`` required - because a small model's tokenizer is part of the
model and should not depend on a large binary wheel. When the Rust
``tokenizers`` package *is* installed, :func:`train_tokenizer` uses it for
training only; the resulting merges are loaded back into the same pure-Python
class, so the runtime behaviour is identical either way.

Why byte-level
--------------
Byte-level BPE cannot produce an unknown token. Every possible input encodes,
including emoji, code, and mid-word Unicode. For a small model trained on a
narrow corpus that robustness is worth more than the small compression loss
versus a word-level vocabulary.

Example
-------
>>> tok = BPETokenizer.train(["hello world", "hello there"], vocab_size=300)
>>> tok.decode(tok.encode("hello world"))
'hello world'
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from minimodel.core.io_utils import atomic_write_text
from minimodel.core.logging_utils import get_logger

__all__ = [
    "DEFAULT_SPECIAL_TOKENS",
    "GPT_SPLIT_PATTERN",
    "BPETokenizer",
    "bytes_to_unicode",
    "train_tokenizer",
]

logger = get_logger(__name__)

# `re` has no \p{L}/\p{N} classes, so letters are spelled `[^\W\d_]` and the
# "symbol" branch uses a negative lookahead instead of a subtracted class. The
# trailing `.` alternative guarantees the pattern is total: every character of
# the input lands in exactly one piece, which matters because anything the
# pattern fails to match would be silently dropped from the corpus.
_LETTER = r"[^\W\d_]"

#: GPT-2/GPT-4 style pre-tokenization pattern. Splitting on this before BPE keeps
#: merges from spanning word boundaries, which is what makes the learned vocab
#: transferable across domains.
GPT_SPLIT_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"
    rf"| ?{_LETTER}+"
    r"| ?\d{1,3}"
    rf"| ?(?:(?!{_LETTER}|\d|\s).)+"
    r"|\s+(?!\S)"
    r"|\s+"
    r"|.",
    re.UNICODE | re.DOTALL,
)

#: Special tokens present in every tokenizer this toolkit trains.
#:
#: The chat and reasoning markers are reserved up front even for base models so
#: that instruction tuning or reasoning distillation never has to resize the
#: embedding matrix later.
DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|endoftext|>",
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|think|>",
    "<|/think|>",
    "<|tool|>",
    "<|/tool|>",
)


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Reversible map from the 256 byte values to printable Unicode characters.

    This is the GPT-2 trick: it keeps the vocabulary and merge list readable as
    text (and JSON-safe) while remaining a bijection, so decoding is exact.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapped = list(printable)
    shift = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + shift)
            shift += 1
    return {b: chr(c) for b, c in zip(printable, mapped, strict=True)}


@lru_cache(maxsize=1)
def _unicode_to_bytes() -> dict[str, int]:
    return {v: k for k, v in bytes_to_unicode().items()}


def _get_pairs(symbols: Sequence[str]) -> set[tuple[str, str]]:
    """All adjacent symbol pairs in ``symbols``."""
    return {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}


class BPETokenizer:
    """A byte-level byte-pair-encoding tokenizer.

    Attributes
    ----------
    vocab:
        Mapping from token string (in the byte-to-unicode alphabet) to id.
    merges:
        Ordered list of merge rules. Earlier merges have higher priority.
    special_tokens:
        Mapping from literal special-token text to id. Special tokens are
        matched before BPE and are never split.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: Sequence[tuple[str, str]],
        special_tokens: dict[str, int] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = [tuple(m) for m in merges]
        self.special_tokens = dict(special_tokens or {})
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.id_to_token_map = {i: t for t, i in self.vocab.items()}
        self.id_to_token_map.update({i: t for t, i in self.special_tokens.items()})
        self._special_pattern = (
            re.compile(
                "("
                + "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
                + ")"
            )
            if self.special_tokens
            else None
        )
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """Total number of ids, including special tokens."""
        return len(self.vocab) + len(self.special_tokens)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"

    @property
    def eos_id(self) -> int:
        """Id of ``<|endoftext|>``, used to separate documents."""
        return self.special_tokens.get("<|endoftext|>", 0)

    @property
    def pad_id(self) -> int:
        """Id of ``<|pad|>``; falls back to the EOS id when absent."""
        return self.special_tokens.get("<|pad|>", self.eos_id)

    @property
    def bos_id(self) -> int:
        """Id used to start a sequence. Shares the EOS token by convention."""
        return self.eos_id

    def token_to_id(self, token: str) -> int | None:
        """Look up a token string, returning ``None`` when absent."""
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.vocab.get(token)

    def id_to_token(self, token_id: int) -> str:
        """Reverse lookup; unknown ids render as ``<|unk:N|>``."""
        return self.id_to_token_map.get(int(token_id), f"<|unk:{token_id}|>")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _bpe(self, token: str) -> list[str]:
        """Apply the merge list to one pre-tokenized piece."""
        symbols = list(token)
        if len(symbols) < 2:
            return symbols
        while True:
            pairs = _get_pairs(symbols)
            candidate = min(pairs, key=lambda p: self.ranks.get(p, float("inf")), default=None)
            if candidate is None or candidate not in self.ranks:
                break
            first, second = candidate
            merged: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
            if len(symbols) == 1:
                break
        return symbols

    def _encode_chunk(self, text: str) -> list[int]:
        """Encode text that is known to contain no special tokens."""
        cached = self._cache.get(text)
        if cached is not None:
            return list(cached)
        byte_map = bytes_to_unicode()
        ids: list[int] = []
        for piece in GPT_SPLIT_PATTERN.findall(text):
            mapped = "".join(byte_map[b] for b in piece.encode("utf-8"))
            for symbol in self._bpe(mapped):
                token_id = self.vocab.get(symbol)
                if token_id is None:
                    # Fall back to single characters, which always exist because
                    # the base vocabulary covers all 256 byte values.
                    ids.extend(self.vocab[ch] for ch in symbol)
                else:
                    ids.append(token_id)
        if len(text) <= 64:
            self._cache[text] = list(ids)
        return ids

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        allow_special: bool = True,
    ) -> list[int]:
        """Encode ``text`` into token ids.

        Parameters
        ----------
        add_bos, add_eos:
            Wrap the result with the begin/end-of-text token.
        allow_special:
            When true, literal special-token text such as ``"<|user|>"`` in the
            input is encoded as that single token. Set to false for untrusted
            input so a user cannot inject role markers.
        """
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        if allow_special and self._special_pattern is not None:
            for part in self._special_pattern.split(text):
                if not part:
                    continue
                if part in self.special_tokens:
                    ids.append(self.special_tokens[part])
                else:
                    ids.extend(self._encode_chunk(part))
        else:
            ids.extend(self._encode_chunk(text))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_batch(self, texts: Iterable[str], **kwargs: Any) -> list[list[int]]:
        """Encode many strings with the same options."""
        return [self.encode(text, **kwargs) for text in texts]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        """Decode ids back to text.

        Invalid UTF-8 (which can happen mid-token during streaming generation)
        is replaced rather than raising.
        """
        reverse = _unicode_to_bytes()
        buffer = bytearray()
        chunks: list[str] = []
        for token_id in ids:
            token_id = int(token_id)
            if (
                token_id in self.id_to_token_map
                and self.id_to_token_map[token_id] in self.special_tokens
            ):
                if buffer:
                    chunks.append(buffer.decode("utf-8", errors="replace"))
                    buffer = bytearray()
                if not skip_special:
                    chunks.append(self.id_to_token_map[token_id])
                continue
            token = self.id_to_token_map.get(token_id)
            if token is None:
                continue
            buffer.extend(reverse.get(ch, ord("?")) for ch in token)
        if buffer:
            chunks.append(buffer.decode("utf-8", errors="replace"))
        return "".join(chunks)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 4096,
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
        min_frequency: int = 2,
        max_token_length: int | None = None,
        verbose: bool = False,
    ) -> BPETokenizer:
        """Train a byte-level BPE tokenizer.

        Parameters
        ----------
        texts:
            Training corpus, one document per item.
        vocab_size:
            Target total vocabulary size, *including* the 256 byte tokens and
            the special tokens.
        min_frequency:
            Stop merging once the best pair occurs fewer than this many times.
        max_token_length:
            Optional cap on merged token length, which prevents a handful of
            very long boilerplate strings from eating the vocabulary.

        Notes
        -----
        Words are counted once and merged as frequency-weighted symbol tuples,
        so training cost scales with the number of *distinct* words rather than
        with corpus size.
        """
        specials = list(dict.fromkeys(special_tokens))
        byte_map = bytes_to_unicode()
        base_alphabet = [byte_map[b] for b in range(256)]

        budget = vocab_size - len(specials) - len(base_alphabet)
        if budget < 0:
            raise ValueError(
                f"vocab_size={vocab_size} is too small; it must exceed "
                f"{len(specials) + len(base_alphabet)} (specials + 256 byte tokens)"
            )

        word_counts: Counter[str] = Counter()
        for text in texts:
            for piece in GPT_SPLIT_PATTERN.findall(text):
                word_counts["".join(byte_map[b] for b in piece.encode("utf-8"))] += 1

        words: list[list[str]] = [list(word) for word in word_counts]
        counts: list[int] = list(word_counts.values())

        merges: list[tuple[str, str]] = []
        for step in range(budget):
            pair_counts: Counter[tuple[str, str]] = Counter()
            for symbols, freq in zip(words, counts, strict=True):
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] += freq
            if not pair_counts:
                break
            best, best_count = pair_counts.most_common(1)[0]
            if best_count < min_frequency:
                break
            if max_token_length is not None and len(best[0]) + len(best[1]) > max_token_length:
                # Skip this pair permanently by merging the next best candidate.
                candidates = [
                    (pair, count)
                    for pair, count in pair_counts.most_common(32)
                    if len(pair[0]) + len(pair[1]) <= max_token_length
                ]
                if not candidates:
                    break
                best, best_count = candidates[0]
            merges.append(best)
            joined = best[0] + best[1]
            for idx, symbols in enumerate(words):
                if len(symbols) < 2:
                    continue
                merged: list[str] = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == best[0] and symbols[i + 1] == best[1]:
                        merged.append(joined)
                        i += 2
                    else:
                        merged.append(symbols[i])
                        i += 1
                words[idx] = merged
            if verbose and (step + 1) % 500 == 0:
                logger.info("bpe merge %d/%d (%r, count=%d)", step + 1, budget, joined, best_count)

        vocab: dict[str, int] = {}
        next_id = len(specials)
        for symbol in base_alphabet:
            vocab[symbol] = next_id
            next_id += 1
        for first, second in merges:
            token = first + second
            if token not in vocab:
                vocab[token] = next_id
                next_id += 1
        special_ids = {token: idx for idx, token in enumerate(specials)}
        return cls(vocab, merges, special_ids)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialisable representation."""
        return {
            "version": 1,
            "type": "byte_bpe",
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": [" ".join(pair) for pair in self.merges],
        }

    def save(self, path: str | Path) -> Path:
        """Write the tokenizer to a JSON file."""
        path = Path(path)
        if path.is_dir():
            path = path / "tokenizer.json"
        atomic_write_text(path, json.dumps(self.to_dict(), ensure_ascii=False, indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Read a tokenizer written by :meth:`save`.

        A Hugging Face ``tokenizers`` JSON file with a byte-level BPE model is
        also accepted, which makes it easy to start from an existing tokenizer.
        """
        path = Path(path)
        if path.is_dir():
            path = path / "tokenizer.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("type") == "byte_bpe":
            merges = [tuple(m.split(" ", 1)) for m in data["merges"]]
            return cls(data["vocab"], merges, data.get("special_tokens", {}))
        if "model" in data and data["model"].get("type") == "BPE":
            model = data["model"]
            merges_raw = model.get("merges", [])
            merges = [
                tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in merges_raw
            ]
            specials = {entry["content"]: entry["id"] for entry in data.get("added_tokens", [])}
            vocab = {k: v for k, v in model["vocab"].items() if k not in specials}
            return cls(vocab, merges, specials)
        raise ValueError(f"{path} is not a recognised tokenizer file")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def compression_ratio(self, texts: Iterable[str]) -> float:
        """Average bytes per token on ``texts``.

        A well-trained English tokenizer lands around 4 bytes/token; much lower
        means the vocabulary is too small for the corpus.
        """
        total_bytes = 0
        total_tokens = 0
        for text in texts:
            total_bytes += len(text.encode("utf-8"))
            total_tokens += len(self.encode(text))
        if total_tokens == 0:
            return 0.0
        return total_bytes / total_tokens


def train_tokenizer(
    texts: Iterable[str],
    *,
    vocab_size: int = 4096,
    special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    min_frequency: int = 2,
    backend: str = "auto",
    verbose: bool = False,
) -> BPETokenizer:
    """Train a tokenizer, using the fast Rust backend when it is available.

    Parameters
    ----------
    backend:
        ``"auto"`` (default) uses ``tokenizers`` if it is installed and falls
        back to the pure-Python trainer, ``"python"`` forces the pure trainer,
        ``"fast"`` requires ``tokenizers`` and raises if it is missing.

    The two backends produce equivalent tokenizers; only training speed differs
    (the Rust one is roughly two orders of magnitude faster on large corpora).
    """
    if backend not in {"auto", "python", "fast"}:
        raise ValueError(f"unknown tokenizer backend {backend!r}")

    if backend in {"auto", "fast"}:
        try:
            return _train_with_fast_backend(
                texts,
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                min_frequency=min_frequency,
            )
        except ImportError:
            if backend == "fast":
                raise
            logger.info("`tokenizers` not installed, using the pure-Python BPE trainer")

    return BPETokenizer.train(
        texts,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        verbose=verbose,
    )


def _train_with_fast_backend(
    texts: Iterable[str],
    *,
    vocab_size: int,
    special_tokens: Sequence[str],
    min_frequency: int,
) -> BPETokenizer:
    """Train with the Rust ``tokenizers`` package and convert the result."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(list(texts), trainer=trainer)

    raw = json.loads(tokenizer.to_str())
    model = raw["model"]
    merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in model["merges"]]
    special_ids = {token: idx for idx, token in enumerate(special_tokens)}
    vocab = {k: v for k, v in model["vocab"].items() if k not in special_ids}
    return BPETokenizer(vocab, merges, special_ids)
