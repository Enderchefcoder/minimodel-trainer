"""Tokenizer training, encoding and chat templating.

The toolkit ships its own byte-level BPE implementation so that a trained model
and its tokenizer are a single self-contained artifact with no heavyweight
runtime dependency. See :mod:`minimodel.tokenization.tokenize` for the tokenizer
itself and :mod:`minimodel.tokenization.chat` for instruction-tuning templates.
"""

from __future__ import annotations

from minimodel.tokenization.chat import (
    IGNORE_INDEX,
    ChatTemplate,
    Message,
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

__all__ = [
    "DEFAULT_SPECIAL_TOKENS",
    "GPT_SPLIT_PATTERN",
    "IGNORE_INDEX",
    "BPETokenizer",
    "ChatTemplate",
    "Message",
    "RenderedChat",
    "bytes_to_unicode",
    "normalize_messages",
    "train_tokenizer",
]
