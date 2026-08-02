"""Turn raw JSONL records into packed token shards.

Three packing modes, one per training stage:

``text`` (pretraining)
    Documents are concatenated with an end-of-text token between them and the
    stream is cut into fixed-length windows at load time. No padding is ever
    written, so every stored token is a token the model trains on.

``chat`` / ``instruction`` (SFT and CoT)
    Each conversation is rendered through the chat template, producing a token
    array and a parallel label array where prompt positions are ``-100``. Short
    examples are packed end to end into the same stream; the label mask keeps
    the boundaries meaningful, so no attention-mask bookkeeping is needed.

``preference`` (DPO/SPIN)
    Chosen and rejected completions are written to a JSONL file rather than a
    binary shard, because pair lengths vary too much for packing to help.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from minimodel.core.io_utils import ensure_dir, human_count, read_jsonl, write_json
from minimodel.core.logging_utils import get_logger
from minimodel.datasets.registry import DatasetSpec, get_dataset
from minimodel.datasets.shards import ShardWriter, choose_dtype
from minimodel.tokenization.chat import IGNORE_INDEX, ChatTemplate, normalize_messages
from minimodel.tokenization.tokenize import BPETokenizer

__all__ = [
    "extract_text",
    "tokenize_chat_records",
    "tokenize_jsonl",
    "tokenize_preference_records",
    "tokenize_text_records",
]

logger = get_logger(__name__)


def extract_text(record: dict[str, Any], field: str = "text") -> str | None:
    """Pull the text out of a record, trying a few common field names.

    >>> extract_text({"content": "hi"})
    'hi'
    """
    for candidate in (field, "text", "content", "raw_content", "document", "body"):
        value = record.get(candidate)
        if isinstance(value, str) and value.strip():
            return value
    return None


def tokenize_text_records(
    records: Iterable[dict[str, Any]],
    tokenizer: BPETokenizer,
    output_dir: str | Path,
    *,
    text_field: str = "text",
    shard_tokens: int = 100_000_000,
    add_eos: bool = True,
    source: str | None = None,
    progress_every: int = 10_000,
) -> dict[str, Any]:
    """Tokenize plain-text records into token shards."""
    writer = ShardWriter(
        output_dir,
        dtype=choose_dtype(tokenizer.vocab_size),
        shard_tokens=shard_tokens,
        supervised=False,
        vocab_size=tokenizer.vocab_size,
        source=source,
    )
    skipped = 0
    with writer:
        for index, record in enumerate(records):
            text = extract_text(record, text_field)
            if text is None:
                skipped += 1
                continue
            writer.write_document(tokenizer.encode(text, add_eos=add_eos, allow_special=False))
            if progress_every and (index + 1) % progress_every == 0:
                logger.info("tokenized %s documents", human_count(index + 1))
    stats = writer.index.to_dict()
    stats["skipped"] = skipped
    return stats


def tokenize_chat_records(
    records: Iterable[dict[str, Any]],
    tokenizer: BPETokenizer,
    output_dir: str | Path,
    *,
    template: ChatTemplate | None = None,
    max_length: int | None = None,
    shard_tokens: int = 50_000_000,
    source: str | None = None,
    drop_unsupervised: bool = True,
) -> dict[str, Any]:
    """Tokenize instruction/chat records into token + label shards.

    Parameters
    ----------
    max_length:
        Truncate rendered conversations to this many tokens. Long reasoning
        traces are the usual reason to set it.
    drop_unsupervised:
        Skip examples where truncation removed every supervised position, which
        would otherwise contribute a NaN loss.
    """
    template = template or ChatTemplate(tokenizer)
    writer = ShardWriter(
        output_dir,
        dtype=choose_dtype(tokenizer.vocab_size),
        shard_tokens=shard_tokens,
        supervised=True,
        vocab_size=tokenizer.vocab_size,
        source=source,
    )
    skipped = 0
    truncated = 0
    with writer:
        for record in records:
            try:
                rendered = template.render(record)
            except (ValueError, TypeError, KeyError):
                skipped += 1
                continue
            input_ids = rendered.input_ids
            labels = rendered.labels
            if max_length is not None and len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
                truncated += 1
            if drop_unsupervised and all(label == IGNORE_INDEX for label in labels):
                skipped += 1
                continue
            writer.write_document(input_ids, labels)
    stats = writer.index.to_dict()
    stats.update({"skipped": skipped, "truncated": truncated})
    return stats


def tokenize_preference_records(
    records: Iterable[dict[str, Any]],
    tokenizer: BPETokenizer,
    output_path: str | Path,
    *,
    template: ChatTemplate | None = None,
    max_length: int | None = 1024,
    prompt_field: str = "prompt",
) -> dict[str, Any]:
    """Render preference pairs to a JSONL file of token ids.

    Each output line has ``prompt_ids``, ``chosen_ids``, ``chosen_labels``,
    ``rejected_ids`` and ``rejected_labels``, which is exactly what the DPO and
    SPIN trainers consume.
    """
    template = template or ChatTemplate(tokenizer)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            prompt = record.get(prompt_field) or record.get("question") or record.get("instruction")
            chosen = record.get("chosen")
            rejected = record.get("rejected")
            if chosen is None or rejected is None:
                skipped += 1
                continue

            # Some datasets store chosen/rejected as full message lists rather
            # than as bare answer strings.
            def _answer_text(value: Any) -> str:
                if isinstance(value, str):
                    return value
                messages = normalize_messages(value)
                for message in reversed(messages):
                    if message["role"] == "assistant":
                        return str(message["content"])
                return str(messages[-1]["content"]) if messages else ""

            if prompt is None and not isinstance(chosen, str):
                messages = normalize_messages(chosen)
                user_turns = [m for m in messages if m["role"] == "user"]
                prompt = user_turns[-1]["content"] if user_turns else None
            if prompt is None:
                skipped += 1
                continue

            base = [{"role": "user", "content": str(prompt)}]
            chosen_render = template.render(
                [*base, {"role": "assistant", "content": _answer_text(chosen)}]
            )
            rejected_render = template.render(
                [*base, {"role": "assistant", "content": _answer_text(rejected)}]
            )
            prompt_ids = template.render_prompt(base)

            row = {
                "prompt_ids": prompt_ids[:max_length] if max_length else prompt_ids,
                "chosen_ids": chosen_render.input_ids[:max_length]
                if max_length
                else chosen_render.input_ids,
                "chosen_labels": chosen_render.labels[:max_length]
                if max_length
                else chosen_render.labels,
                "rejected_ids": rejected_render.input_ids[:max_length]
                if max_length
                else rejected_render.input_ids,
                "rejected_labels": rejected_render.labels[:max_length]
                if max_length
                else rejected_render.labels,
            }
            handle.write(json.dumps(row) + "\n")
            written += 1

    stats = {"path": str(output_path), "pairs": written, "skipped": skipped}
    write_json(output_path.with_suffix(".index.json"), stats)
    logger.info("wrote %s preference pairs to %s", human_count(written), output_path)
    return stats


def _iter_source(
    source: str | Path | Iterable[dict[str, Any]], limit: int | None
) -> Iterator[dict[str, Any]]:
    """Iterate records from a JSONL path or an in-memory iterable."""
    if isinstance(source, (str, Path)):
        iterator: Iterable[dict[str, Any]] = read_jsonl(source)
    else:
        iterator = source
    for index, record in enumerate(iterator):
        if limit is not None and index >= limit:
            return
        yield record


def tokenize_jsonl(
    source: str | Path | Iterable[dict[str, Any]],
    tokenizer: BPETokenizer,
    output_dir: str | Path,
    *,
    format: str = "text",
    text_field: str = "text",
    template: ChatTemplate | None = None,
    max_length: int | None = None,
    limit: int | None = None,
    shard_tokens: int = 100_000_000,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Tokenize a JSONL file (or record iterable) according to ``format``.

    ``format`` is one of ``text``, ``chat``/``instruction``, or ``preference``.
    """
    records = _iter_source(source, limit)
    normalized_format = format.strip().lower()

    if normalized_format in {"text", "raw"}:
        return tokenize_text_records(
            records,
            tokenizer,
            output_dir,
            text_field=text_field,
            shard_tokens=shard_tokens,
            source=source_name,
        )
    if normalized_format in {"chat", "instruction", "cot", "sft"}:
        return tokenize_chat_records(
            records,
            tokenizer,
            output_dir,
            template=template,
            max_length=max_length,
            shard_tokens=shard_tokens,
            source=source_name,
        )
    if normalized_format in {"preference", "dpo"}:
        output = Path(output_dir)
        target = output / "pairs.jsonl" if output.suffix == "" else output
        return tokenize_preference_records(
            records, tokenizer, target, template=template, max_length=max_length
        )
    raise ValueError(
        f"unknown tokenization format {format!r}; expected text, chat, instruction or preference"
    )


def tokenize_registered(
    name: str,
    tokenizer: BPETokenizer,
    *,
    raw_dir: str | Path = "data/raw",
    output_root: str | Path = "data/tokenized",
    limit: int | None = None,
    max_length: int | None = None,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Tokenize a previously pulled registry dataset using its declared format."""
    spec: DatasetSpec = get_dataset(name, path=registry_path)
    raw_path = Path(raw_dir) / f"{name}.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Run `minimodel data pull {name}` first."
        )
    return tokenize_jsonl(
        raw_path,
        tokenizer,
        Path(output_root) / name,
        format=spec.format,
        text_field=spec.text_field,
        limit=limit,
        max_length=max_length,
        source_name=spec.display,
    )
