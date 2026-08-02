"""Fetch raw datasets into a uniform on-disk JSONL layout.

Everything downstream reads JSONL, one record per line, so this module is the
only place that has to know about Hugging Face datasets, remote URLs or local
files. Records keep their original field names; normalisation to chat messages
happens later, at tokenization time, where the tokenizer and chat template are
available.

Streaming is the default for Hugging Face sources. A pretraining corpus is
typically hundreds of gigabytes and there is no reason to materialise all of it
to pull a 2B-token slice.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from minimodel.core.io_utils import ensure_dir, human_count
from minimodel.core.logging_utils import get_logger
from minimodel.datasets.builtin import builtin_records
from minimodel.datasets.registry import DatasetSpec, get_dataset, resolve_mixture

__all__ = [
    "iter_records",
    "pull_dataset",
    "pull_mixture",
    "write_jsonl_stream",
]

logger = get_logger(__name__)


def _iter_huggingface(spec: DatasetSpec, limit: int | None, streaming: bool) -> Iterator[dict]:
    """Stream records from the Hugging Face hub."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "pulling Hugging Face datasets requires the `datasets` package.\n"
            "Install it with:  pip install 'minimodel-trainer[hf]'"
        ) from exc

    kwargs: dict[str, Any] = {"split": spec.split, "streaming": streaming}
    if spec.config:
        kwargs["name"] = spec.config
    dataset = load_dataset(spec.repo, **kwargs)  # pragma: no cover - network
    for index, record in enumerate(dataset):  # pragma: no cover - network
        if limit is not None and index >= limit:
            break
        yield dict(record)


def _iter_local(spec: DatasetSpec, limit: int | None) -> Iterator[dict]:
    """Read records from a local ``.jsonl``, ``.json`` or ``.txt`` file or directory."""
    if not spec.path:
        raise ValueError(f"dataset {spec.name!r} has source 'local' but no `path`")
    root = Path(spec.path)
    paths = sorted(root.rglob("*")) if root.is_dir() else [root]
    count = 0
    for path in paths:
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
                    count += 1
                    if limit is not None and count >= limit:
                        return
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else [payload]
            for record in records:
                yield dict(record)
                count += 1
                if limit is not None and count >= limit:
                    return
        elif suffix in {".txt", ".md"}:
            # Blank-line-separated paragraphs become documents.
            text = path.read_text(encoding="utf-8", errors="replace")
            for block in text.split("\n\n"):
                block = block.strip()
                if not block:
                    continue
                yield {"text": block}
                count += 1
                if limit is not None and count >= limit:
                    return


def _iter_url(spec: DatasetSpec, limit: int | None) -> Iterator[dict]:
    """Download a JSONL or plain-text file over HTTP and yield its records."""
    import requests

    if not spec.url:
        raise ValueError(f"dataset {spec.name!r} has source 'url' but no `url`")
    response = requests.get(spec.url, stream=True, timeout=60)  # pragma: no cover - network
    response.raise_for_status()  # pragma: no cover - network
    count = 0  # pragma: no cover - network
    for raw in response.iter_lines(decode_unicode=True):  # pragma: no cover - network
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            record = {"text": raw}
        yield record if isinstance(record, dict) else {"text": str(record)}
        count += 1
        if limit is not None and count >= limit:
            return


def iter_records(
    spec: DatasetSpec, *, limit: int | None = None, streaming: bool = True
) -> Iterator[dict]:
    """Yield raw records for ``spec`` from whichever backend it declares."""
    if spec.source == "builtin":
        records = builtin_records(spec.stage if spec.stage != "eval" else "pretrain")
        for index, record in enumerate(records):
            if limit is not None and index >= limit:
                return
            yield record
    elif spec.source == "huggingface":
        yield from _iter_huggingface(spec, limit, streaming)
    elif spec.source == "local":
        yield from _iter_local(spec, limit)
    elif spec.source == "url":
        yield from _iter_url(spec, limit)
    else:
        raise ValueError(f"unknown dataset source {spec.source!r} for {spec.name!r}")


def write_jsonl_stream(records: Iterable[dict], path: str | Path) -> int:
    """Write records to ``path`` as JSON Lines and return how many were written.

    Written incrementally so that pulling a very large slice never needs the
    whole thing in memory.
    """
    path = Path(path)
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def pull_dataset(
    name: str,
    output_dir: str | Path = "data/raw",
    *,
    limit: int | None = None,
    streaming: bool = True,
    registry_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Materialise one registered dataset as ``<output_dir>/<name>.jsonl``.

    Parameters
    ----------
    name:
        Registry key, e.g. ``"cosmopedia-v2"``.
    limit:
        Stop after this many records. Strongly recommended for web corpora.
    streaming:
        Stream from the hub instead of downloading the full dataset.
    overwrite:
        Re-pull even if the output file already exists.

    Returns
    -------
    Path
        The JSONL file that was written.
    """
    spec = get_dataset(name, path=registry_path)
    destination = Path(output_dir) / f"{name}.jsonl"
    if destination.exists() and not overwrite:
        logger.info("%s already exists, skipping (use overwrite=True to refetch)", destination)
        return destination

    logger.info("pulling %s from %s", name, spec.display)
    count = write_jsonl_stream(iter_records(spec, limit=limit, streaming=streaming), destination)
    logger.info("wrote %s records to %s", human_count(count), destination)

    meta = {
        "dataset": spec.to_dict(),
        "records": count,
        "limit": limit,
    }
    (destination.with_suffix(".meta.json")).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def pull_mixture(
    name: str,
    output_dir: str | Path = "data/raw",
    *,
    total_records: int | None = None,
    streaming: bool = True,
    registry_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Pull every dataset in a mixture, splitting ``total_records`` by weight.

    Returns a mapping from dataset name to the JSONL file that was written.
    """
    components = resolve_mixture(name, path=registry_path)
    outputs: dict[str, Path] = {}
    for spec, weight in components:
        limit = int(total_records * weight) if total_records else None
        outputs[spec.name] = pull_dataset(
            spec.name,
            output_dir,
            limit=limit,
            streaming=streaming,
            registry_path=registry_path,
            overwrite=overwrite,
        )
    return outputs
