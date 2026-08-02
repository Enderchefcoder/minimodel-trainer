"""Shared infrastructure used by every other subpackage.

Nothing in :mod:`minimodel.core` knows about models or training loops; it only
provides the plumbing they all need: configuration, registries, structured
logging, deterministic seeding, device/dtype resolution, filesystem helpers and
distributed-training utilities.
"""

from __future__ import annotations

from minimodel.core.config import Config, ConfigError, load_config, merge_configs, parse_overrides
from minimodel.core.distributed import (
    barrier,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    setup_distributed,
    teardown_distributed,
)
from minimodel.core.devices import (
    autocast_context,
    describe_device,
    resolve_device,
    resolve_dtype,
)
from minimodel.core.io_utils import (
    append_jsonl,
    atomic_write_text,
    ensure_dir,
    human_bytes,
    human_count,
    human_duration,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from minimodel.core.logging_utils import JsonlLogger, get_logger, setup_logging
from minimodel.core.registry import Registry
from minimodel.core.seeding import get_rng_state, seed_worker, set_rng_state, set_seed

__all__ = [
    "Config",
    "ConfigError",
    "JsonlLogger",
    "Registry",
    "append_jsonl",
    "atomic_write_text",
    "autocast_context",
    "barrier",
    "describe_device",
    "ensure_dir",
    "get_logger",
    "get_rank",
    "get_rng_state",
    "get_world_size",
    "human_bytes",
    "human_count",
    "human_duration",
    "is_distributed",
    "is_main_process",
    "load_config",
    "merge_configs",
    "parse_overrides",
    "read_json",
    "read_jsonl",
    "resolve_device",
    "resolve_dtype",
    "seed_worker",
    "set_rng_state",
    "set_seed",
    "setup_distributed",
    "setup_logging",
    "teardown_distributed",
    "write_json",
    "write_jsonl",
]
