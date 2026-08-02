"""Configuration loading and manipulation.

Every entry point in the toolkit is driven by a YAML file. This module turns
those files into :class:`Config` objects, which behave like dictionaries but
additionally support:

* dotted access (``cfg["model.dim"]`` and ``cfg.get("model.dim", 128)``),
* recursive merging of a base config with overrides,
* ``extends:`` inheritance so recipes can build on each other,
* ``${VAR}`` environment-variable interpolation,
* command-line overrides of the form ``--set training.lr=3e-4``.

Example
-------
>>> cfg = Config({"model": {"dim": 128}})
>>> cfg["model.dim"]
128
>>> cfg.set("model.dim", 256)
>>> cfg["model.dim"]
256
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Iterator, Mapping, MutableMapping
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Config",
    "ConfigError",
    "load_config",
    "merge_configs",
    "parse_overrides",
    "parse_scalar",
]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

# Guards against `extends:` cycles and runaway inheritance chains.
_MAX_EXTENDS_DEPTH = 16


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or inconsistent."""


def parse_scalar(text: str) -> Any:
    """Convert a command-line scalar into a Python value.

    Tries YAML scalar rules first (so ``true``/``null``/``3e-4``/``[1, 2]`` all
    work) and falls back to the raw string.

    >>> parse_scalar("3e-4")
    0.0003
    >>> parse_scalar("true")
    True
    >>> parse_scalar("hello")
    'hello'
    """
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    # `yaml.safe_load("")` returns None, but an explicitly empty override should
    # stay an empty string rather than silently becoming null.
    if value is None and text.strip() != "" and text.strip().lower() not in {"null", "~", "none"}:
        return text
    if text.strip() == "":
        return ""
    return value


def _interpolate_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` inside strings."""
    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            resolved = os.environ.get(name)
            if resolved is None:
                if default is None:
                    raise ConfigError(
                        f"environment variable {name!r} referenced in config is not set "
                        f"(use ${{{name}:-default}} to provide a fallback)"
                    )
                resolved = default
            return resolved

        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, Mapping):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def merge_configs(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` on top of ``base`` without mutating either.

    Nested mappings are merged key by key; every other type (including lists) is
    replaced wholesale.

    >>> merge_configs({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}})
    {'a': {'b': 9, 'c': 2}}
    """
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = merge_configs(existing, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_overrides(pairs: list[str] | tuple[str, ...] | None) -> dict[str, Any]:
    """Turn ``["a.b=1", "c=hello"]`` into a nested dictionary.

    >>> parse_overrides(["training.lr=3e-4", "model.dim=256"])
    {'training': {'lr': 0.0003}, 'model': {'dim': 256}}
    """
    nested: dict[str, Any] = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise ConfigError(f"override {pair!r} is not of the form key=value")
        key, _, raw = pair.partition("=")
        key = key.strip()
        if not key:
            raise ConfigError(f"override {pair!r} has an empty key")
        cursor = nested
        parts = key.split(".")
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[parts[-1]] = parse_scalar(raw)
    return nested


class Config(MutableMapping[str, Any]):
    """A nested configuration mapping with dotted-path access.

    ``Config`` intentionally stays a plain mapping rather than a schema-checked
    dataclass: recipes evolve quickly and unknown keys should not be fatal.
    Consumers convert the sections they care about into dataclasses via
    :meth:`section`.
    """

    def __init__(self, data: Mapping[str, Any] | None = None, *, source: str | None = None):
        self._data: dict[str, Any] = copy.deepcopy(dict(data or {}))
        self.source = source

    # ------------------------------------------------------------------
    # MutableMapping protocol
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        cursor: Any = self._data
        for part in str(key).split("."):
            if not isinstance(cursor, Mapping) or part not in cursor:
                raise KeyError(key)
            cursor = cursor[part]
        return cursor

    def __setitem__(self, key: str, value: Any) -> None:
        parts = str(key).split(".")
        cursor = self._data
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[parts[-1]] = value

    def __delitem__(self, key: str) -> None:
        parts = str(key).split(".")
        cursor: Any = self._data
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise KeyError(key)
            cursor = cursor[part]
        if not isinstance(cursor, dict) or parts[-1] not in cursor:
            raise KeyError(key)
        del cursor[parts[-1]]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        where = f", source={self.source!r}" if self.source else ""
        return f"Config({self._data!r}{where})"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Return ``key`` (dotted path allowed) or ``default`` when missing."""
        try:
            return self[key]
        except KeyError:
            return default

    def require(self, key: str) -> Any:
        """Return ``key`` or raise :class:`ConfigError` with a helpful message."""
        try:
            return self[key]
        except KeyError as exc:
            where = f" in {self.source}" if self.source else ""
            raise ConfigError(f"required config key {key!r} is missing{where}") from exc

    def set(self, key: str, value: Any) -> None:
        """Alias of ``cfg[key] = value`` that reads better in scripts."""
        self[key] = value

    def setdefault_path(self, key: str, value: Any) -> Any:
        """Set ``key`` to ``value`` only if it is not already present."""
        try:
            return self[key]
        except KeyError:
            self[key] = value
            return value

    def section(self, key: str, default: Mapping[str, Any] | None = None) -> Config:
        """Return a sub-tree as its own :class:`Config`.

        A missing key yields an empty (or ``default``-populated) config rather
        than raising, which keeps optional recipe sections ergonomic.
        """
        value = self.get(key, default if default is not None else {})
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ConfigError(f"config key {key!r} is not a mapping (got {type(value).__name__})")
        return Config(value, source=self.source)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy as a plain nested :class:`dict`."""
        return copy.deepcopy(self._data)

    def merged(self, override: Mapping[str, Any]) -> Config:
        """Return a new config with ``override`` merged on top of this one."""
        return Config(merge_configs(self._data, dict(override)), source=self.source)

    def update_from_overrides(self, pairs: list[str] | tuple[str, ...] | None) -> Config:
        """Return a new config with ``key=value`` CLI overrides applied."""
        return self.merged(parse_overrides(pairs))

    def flatten(self) -> dict[str, Any]:
        """Flatten to ``{"a.b.c": value}`` form, useful for logging and tables."""

        def _walk(node: Any, prefix: str) -> Iterator[tuple[str, Any]]:
            if isinstance(node, Mapping):
                for key, value in node.items():
                    yield from _walk(value, f"{prefix}.{key}" if prefix else str(key))
            else:
                yield prefix, node

        return dict(_walk(self._data, ""))

    def to_yaml(self) -> str:
        """Serialise back to YAML text."""
        return yaml.safe_dump(self._data, sort_keys=False, default_flow_style=False)

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON, stringifying anything YAML-only."""
        return json.dumps(self._data, indent=indent, default=str, sort_keys=False)

    def save(self, path: str | Path) -> Path:
        """Write the config to ``path`` as YAML and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        overrides: Mapping[str, Any] | None = None,
        cli_overrides: list[str] | tuple[str, ...] | None = None,
    ) -> Config:
        """Load a YAML config file, resolving ``extends:`` and interpolation."""
        return load_config(path, overrides=overrides, cli_overrides=cli_overrides)

    @classmethod
    def from_yaml_string(cls, text: str, *, source: str | None = None) -> Config:
        """Parse a YAML document from a string."""
        data = yaml.safe_load(text) or {}
        if not isinstance(data, Mapping):
            raise ConfigError("top-level config must be a mapping")
        return cls(_interpolate_env(dict(data)), source=source)


def _resolve_extends(path: Path, depth: int = 0) -> dict[str, Any]:
    """Load ``path`` and recursively merge whatever it ``extends``."""
    if depth > _MAX_EXTENDS_DEPTH:
        raise ConfigError(f"`extends` chain deeper than {_MAX_EXTENDS_DEPTH} while loading {path}")
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"top-level config in {path} must be a mapping")
    data = dict(raw)

    parents = data.pop("extends", None)
    if parents is None:
        return data
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list):
        raise ConfigError(f"`extends` in {path} must be a string or list of strings")

    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = (path.parent / parent_path).resolve()
        merged = merge_configs(merged, _resolve_extends(parent_path, depth + 1))
    return merge_configs(merged, data)


def load_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    cli_overrides: list[str] | tuple[str, ...] | None = None,
) -> Config:
    """Load ``path`` into a :class:`Config`.

    Parameters
    ----------
    path:
        YAML file to read. Relative ``extends:`` entries are resolved against
        the directory containing the file.
    overrides:
        Nested mapping merged on top of the file contents.
    cli_overrides:
        ``["a.b=1"]``-style strings, applied after ``overrides``.
    """
    path = Path(path)
    data = _resolve_extends(path)
    if overrides:
        data = merge_configs(data, dict(overrides))
    if cli_overrides:
        data = merge_configs(data, parse_overrides(list(cli_overrides)))
    return Config(_interpolate_env(data), source=str(path))
