"""A tiny name -> factory registry.

Registries let YAML recipes refer to code by name (``optimizer: muon``,
``family: looped-transformer``) without the loader importing every possible
implementation up front.

>>> losses = Registry("loss")
>>> @losses.register("mse")
... def mse(a, b):
...     return ((a - b) ** 2).mean()
>>> losses.get("mse") is mse
True
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Generic, TypeVar

__all__ = ["Registry"]

T = TypeVar("T")


class Registry(Generic[T]):
    """Case-insensitive mapping from a name to a factory or class.

    Names are normalised by lowercasing and by treating ``-`` and ``_`` as the
    same character, so ``looped-transformer`` and ``looped_transformer`` resolve
    to the same entry.
    """

    def __init__(self, name: str):
        self.name = name
        self._entries: dict[str, T] = {}
        self._aliases: dict[str, str] = {}

    @staticmethod
    def normalize(key: str) -> str:
        """Return the canonical lookup form of ``key``."""
        return str(key).strip().lower().replace("-", "_")

    def register(
        self, name: str, *, aliases: tuple[str, ...] = (), overwrite: bool = False
    ) -> Callable[[T], T]:
        """Return a decorator that registers the decorated object under ``name``."""

        def decorator(obj: T) -> T:
            self.add(name, obj, aliases=aliases, overwrite=overwrite)
            return obj

        return decorator

    def add(
        self, name: str, obj: T, *, aliases: tuple[str, ...] = (), overwrite: bool = False
    ) -> None:
        """Register ``obj`` under ``name`` (plus optional ``aliases``)."""
        key = self.normalize(name)
        if key in self._entries and not overwrite:
            raise KeyError(f"{self.name!r} registry already contains {name!r}")
        self._entries[key] = obj
        for alias in aliases:
            alias_key = self.normalize(alias)
            if alias_key in self._aliases and not overwrite:
                raise KeyError(f"{self.name!r} registry already contains alias {alias!r}")
            self._aliases[alias_key] = key

    def get(self, name: str) -> T:
        """Look up ``name``, raising a ``KeyError`` that lists valid options."""
        key = self.normalize(name)
        key = self._aliases.get(key, key)
        if key not in self._entries:
            options = ", ".join(sorted(self._entries)) or "<empty>"
            raise KeyError(f"unknown {self.name} {name!r}; available: {options}")
        return self._entries[key]

    def create(self, name: str, *args, **kwargs):
        """Look up ``name`` and immediately call it."""
        factory = self.get(name)
        return factory(*args, **kwargs)  # type: ignore[operator]

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        key = self.normalize(name)
        return self._aliases.get(key, key) in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._entries))

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, entries={sorted(self._entries)})"

    def names(self) -> list[str]:
        """All canonical registered names, sorted."""
        return sorted(self._entries)

    def aliases(self) -> Mapping[str, str]:
        """Mapping of alias -> canonical name."""
        return dict(self._aliases)
