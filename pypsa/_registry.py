# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The base of the registries a network keeps in `n.meta`, as its composite types and its stored overrides."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pypsa import Network


def jsonable(content: Any, what: str) -> Any:
    """Return `content` as JSON reads it back, a deep copy.

    Raises:
        ValueError: `content` changes under the round trip, as a NumPy number,
            a date, a tuple or a key that is no str does.

    """
    try:
        read = json.loads(json.dumps(content))
    except (TypeError, ValueError):
        read = None
    if read != content:
        msg = f"{what} holds JSON values only: mappings with str keys, lists, str, numbers, bool and null."
        raise ValueError(msg)
    return read


class Registry[T](Mapping[str, T]):
    """Named entries stored in order under one key of `n.meta`, so copies and saved networks keep them."""

    META: ClassVar[str]
    NOUN: ClassVar[str]

    def __init__(self, n: Network) -> None:
        """Bind the registry to the network."""
        self.n = n

    def __repr__(self) -> str:
        """List the registered names."""
        return f"{type(self).__name__} {list(self)}"

    def _entries(self) -> dict[str, Any]:
        """Return the stored entries, by name in order.

        Raises:
            ValueError: `n.meta` holds something else than a mapping of names to mappings.

        """
        entries = self.n.meta.get(self.META, {})
        if not isinstance(entries, Mapping) or not all(
            isinstance(e, Mapping) for e in entries.values()
        ):
            msg = f"`n.meta[{self.META!r}]` maps each name to a mapping, not {type(entries).__name__}: {str(entries)[:200]}"
            raise ValueError(msg)
        return dict(entries)

    def _read(self, entry: Mapping[str, Any]) -> T:
        """Return what a stored entry gives."""
        raise NotImplementedError

    def _content(self, entry: Mapping[str, Any]) -> Any:
        """Return what tells a stored entry from another, by which a name is re-registered or merged."""
        return entry

    def __getitem__(self, name: str) -> T:
        """Return a registered entry.

        Raises:
            KeyError: None of this name is registered.

        """
        entries = self._entries()
        if name not in entries:
            msg = f"No {self.NOUN} {name!r} is registered."
            raise KeyError(msg)
        return self._read(entries[name])

    def __getattr__(self, name: str) -> T:
        """Return a registered entry."""
        if name.startswith("_") or name == "n" or name not in self:
            msg = f"No {self.NOUN} {name!r} is registered."
            raise AttributeError(msg)
        return self[name]

    def __contains__(self, name: object) -> bool:
        """Whether an entry of this name is registered."""
        return name in self._entries()

    def __iter__(self) -> Iterator[str]:
        """Iterate the registered names in order."""
        return iter(self._entries())

    def __len__(self) -> int:
        """Count the registered entries."""
        return len(self._entries())

    def __eq__(self, other: object) -> bool:
        """Whether both hold the same entries in the same order, since a later one wins."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return list(self._entries().items()) == list(other._entries().items())

    def _check_name(self, name: Any) -> None:
        """Refuse a name that is no identifier or that shadows an attribute of the registry.

        Raises:
            ValueError: The name is refused.

        """
        if not isinstance(name, str) or not name.isidentifier() or name in dir(self):
            msg = f"`{type(self).__name__}` names each entry by an identifier that is no attribute of it, not {name!r}."
            raise ValueError(msg)

    def _holds(self, name: str, content: Any) -> bool:
        """Return whether `name` is registered with `content`.

        Raises:
            ValueError: `name` is registered with other content.

        """
        entries = self._entries()
        if name not in entries:
            return False
        if self._content(entries[name]) != content:
            msg = f"Another {self.NOUN} {name!r} is registered."
            raise ValueError(msg)
        return True

    def _write(self, entries: dict[str, Any]) -> None:
        """Store `entries`, dropping the key of `n.meta` where there are none."""
        if entries:
            self.n.meta[self.META] = entries
        else:
            self.n.meta.pop(self.META, None)

    def _set(self, entries: dict[str, Any]) -> None:
        """Store `entries` where the build stack of the network holds with them.

        Raises:
            ValueError: The stack does not hold. The registry is left as it was.

        """
        from pypsa.optimization.spec.build import holding  # noqa: PLC0415

        before = dict(self.n.meta)
        self._write(entries)
        try:
            holding(self.n)
        except ValueError:
            self.n.meta.clear()
            self.n.meta.update(before)
            raise

    def _union(self, other: Registry[T]) -> dict[str, Any]:
        """Return the entries of both, those of `other` that this lacks after its own.

        Raises:
            ValueError: A name holds different content in both.

        """
        entries, others = self._entries(), other._entries()
        clash = [
            k
            for k, e in others.items()
            if k in entries and self._content(entries[k]) != self._content(e)
        ]
        if clash:
            msg = f"Both networks register a different {self.NOUN} under the names {clash}."
            raise ValueError(msg)
        return copy.deepcopy(
            entries | {k: e for k, e in others.items() if k not in entries}
        )


def registries(n: Network) -> tuple[Registry, ...]:
    """Return the registries of a network."""
    return n.composites, n.optimize.overrides


def united(n: Network, other: Network) -> dict[str, dict[str, Any]]:
    """Return the entries of the registries of both networks by key of `n.meta`, those of `other` after those of `n`.

    Raises:
        ValueError: A name holds different content in both, or the build
            stack does not hold with the entries of both.

    """
    from pypsa.optimization.spec.build import holding  # noqa: PLC0415

    unions = {
        mine.META: mine._union(theirs)
        for mine, theirs in zip(registries(n), registries(other), strict=True)
    }
    unions = {k: v for k, v in unions.items() if v}
    before = n.meta
    n.meta = before | unions
    try:
        holding(n)
    except ValueError as e:
        msg = f"The composite types and stored overrides of both networks do not build together: {e}"
        raise ValueError(msg) from e
    finally:
        n.meta = before
    return unions
