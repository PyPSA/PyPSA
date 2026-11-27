# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The composite types and stored overrides a network registers: their stored state and every change to it.

A network keeps them in `n._registries`, by kind and by name in the order they
were registered. A build lays them over the formulation in that order, the
composite types before the stored overrides, so an entry may build on any
entry laid before it. Every change is tried first: the network keeps it only
where the default formulation holds with it. A saved network holds the
registries under the meta key `"_pypsa"`.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pypsa import Network
    from pypsa.network.abstract import _NetworkABC

ROOT = "_pypsa"
COMPOSITES = "composites"
OVERRIDES = "overrides"
#: What an entry of each kind is called, the kinds in the order a build lays them.
NOUNS = {COMPOSITES: "composite type", OVERRIDES: "override"}
#: The key of a stored entry of each kind that holds the math its author wrote.
MATH = {COMPOSITES: "math", OVERRIDES: "patch"}


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


def entries(n: Network, kind: str) -> dict[str, Any]:
    """Return the stored entries of a kind, by name in the order they were registered.

    Raises:
        ValueError: The registries of the network, or its entries of this
            kind, are no mapping of names to mappings, as a saved network
            with a malformed meta key `"_pypsa"` gives.

    """
    root = n._registries
    nested = isinstance(root, Mapping)
    stored = root.get(kind, {}) if nested else root
    if not isinstance(stored, Mapping) or not all(
        isinstance(e, Mapping) for e in stored.values()
    ):
        where = f"[{ROOT!r}][{kind!r}]" if nested else f"[{ROOT!r}]"
        msg = f"The saved meta {where} maps each name to a mapping, not {type(stored).__name__}: {str(stored)[:200]}"
        raise ValueError(msg)
    return dict(stored)


def entry(n: Network, kind: str, name: str) -> Any:
    """Return the stored entry of a name.

    Raises:
        KeyError: None of this name is registered.

    """
    stored = entries(n, kind)
    if name not in stored:
        msg = f"No {NOUNS[kind]} {name!r} is registered."
        raise KeyError(msg)
    return stored[name]


def equal(n: Network, other: Network, kind: str) -> bool:
    """Whether both networks hold the same entries of a kind in the same order, since a later one wins."""
    return list(entries(n, kind).items()) == list(entries(other, kind).items())


def check_free(n: Network, kind: str, name: str, overwrite: bool) -> None:
    """Refuse a name that is registered, unless its entry is to be replaced.

    Raises:
        ValueError: The name is registered and `overwrite` is False.

    """
    if name in entries(n, kind) and not overwrite:
        msg = f"The {NOUNS[kind]} {name!r} is registered already. Pass `overwrite=True` to replace it."
        raise ValueError(msg)


def below(n: Network, kind: str, name: str) -> set[str]:
    """Return the keys the default formulation declares below the entry `name` of a kind, or below a new entry.

    Raises:
        ValueError: The formulation does not hold with the entries below it.

    """
    return _keys(n, _lower(n, kind, name))


def register(n: Network, kind: str, name: str, content: Any) -> None:
    """Register `content` under a name, in the place of the entry of that name where there is one.

    Raises:
        ValueError: The default formulation does not hold with it, as where
            an entry laid after it builds on what it replaces. Nothing is
            registered then.

    """
    root = _with(n, kind, entries(n, kind) | {name: content})
    _keys(n, root)
    n._registries = root


def unregister(n: Network, kind: str, name: str) -> None:
    """Unregister an entry, also where the formulation does not hold with it.

    Raises:
        KeyError: None of this name is registered.
        ValueError: Something needs the entry; `_drop` lists the cases.
            Nothing is unregistered then.

    """
    entry(n, kind, name)
    _drop(n, kind, [name])


def clear(n: Network, kind: str) -> None:
    """Unregister all entries of a kind, also where the formulation does not hold with them.

    Raises:
        ValueError: Something needs one of them; `_drop` lists the cases.
            Nothing is unregistered then.

    """
    _drop(n, kind, list(entries(n, kind)))


def _drop(n: Network, kind: str, names: Collection[str]) -> None:
    """Unregister the entries `names` of a kind unless something needs them.

    Raises:
        ValueError: Instances of a composite type exist; an entry that stays,
            a later one of its kind or a stored override, builds on one of
            them; or the formulation held with them and does not without.

    """
    if kind == COMPOSITES:
        from pypsa.composites import instances  # noqa: PLC0415

        for name in names:
            if found := instances(n, name):
                msg = f"The {name} instances {found} exist; remove them first with `n.remove({name!r}, ...)`."
                raise ValueError(msg)
    stored = entries(n, kind)
    rest = _with(n, kind, {k: e for k, e in stored.items() if k not in names})
    try:
        _keys(n, rest)
    except ValueError as e:
        for name in names:
            healthy = _builds(n, _lower(n, kind, name, with_it=True))
            if healthy and (dependents := _dependents(n, kind, name, names)):
                msg = f"{dependents} build on {name!r}; unregister them first."
                raise ValueError(msg) from e
        if _builds(n, n._registries):
            raise
    n._registries = rest


def _dependents(n: Network, kind: str, name: str, gone: Collection[str]) -> list[str]:
    """Return the entries that a build lays after the entry `name` and that build on it, without those of its kind in `gone`.

    An entry builds on it where its math writes a key that `name` writes, or
    names a key that `name` creates.
    """
    from pypsa.optimization.spec import written  # noqa: PLC0415

    stored = entries(n, kind)
    if kind == COMPOSITES:
        from pypsa.composites import CompositeType  # noqa: PLC0415

        touched = set(written(CompositeType.read(stored[name]).patch()))
        created = touched
    else:
        touched = set(written(stored[name]["patch"]))
        created = set(stored[name]["creates"])
    words = [key.rpartition(".")[2] for key in created]
    names = list(stored)
    later = [
        (k, stored[k].get(MATH[kind], {}))
        for k in names[names.index(name) + 1 :]
        if k not in gone
    ]
    for after in list(NOUNS)[list(NOUNS).index(kind) + 1 :]:
        later += [(k, e.get(MATH[after], {})) for k, e in entries(n, after).items()]
    return [
        k
        for k, math in later
        if touched & written(math).keys()
        or any(re.search(rf"\b{re.escape(w)}\b", json.dumps(math)) for w in words)
    ]


def _lower(n: Network, kind: str, name: str, with_it: bool = False) -> dict[str, Any]:
    """Return the registries a build lays before the entry `name` of a kind, with the entry on request.

    Where no entry has this name, they are those it lays before a new one.
    """
    root: dict[str, Any] = {}
    for k in NOUNS:
        root[k] = entries(n, k)
        if k == kind:
            names = list(root[k])
            end = names.index(name) + with_it if name in names else len(names)
            root[k] = {x: root[k][x] for x in names[:end]}
            break
    return root


def _with(n: Network, kind: str, stored: Mapping[str, Any]) -> dict[str, Any]:
    """Return the registries of `n` with `stored` as its entries of a kind, the kind left out where there are none."""
    root = {k: v for k, v in n._registries.items() if k != kind}
    return root | {kind: dict(stored)} if stored else root


def _keys(n: Network, root: dict[str, Any]) -> set[str]:
    """Return the keys the default formulation declares with the registries `root`, which `n` does not keep.

    Raises:
        ValueError: The formulation does not hold with them.

    """
    from pypsa.optimization.spec import holding  # noqa: PLC0415

    before = n._registries
    n._registries = root
    try:
        return holding(n)
    finally:
        n._registries = before


def _builds(n: Network, root: dict[str, Any]) -> bool:
    """Whether the default formulation holds with the registries `root`."""
    try:
        _keys(n, root)
    except ValueError:
        return False
    return True


def _told(kind: str, stored: Any) -> Any:
    """Return what tells a stored entry from another of its name: the patch of an override, all of a composite type.

    The `edits` and `creates` of an override follow from what it was registered over.
    """
    return stored[MATH[kind]] if kind == OVERRIDES else stored


def united(n: Network, other: Network) -> dict[str, dict[str, Any]]:
    """Return the registries of both networks as a network keeps them, the entries of `other` after those of `n`.

    Raises:
        ValueError: A name holds a different entry in both, an override told
            by its patch, or the default formulation does not hold with the
            entries of both.

    """
    root = {}
    for kind, noun in NOUNS.items():
        mine, theirs = entries(n, kind), entries(other, kind)
        shared = mine.keys() & theirs.keys()
        clash = sorted(
            k for k in shared if _told(kind, mine[k]) != _told(kind, theirs[k])
        )
        if clash:
            msg = f"Both networks register a different {noun} under the names {clash}."
            raise ValueError(msg)
        if both := mine | {k: e for k, e in theirs.items() if k not in mine}:
            root[kind] = copy.deepcopy(both)
    try:
        _keys(n, root)
    except ValueError as e:
        msg = f"The composite types and stored overrides of both networks do not build together: {e}"
        raise ValueError(msg) from e
    return root


def carry(n: _NetworkABC, new: _NetworkABC) -> None:
    """Give `new` a deep copy of the registries of `n`.

    Parameters
    ----------
    n : Network
        The network to take the composite types and stored overrides from.
    new : Network
        The network built from `n`, which keeps a copy that shares nothing with `n`.

    """
    new._registries = copy.deepcopy(n._registries)
