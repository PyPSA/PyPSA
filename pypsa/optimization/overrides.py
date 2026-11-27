# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Overrides stored on a network, which every build of `n.optimize()` applies to the formulation.

Each is a `mathspec.override` patch, as the argument `overrides` of
`n.optimize()` takes. The network keeps it, so copies and saved networks keep
it, and a loaded network builds the model its author built. A saved network
holds it under the meta key `"_pypsa"`.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pypsa import _registry
from pypsa._registry import OVERRIDES, jsonable
from pypsa.optimization.spec import (
    fingerprint,
    is_path,
    laid,
    read_override,
    written,
)

if TYPE_CHECKING:
    from pypsa import Network
    from pypsa.optimization.spec import Layer, Override


class Overrides(Mapping[str, dict[str, Any]]):
    """The overrides stored on a network.

    `n.optimize.overrides` reads as a mapping of each name to its patch, in
    the order the overrides were registered. Change it with `register`,
    `unregister` and `clear`.

    Every build applies the stored overrides in their order, after the
    composite types and before the `overrides` of `n.optimize()`, so a
    later one wins. An override may change, remove and add entries of the
    formulation. PyPSA records which entries it changes and adds. A build
    refuses an override when an entry it changes no longer exists or PyPSA
    changed it, or when an entry it adds exists already.

    Examples
    --------
    >>> n = pypsa.Network()
    >>> no_ramp = {"constraints": {"Generator_p_ramp_limit_up": None}}
    >>> n.optimize.overrides.register(no_ramp, name="no_ramp")
    >>> n.optimize.overrides
    Overrides ['no_ramp']
    >>> n.optimize.overrides["no_ramp"]
    {'constraints': {'Generator_p_ramp_limit_up': None}}
    >>> n.optimize.overrides.unregister("no_ramp")
    >>> len(n.optimize.overrides)
    0

    """

    def __init__(self, n: Network) -> None:
        """Bind the registry to the network."""
        self._n = n

    def __getitem__(self, name: str) -> dict[str, Any]:
        """Return the patch of a stored override.

        Parameters
        ----------
        name : str
            The name of the override.

        Returns
        -------
        dict
            A copy of the patch.

        Raises
        ------
        KeyError
            No override of this name is registered.

        """
        return copy.deepcopy(_registry.entry(self._n, OVERRIDES, name)["patch"])

    def __iter__(self) -> Iterator[str]:
        """Iterate the names of the stored overrides in order."""
        return iter(_registry.entries(self._n, OVERRIDES))

    def __len__(self) -> int:
        """Count the stored overrides."""
        return len(_registry.entries(self._n, OVERRIDES))

    def __eq__(self, other: object) -> bool:
        """Whether both hold the same overrides in the same order."""
        if not isinstance(other, Overrides):
            return NotImplemented
        return _registry.equal(self._n, other._n, OVERRIDES)

    def __repr__(self) -> str:
        """List the names of the stored overrides."""
        return f"Overrides {list(self)}"

    def register(
        self, override: Override, name: str | None = None, overwrite: bool = False
    ) -> None:
        """Store an override on the network.

        PyPSA checks the override against the default formulation, the
        composite types and the stored overrides. An override of linearised
        unit commitment or of piecewise options must stay in the `overrides`
        of `n.optimize()`. If PyPSA refuses the override, the network stays
        unchanged.

        Parameters
        ----------
        override : str | Path | dict
            A `mathspec.override` patch as the argument `overrides` of
            `n.optimize()` takes it: a YAML file, YAML text or a mapping.
        name : str, optional
            The name to register it under, a Python identifier. A file takes
            its stem by default. Text and a mapping need a `name`.
        overwrite : bool, default False
            Whether to replace a stored override of this name. The new
            override takes its place in the order.

        Raises
        ------
        FileNotFoundError
            A `str` names no file and is no YAML text of a mapping.
        ValueError
            The name is missing or not an identifier, an override of this name
            is registered and `overwrite` is False, the patch holds a value
            that JSON cannot store, the same patch is registered under another
            name, or a stored override no longer fits.
        mathspec.errors.LanguageError
            The patch does not fit the formulation, as when it removes an entry
            that does not exist, or a later stored override does not fit with
            it.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> up = {"constraints": {"Generator_p_ramp_limit_up": None}}
        >>> down = {"constraints": {"Generator_p_ramp_limit_down": None}}
        >>> n.optimize.overrides.register(up, name="no_ramp")
        >>> n.optimize.overrides.register(down, name="no_ramp", overwrite=True)
        >>> n.optimize.overrides["no_ramp"]
        {'constraints': {'Generator_p_ramp_limit_down': None}}

        """
        layer = read_override(override)
        stem = Path(override).stem if name is None and is_path(override) else None
        name = stem if name is None else name
        if name is None:
            msg = "Pass `name` to register an override given as text or a mapping."
            raise ValueError(msg)
        if not isinstance(name, str) or not name.isidentifier():
            msg = f"{name!r} is not a valid override name. Use a Python identifier, such as 'no_ramps'."
            if stem is not None:
                msg += " Pass `name` to register the file under another name."
            raise ValueError(msg)
        _registry.check_free(self._n, OVERRIDES, name, overwrite)
        patch = jsonable(layer, f"The override {name!r}")
        stored = _registry.entries(self._n, OVERRIDES)
        twin = [k for k, e in stored.items() if k != name and e["patch"] == patch]
        if twin:
            msg = f"The override {name!r} is registered already as {twin[0]!r}. Use that name, or unregister it first."
            raise ValueError(msg)
        keys = _registry.below(self._n, OVERRIDES, name)
        entries = written(patch).keys()
        content = {
            "patch": patch,
            "edits": {k: fingerprint(k) for k in sorted(entries & keys)},
            "creates": sorted(entries - keys),
        }
        _registry.register(self._n, OVERRIDES, name, content)

    def unregister(self, name: str) -> None:
        """Unregister a stored override, also one that no longer fits the formulation.

        Parameters
        ----------
        name : str
            The name of the override.

        Raises
        ------
        KeyError
            No override of this name is registered.
        ValueError
            A later override depends on it. Nothing is unregistered then.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> up = {"constraints": {"Generator_p_ramp_limit_up": None}}
        >>> n.optimize.overrides.register(up, name="no_ramp")
        >>> n.optimize.overrides.unregister("no_ramp")
        >>> n.optimize.overrides
        Overrides []

        """
        _registry.unregister(self._n, OVERRIDES, name)

    def clear(self) -> None:
        """Unregister all stored overrides, also those that no longer fit the formulation.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> up = {"constraints": {"Generator_p_ramp_limit_up": None}}
        >>> n.optimize.overrides.register(up, name="no_ramp")
        >>> n.optimize.overrides.clear()
        >>> n.optimize.overrides
        Overrides []

        """
        _registry.clear(self._n, OVERRIDES)


def layers(n: Network, keys: set[str], plain: bool) -> tuple[list[Layer], set[str]]:
    """Return the stored patches of `n`, each with the label a refusal names it by, and the keys the spec declares under them.

    Parameters
    ----------
    n : Network
        The network that stores the overrides.
    keys : set of str
        The keys the spec declares below the stored overrides, as
        `section.key`.
    plain : bool
        Whether the build is without linearised unit commitment and
        piecewise options, as registration checks.

    Raises
    ------
    ValueError
        An override edits a key the spec below it no longer declares, which
        would add the entry, or one PyPSA changed since it was registered; or
        it creates one it now declares, which would edit it.

    """
    stored = []
    for name, entry in _registry.entries(n, OVERRIDES).items():
        if not {"patch", "edits", "creates"} <= entry.keys():
            msg = f"The stored override {name!r} lacks `patch`, `edits` or `creates`. Register its patch again with `overwrite=True`."
            raise ValueError(msg)
        gone = [k for k in entry["edits"] if k not in keys]
        taken = [k for k in entry["creates"] if k in keys]
        changed = [
            k for k, h in entry["edits"].items() if k in keys and h != fingerprint(k)
        ]
        drift = []
        if gone:
            drift.append(f"changes {gone}, which no longer exist")
        if taken:
            drift.append(f"adds {taken}, which exist already")
        if changed:
            drift.append(f"changes {changed}, which PyPSA now defines differently")
        if drift:
            advice = (
                "The formulation or the composite types changed since you registered it. Check the override and register it again with `overwrite=True`."
                if plain or changed
                else "Stored overrides must work without linearised unit commitment and piecewise options. Pass this override to `n.optimize(overrides=[...])` instead."
            )
            msg = f"The stored override {name!r} {' and '.join(drift)}. {advice}"
            raise ValueError(msg)
        keys = laid(keys, entry["patch"])
        stored.append((f"the stored override {name!r}", entry["patch"]))
    return stored, keys
