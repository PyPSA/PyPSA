# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Overrides stored on a network, which every build of `n.optimize()` lays over the formulation.

Each is a `mathspec.override` patch, as the argument `overrides` of
`n.optimize()` takes. It is stored in `n.meta["_pypsa"]`, so copies and saved
networks keep it, and a loaded network builds the model its author built.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pypsa._registry import Registry, jsonable
from pypsa.optimization.spec.build import _layer, holding, is_path, laid, written

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pypsa import Network
    from pypsa.optimization.spec.build import Layer, Override

META = "overrides"


def _label(name: str) -> str:
    return f"the stored override {name!r}"


def _builds_on(patch: Mapping[str, Any], touched: set[str], created: list[str]) -> bool:
    """Whether `patch` writes a key of `touched` or reads one of `created`."""
    if touched & written(patch).keys():
        return True
    text = json.dumps(patch)
    return any(re.search(rf"\b{re.escape(c)}\b", text) for c in created)


def _builds(n: Network) -> bool:
    """Whether the stack of `n` holds."""
    try:
        holding(n)
    except ValueError:
        return False
    return True


class Overrides(Registry[dict[str, Any]]):
    """The overrides stored on a network, read from `n.meta`.

    A build lays them over the formulation after the composite types and
    before the per-call `overrides` of `n.optimize()`, in the order they were
    registered, so a later one wins. They may edit, remove and add entries.
    Each entry stores the patch and the keys it edits and creates. A build
    refuses an override whose edited keys the formulation below it no longer
    declares, or whose created keys it now declares.

    Examples
    --------
    >>> n.optimize.overrides.register("full_load_hours.yaml")  # doctest: +SKIP
    >>> n.optimize.overrides.register(
    ...     {"constraints": {"Generator_p_ramp_limit_up": None}}, name="no_ramp"
    ... )  # doctest: +SKIP
    >>> n.optimize.overrides  # doctest: +SKIP
    Overrides ['full_load_hours', 'no_ramp']
    >>> n.optimize.overrides.no_ramp  # doctest: +SKIP
    {'constraints': {'Generator_p_ramp_limit_up': None}}
    >>> n.optimize.overrides.remove("no_ramp")  # doctest: +SKIP
    >>> n.optimize.overrides.clear()  # doctest: +SKIP

    """

    META = META
    NOUN = "override"

    def _read(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(entry["patch"])

    def _content(self, entry: Mapping[str, Any]) -> Any:
        return entry["patch"]

    def register(self, override: Override, name: str | None = None) -> dict[str, Any]:
        """Store an override from a YAML file, YAML text or mapping.

        The override is checked laid over the default formulation, the
        composite types and the stored overrides. An override of linearised
        unit commitment or of piecewise options stays per-call. Registering a
        name again with the same patch does nothing.

        Parameters
        ----------
        override : str | Path | dict
            A `mathspec.override` patch, as the argument `overrides` of
            `n.optimize()` takes.
        name : str, optional
            The name to store it under, by default the stem of the file.
            Text and a mapping take one.

        Returns
        -------
        dict
            The stored patch.

        Raises
        ------
        ValueError
            The name is missing, no identifier or taken by another patch;
            the patch holds other values than JSON or is stored under another
            name; or a stored override no longer holds.
        mathspec.errors.LanguageError
            The patch does not hold with the formulation.

        """
        if name is None and is_path(override):
            name = Path(override).stem
        if name is None:
            msg = "An override given as text or a mapping takes a `name`."
            raise ValueError(msg)
        self._check_name(name)
        patch = jsonable(_layer(override), f"The override {name!r}")
        if self._holds(name, patch):
            return self[name]
        twin = [k for k, e in self._entries().items() if e["patch"] == patch]
        if twin:
            msg = f"The override {name!r} is stored as {twin[0]!r}."
            raise ValueError(msg)
        keys = holding(self.n, [(_label(name), patch)])
        entries = written(patch).keys()
        stored = {
            "patch": patch,
            "edits": sorted(entries & keys),
            "creates": sorted(entries - keys),
        }
        self._write(self._entries() | {name: stored})
        return self[name]

    def remove(self, name: str) -> None:
        """Remove a stored override.

        Raises:
            KeyError: No override of this name is stored.
            ValueError: The stack held with it and does not without it, as a
                later override builds on it. Nothing is removed then.

        """
        entries = self._entries()
        if name not in entries:
            msg = f"No override {name!r} is stored."
            raise KeyError(msg)
        rest = {k: e for k, e in entries.items() if k != name}
        try:
            self._set(rest)
        except ValueError as e:
            if not _builds(self.n):
                self._write(rest)
                return
            touched = set(written(entries[name]["patch"]))
            created = [k.rpartition(".")[2] for k in entries[name]["creates"]]
            later = list(entries)[list(entries).index(name) + 1 :]
            dependents = [
                k for k in later if _builds_on(entries[k]["patch"], touched, created)
            ]
            if not dependents:
                raise
            msg = f"The overrides {dependents} build on {name!r}; remove them first."
            raise ValueError(msg) from e

    def clear(self) -> None:
        """Remove all stored overrides, also where they no longer build.

        Raises:
            ValueError: The stack held with them and does not without, as a
                composite type builds on them. Nothing is removed then.

        """
        try:
            self._set({})
        except ValueError:
            if _builds(self.n):
                raise
            self._write({})

    def layers(self, keys: set[str], plain: bool) -> tuple[list[Layer], set[str]]:
        """Return the stored patches, each with the label a refusal names it by, and the keys the spec declares under them.

        Parameters
        ----------
        keys : set of str
            The keys the spec declares below the stored overrides, as
            `section.key`.
        plain : bool
            Whether the build is without linearised unit commitment and
            piecewise options, as registration checks.

        Raises
        ------
        ValueError
            An override edits a key the spec below it no longer declares,
            which would add the entry, or creates one it now declares, which
            would edit it.

        """
        layers = []
        for name, entry in self._entries().items():
            gone = [k for k in entry["edits"] if k not in keys]
            taken = [k for k in entry["creates"] if k in keys]
            drift = []
            if gone:
                drift.append(
                    f"edits {gone}, which the spec below it no longer declares"
                )
            if taken:
                drift.append(f"creates {taken}, which the spec below it now declares")
            if drift:
                advice = (
                    "The formulation or the composite types differ from those it was registered with; remove it, check it and register it again."
                    if plain
                    else "Stored overrides are checked without linearised unit commitment and piecewise options; pass it to this build as a per-call override instead."
                )
                msg = f"The stored override {name!r} {' and '.join(drift)}. {advice}"
                raise ValueError(msg)
            keys = laid(keys, entry["patch"])
            layers.append((_label(name), entry["patch"]))
        return layers, keys
