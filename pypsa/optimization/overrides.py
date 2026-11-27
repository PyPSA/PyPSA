# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Overrides stored on a network, which every build of `n.optimize()` lays over the formulation.

Each is a `mathspec.override` patch, as the argument `overrides` of
`n.optimize()` takes. The network keeps it, so copies and saved networks keep
it, and a loaded network builds the model its author built. A saved network
holds it under the meta key `"_pypsa"`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pypsa._registry import Registry, builds_on, jsonable
from pypsa.optimization.spec import (
    fingerprint,
    holding,
    is_path,
    laid,
    read_override,
    written,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pypsa.optimization.spec import Layer, Override

META = "overrides"


def _label(name: str) -> str:
    return f"the stored override {name!r}"


class Overrides(Registry[dict[str, Any]]):
    """The overrides stored on a network.

    A build lays them over the formulation after the composite types and
    before the per-call `overrides` of `n.optimize()`, in the order they were
    registered, so a later one wins. They may edit, remove and add entries.
    Each entry stores the patch, the keys it edits with a fingerprint of their
    content, and the keys it creates. A build refuses an override whose
    edited keys the formulation below it no longer declares or PyPSA changed,
    or whose created keys it now declares.

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
        FileNotFoundError
            A `str` names no file and is no YAML text of a mapping.
        ValueError
            The name is missing, no identifier or taken by another patch;
            the patch holds other values than JSON or is stored under another
            name; or a stored override no longer holds.
        mathspec.errors.LanguageError
            The patch does not hold with the formulation.

        """
        layer = read_override(override)
        stem = None
        if name is None and is_path(override):
            name = stem = Path(override).stem
        if name is None:
            msg = "An override given as text or a mapping takes a `name`."
            raise ValueError(msg)
        try:
            self._check_name(name)
        except ValueError as e:
            if stem is None:
                raise
            msg = f"{e} Pass `name` to store the file under another name."
            raise ValueError(msg) from e
        patch = jsonable(layer, f"The override {name!r}")
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
            "edits": {k: fingerprint(k) for k in sorted(entries & keys)},
            "creates": sorted(entries - keys),
        }
        self._write(self._entries() | {name: stored})
        return self[name]

    def remove(self, name: str) -> None:
        """Remove a stored override, also where it no longer builds.

        Raises:
            KeyError: No override of this name is stored.
            ValueError: A later override builds on it, or the stack held with
                it and does not without it. Nothing is removed then.

        """
        self._remove(name)

    def _dependents(self, name: str, entries: dict[str, Any]) -> list[str]:
        touched = set(written(entries[name]["patch"]))
        created = [k.rpartition(".")[2] for k in entries[name]["creates"]]
        later = list(entries)[list(entries).index(name) + 1 :]
        return [k for k in later if builds_on(entries[k]["patch"], touched, created)]

    def clear(self) -> None:
        """Remove all stored overrides, also where they no longer build.

        Raises:
            ValueError: The stack held with them and does not without, as a
                composite type builds on them. Nothing is removed then.

        """
        try:
            self._set({})
        except ValueError:
            if self._builds():
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
            which would add the entry, or one PyPSA changed since it was
            registered; or it creates one it now declares, which would edit it.

        """
        layers = []
        for name, entry in self._entries().items():
            if not {"patch", "edits", "creates"} <= entry.keys():
                msg = f"The stored override {name!r} lacks the `patch`, `edits` and `creates` that `register` stores; remove it and register its patch again."
                raise ValueError(msg)
            gone = [k for k in entry["edits"] if k not in keys]
            taken = [k for k in entry["creates"] if k in keys]
            changed = [
                k
                for k, h in entry["edits"].items()
                if k in keys and h != fingerprint(k)
            ]
            drift = []
            if gone:
                drift.append(
                    f"edits {gone}, which the spec below it no longer declares"
                )
            if taken:
                drift.append(f"creates {taken}, which the spec below it now declares")
            if changed:
                drift.append(f"edits {changed}, which PyPSA changed since")
            if drift:
                advice = (
                    "The formulation or the composite types differ from those it was registered with; remove it, check it and register it again."
                    if plain or changed
                    else "Stored overrides are checked without linearised unit commitment and piecewise options; pass it to this build as a per-call override instead."
                )
                msg = f"The stored override {name!r} {' and '.join(drift)}. {advice}"
                raise ValueError(msg)
            keys = laid(keys, entry["patch"])
            layers.append((_label(name), entry["patch"]))
        return layers, keys
