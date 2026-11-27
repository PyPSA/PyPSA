# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The spec of `n.optimize()` and the layers a build lays over it.

It owns the rendered spec and its caches, the overrides as layers, the stack
of piecewise options, composite types and stored overrides of a network, the
keys a spec declares under its layers, and the fingerprints of its entries.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
import re
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

from pypsa.optimization.spec._piecewise import _piecewise_patch
from pypsa.optimization.spec._render import Names, Scalings, formulated, render

if TYPE_CHECKING:
    from collections.abc import Sequence
    from importlib.resources.abc import Traversable

    from mathspec.spec import Spec

    from pypsa import Network
    from pypsa.optimization.piecewise import PiecewiseOptions

SPEC = files("pypsa.optimization.spec")


@functools.cache
def _unit_dims() -> tuple[str, ...]:
    """Return the spec's unit dims, one per component with a formulation, which every entry lists first."""
    import pypsa  # noqa: PLC0415

    return tuple(sorted(c._unit_dim for c in formulated(pypsa.Network())))


@functools.cache
def _rendered(linearized_unit_commitment: bool) -> tuple[Spec, Names, Scalings]:
    """Return the spec, its PyPSA names and its scaling rules, rendered once per process."""
    return render(SPEC, _patch(linearized_unit_commitment))


#: A layer over the spec: a YAML file, YAML text or a mapping, written as a `mathspec.override` patch.
Override = str | Path | Mapping[str, Any]


def is_path(override: Override) -> TypeGuard[str | Path]:
    """Return whether *override* names a file: a path, or a `str` that names an existing file."""
    return isinstance(override, Path) or (
        isinstance(override, str) and os.path.isfile(override)  # noqa: PTH113
    )


def read_override(override: Override) -> dict[str, Any]:
    """Return *override* as the mapping it states.

    Raises:
        FileNotFoundError: A `str` names no file and is no YAML text of a mapping.
        TypeError: *override* is not a path, YAML text or mapping, or states
            no mapping.

    """
    import yaml  # noqa: PLC0415

    if isinstance(override, Mapping):
        return dict(override)
    if is_path(override):
        stated = yaml.safe_load(Path(override).read_text())
    elif isinstance(override, str):
        error = None
        try:
            stated = yaml.safe_load(override)
        except yaml.YAMLError as e:
            stated, error = None, e
        if not isinstance(stated, Mapping):
            msg = f"No file {override!r} exists, and the string is not YAML text of an override either."
            msg += f" YAML reads: {error}" if error else ""
            raise FileNotFoundError(msg) from error
    else:
        msg = f"An override is a YAML path, YAML text or a mapping, not {type(override).__name__}."
        raise TypeError(msg)
    if not isinstance(stated, Mapping):
        msg = f"An override states a mapping, not {type(stated).__name__}."
        raise TypeError(msg)
    return dict(stated)


def _expanded(s: Spec) -> tuple[Spec, Spec]:
    return s, s.expand("piecewise")


@functools.cache
def _base(linearized_unit_commitment: bool) -> tuple[Spec, Spec]:
    return _expanded(_rendered(linearized_unit_commitment)[0])


#: The most patched formulations kept in memory; the oldest go first.
KEEP = 8
_PATCHED: dict[str, tuple[Spec, Spec]] = {}
Layer = tuple[str, dict[str, Any]]


def _formulation(
    linearized_unit_commitment: bool = False, layers: Sequence[Layer] = ()
) -> tuple[Spec, Spec]:
    """Return the spec with *layers* laid over it in turn, and its piecewise curves written out, which linopy builds from.

    Raises:
        mathspec.errors.LanguageError: A layer does not hold; the message names it by its label.

    """
    import mathspec  # noqa: PLC0415
    from mathspec.errors import LanguageError  # noqa: PLC0415

    base = _base(linearized_unit_commitment)
    if not layers:
        return base
    labels = [label for label, _ in layers]
    patches = [patch for _, patch in layers]
    key = json.dumps(
        [linearized_unit_commitment, patches], sort_keys=True, default=repr
    )
    if key not in _PATCHED:
        try:
            patched = mathspec.override(base[0], patches)
        except LanguageError as e:
            named = re.sub(
                r"(?:patch )?'#(\d+)'", lambda m: labels[int(m[1]) - 1], str(e)
            )
            raise LanguageError(named) from None
        _PATCHED[key] = _expanded(patched)
        while len(_PATCHED) > KEEP:
            del _PATCHED[next(iter(_PATCHED))]
    return _PATCHED[key]


def _piecewise_layers(
    linearized_unit_commitment: bool, piecewise_options: Sequence[PiecewiseOptions]
) -> list[Layer]:
    """Return the layer of the piecewise options, none without options."""
    if not piecewise_options:
        return []
    base = _base(linearized_unit_commitment)[0]
    return [("the piecewise options", _piecewise_patch(base, piecewise_options))]


def _per_call(overrides: Sequence[Override]) -> list[Layer]:
    """Return the layer of each per-call override, a path named by itself.

    Raises:
        TypeError: `overrides` is a single override, not a list of them.

    """
    if isinstance(overrides, str | Path | Mapping):
        msg = "`overrides` takes a list of overrides; put a single one in a list."
        raise TypeError(msg)
    return [
        (_per_call_label(k, o), read_override(o)) for k, o in enumerate(overrides, 1)
    ]


def _per_call_label(place: int, override: Override) -> str:
    if is_path(override):
        return f"the override {str(override)!r}"
    return f"the override #{place} of `overrides`"


def written(patch: Mapping[str, Any]) -> dict[str, Any]:
    """Return what a patch writes by `<section>.<key>`, and the objective by `objective`."""
    entries = {
        f"{section}.{key}": entry
        for section, keyed in patch.items()
        if section != "objective" and isinstance(keyed, Mapping)
        for key, entry in keyed.items()
    }
    return entries | ({"objective": patch["objective"]} if "objective" in patch else {})


def laid(keys: set[str], patch: Mapping[str, Any]) -> set[str]:
    """Return the keys a spec declares once `patch` is laid over it."""
    entries = written(patch)
    return (keys | entries.keys()) - {k for k, e in entries.items() if e is None}


@functools.cache
def _keys(linearized_unit_commitment: bool) -> frozenset[str]:
    """Return the keys of the spec, as `written` names them."""
    return frozenset(written(_base(linearized_unit_commitment)[0].to_dict()))


@functools.cache
def fingerprint(key: str) -> str:
    """Return a hash of the entry `key` of the default spec as `written` names it, without its description."""
    entry = written(_base(False)[0].to_dict()).get(key)
    if isinstance(entry, Mapping):
        entry = {k: v for k, v in entry.items() if k != "description"}
    text = json.dumps(entry, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _stack(
    n: Network,
    linearized_unit_commitment: bool,
    piecewise_options: Sequence[PiecewiseOptions] = (),
) -> tuple[list[Layer], set[str]]:
    """Return the layers a build of `n` lays over the spec before per-call overrides, and the keys the spec declares under them.

    They are the piecewise options, the composite types and the stored overrides, in turn.

    Raises:
        ValueError: A stored override no longer edits or creates what it was
            registered to, or PyPSA changed an entry it edits.

    """
    plain = not linearized_unit_commitment and not piecewise_options
    if not plain:
        _stack(n, linearized_unit_commitment=False)
    layers = _piecewise_layers(linearized_unit_commitment, piecewise_options)
    layers += [
        (f"the composite type {k!r}", p) for k, p in n.composites.patches().items()
    ]
    keys = functools.reduce(
        laid, [p for _, p in layers], set(_keys(linearized_unit_commitment))
    )
    stored, keys = n.optimize.overrides.layers(keys, plain)
    return layers + stored, keys


def holding(n: Network, layers: Sequence[Layer] = ()) -> set[str]:
    """Return the keys the default spec declares under *layers* laid over the stack of `n`.

    Raises:
        ValueError: The default formulation does not hold them.

    """
    stack, keys = _stack(n, linearized_unit_commitment=False)
    _formulation(False, [*stack, *layers])
    return keys


def spec(
    linearized_unit_commitment: bool = False,
    piecewise_options: Sequence[PiecewiseOptions] = (),
    overrides: Sequence[Override] = (),
) -> Spec:
    """Return the spec of `n.optimize()` for a network without composites and stored overrides, patched for linearised unit commitment and piecewise options on request, with *overrides* laid over it in turn."""
    piecewise = _piecewise_layers(linearized_unit_commitment, piecewise_options)
    layers = [*piecewise, *_per_call(overrides)]
    return _formulation(linearized_unit_commitment, layers)[0]


def _patch(linearized_unit_commitment: bool) -> Traversable | None:
    """Return the patch folder of linearised unit commitment, where asked for."""
    return SPEC / "linearized_uc" if linearized_unit_commitment else None


def spec_names(linearized_unit_commitment: bool = False) -> Names:
    """Return the PyPSA names the spec states, patched for linearised unit commitment on request."""
    return copy.deepcopy(_rendered(linearized_unit_commitment)[1])


def spec_scalings(linearized_unit_commitment: bool = False) -> Scalings:
    """Return the scaling rules the spec states, patched for linearised unit commitment on request."""
    return dict(_rendered(linearized_unit_commitment)[2])


def _repeated(n: Network, per_call: Sequence[Layer]) -> list[str]:
    """Return where a per-call override removes an entry a stored override removes already."""
    stored = {k: _removals(p) for k, p in n.optimize.overrides.items()}
    return [
        f"{label} removes {both}, which the stored override {k!r} removes already"
        for label, patch in per_call
        for k, removed in stored.items()
        if (both := sorted(_removals(patch) & removed))
    ]


def _removals(patch: Mapping[str, Any]) -> set[str]:
    """Return the keys a patch removes."""
    return {k for k, entry in written(patch).items() if entry is None}
