# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Render a spec from its system file, each component's traits and its own file.

A file under `traits/` states a formulation once for every component that
declares the trait in `_formulation`. It is a spec of its own, which reads
under `given:` what another file declares, written over placeholder names.
`mathspec.rename` gives it the names of each component. A word of a name,
between `_` or `-`, is a placeholder if it is one of these:

- `Unit`: the component's name, as `StorageUnit`.
- `UNIT`: its unit dimension, as `storage_unit`.
- `NOM`: its nominal attribute, as `p_nom`.
- `P`: its power variable, `p` or `p_dispatch`, where it has one.
- `LEVEL`: its storage level variable, `e` or `state_of_charge`, and
  `CYCLIC` its cyclic flag, where it stores energy.
- `COEF`: the attribute that scales `p` at each port, `efficiency` or
  `rate`, where it has ports.

A component lacks a placeholder it has no value for, so a trait that uses
one fails to render for it. A file under `components/` holds, under the
component's own names, what it shares with no other.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

#: A PyPSA name: one string, or one per transmission loss mode.
Name = str | dict[str, str]
#: The PyPSA names of the spec's entries, by section and entry.
Names = dict[str, dict[str, Name]]
#: The sections whose entries may carry a `name:`.
NAMED = ("variables", "constraints", "piecewise")
#: A word of a name, between `_` or `-`.
WORD = re.compile(r"[^\W_]+")
#: The words that stand for a value of the component.
PLACEHOLDERS = frozenset({"Unit", "UNIT", "NOM", "P", "LEVEL", "CYCLIC", "COEF"})

if TYPE_CHECKING:
    from collections.abc import Callable
    from importlib.resources.abc import Traversable

    from mathspec.spec import Spec

    from pypsa import Network
    from pypsa.components.components import Components


def formulated(n: Network, trait: str = "") -> list[Components]:
    """Return the components of *n* that declare a formulation, or the trait *trait*."""
    components = [c for c in n.components.values() if c._formulation]
    return [c for c in components if not trait or trait in c._formulation]


def symbols(c: Components) -> dict[str, str]:
    """Return what the placeholders of a trait stand for in component *c*."""
    found = {"Unit": c.name, "UNIT": c._unit_dim, "NOM": c._operational_attrs["nom"]}
    power = [v for v in c._operational_variables if v.startswith("p")]
    if power:
        found["P"] = power[0]
    if c._level_attrs:
        initial, found["CYCLIC"] = c._level_attrs
        found["LEVEL"] = initial.removesuffix("_initial")
    if coefficient := getattr(c, "_coefficient_attr", None):
        found["COEF"] = coefficient
    return found


def renamer(c: Components) -> Callable[[str], str]:
    """Return the rule that gives a name over placeholders its name in component *c*.

    Raises:
        KeyError: The name holds a placeholder *c* has no value for.

    """
    found = symbols(c)

    def word(match: re.Match[str]) -> str:
        return found[match[0]] if match[0] in PLACEHOLDERS else match[0]

    return lambda name: WORD.sub(word, name)


def _load(path: Traversable, rename: Callable[[str], str]) -> tuple[dict, Names]:
    """Return the spec in *path* without its `name:` fields, and those names renamed by *rename*.

    The closed mathspec schema refuses `name:`, and `mathspec.rename` keeps
    the label of a piecewise link, so both are renamed here.
    """
    import yaml  # noqa: PLC0415

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    data = yaml.load(path.read_text(), Loader=loader)
    names: Names = {}
    for section in NAMED:
        entries = data.get(section, {})
        named = {k: v.pop("name") for k, v in entries.items() if "name" in v}
        names[section] = {
            rename(k): rename(v)
            if isinstance(v, str)
            else {m: rename(n) for m, n in v.items()}
            for k, v in named.items()
        }
    for entry in data.get("piecewise", {}).values():
        if "links" in entry:
            entry["links"] = {rename(k): v for k, v in entry["links"].items()}
    return data, names


def _claim(entry: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in entry.items() if k != "description"}


def _merge(spec: dict[str, Any], fragment: dict[str, Any], origin: str) -> None:
    """Add the entries of *fragment* to *spec*, but its `given:`.

    A dimension or relation may repeat where it states what it stated before.
    """
    for section, body in fragment.items():
        if not isinstance(body, dict):
            spec.setdefault(section, body)
            continue
        if section == "given":
            continue
        held = spec.setdefault(section, {})
        shared = section in ("dimensions", "relations")
        twice = [
            k
            for k in body
            if k in held and not (shared and _claim(held[k]) == _claim(body[k]))
        ]
        if twice:
            msg = f"{origin} declares {sorted(twice)} in `{section}`, which another file declares too."
            raise ValueError(msg)
        held |= {k: v for k, v in body.items() if k not in held}


def _renamed(patch: Any, rename: Callable[[str], str]) -> Any:
    """Return *patch* with *rename* applied to each key and text but descriptions, as `mathspec.rename` takes no patch."""
    if isinstance(patch, dict):
        return {
            rename(k): v if k == "description" else _renamed(v, rename)
            for k, v in patch.items()
        }
    if isinstance(patch, list):
        return [_renamed(v, rename) for v in patch]
    return rename(patch) if isinstance(patch, str) else patch


def render(root: Traversable, patch: Traversable | None = None) -> tuple[Spec, Names]:
    """Return the spec *root* states and its PyPSA names, with the patch in *patch* laid over.

    It is `system.yaml`, then for each component every trait file under
    `traits/` it declares, renamed, and its own file under `components/`,
    wherever *root* has one. A file under *patch* at one of these paths is
    part of the patch. An entry of a variable, constraint or piecewise
    curve may state its PyPSA name in `name:`, one per loss mode where
    PyPSA names it by mode; one without keeps its key as name.

    The files are joined as a dict, not by `mathspec.merge`, which needs
    each file to load on its own: `system.yaml` and the component files
    read what the traits declare.

    Raises:
        ValueError: Two files declare one entry.
        mathspec.errors.LanguageError: A trait does not load on its own.

    """
    import mathspec  # noqa: PLC0415

    import pypsa  # noqa: PLC0415

    paths: list[tuple[str, Callable[[str], str]]] = [("system.yaml", lambda n: n)]
    for c in formulated(pypsa.Network()):
        rename = renamer(c)
        paths += [(f"traits/{t}.yaml", rename) for t in c._formulation]
        paths.append((f"components/{c._unit_dim}.yaml", rename))
    spec: dict[str, Any] = {}
    patched: dict[str, Any] = {}
    names: Names = {section: {} for section in NAMED}
    for path, rename in paths:
        if (root / path).is_file():
            fragment, found = _load(root / path, rename)
            if path.startswith("traits/"):
                fragment = mathspec.rename(fragment, rename).to_dict()
            _merge(spec, fragment, str(root / path))
            names = {s: names[s] | found[s] for s in NAMED}
        if patch is not None and (patch / path).is_file():
            fragment, found = _load(patch / path, rename)
            _merge(patched, _renamed(fragment, rename), str(patch / path))
            names = {s: names[s] | found[s] for s in NAMED}
    rendered = mathspec.to_spec(spec)
    return mathspec.override(rendered, [patched]) if patched else rendered, names
