# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Render a spec from its system file, each component's traits and its own file.

A file under `traits/` states a formulation once for every component that
declares the trait in `_formulation`. A component renders a trait whole,
so a trait holds what one set of components shares, stated alike for
each, as `transmission_expansion` apart from `capacity`. Within that set,
traits split by topic, as `ramping` apart from `commitment`. An entry
that one component alone holds, or that components state each in their
own way, is in their files. A trait is named after what it holds, as
`operational_limit` after the global constraint type it serves. It is a
spec of its own, which reads
under `given:` what another file declares, written over placeholder names.
`mathspec.rename` gives it the names of each component, and its
descriptions too. A word of a name or description, between `_` or `-`,
is a placeholder if it is one of these:

- `COMPONENT`: the component's name, as `StorageUnit`.
- `UNIT`: its unit dimension, as `StorageUnit`.
- `NOM`: its nominal capacity, `p_nom`, `s_nom` or `e_nom`.
- `POWER`: its power variable, `p` or `p_dispatch`, where it has one.
- `LEVEL`: its storage level variable, `e` or `state_of_charge`, and
  `CYCLIC` its cyclic flag, where it stores energy.
- `COEF`: the attribute that scales `p` at each port, `efficiency` or
  `rate`, where it has ports.

A component lacks a placeholder it has no value for, so a trait that uses
one fails to render for it. Placeholders are upper case, so no PyPSA
attribute holds one. A trait writes `p` or `s` as is where only
components with that base declare it, as `s_max_pu` in `passive_branch`:
a placeholder there would hide which attributes the trait reads. A file
under `components/` holds, under the component's own names, what it
shares with no other.
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
#: The rules an entry may name in `scaling:`, by section: a variable for the
#: factor of its columns, a constraint for that of its rows, a piecewise curve
#: for that of the rows that hold its cost.
SCALINGS = {
    "variables": frozenset({"bus", "objective", "none"}),
    "constraints": frozenset(
        {"bus", "objective", "none", "cycle", "carrier", "global"}
    ),
    "piecewise": frozenset({"bus", "objective", "none"}),
}
#: The scaling rule of each entry that names one, by its section and key.
Scalings = dict[tuple[str, str], str]
#: A word of a name, between `_` or `-`.
WORD = re.compile(r"[^\W_]+")
#: The words that stand for a value of the component.
PLACEHOLDERS = frozenset(
    {"COMPONENT", "UNIT", "NOM", "POWER", "LEVEL", "CYCLIC", "COEF"}
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from importlib.resources.abc import Traversable

    from mathspec.spec import Spec

    from pypsa import Network
    from pypsa.components.components import Components


def snake(name: str) -> str:
    """Return *name* in snake case, the stem of a component's own spec file, as `storage_unit`."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def formulated(n: Network, trait: str = "") -> list[Components]:
    """Return the components of *n* that declare a formulation, or the trait *trait*."""
    components = [c for c in n.components.values() if c._formulation]
    return [c for c in components if not trait or trait in c._formulation]


def symbols(c: Components) -> dict[str, str]:
    """Return what the placeholders of a trait stand for in component *c*."""
    found = {
        "COMPONENT": c.name,
        "UNIT": c._unit_dim,
        "NOM": c._operational_attrs["nom"],
    }
    if c._power_variable:
        found["POWER"] = c._power_variable
    if c._level_attrs:
        initial, found["CYCLIC"] = c._level_attrs
        found["LEVEL"] = initial.removesuffix("_initial")
    if coefficient := getattr(c, "_coefficient_attr", None):
        found["COEF"] = coefficient
    return found


def renamer(c: Components) -> Callable[[str], str]:
    """Return the rule that gives a name over placeholders its name in component *c*.

    Raises
    ------
    KeyError
        The name holds a placeholder *c* has no value for.

    """
    found = symbols(c)

    def word(match: re.Match[str]) -> str:
        return found[match[0]] if match[0] in PLACEHOLDERS else match[0]

    return lambda name: WORD.sub(word, name)


def load(
    path: Traversable, rename: Callable[[str], str]
) -> tuple[dict, Names, Scalings]:
    """Return the spec in *path* without its `name:` and `scaling:` fields, and those fields by section and renamed key.

    The closed mathspec schema refuses both fields, and `mathspec.rename`
    keeps the label of a piecewise link, so all are renamed here.

    Raises
    ------
    ValueError
        A `scaling:` names no rule `SCALINGS` allows in its section.

    """
    import yaml  # noqa: PLC0415

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    data = yaml.load(path.read_text(), Loader=loader)  # noqa: S506
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
    scalings: Scalings = {}
    for section, rules in SCALINGS.items():
        entries = data.get(section, {})
        scaled = {k: v.pop("scaling") for k, v in entries.items() if "scaling" in v}
        unknown = sorted(set(scaled.values()) - rules)
        if unknown:
            msg = f"{path} names the scaling rules {unknown} in `{section}`; use one of {sorted(rules)}."
            raise ValueError(msg)
        scalings |= {(section, rename(k)): rule for k, rule in scaled.items()}
    for entry in data.get("piecewise", {}).values():
        if "links" in entry:
            entry["links"] = {rename(k): v for k, v in entry["links"].items()}
    return data, names, scalings


def _claim(entry: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in entry.items() if k != "description"}


def _merge(spec: dict[str, Any], fragment: dict[str, Any], origin: str) -> None:
    """Add the entries of *fragment* to *spec*, but its `given:`.

    A dimension, relation or macro may repeat where it states what it stated before.
    """
    for section, body in fragment.items():
        if not isinstance(body, dict):
            spec.setdefault(section, body)
            continue
        if section == "given":
            continue
        held = spec.setdefault(section, {})
        shared = section in ("dimensions", "relations", "macros")
        twice = [
            k
            for k in body
            if k in held and not (shared and _claim(held[k]) == _claim(body[k]))
        ]
        if twice:
            msg = f"{origin} declares {sorted(twice)} in `{section}`, which another file declares too."
            raise ValueError(msg)
        held |= {k: v for k, v in body.items() if k not in held}


def renamed(patch: Any, rename: Callable[[str], str]) -> Any:
    """Return *patch* with *rename* applied to each key and text, as `mathspec.rename` takes no patch."""
    if isinstance(patch, dict):
        return {rename(k): renamed(v, rename) for k, v in patch.items()}
    if isinstance(patch, list):
        return [renamed(v, rename) for v in patch]
    return rename(patch) if isinstance(patch, str) else patch


def _described(spec: Any, rename: Callable[[str], str]) -> Any:
    """Return *spec* with *rename* applied to each description, which `mathspec.rename` keeps as written."""
    if isinstance(spec, dict):
        return {
            k: rename(v) if k == "description" else _described(v, rename)
            for k, v in spec.items()
        }
    if isinstance(spec, list):
        return [_described(v, rename) for v in spec]
    return spec


def render(
    root: Traversable, patch: Traversable | None = None
) -> tuple[Spec, Names, Scalings]:
    """Return the spec *root* states, its PyPSA names and scaling rules, with the patch in *patch* laid over.

    It is `system.yaml`, then for each component every trait file under
    `traits/` it declares, renamed, and its own file under `components/`;
    *root* holds each. A file under *patch* at one of these paths is part
    of the patch. An entry of a variable, constraint or piecewise
    curve may state its PyPSA name in `name:`, one per loss mode where
    PyPSA names it by mode; one without keeps its key as name. It may
    also name in `scaling:` one of the rules `SCALINGS` allows in its
    section: a variable for its column factors, a constraint for its row
    factors, a piecewise curve for the factors of the rows that hold its
    cost. An entry without one takes the rule its structure gives, which
    `pypsa.optimization.spec.scaling` states.

    The files are joined as a dict, not by `mathspec.merge`, which needs
    each file to load on its own: `system.yaml` and the component files
    read what the traits declare.

    Raises
    ------
    FileNotFoundError
        *root* lacks one of these files.
    ValueError
        Two files declare one entry.
    mathspec.errors.LanguageError
        A trait does not load on its own.

    """
    import mathspec  # noqa: PLC0415

    import pypsa  # noqa: PLC0415

    paths: list[tuple[str, Callable[[str], str]]] = [("system.yaml", lambda n: n)]
    for c in formulated(pypsa.Network()):
        rename = renamer(c)
        paths += [(f"traits/{t}.yaml", rename) for t in c._formulation]
        paths.append((f"components/{snake(c.name)}.yaml", rename))
    spec: dict[str, Any] = {}
    patched: dict[str, Any] = {}
    names: Names = {section: {} for section in NAMED}
    scalings: Scalings = {}
    for path, rename in paths:
        if not (root / path).is_file():
            msg = f"The spec at {root} has no {path}, which the components with a formulation need."
            raise FileNotFoundError(msg)
        fragment, found, scaled = load(root / path, rename)
        if path.startswith("traits/"):
            fragment = mathspec.rename(fragment, rename).to_dict()
            fragment = _described(fragment, rename)
        _merge(spec, fragment, str(root / path))
        names = {s: names[s] | found[s] for s in NAMED}
        scalings |= scaled
        if patch is not None and (patch / path).is_file():
            fragment, found, scaled = load(patch / path, rename)
            _merge(patched, renamed(fragment, rename), str(patch / path))
            names = {s: names[s] | found[s] for s in NAMED}
            scalings |= scaled
    rendered = mathspec.to_spec(spec)
    overridden = mathspec.override(rendered, [patched]) if patched else rendered
    return overridden, names, scalings
