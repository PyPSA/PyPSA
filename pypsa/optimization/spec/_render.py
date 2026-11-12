# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Render a spec from its system file, each component's traits and its own file.

A file under `traits/` states a formulation once for every component that
declares the trait in `_formulation`. It is plain mathspec with these
placeholders, which `string.Template` fills per component:

- `${C}`: the component's name, as `StorageUnit`.
- `${c}`: its unit dimension, as `storage_unit`.
- `${nom}`: its nominal attribute, as `p_nom`.
- `${unit}`: its unit dimension in prose, as `storage unit`.
- `${p}`: its power variable, `p` or `p_dispatch`, where it has one.
- `${level}`: its storage level variable, `e` or `state_of_charge`, and
  `${cyclic}` its cyclic flag, where it stores energy.
- `${coef}`: the attribute that scales `p` at each port, `efficiency` or
  `rate`, where it has ports.

A component lacks a placeholder it has no value for, so a trait that uses
one fails to render for it. A file under `components/` holds what a
component shares with no other.
"""

from __future__ import annotations

from string import Template
from typing import TYPE_CHECKING, Any

#: A PyPSA name: one string, or one per transmission loss mode.
Name = str | dict[str, str]
#: The PyPSA names of the spec's entries, by section and entry.
Names = dict[str, dict[str, Name]]
#: The sections whose entries may carry a `name:`.
NAMED = ("variables", "constraints", "piecewise")

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

    from pypsa import Network
    from pypsa.components.components import Components


def formulated(n: Network, trait: str = "") -> list[Components]:
    """Return the components of *n* that declare a formulation, or the trait *trait*."""
    components = [c for c in n.components.values() if c._formulation]
    return [c for c in components if not trait or trait in c._formulation]


def symbols(c: Components) -> dict[str, str]:
    """Return what the placeholders of a trait stand for in component *c*."""
    dim = c._unit_dim
    nom = c._operational_attrs["nom"]
    found = {"C": c.name, "c": dim, "nom": nom, "unit": dim.replace("_", " ")}
    power = [v for v in c._operational_variables if v.startswith("p")]
    if power:
        found["p"] = power[0]
    if c._level_attrs:
        initial, found["cyclic"] = c._level_attrs
        found["level"] = initial.removesuffix("_initial")
    if coefficient := getattr(c, "_coefficient_attr", None):
        found["coef"] = coefficient
    return found


def _merge(spec: dict[str, Any], fragment: dict[str, Any], origin: str) -> None:
    for section, body in fragment.items():
        held = spec.setdefault(section, {})
        if twice := held.keys() & body.keys():
            msg = f"{origin} declares {sorted(twice)} in `{section}`, which another file declares too."
            raise ValueError(msg)
        held |= body


def _names(spec: dict[str, Any]) -> Names:
    """Pop the `name:` of each entry of *spec*, which the closed mathspec schema refuses."""
    names: Names = {}
    for section in NAMED:
        entries = spec.get(section, {})
        names[section] = {k: v.pop("name") for k, v in entries.items() if "name" in v}
    return names


def render(root: Traversable) -> tuple[dict[str, Any], Names]:
    """Return the spec *root* states, as a dict for `mathspec.to_spec`, and its PyPSA names.

    It is `system.yaml`, then for each component every trait file under
    `traits/` it declares and its own file under `components/`, wherever
    *root* has one. An entry of a variable, constraint or piecewise curve
    may state its PyPSA name in `name:`, one per loss mode where PyPSA
    names it by mode; one without keeps its key as name.

    Raises:
        ValueError: Two files declare one entry.

    """
    from mathspec._yaml import parse_yaml  # noqa: PLC0415

    import pypsa  # noqa: PLC0415

    system = root / "system.yaml"
    spec = parse_yaml(system.read_text(), str(system))
    for c in formulated(pypsa.Network()):
        mapping = symbols(c)
        traits = [root / "traits" / f"{t}.yaml" for t in c._formulation]
        for path in [*traits, root / "components" / f"{mapping['c']}.yaml"]:
            if path.is_file():
                text = Template(path.read_text()).substitute(mapping)
                _merge(spec, parse_yaml(text, str(path)), str(path))
    return spec, _names(spec)
