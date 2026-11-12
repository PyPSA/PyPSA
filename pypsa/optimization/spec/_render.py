# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Render a spec from its system file, each component's traits and its own file.

A file under `traits/` states a formulation once for every component that
declares the trait in `_formulation`. It is plain mathspec in which `${C}`
stands for the component's name, `${c}` for its unit dimension, `${nom}` for
its nominal attribute and `${unit}` for its unit dimension in prose. A file
under `components/` holds what a component shares with no other.
"""

from __future__ import annotations

from string import Template
from typing import TYPE_CHECKING, Any

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
    return {"C": c.name, "c": dim, "nom": nom, "unit": dim.replace("_", " ")}


def _merge(spec: dict[str, Any], fragment: dict[str, Any], origin: str) -> None:
    for section, body in fragment.items():
        held = spec.setdefault(section, {})
        if twice := held.keys() & body.keys():
            msg = f"{origin} declares {sorted(twice)} in `{section}`, which another file declares too."
            raise ValueError(msg)
        held |= body


def render(root: Traversable) -> dict[str, Any]:
    """Return the spec *root* states, as a dict for `mathspec.to_spec`.

    It is `system.yaml`, then for each component every trait file under
    `traits/` it declares and its own file under `components/`, wherever
    *root* has one.

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
    return spec
