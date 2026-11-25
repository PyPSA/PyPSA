# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Composites: reusable groups of components with the math that ties them.

A composite type is defined once, as a YAML file, YAML text or a mapping, and
registered on a network with `n.composites.register`. Each instance added with
`n.add("<type>", ...)` is nothing but its member components, tagged with the
columns `composite`, `composite_type` and `composite_member`. The definition is
stored in `n.meta`, so copies and saved networks keep their composites.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.components.types import component_types_df
from pypsa.statistics.grouping import groupers

if TYPE_CHECKING:
    from pypsa import Network
    from pypsa.optimization.spec.build import Override
    from pypsa.optimization.spec.sources import NetworkSources

#: The key of `n.meta` that holds the registered definitions.
META = "composites"
#: The sections of a definition.
SECTIONS = {"name", "description", "parameters", "components", "math"}
#: The sections of the math of a definition.
MATH = {"variables", "expressions", "constraints"}
#: The parameter names a type declares itself.
RESERVED = {"members"}


def _dim(class_name: str) -> str:
    """Return the spec dimension over the components of a class."""
    return "bus" if class_name == "Bus" else class_name


def _text(entry: Any) -> Iterator[str]:
    """Yield the strings of a math entry, its descriptions left out."""
    if isinstance(entry, Mapping):
        for key, value in entry.items():
            if key != "description":
                yield from _text(value)
    elif isinstance(entry, list):
        for value in entry:
            yield from _text(value)
    elif isinstance(entry, str):
        yield entry


@dataclass(frozen=True)
class CompositeType:
    """The definition of a composite type.

    Parameters
    ----------
    name : str
        The type's name, as `battery`. It names its dimension in the spec and
        prefixes every key of its math.
    parameters : dict
        The parameters `add` takes, each with its default. A default of None
        makes the parameter required.
    components : dict
        The members per component class, each a template of attributes. A
        value `$<parameter>` takes the parameter, `@<member>` the name of
        another member of the same instance.
    math : dict
        The variables, expressions and constraints of the type, as a spec
        patch. A parameter `<name>_<parameter>` and the relation
        `<name>_members`, which maps each instance to its members, are
        declared for it.
    description : str
        What an instance is.

    """

    name: str
    parameters: dict[str, Any]
    components: dict[str, dict[str, dict[str, Any]]]
    math: dict[str, dict[str, Any]]
    description: str = ""

    @classmethod
    def read(cls, definition: Override) -> CompositeType:
        """Return the type a YAML file, YAML text or mapping defines.

        Raises:
            ValueError: The definition is malformed or clashes with the
                formulation.

        """
        from pypsa.optimization.spec.build import _layer  # noqa: PLC0415

        d = _layer(definition)
        unknown = d.keys() - SECTIONS
        if unknown or "name" not in d:
            msg = f"A composite definition has a name and the sections {sorted(SECTIONS - {'name'})}, not {sorted(unknown)}."
            raise ValueError(msg)
        t = cls(
            name=d["name"],
            parameters=dict(d.get("parameters") or {}),
            components={k: dict(v) for k, v in (d.get("components") or {}).items()},
            math={k: dict(v) for k, v in (d.get("math") or {}).items()},
            description=d.get("description") or "",
        )
        t._check()
        return t

    def to_dict(self) -> dict[str, Any]:
        """Return the definition as a mapping `read` takes back."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "components": self.components,
            "math": self.math,
        }

    @property
    def members(self) -> dict[str, str]:
        """Return the class of each member."""
        return {m: c for c, ms in self.components.items() for m in ms}

    @property
    def anchor(self) -> str:
        """Return the first member, which carries the parameters of the math."""
        return next(iter(self.members))

    @property
    def math_parameters(self) -> list[str]:
        """Return the parameters the math reads as `<name>_<parameter>`."""
        text = " ".join(_text(self.math))
        used = set(re.findall(rf"\b{self.name}_(\w+)\b", text))
        return [p for p in self.parameters if p in used]

    def column(self, parameter: str) -> str:
        """Return the column of the anchor that holds a parameter of the math."""
        return f"{self.name}_{parameter}"

    def patch(self) -> dict[str, Any]:
        """Return the spec patch of the type: its declarations and its math."""
        t = self.name
        parameters = {}
        for p in self.math_parameters:
            dtype = {bool: "bool", str: "str"}.get(type(self.parameters[p]))
            parameters[f"{t}_{p}"] = {"dims": ["scenario", t]} | (
                {"dtype": dtype} if dtype else {}
            )
        declared = {
            "dimensions": {t: {"description": self.description or f"{t} composites"}},
            "relations": {
                f"{t}_members": {
                    "description": f"the members of each {t}",
                    "key": t,
                    "values": {m: _dim(c) for m, c in self.members.items()},
                }
            },
            "parameters": parameters,
        }
        return declared | self.math

    def _check(self) -> None:
        from pypsa.optimization.spec._render import PLACEHOLDERS  # noqa: PLC0415
        from pypsa.optimization.spec.build import spec  # noqa: PLC0415

        t = self.name
        base = spec()
        taken = {*component_types_df.index, *component_types_df.list_name}
        taken |= {*base.dimensions, *PLACEHOLDERS}
        if not isinstance(t, str) or not t.isidentifier() or t in taken:
            msg = f"A composite type is named by an identifier that is no component, dimension or placeholder, not {t!r}."
            raise ValueError(msg)
        unknown = [c for c in self.components if _dim(c) not in base.dimensions]
        if unknown or not self.components:
            msg = f"The components of {t} are listed by component class, one the formulation models, not {unknown}."
            raise ValueError(msg)
        if RESERVED & self.parameters.keys():
            msg = f"{t} cannot name a parameter {sorted(RESERVED)}."
            raise ValueError(msg)
        members = [m for ms in self.components.values() for m in ms]
        if len(set(members)) < len(members) or not all(
            m.isidentifier() for m in members
        ):
            msg = f"The members of {t} have unique identifiers as names, not {members}."
            raise ValueError(msg)
        for template in (a for ms in self.components.values() for a in ms.values()):
            for value in template.values():
                if isinstance(value, str) and (
                    (value.startswith("$") and value[1:] not in self.parameters)
                    or (value.startswith("@") and value[1:] not in members)
                ):
                    msg = f"{t} refers to {value}, which it does not define."
                    raise ValueError(msg)
        if not self.math.keys() <= MATH:
            msg = f"The math of {t} has the sections {sorted(MATH)}, not {sorted(self.math)}."
            raise ValueError(msg)
        existing = {k for section in MATH for k in getattr(base, section)}
        for key, entry in ((k, e) for s in self.math.values() for k, e in s.items()):
            if not key.startswith(f"{t}_") or key in existing or entry is None:
                msg = f"The math of {t} adds entries keyed `{t}_<name>`; it cannot add {key!r}, nor change or remove the formulation."
                raise ValueError(msg)


class Composite:
    """The instances of one composite type on a network."""

    def __init__(self, n: Network, ctype: CompositeType) -> None:
        """Bind the type to the network."""
        self.n = n
        self.type = ctype

    def __repr__(self) -> str:
        """Name the type and count its instances."""
        return f"Composite '{self.type.name}' with {len(self.members)} instances"

    @property
    def name(self) -> str:
        """Return the type's name."""
        return self.type.name

    def add(self, name: str | int | Sequence[int | str], **kwargs: Any) -> pd.Index:
        """Add instances: each member of each instance, tagged with its instance.

        Parameters
        ----------
        name : str or sequence of str
            The names of the instances. A member is named `<instance> <member>`.
        **kwargs
            The parameters of the type. A value is a scalar for all instances,
            a sequence or a Series over the instances, or for a time-varying
            attribute a DataFrame with the instances as columns.

        Returns
        -------
        pandas.Index
            The names of the instances.

        Raises
        ------
            ValueError: A parameter is unknown or missing, or an instance name
                is taken.

        """
        names = pd.Index([name] if isinstance(name, str | int) else list(name)).astype(
            str
        )
        unknown = kwargs.keys() - self.type.parameters.keys()
        values = self.type.parameters | kwargs
        missing = [p for p, v in values.items() if v is None]
        if unknown or missing:
            msg = f"{self.name} takes the parameters {sorted(self.type.parameters)}; unknown {sorted(unknown)}, missing {missing}."
            raise ValueError(msg)
        taken = names.intersection(self.n.composites.instances)
        if names.has_duplicates or len(taken):
            msg = f"Composite instances have unique names; {list(taken) or list(names[names.duplicated()])} are taken."
            raise ValueError(msg)
        laid = [self._laid(m, c, values, names) for m, c in self.type.members.items()]
        for member, class_name, labels, attrs in laid:
            self.n.add(
                class_name,
                labels,
                **attrs,
                composite=names,
                composite_type=self.name,
                composite_member=member,
            )
        return names

    def _laid(
        self, member: str, class_name: str, values: dict[str, Any], names: pd.Index
    ) -> tuple[str, str, pd.Index, dict[str, Any]]:
        """Return a member of the instances with its names and attributes.

        Raises:
            ValueError: A member of that name exists.

        """
        from pypsa.optimization.spec.sources import of_scenario  # noqa: PLC0415

        labels = names + f" {member}"
        exist = labels.intersection(
            of_scenario(self.n.c[class_name].static, self.n).index
        )
        if len(exist):
            msg = f"The {class_name} components {list(exist)} exist."
            raise ValueError(msg)
        template = self.type.components[class_name][member]
        attrs = {a: self._value(v, values, names, labels) for a, v in template.items()}
        if member == self.type.anchor:
            for p in self.type.math_parameters:
                attrs[self.type.column(p)] = self._per_instance(
                    values[p], names, labels, p
                )
        return member, class_name, labels, attrs

    def _value(
        self, value: Any, values: dict[str, Any], names: pd.Index, labels: pd.Index
    ) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            return self._per_instance(values[value[1:]], names, labels)
        if isinstance(value, str) and value.startswith("@"):
            return (names + f" {value[1:]}").to_numpy(dtype=object)
        return value

    @staticmethod
    def _per_instance(
        value: Any, names: pd.Index, labels: pd.Index, scalar: str = ""
    ) -> Any:
        """Return a parameter laid over the members: by instance where it varies by instance.

        Raises:
            ValueError: A math parameter is a time series, or a value by
                instance leaves an instance out.

        """
        if isinstance(value, pd.DataFrame) and scalar:
            msg = f"The math parameter {scalar} takes one value per instance, not a time series."
            raise ValueError(msg)
        by_instance = isinstance(value, pd.DataFrame) or (
            isinstance(value, pd.Series) and value.index.isin(names).any()
        )
        if not by_instance:
            return value
        index = value.columns if isinstance(value, pd.DataFrame) else value.index
        if not names.isin(index).all():
            msg = f"A value by instance covers every instance; it lacks {list(names.difference(index))}."
            raise ValueError(msg)
        if isinstance(value, pd.DataFrame):
            return value[names].set_axis(labels, axis=1)
        return value.reindex(names).to_numpy()

    @property
    def members(self) -> pd.DataFrame:
        """Return the members of each instance: instances by member, each a component name.

        Raises:
            ValueError: An instance lacks a member.

        """
        from pypsa.optimization.spec.sources import of_scenario  # noqa: PLC0415

        columns = {}
        for member, class_name in self.type.members.items():
            static = of_scenario(self.n.c[class_name].static, self.n)
            if "composite_type" not in static:
                columns[member] = pd.Series(dtype=object)
                continue
            tagged = static[
                (static.composite_type == self.name)
                & (static.composite_member == member)
            ]
            columns[member] = pd.Series(tagged.index, index=tagged.composite)
        members = pd.DataFrame(columns)
        members.index = pd.Index(members.index, dtype=object, name=self.name)
        incomplete = members.isna().any(axis=1)
        if incomplete.any():
            msg = f"The {self.name} instances {list(members.index[incomplete])} lack members; remove or add them in full."
            raise ValueError(msg)
        return members.astype(object)

    @property
    def static(self) -> pd.DataFrame:
        """Return the parameters of each instance, read from its members."""
        from pypsa.optimization.spec.sources import of_scenario  # noqa: PLC0415

        members = self.members
        read = {}
        for p in self.type.parameters:
            member, attr = self._home(p)
            static = of_scenario(self.n.c[self.type.members[member]].static, self.n)
            read[p] = static[attr].reindex(members[member]).to_numpy()
        return pd.DataFrame(read, index=members.index)

    def _home(self, parameter: str) -> tuple[str, str]:
        """Return the member and attribute that hold a parameter."""
        if parameter in self.type.math_parameters:
            return self.type.anchor, self.type.column(parameter)
        for member, template in (
            (m, a) for ms in self.type.components.values() for m, a in ms.items()
        ):
            for attr, value in template.items():
                if value == f"${parameter}":
                    return member, attr
        msg = f"{self.name} stores no parameter {parameter}."
        raise KeyError(msg)

    def providers(self) -> dict[str, Callable[[NetworkSources, str], Any]]:
        """Return the data of the type's dimension, relation and parameters for a build."""
        t = self.name
        providers: dict[str, Callable[[NetworkSources, str], Any]] = {
            t: lambda s, name: self._built(s).index,
            f"{t}_members": lambda s, name: self._built(s).reset_index(),
        }
        for p in self.type.math_parameters:
            providers[f"{t}_{p}"] = self._parameter
        return providers

    def _built(self, s: NetworkSources) -> pd.DataFrame:
        """Return the members of the instances whose members are all in the build; every bus is.

        Raises:
            ValueError: Some members of an instance are in the build, some not.

        """

        def built() -> pd.DataFrame:
            members = self.members
            inside = pd.DataFrame(
                {
                    m: members[m].isin(s.coords(c))
                    for m, c in self.type.members.items()
                    if c != "Bus"
                },
                index=members.index,
            )
            partly = inside.any(axis=1) & ~inside.all(axis=1)
            if partly.any():
                msg = f"The {self.name} instances {list(members.index[partly])} have members outside the build; activate or deactivate them in full."
                raise ValueError(msg)
            return members[inside.all(axis=1)]

        return s.memo(f"{self.name} members", built)

    def _parameter(self, s: NetworkSources, name: str) -> xr.DataArray:
        """Return a parameter `<type>_<parameter>` of the math over the scenarios and the instances in the build."""
        from pypsa.optimization.spec.sources import of_scenario  # noqa: PLC0415

        column = self.type.column(name.removeprefix(f"{self.name}_"))
        anchors = self._built(s)[self.type.anchor]
        static = self.n.c[self.type.members[self.type.anchor]].static
        values = [
            of_scenario(static, self.n, scenario)[column].reindex(anchors).to_numpy()
            for scenario in s.scenarios
        ]
        coords = {"scenario": s.scenarios, self.name: anchors.index}
        return xr.DataArray(np.array(values, dtype=s.dtype(name)), coords, list(coords))


class Composites(Mapping[str, Composite]):
    """The composite types registered on a network, read from `n.meta`.

    Examples
    --------
    >>> n.composites.register("battery.yaml")  # doctest: +SKIP
    >>> n.add("battery", ["north", "south"], bus=["north", "south"])  # doctest: +SKIP
    >>> n.composites.battery.members  # doctest: +SKIP

    """

    def __init__(self, n: Network) -> None:
        """Bind the registry to the network."""
        self.n = n

    def __repr__(self) -> str:
        """List the registered types."""
        return f"Composites {list(self)}"

    def _definitions(self) -> dict[str, Any]:
        return self.n.meta.get(META, {})

    def __getitem__(self, name: str) -> Composite:
        """Return the instances of a registered type."""
        definition = json.dumps(self._definitions()[name], sort_keys=True)
        return Composite(self.n, _parsed(definition))

    def __contains__(self, name: object) -> bool:
        """Whether a type of this name is registered."""
        return name in self._definitions()

    def __getattr__(self, name: str) -> Composite:
        """Return the instances of a registered type."""
        if name.startswith("_") or name not in self._definitions():
            msg = f"No composite type {name!r} is registered."
            raise AttributeError(msg)
        return self[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate the registered type names."""
        return iter(self._definitions())

    def __len__(self) -> int:
        """Count the registered types."""
        return len(self._definitions())

    def register(self, definition: Override) -> Composite:
        """Register a composite type from a YAML file, YAML text or mapping.

        Raises:
            ValueError: The definition is malformed, or another type of the
                same name is registered.

        """
        t = CompositeType.read(definition)
        known = self._definitions().get(t.name)
        if known is not None and known != t.to_dict():
            msg = f"Another composite type {t.name!r} is registered."
            raise ValueError(msg)
        self.n.meta.setdefault(META, {})[t.name] = t.to_dict()
        return self[t.name]

    @property
    def instances(self) -> pd.Index:
        """Return the names of all instances of all types."""
        return pd.Index([i for c in self.values() for i in c.members.index])

    def patches(self) -> list[dict[str, Any]]:
        """Return the spec patch of each registered type."""
        return [c.type.patch() for c in self.values()]

    def providers(self) -> dict[str, Callable[[NetworkSources, str], Any]]:
        """Return the data of every registered type for a build."""
        return {k: f for c in self.values() for k, f in c.providers().items()}


@functools.lru_cache(maxsize=64)
def _parsed(definition: str) -> CompositeType:
    """Return the type a definition in JSON states, checked once per process."""
    return CompositeType.read(json.loads(definition))


def _composite(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> pd.Series:
    """Group components by the composite instance they belong to; `""` for none."""
    static = n.c[c].static
    tags = static["composite"] if "composite" in static else pd.Series("", static.index)
    return tags.fillna("").rename("composite")


groupers.add_grouper("composite", _composite)
