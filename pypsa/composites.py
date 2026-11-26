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

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from pypsa._registry import Registry, builds_on, jsonable
from pypsa.components.types import component_types_df
from pypsa.network.transform import _build_suffixed_names

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
#: The parameter names a type cannot take: its relation and the arguments of `n.add`.
RESERVED = {"members", "class_name", "name", "suffix", "overwrite", "return_names"}
#: The columns that tag a member.
TAGS = ["composite", "composite_type", "composite_member"]


def _dim(class_name: str) -> str:
    """Return the spec dimension over the components of a class."""
    return "bus" if class_name == "Bus" else class_name


def _tags(n: Network) -> pd.DataFrame:
    """Return every member of a composite: its class, its name and its tags."""
    from pypsa.optimization.spec.sources import of_scenario  # noqa: PLC0415

    frames = [
        of_scenario(c.static, n)[TAGS].assign(class_name=c.name)
        for c in n.components
        if "composite_type" in c.static
    ]
    if not frames:
        return pd.DataFrame(columns=["name", *TAGS, "class_name"])
    tags = pd.concat(frames).rename_axis("name").reset_index()
    return tags[tags.composite_type.notna() & (tags.composite_type != "")]


def _instances(n: Network, ctype: str) -> list[str]:
    """Return the names of the instances of a composite type."""
    tags = _tags(n)
    return list(tags.composite[tags.composite_type == ctype].unique())


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
            msg = f"A composite definition has a name and the sections {sorted(SECTIONS - {'name'})}; it has {sorted(d)}."
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

    @property
    def members(self) -> dict[str, str]:
        """Return the class of each member."""
        return {m: c for c, ms in self.components.items() for m in ms}

    @property
    def templates(self) -> dict[str, dict[str, Any]]:
        """Return the template of each member."""
        return {m: a for ms in self.components.values() for m, a in ms.items()}

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

    @property
    def homes(self) -> dict[str, tuple[str, str]]:
        """Return the member and the attribute that hold each parameter."""
        homes = {
            v[1:]: (m, a)
            for m, template in self.templates.items()
            for a, v in template.items()
            if isinstance(v, str) and v.startswith("$")
        }
        return homes | {p: (self.anchor, self.column(p)) for p in self.math_parameters}

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
        taken |= {*base.dimensions, *PLACEHOLDERS, *dir(Composites)}
        if not isinstance(t, str) or not t.isidentifier() or t in taken:
            msg = f"A composite type is named by an identifier that is no component, dimension, placeholder or attribute of `n.composites`, not {t!r}."
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
        refs = {
            v
            for template in self.templates.values()
            for v in template.values()
            if isinstance(v, str) and v[:1] in ("$", "@")
        }
        known = {f"${p}" for p in self.parameters} | {f"@{m}" for m in members}
        if refs - known:
            msg = f"{t} refers to {sorted(refs - known)}, which it does not define."
            raise ValueError(msg)
        if not self.math.keys() <= MATH:
            msg = f"The math of {t} has the sections {sorted(MATH)}, not {sorted(self.math)}."
            raise ValueError(msg)
        existing = {k for section in MATH for k in getattr(base, section)}
        for key, entry in ((k, e) for s in self.math.values() for k, e in s.items()):
            if not key.startswith(f"{t}_") or key in existing or entry is None:
                msg = f"The math of {t} adds entries keyed `{t}_<name>`; it cannot add {key!r}, nor change or remove the formulation."
                raise ValueError(msg)
        math = set(self.math_parameters)
        templated = {v[1:] for v in refs if v.startswith("$")}
        if math & templated:
            msg = f"A template of {t} sets {sorted(math & templated)}, so the math reads it from the member, as `at(Link_efficiency, by={t}_members[charger])`, not as `{t}_<parameter>`."
            raise ValueError(msg)
        if unused := sorted(self.parameters.keys() - math - templated):
            msg = (
                f"Neither a template nor the math of {t} uses the parameters {unused}."
            )
            raise ValueError(msg)
        if untyped := [p for p in math if self.parameters[p] is None]:
            msg = f"The math of {t} reads {untyped}, so they take a default, which sets their dtype."
            raise ValueError(msg)


class Composite:
    """The instances of one composite type on a network."""

    def __init__(self, n: Network, ctype: CompositeType) -> None:
        """Bind the type to the network."""
        self.n = n
        self.type = ctype

    def __repr__(self) -> str:
        """Name the type and count its instances."""
        return f"Composite '{self.name}' with {len(_instances(self.n, self.name))} instances"

    @property
    def name(self) -> str:
        """Return the type's name."""
        return self.type.name

    def add(
        self,
        name: str | int | Sequence[int | str],
        suffix: str | Sequence[str] = "",
        **kwargs: Any,
    ) -> pd.Index:
        """Add instances: each member of each instance, tagged with its instance.

        Parameters
        ----------
        name : str or sequence of str
            The names of the instances. A member is named `<instance> <member>`.
        suffix : str or sequence of str, default ""
            Suffix added to each instance name, as in `n.add`.
        **kwargs
            The parameters of the type. A value is a scalar for all instances,
            a sequence or a Series over the instances, or for a time-varying
            attribute a DataFrame with the instances as columns, or a Series
            over the snapshots for all of them.

        Returns
        -------
        pandas.Index
            The names of the instances.

        Raises
        ------
            ValueError: A parameter is unknown or missing, or an instance name
                is empty or taken. Nothing is added then.

        """
        names = _build_suffixed_names(name, suffix)
        unknown = kwargs.keys() - self.type.parameters.keys()
        values = self.type.parameters | kwargs
        missing = [p for p, v in values.items() if v is None]
        if unknown or missing:
            msg = f"{self.name} takes the parameters {sorted(self.type.parameters)}; unknown {sorted(unknown)}, missing {missing}."
            raise ValueError(msg)
        taken = names.intersection(self.n.composites.instances)
        bad = [*taken, *names[names.duplicated() | (names == "")]]
        if bad:
            msg = f"Composite instances have unique, non-empty names; {bad} are taken or empty."
            raise ValueError(msg)
        laid = [self._laid(m, c, values, names) for m, c in self.type.members.items()]
        added: list[tuple[str, pd.Index]] = []
        try:
            for member, class_name, labels, attrs in laid:
                self.n.add(
                    class_name,
                    labels,
                    **attrs,
                    composite=names,
                    composite_type=self.name,
                    composite_member=member,
                )
                added.append((class_name, labels))
        except Exception:
            for class_name, labels in added:
                self.n.remove(class_name, labels)
            raise
        static = self.n.c[self.type.members[self.type.anchor]].static
        for p in self.type.math_parameters:
            column = self.type.column(p)
            static[column] = (
                static[column].fillna(self.type.parameters[p]).infer_objects()
            )
        return names

    def remove(
        self, name: str | int | Sequence[int | str], suffix: str | Sequence[str] = ""
    ) -> None:
        """Remove instances: every member of each.

        Raises:
            ValueError: An instance does not exist.

        """
        names = _build_suffixed_names(name, suffix)
        tags = _tags(self.n)
        tags = tags[tags.composite_type == self.name]
        if len(unknown := names.difference(tags.composite)):
            msg = f"There are no {self.name} instances {list(unknown)}."
            raise ValueError(msg)
        for class_name, rows in tags[tags.composite.isin(names)].groupby("class_name"):
            self.n.remove(class_name, rows.name)

    def _laid(
        self, member: str, class_name: str, values: dict[str, Any], names: pd.Index
    ) -> tuple[str, str, pd.Index, dict[str, Any]]:
        """Return a member of the instances with its names and attributes.

        Raises:
            ValueError: A member of that name exists.

        """
        labels = names + f" {member}"
        if len(exist := labels.intersection(self.n.c[class_name].names)):
            msg = f"The {class_name} components {list(exist)} exist."
            raise ValueError(msg)
        template = self.type.templates[member]
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

    def _per_instance(
        self, value: Any, names: pd.Index, labels: pd.Index, parameter: str = ""
    ) -> Any:
        """Return a parameter laid over the members: by instance where it is a Series or a DataFrame.

        A Series over the snapshots is a time series for all instances.

        Raises:
            ValueError: A math parameter is a time series, or a value by
                instance leaves an instance out.

        """
        frame = isinstance(value, pd.DataFrame)
        series = isinstance(value, pd.Series)
        timed = frame or (series and value.index.equals(self.n.snapshots))
        if parameter and timed:
            msg = f"The math parameter {parameter} takes one value per instance, not a time series."
            raise ValueError(msg)
        if series and timed:
            return pd.DataFrame(dict.fromkeys(labels, value))
        if not (frame or series):
            return value
        index = value.columns if frame else value.index
        if not names.isin(index).all():
            msg = f"A value by instance covers every instance; it lacks {list(names.difference(index))}."
            raise ValueError(msg)
        if frame:
            return value[names].set_axis(labels, axis=1)
        return value.reindex(names).to_numpy()

    @property
    def members(self) -> pd.DataFrame:
        """Return the members of each instance: instances by member, each a component name.

        Raises:
            ValueError: An instance lacks a member.

        """
        tags = _tags(self.n)
        tags = tags[tags.composite_type == self.name]
        members = (
            tags.pivot(index="composite", columns="composite_member", values="name")
            .reindex(index=tags.composite.unique(), columns=list(self.type.members))
            .rename_axis(index=self.name, columns=None)
        )
        incomplete = members.isna().any(axis=1)
        if incomplete.any():
            msg = f"The {self.name} instances {list(members.index[incomplete])} lack members; remove or add them in full."
            raise ValueError(msg)
        members.index = members.index.astype(object)
        return members.astype(object)

    @property
    def static(self) -> pd.DataFrame:
        """Return the parameters of each instance, read from its members."""
        from pypsa.optimization.spec.sources import of_scenario  # noqa: PLC0415

        members = self.members
        read = {}
        for p, (member, attr) in self.type.homes.items():
            static = of_scenario(self.n.c[self.type.members[member]].static, self.n)
            read[p] = static[attr].reindex(members[member]).to_numpy()
        return pd.DataFrame(read, index=members.index)[list(self.type.parameters)]

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


class Composites(Registry[Composite]):
    """The composite types registered on a network, read from `n.meta`.

    Examples
    --------
    >>> n.composites.register("battery.yaml")  # doctest: +SKIP
    >>> n.add("battery", ["north", "south"], bus=["north", "south"])  # doctest: +SKIP
    >>> n.composites.battery.members  # doctest: +SKIP
    >>> n.remove("battery", ["north", "south"])  # doctest: +SKIP
    >>> n.composites.remove("battery")  # doctest: +SKIP

    """

    META = META
    NOUN = "composite type"

    def _read(self, entry: Mapping[str, Any]) -> Composite:
        return Composite(self.n, CompositeType.read(entry))

    def register(self, definition: Override) -> Composite:
        """Register a composite type from a YAML file, YAML text or mapping.

        The definition is checked with the formulation, the other composite
        types and the stored overrides of the network.

        Raises:
            ValueError: The definition is malformed, another type of the
                same name is registered, or a stored override creates an
                entry the type declares.
            mathspec.errors.LanguageError: The math does not hold with the
                formulation.

        """
        t = CompositeType.read(definition)
        content = jsonable(asdict(t), f"The composite type {t.name!r}")
        if not self._holds(t.name, content):
            self._set(self._entries() | {t.name: content})
        return self[t.name]

    def remove(self, name: str) -> None:
        """Unregister a composite type, also where its math no longer builds.

        Raises:
            KeyError: No composite type of this name is registered.
            ValueError: Instances of the type exist, or a later type or a
                stored override builds on it. Nothing is removed then.

        """
        if name in self and (instances := _instances(self.n, name)):
            msg = f"The {name} instances {instances} exist; remove them first with `n.remove({name!r}, ...)`."
            raise ValueError(msg)
        self._remove(name)

    def _dependents(self, name: str, entries: dict[str, Any]) -> list[str]:
        from pypsa.optimization.spec.build import written  # noqa: PLC0415

        touched = set(written(CompositeType.read(entries[name]).patch()))
        created = [k.rpartition(".")[2] for k in touched]
        later = list(entries)[list(entries).index(name) + 1 :]
        types = [k for k in later if builds_on(entries[k]["math"], touched, created)]
        overrides = self.n.optimize.overrides
        return types + [
            k for k, p in overrides.items() if builds_on(p, touched, created)
        ]

    @property
    def instances(self) -> pd.Index:
        """Return the names of all instances of all types."""
        return pd.Index(_tags(self.n).composite.unique(), dtype=object)

    def patches(self) -> dict[str, dict[str, Any]]:
        """Return the spec patch of each registered type, by name."""
        return {k: c.type.patch() for k, c in self.items()}

    def providers(self) -> dict[str, Callable[[NetworkSources, str], Any]]:
        """Return the data of every registered type for a build.

        Raises:
            ValueError: Components are tagged with a type that is not
                registered.

        """
        unknown = set(_tags(self.n).composite_type) - set(self)
        if unknown:
            msg = f"Components are tagged with the composite types {sorted(unknown)}, which are not registered; register them with `n.composites.register`."
            raise ValueError(msg)
        return {k: f for c in self.values() for k, f in c.providers().items()}
