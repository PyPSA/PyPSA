# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Composites: reusable groups of components with the math that ties them.

A composite type is defined once, as a YAML file, YAML text or a mapping, and
registered on a network with `n.composites.register`. Each instance added with
`n.add("<type>", ...)` is nothing but its member components, tagged with the
columns `composite`, `composite_type` and `composite_member`. The definition is
kept by the network, so copies and saved networks keep their composites. A
saved network holds it under the meta key `"_pypsa"`.

`Composites` and `Composite` are what a user reads and calls. The stored state
and the rules of every change to it are in `pypsa._registry`; the functions
`patches` and `providers` give a build the math and the data of the types.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from pypsa import _registry
from pypsa._registry import COMPOSITES, jsonable
from pypsa.components.types import all_components
from pypsa.components.types import get as component_type
from pypsa.network.transform import _build_suffixed_names
from pypsa.optimization.spec import read_override, spec
from pypsa.optimization.spec._render import PLACEHOLDERS
from pypsa.optimization.spec.sources import of_scenario

if TYPE_CHECKING:
    from pypsa import Network
    from pypsa.optimization.spec import Override
    from pypsa.optimization.spec.sources import NetworkSources

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
    frames = [
        of_scenario(c.static, n)[TAGS].assign(class_name=c.name)
        for c in n.components
        if "composite_type" in c.static
    ]
    if not frames:
        return pd.DataFrame(columns=["name", *TAGS, "class_name"])
    tags = pd.concat(frames).rename_axis("name").reset_index()
    return tags[tags.composite_type.notna() & (tags.composite_type != "")]


def _components() -> set[str]:
    """Return the names and list names of every component type, custom ones included."""
    return {*all_components, *(c.name for c in all_components.values())}


def instances(n: Network, ctype: str | None = None) -> list[str]:
    """Return the names of the instances of a composite type, or of all types."""
    tags = _tags(n)
    if ctype is not None:
        tags = tags[tags.composite_type == ctype]
    return list(tags.composite.unique())


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
        d = copy.deepcopy(read_override(definition))
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
        own = {k.removeprefix(f"{self.name}_") for s in self.math.values() for k in s}
        return [p for p in self.parameters if p in used - own]

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

    @property
    def entries(self) -> set[str]:
        """Return the names the type declares in the spec."""
        return {k for section in self.patch().values() for k in section}

    def _check(self) -> None:
        t = self.name
        base = spec()
        taken = _components() | {*base.dimensions, *PLACEHOLDERS}
        if not isinstance(t, str) or not t.isidentifier() or t in taken:
            msg = f"A composite type is named by an identifier that is no component, dimension or placeholder, not {t!r}; rename the type."
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
        anchor = self.members[self.anchor]
        used = {*component_type(anchor).defaults.index, *TAGS}
        if clash := sorted({self.column(p) for p in math} & used):
            msg = f"The math parameters of {t} take the columns {clash} of the {anchor} components, which PyPSA uses; rename them."
            raise ValueError(msg)
        if untyped := [p for p in math if self.parameters[p] is None]:
            msg = f"The math of {t} reads {untyped}, so they take a default, which sets their dtype."
            raise ValueError(msg)


class Composite:
    """A composite type registered on a network, with its instances.

    `n.composites["<type>"]` returns it. It reads only: add instances with
    `n.add("<type>", ...)` and remove them with `n.remove("<type>", ...)`.

    Examples
    --------
    >>> n = pypsa.Network()
    >>> n.add("Bus", "north")
    >>> n.composites.register(
    ...     {
    ...         "name": "battery",
    ...         "parameters": {"bus": None, "efficiency": 0.9},
    ...         "components": {
    ...             "Bus": {"dc": {}},
    ...             "Link": {
    ...                 "charger": {"bus0": "$bus", "bus1": "@dc"},
    ...                 "discharger": {
    ...                     "bus0": "@dc",
    ...                     "bus1": "$bus",
    ...                     "efficiency": "$efficiency",
    ...                 },
    ...             },
    ...         },
    ...     }
    ... )
    >>> n.add("battery", "bat", bus="north")
    >>> n.composites["battery"]
    Composite 'battery' with 1 instances
    >>> n.composites["battery"].members
             dc      charger      discharger
    battery
    bat  bat dc  bat charger  bat discharger
    >>> n.composites["battery"].static
               bus  efficiency
    battery
    bat      north         0.9

    """

    def __init__(self, n: Network, name: str) -> None:
        """Bind the type of this name to the network."""
        self._n = n
        self._name = name

    def __repr__(self) -> str:
        """Name the type and count its instances."""
        return f"Composite '{self.name}' with {len(instances(self._n, self.name))} instances"

    @property
    def name(self) -> str:
        """Return the name of the type."""
        return self._name

    @property
    def definition(self) -> dict[str, Any]:
        """Return the definition of the type as `n.composites.register` takes it, a copy."""
        return copy.deepcopy(_registry.entry(self._n, COMPOSITES, self.name))

    @cached_property
    def _type(self) -> CompositeType:
        """Return the type, read from the network."""
        return CompositeType.read(_registry.entry(self._n, COMPOSITES, self.name))

    def _add(
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
        unknown = kwargs.keys() - self._type.parameters.keys()
        values = self._type.parameters | kwargs
        missing = [p for p, v in values.items() if v is None]
        if unknown or missing:
            msg = f"{self.name} takes the parameters {sorted(self._type.parameters)}; unknown {sorted(unknown)}, missing {missing}."
            raise ValueError(msg)
        taken = names.intersection(instances(self._n))
        bad = [*taken, *names[names.duplicated() | (names == "")]]
        if bad:
            msg = f"Composite instances have unique, non-empty names; {bad} are taken or empty."
            raise ValueError(msg)
        laid = [
            self._laid(m, c, values, names, suffix)
            for m, c in self._type.members.items()
        ]
        columns = {c: self._n.c[c].static.columns for c in self._type.components}
        added: list[tuple[str, pd.Index]] = []
        try:
            for member, class_name, labels, attrs in laid:
                self._n.add(
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
                self._n.remove(class_name, labels)
            for class_name, before in columns.items():
                static = self._n.c[class_name].static
                static.drop(columns=static.columns.difference(before), inplace=True)
            raise
        return names

    def _remove(
        self, name: str | int | Sequence[int | str], suffix: str | Sequence[str] = ""
    ) -> None:
        """Remove instances: every member of each.

        Raises:
            ValueError: An instance does not exist, or a component carries a
                member the type does not have. Nothing is removed then.

        """
        names = _build_suffixed_names(name, suffix)
        tags = self._tagged()
        if len(unknown := names.difference(tags.composite)):
            msg = f"There are no {self.name} instances {list(unknown)}."
            raise ValueError(msg)
        for class_name, rows in tags[tags.composite.isin(names)].groupby("class_name"):
            self._n.remove(class_name, rows.name)

    def _tagged(self) -> pd.DataFrame:
        """Return the tags of the members of the instances of this type.

        Raises:
            ValueError: A component is tagged with a member the type does
                not have, or with a member of another component class.

        """
        tags = _tags(self._n)
        tags = tags[tags.composite_type == self.name]
        stored = _registry.entry(self._n, COMPOSITES, self.name)
        members = CompositeType(**stored).members
        stray = tags.composite_member.map(members) != tags.class_name
        if stray.any():
            found = list(tags.class_name[stray] + " " + tags.name[stray])
            msg = f"The components {found} are tagged as members of a {self.name}, but a {self.name} has only the members {members}. Correct or clear their `composite_member` tags."
            raise ValueError(msg)
        return tags

    def _laid(
        self,
        member: str,
        class_name: str,
        values: dict[str, Any],
        names: pd.Index,
        suffix: str | Sequence[str],
    ) -> tuple[str, str, pd.Index, dict[str, Any]]:
        """Return a member of the instances with its names and attributes.

        Raises:
            ValueError: A member of that name exists.

        """
        labels = names + f" {member}"
        if len(exist := labels.intersection(self._n.c[class_name].names)):
            msg = f"The {class_name} components {list(exist)} exist."
            raise ValueError(msg)
        template = self._type.templates[member]
        attrs = {
            a: self._value(v, values, names, labels, suffix)
            for a, v in template.items()
        }
        if member == self._type.anchor:
            for p in self._type.math_parameters:
                attrs[self._type.column(p)] = self._per_instance(
                    values[p], names, labels, suffix, p
                )
        return member, class_name, labels, attrs

    def _value(
        self,
        value: Any,
        values: dict[str, Any],
        names: pd.Index,
        labels: pd.Index,
        suffix: str | Sequence[str],
    ) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            return self._per_instance(values[value[1:]], names, labels, suffix)
        if isinstance(value, str) and value.startswith("@"):
            return (names + f" {value[1:]}").to_numpy(dtype=object)
        return value

    def _per_instance(
        self,
        value: Any,
        names: pd.Index,
        labels: pd.Index,
        suffix: str | Sequence[str],
        parameter: str = "",
    ) -> Any:
        """Return a parameter laid over the members: by instance where it is a Series or a DataFrame.

        A Series over the snapshots is a time series for all instances. A
        `suffix` string is added to the instances of a value, as `n.add` does.

        Raises:
            ValueError: A math parameter is a time series, or a value by
                instance leaves an instance out.

        """
        frame = isinstance(value, pd.DataFrame)
        series = isinstance(value, pd.Series)
        timed = frame or (series and value.index.equals(self._n.snapshots))
        if parameter and timed:
            msg = f"The math parameter {parameter} takes one value per instance, not a time series."
            raise ValueError(msg)
        if series and timed:
            return pd.DataFrame(dict.fromkeys(labels, value))
        if not (frame or series):
            return value
        if isinstance(suffix, str):
            value = value.rename(
                lambda s: str(s).removesuffix(suffix) + suffix, axis=int(frame)
            )
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
            ValueError: An instance lacks a member, or a component carries a
                member the type does not have.

        """
        tags = self._tagged()
        members = (
            tags.pivot(index="composite", columns="composite_member", values="name")
            .reindex(index=tags.composite.unique(), columns=list(self._type.members))
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
        """Return the parameters of each instance, read from the static attributes of its members.

        A parameter given as a time series shows its static value, as
        `n.c.<component>.static` does; the series is in the member's `dynamic`.
        """
        members = self.members
        read = {}
        for p, (member, attr) in self._type.homes.items():
            static = of_scenario(self._n.c[self._type.members[member]].static, self._n)
            read[p] = static[attr].reindex(members[member]).to_numpy()
        return pd.DataFrame(read, index=members.index)[list(self._type.parameters)]

    def _providers(self) -> dict[str, Callable[[NetworkSources, str], Any]]:
        """Return the data of the type's dimension, relation and parameters for a build."""
        t = self.name
        providers: dict[str, Callable[[NetworkSources, str], Any]] = {
            t: lambda s, name: self._built(s).index,
            f"{t}_members": lambda s, name: self._built(s).reset_index(),
        }
        for p in self._type.math_parameters:
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
                    for m, c in self._type.members.items()
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
        column = self._type.column(name.removeprefix(f"{self.name}_"))
        anchors = self._built(s)[self._type.anchor]
        static = self._n.c[self._type.members[self._type.anchor]].static
        values = [
            of_scenario(static, self._n, scenario)[column].reindex(anchors).to_numpy()
            for scenario in s.scenarios
        ]
        coords = {"scenario": s.scenarios, self.name: anchors.index}
        return xr.DataArray(np.array(values, dtype=s.dtype(name)), coords, list(coords))


class Composites(Mapping[str, Composite]):
    """The composite types registered on a network.

    `n.composites` reads as a mapping of each type name to its
    [Composite][pypsa.composites.Composite], in the order the types were
    registered. Change it with `register`, `unregister` and `clear`.

    Examples
    --------
    >>> n = pypsa.Network()
    >>> n.add("Bus", "north")
    >>> pair = {
    ...     "name": "pair",
    ...     "parameters": {"bus": None},
    ...     "components": {"Generator": {"up": {"bus": "$bus"}, "down": {"bus": "$bus"}}},
    ... }
    >>> n.composites.register(pair)
    >>> n.composites
    Composites ['pair']
    >>> n.add("pair", ["a", "b"], bus="north")
    >>> n.composites["pair"]
    Composite 'pair' with 2 instances
    >>> n.remove("pair", ["a", "b"])
    >>> n.composites.unregister("pair")
    >>> len(n.composites)
    0

    """

    def __init__(self, n: Network) -> None:
        """Bind the registry to the network."""
        self._n = n

    def __getitem__(self, name: str) -> Composite:
        """Return a registered type with its instances.

        Raises:
            KeyError: No type of this name is registered.
            ValueError: A component type of this name exists.

        """
        _registry.entry(self._n, COMPOSITES, name)
        if name in _components():
            msg = f"{name!r} names both a composite type and a component type, so `n.add` and `n.remove` cannot tell them apart. Rename the component type, or unregister the composite type with `n.composites.unregister({name!r})`."
            raise ValueError(msg)
        return Composite(self._n, name)

    def __iter__(self) -> Iterator[str]:
        """Iterate the names of the registered types in order."""
        return iter(_registry.entries(self._n, COMPOSITES))

    def __len__(self) -> int:
        """Count the registered types."""
        return len(_registry.entries(self._n, COMPOSITES))

    def __eq__(self, other: object) -> bool:
        """Whether both hold the same types in the same order."""
        if not isinstance(other, Composites):
            return NotImplemented
        return _registry.equal(self._n, other._n, COMPOSITES)

    def __repr__(self) -> str:
        """List the names of the registered types."""
        return f"Composites {list(self)}"

    def register(self, definition: Override, overwrite: bool = False) -> None:
        """Register a composite type.

        The definition is checked with the formulation, the other composite
        types and the stored overrides of the network. If it is refused, the
        network stays as it was.

        Parameters
        ----------
        definition : str | Path | dict
            The definition of the type: a YAML file, YAML text or a mapping.
            Its `name` names the type.
        overwrite : bool, default False
            Whether to replace a registered type of this name. The new type
            takes its place in the order. Instances of the type may stay
            where the new type keeps the members and the parameters.

        Raises
        ------
        ValueError
            The definition is malformed; a type of this name is registered
            and `overwrite` is False; the type declares a name another type
            declares; or instances exist that the new type does not fit.
        mathspec.errors.LanguageError
            The math does not hold with the formulation, or a stored override
            does not hold with the type.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> pair = {"name": "pair", "components": {"Generator": {"up": {}, "down": {}}}}
        >>> n.composites.register(pair)
        >>> n.composites.register(pair | {"description": "two generators"}, overwrite=True)
        >>> n.composites["pair"].definition["description"]
        'two generators'

        """
        t = CompositeType.read(definition)
        content = jsonable(asdict(t), f"The composite type {t.name!r}")
        _registry.check_free(self._n, COMPOSITES, t.name, overwrite)
        stored = _registry.entries(self._n, COMPOSITES)
        others = {k: CompositeType(**d) for k, d in stored.items()}
        old = others.pop(t.name, None)
        taken = {k for other in others.values() for k in other.entries}
        if clash := sorted(t.entries & taken):
            msg = f"The composite type {t.name!r} declares {clash}, which another registered type declares."
            raise ValueError(msg)
        fits = old is None or (old.members, old.homes) == (t.members, t.homes)
        if not fits and (found := instances(self._n, t.name)):
            msg = f"The {t.name} instances {found} exist, and the new type has other members or parameters. Remove them first with `n.remove({t.name!r}, ...)`."
            raise ValueError(msg)
        _registry.register(self._n, COMPOSITES, t.name, content)

    def unregister(self, name: str) -> None:
        """Unregister a composite type, also where its math no longer builds.

        Parameters
        ----------
        name : str
            The name of the type.

        Raises
        ------
        KeyError
            No type of this name is registered.
        ValueError
            Instances of the type exist, or a later type or a stored override
            builds on it. Nothing is unregistered then.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> pair = {"name": "pair", "components": {"Generator": {"up": {}, "down": {}}}}
        >>> n.composites.register(pair)
        >>> n.composites.unregister("pair")
        >>> n.composites
        Composites []

        """
        _registry.unregister(self._n, COMPOSITES, name)

    def clear(self) -> None:
        """Unregister all composite types, also where their math no longer builds.

        Raises
        ------
        ValueError
            Instances of a type exist, or a stored override builds on a type.
            Nothing is unregistered then.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> pair = {"name": "pair", "components": {"Generator": {"up": {}, "down": {}}}}
        >>> n.composites.register(pair)
        >>> n.composites.clear()
        >>> n.composites
        Composites []

        """
        _registry.clear(self._n, COMPOSITES)


def patches(n: Network) -> dict[str, dict[str, Any]]:
    """Return the spec patch of each registered type, by name in the order a build lays them."""
    return {k: c._type.patch() for k, c in n.composites.items()}


def providers(n: Network) -> dict[str, Callable[[NetworkSources, str], Any]]:
    """Return the data of every registered type for a build.

    Raises:
        ValueError: Components are tagged with a type that is not
            registered.

    """
    unknown = set(_tags(n).composite_type) - set(n.composites)
    if unknown:
        msg = f"Components are tagged with the composite types {sorted(unknown)}, which are not registered; register them with `n.composites.register`."
        raise ValueError(msg)
    return {k: f for c in n.composites.values() for k, f in c._providers().items()}
