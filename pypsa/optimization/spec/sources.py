# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep: a `pypsa.Network` read as the sources the spec declares.

Every dimension, relation and parameter of the spec is one key of
`NetworkSources`. linopy reads each key once, when the build needs it. A
parameter named `<Component>_<attr>` after a PyPSA attribute is read off
`c.da`, a relation `<Component>_<column>` off a static column, and a
dimension takes the labels its relations map into it. Every other key is
derived by the one function registered for it with `source`, as its
description in the spec states. This module registers the labels and
weightings; a module beside it, one per trait or topic, as `_commitment`,
registers the keys of its trait or topic.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import xarray as xr

from pypsa._options import options
from pypsa.components.common import invariant
from pypsa.optimization.spec._render import (
    PLACEHOLDERS,
    WORD,
    formulated,
    renamer,
    symbols,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Hashable

    from mathspec.spec import Spec

    from pypsa import Network
    from pypsa.components import Components
    from pypsa.components._types.mixin.multiports import _Multiport

#: The scenario label of a network that has none.
BASE_SCENARIO = "base"
#: The period label of a run without investment periods.
NO_PERIOD = 0

_SOURCES: dict[str, Callable[..., Any]] = {}


def source[F: Callable[..., Any]](*keys: str) -> Callable[[F], F]:
    """Register `f` as the one derivation of each key, called as `f(s, name)`.

    A key with the placeholders of a trait, as `COMPONENT_activity_mask`, stands for that
    source of every component with a formulation; `f` takes the component as
    `c` then.

    Raises
    ------
    ValueError
        A key has a derivation already.

    """

    def register(f: F) -> F:
        for key in keys:
            if key in _SOURCES:
                msg = f"{key} is derived by {_SOURCES[key].__qualname__} already."
                raise ValueError(msg)
            _SOURCES[key] = f
        return f

    return register


def _providers(
    components: Collection[Components],
    declared: Collection[str],
    sources: Mapping[str, Callable[..., Any]] = _SOURCES,
) -> dict[str, Callable[..., Any]]:
    """Return the derivation in `sources` of each name in `declared`, a placeholder key stated per component that has its placeholders."""
    general = {
        k: f for k, f in sources.items() if PLACEHOLDERS.isdisjoint(WORD.findall(k))
    }
    stated = {k: f for k, f in sources.items() if k not in general}
    providers = {k: f for k, f in general.items() if k in declared}
    for c in components:
        rename = renamer(c)
        known = symbols(c).keys()
        for key, f in stated.items():
            if not PLACEHOLDERS.intersection(WORD.findall(key)) <= known:
                continue
            name = rename(key)
            if name not in declared:
                continue
            if name in providers:
                msg = f"{name} has two derivations."
                raise ValueError(msg)
            providers[name] = functools.partial(f, c=c)
    return providers


class NetworkSources(Mapping[str, Any]):
    """The sources the spec reads, derived from a network on access.

    Parameters
    ----------
    n : pypsa.Network
        The network to build.
    spec : mathspec.spec.Spec
        The spec the sources are for; its declarations are the keys.
    network_index, model_index : pandas.Index
        The build's snapshots in the network's labelling and in the model's.
    multi_invest : bool
        Whether the run optimises over investment periods.
    transmission_losses : dict | None
        The normalised loss configuration, or None for a lossless run.
    committable_big_m : float | None
        PyPSA's `committable_big_m` keyword.
    cycle_basis_method : str
        The cycle basis method of the KVL rows.
    branch_outages : pandas.Index | None
        The outages of a security-constrained run.
    providers : Mapping | None
        Further derivations of this build, called as `f(sources, name)`.

    """

    def __init__(
        self,
        n: Network,
        spec: Spec,
        network_index: pd.Index,
        model_index: pd.Index,
        *,
        multi_invest: bool = False,
        transmission_losses: dict | None = None,
        committable_big_m: float | None = None,
        cycle_basis_method: str = "bfs-refined",
        branch_outages: Any = None,
        providers: Mapping[str, Callable[..., Any]] | None = None,
    ) -> None:
        """Initialise the sources of one build."""
        self.n = n
        self.spec = spec
        self.network_index = network_index
        self.model_index = model_index
        self.multi_invest = multi_invest
        self.transmission_losses = transmission_losses
        self.committable_big_m = committable_big_m
        self.cycle_basis_method = cycle_basis_method
        self.branch_outages = branch_outages
        self.units = {c._unit_dim: c for c in formulated(n)}
        self._keys = dict.fromkeys(
            [*spec.dimensions, *spec.relations, *spec.parameters]
        )
        self._providers = _providers(self.units.values(), self._keys)
        providers = dict(providers or {})
        if clash := self._providers.keys() & providers.keys():
            msg = f"{sorted(clash)} have two derivations."
            raise ValueError(msg)
        self._providers |= providers
        self._cache: dict[str, Any] = {}

    def __iter__(self) -> Iterator[str]:
        """Iterate the declared names."""
        return iter(self._keys)

    def __len__(self) -> int:
        """Count the declared names."""
        return len(self._keys)

    def __contains__(self, name: object) -> bool:
        """Whether the spec declares `name`."""
        return name in self._keys

    def __getitem__(self, name: str) -> Any:
        """Derive the source declared as `name`."""
        return self.memo(name, lambda: self._derive(name))

    def memo(self, key: str, derive: Callable[[], Any]) -> Any:
        """Return `derive()`, computed once per build under `key`."""
        if key not in self._cache:
            self._cache[key] = derive()
        return self._cache[key]

    def _derive(self, name: str) -> Any:
        derive = self.derivation(name)
        if name in self.spec.parameters and self.unpopulated(name):
            return self.filled(name, "" if self.dtype(name) is object else 0)
        value = derive()
        return self.frame(name, value) if isinstance(value, xr.DataArray) else value

    def derivation(self, name: str) -> Callable[[], Any]:
        """Return the derivation of `name`: its provider, else the PyPSA data it names.

        A parameter `<Component>_<attr>` reads the attribute, or takes its
        `missing:` value where the component lacks it, a relation
        `<Component>_<column>` reads the static column, and a dimension takes
        the labels the relations map into it.

        Raises
        ------
        NotImplementedError
            `name` has none of these.

        """
        if name in self._providers:
            return functools.partial(self._providers[name], self, name)
        c = self.component(name)
        if name in self.spec.parameters and c is not None:
            missing = self.spec.parameters[name].missing
            attr = name.removeprefix(f"{c.name}_")
            absent = attr not in c.static and attr not in c.dynamic
            if absent and missing != "refused":
                value = np.nan if missing == "absent" else missing
                return functools.partial(self.filled, name, value)
            self.attr(c, name)
            return functools.partial(self.attribute, c, name)
        if name in self.spec.relations and c is not None:
            return functools.partial(self.mapping, c, name)
        if name in self.spec.dimensions and self.mapped_into(name):
            return functools.partial(self.mapped, name)
        msg = f"PyPSA has no data for {name!r}."
        if name in self.spec.dimensions:
            msg += f" No relation maps components into it. Declare one, such as `{{key: Link, values: {name}}}`."
        else:
            msg += " Name a parameter `<Component>_<attribute>` after an attribute of the component."
        raise NotImplementedError(msg)

    def mapping(self, c: Components, name: str) -> pd.DataFrame:
        """Map each unit to the label in the static column a relation `<Component>_<column>` names; a blank cell maps it to none.

        Raises
        ------
        NotImplementedError
            The relation is not keyed by the units alone with one column as its
            values, or `c` has no such column.
        ValueError
            The column differs across scenarios.

        """
        column = name.removeprefix(f"{c.name}_")
        (_, key), *values = self.spec.relations[name].pairs
        if key != c._unit_dim or len(values) != 1:
            msg = f"PyPSA has no data for the relation {name!r}. It reads the column {column!r} of {c.name}, so declare it as `{{key: {c._unit_dim}, values: <dimension>}}`."
            raise NotImplementedError(msg)
        if column not in c.static:
            msg = f"PyPSA has no data for the relation {name!r}: {c.name} has no column {column!r}. Add the column to the {c.name} components."
            raise NotImplementedError(msg)
        labels = invariant(c.static[column]).loc[self.static(c).index]
        labels = labels[~_blank(labels)]
        return pd.DataFrame(
            {
                key: labels.index.to_numpy(dtype=object),
                values[0][0]: labels.to_numpy(dtype=object),
            }
        )

    def mapped_into(self, dim: str) -> list[tuple[str, str]]:
        """Return the relations, with their column, that map units into dimension `dim`."""
        relations = self.spec.relations.items()
        return [
            (k, role)
            for k, r in relations
            for role, d in r.pairs[len(r.key_roles) :]
            if d == dim
        ]

    def mapped(self, dim: str) -> pd.Index:
        """Return the labels the relations map into dimension `dim`, in their order of first appearance."""
        labels = pd.concat([self[k][role] for k, role in self.mapped_into(dim)])
        return pd.Index(labels.unique(), dtype=object)

    def component(self, name: str) -> Components | None:
        """Return the component a source name opens with, None for another name."""
        prefix = name.partition("_")[0]
        return next((c for c in self.units.values() if c.name == prefix), None)

    # -- labels -----------------------------------------------------------

    @functools.cached_property
    def scenarios(self) -> pd.Index:
        """The scenario labels, one placeholder for a network without scenarios."""
        if self.n.has_scenarios:
            return pd.Index(self.n.scenarios, name="scenario")
        return pd.Index([BASE_SCENARIO], name="scenario")

    @functools.cached_property
    def snapshots(self) -> pd.Index:
        """The model's snapshot labels, as objects so any PyPSA label passes."""
        return pd.Index(
            self.model_index.to_numpy(dtype=object), dtype=object, name=None
        )

    @functools.cached_property
    def positions(self) -> np.ndarray:
        """The positions of the build's snapshots in `n.snapshots`."""
        return self.n.snapshots.get_indexer(self.network_index)

    @functools.cached_property
    def every_snapshot(self) -> bool:
        """Whether the build spans every snapshot of `n`, in its order."""
        positions = self.positions
        return len(positions) == len(self.n.snapshots) and bool(
            (positions == np.arange(len(positions))).all()
        )

    @functools.cached_property
    def periods(self) -> pd.Index:
        """The investment periods the build spans."""
        if self.multi_invest:
            return pd.Index(self.network_index.unique("period"), dtype=int)
        return pd.Index([NO_PERIOD], dtype=int)

    @functools.cached_property
    def period_of(self) -> np.ndarray:
        """The period label of each build snapshot."""
        if self.multi_invest:
            return np.asarray(self.network_index.get_level_values("period"), dtype=int)
        return np.full(len(self.network_index), NO_PERIOD, dtype=int)

    def names(self, c: Components) -> pd.Index:
        """Return the labels of a component's units."""
        return self.coords(c._unit_dim)

    def ports(self, c: Components) -> pd.DataFrame:
        """Return the ports of a multiport read long: one row per unit and set `bus<k>`, with its number `k`."""
        return self.memo(
            f"{c.name} ports", lambda: _ports_long(self, cast("_Multiport", c))
        )

    # -- arrays -----------------------------------------------------------

    def declared(self, name: str) -> tuple[str, ...]:
        """Return the declared dims of a parameter."""
        return tuple(self.spec.parameters[name].dims)

    def coords(self, dim: str) -> pd.Index:
        """Return the labels of a dimension, one index per build, named after it.

        Every array of the build shares it, so its labels are hashed once.
        """
        return self.memo(f"{dim} labels", lambda: self[dim].rename(dim))

    def frame(self, name: str, da: xr.DataArray) -> xr.DataArray:
        """`da` broadcast over and laid out in the declared dims of `name`."""
        for dim in self.missing(name, da.dims):
            da = da.expand_dims({dim: self.coords(dim)})
        return da.transpose(*self.declared(name)).rename(name)

    def missing(self, name: str, laid: Collection[Hashable]) -> list[str]:
        """Return the declared dims of `name` that `laid` lacks.

        Raises
        ------
        ValueError
            `laid` holds a dim `name` does not declare.

        """
        dims = self.declared(name)
        extra = [d for d in laid if d not in dims]
        if extra:
            msg = f"{name}: dims {extra} are not declared ({dims})"
            raise ValueError(msg)
        return [d for d in dims if d not in laid]

    def unpopulated(self, name: str) -> bool:
        """Whether a declared dim of parameter `name` has no label, so it has no cell to read."""
        return any(not len(self.coords(dim)) for dim in self.declared(name))

    def dtype(self, name: str) -> type:
        """Return the numpy dtype of the declared dtype of parameter `name`."""
        declared = self.spec.parameters[name].dtype
        return {"bool": bool, "int": int, "str": object}.get(declared, float)

    def filled(self, name: str, value: Any) -> xr.DataArray:
        """Return a grid over the declared dims of `name`, every cell `value`; NaN makes an int or bool grid float."""
        dims = self.declared(name)
        coords = {d: self.coords(d) for d in dims}
        dtype = self.dtype(name)
        if dtype in (int, bool) and isinstance(value, float) and np.isnan(value):
            dtype = float
        shape = [len(c) for c in coords.values()]
        return xr.DataArray(np.full(shape, value, dtype=dtype), coords, dims, name)

    def static(self, c: Components) -> pd.DataFrame:
        """Return the static data of a component's units, of the first scenario."""
        return self.memo(f"{c.name} static", lambda: self._static(c))

    def _static(self, c: Components) -> pd.DataFrame:
        static = of_scenario(c.static, self.n)
        return static[static.index.isin(self.names(c))]

    def column(self, c: Components, attr: str, dtype: Any) -> xr.DataArray:
        """Return a static column of a component's units over its unit dim."""
        column = self.static(c)[attr]
        index = {c._unit_dim: self.labels(c, column.index)}
        return xr.DataArray(column.to_numpy(dtype=dtype), index, c._unit_dim)

    def labels(self, c: Components, index: pd.Index) -> pd.Index | np.ndarray:
        """Return the unit labels in `index`: the build's shared index where they are its units."""
        names = self.names(c)
        return names if index.equals(names) else index.to_numpy(dtype=object)

    def unit_array(
        self,
        c: Components,
        da: xr.DataArray,
        snapshots: pd.Index | np.ndarray | None = None,
    ) -> xr.DataArray:
        """Return a `c.da` array relabelled into the spec's dims.

        Its snapshots are the build's, or *snapshots* where given, which
        labels the snapshots of `da` as they are.
        """
        dim = c._unit_dim
        if "name" in da.dims:
            names = self.names(c)
            if not da.indexes["name"].equals(names):
                da = da.sel(name=da.indexes["name"].isin(names))
            labels = self.labels(c, da.indexes["name"])
            da = da.rename(name=dim).assign_coords({dim: labels})
        if "snapshot" in da.dims:
            if snapshots is None:
                snapshots = self.snapshots
                if da.sizes["snapshot"] == len(self.n.snapshots):
                    da = da.isel(snapshot=self.positions)
            tied = [
                k
                for k in da.coords
                if k == "snapshot" or (k not in da.dims and "snapshot" in da[k].dims)
            ]
            da = da.drop_vars(tied).assign_coords(snapshot=snapshots)
        if "scenario" not in da.dims and self.n.has_scenarios:
            da = da.expand_dims(scenario=self.scenarios)
        elif "scenario" in da.dims and not self.n.has_scenarios:
            da = da.squeeze("scenario", drop=True)
        if not self.n.has_scenarios:
            da = da.expand_dims(scenario=self.scenarios)
        return da

    def da(self, c: Components, attr: str) -> xr.DataArray:
        """Return a PyPSA attribute of a component in the spec's dims."""
        values = self.numeric(c, attr)
        if values is None:
            return self.read(c, attr)
        dims = self.laid(c, values)
        coords = {d: self.coords(d) for d in dims}
        return xr.DataArray(values, coords, dims, attr)

    def read(self, c: Components, attr: str) -> xr.DataArray:
        """Return a PyPSA attribute of a component in the spec's dims, read through `c.da`."""
        raw = self.unit_array(c, c.da[attr])
        return raw.drop_vars([k for k in raw.coords if k not in raw.dims])

    def laid(self, c: Components, values: np.ndarray) -> tuple[str, ...]:
        """Return the dims `values` lays a PyPSA attribute over."""
        varying = ("snapshot",) if values.ndim == 3 else ()
        return ("scenario", *varying, c._unit_dim)

    def kept(self, c: Components) -> np.ndarray | None:
        """Return the positions of the build's units in `c.static`, in their order, None where they are all of it.

        Raises
        ------
        ValueError
            A unit of the build is not in `c.static`.

        """
        return self.memo(f"{c.name} kept", lambda: self._kept(c))

    def _kept(self, c: Components) -> np.ndarray | None:
        names = self.names(c)
        if c.static.index.equals(names):
            return None
        kept = c.static.index.get_indexer(names)
        if (kept == -1).any():
            msg = f"{c.name} has no static data of {list(names[kept == -1])}"
            raise ValueError(msg)
        return kept

    def numeric(self, c: Components, attr: str) -> np.ndarray | None:
        """Return a numeric PyPSA attribute over (scenario, snapshot, unit) or (scenario, unit), read off pandas.

        It holds what `unit_array` makes of `c.da[attr]`, without a step
        through xarray. None for what only that reads: a network with
        scenarios, the activity mask, labels and an attribute the
        component lacks. Under `options.debug.runtime_verification` it is
        None too, so every read passes the integrity check of `c.da`.
        """
        if self.n.has_scenarios or options.debug.runtime_verification:
            return None
        if attr == "active" or attr not in c.static:
            return None
        column = c.static[attr].to_numpy()
        if attr not in c.dynamic.keys():
            values = column
        elif c.dynamic[attr].columns.empty:
            values = np.broadcast_to(column, (len(self.n.snapshots), column.size))
        else:
            values = c._as_dynamic(attr).to_numpy()
        if values.dtype.kind not in "biuf":
            return None
        kept = self.kept(c)
        if kept is not None:
            values = values[..., kept]
        if values.ndim == 2 and not self.every_snapshot:
            values = values[self.positions]
        return values[np.newaxis]

    def attr(self, c: Components, name: str) -> str:
        """Return the PyPSA attribute a parameter `<Component>_<attr>` names.

        Raises
        ------
        NotImplementedError
            `c` has no such attribute.
        ValueError
            The attribute varies by snapshot, the spec holds it static.

        """
        attr = name.removeprefix(f"{c.name}_")
        if attr not in c.static and attr not in c.dynamic:
            msg = f"PyPSA has no data for the parameter {name!r}: {c.name} has no attribute {attr!r}. Add it as a column of the {c.name} components."
            raise NotImplementedError(msg)
        varies = attr == "active" or attr in c.dynamic.keys()
        if varies and "snapshot" not in self.declared(name):
            msg = f"{name} varies by snapshot in the network, but its spec entry has no `snapshot` dim. Add `snapshot` to its `dims`."
            raise ValueError(msg)
        return attr

    def attribute(self, c: Components, name: str) -> xr.DataArray:
        """Return a parameter `<Component>_<attr>` named after a PyPSA attribute, read off `c.da`."""
        attr = self.attr(c, name)
        values = self.numeric(c, attr)
        if values is not None:
            return self.framed(name, self.cast(name, values), self.laid(c, values))
        da = self.cast(name, self.read(c, attr))
        if "scenario" in da.dims and "scenario" not in self.declared(name):
            da = da.isel(scenario=0, drop=True)
        return self.frame(name, da)

    def cast[A: (np.ndarray, xr.DataArray)](self, name: str, values: A) -> A:
        """Return `values` in the declared dtype of parameter `name`, an int with a NaN as it is."""
        declared = self.spec.parameters[name].dtype
        if declared in ("bool", "float"):
            return values.astype(self.dtype(name))
        if declared == "int" and not pd.isna(values).any():
            return values.astype(int)
        return values

    def framed(
        self, name: str, values: np.ndarray, laid: tuple[str, ...]
    ) -> xr.DataArray:
        """`values` over `laid` broadcast over and laid out in the declared dims of `name`, as `frame` lays an array."""
        dims = self.declared(name)
        if "scenario" not in dims:
            values, laid = values[0], laid[1:]
        for dim in self.missing(name, laid):
            shape = (len(self.coords(dim)), *values.shape)
            values, laid = np.broadcast_to(values, shape), (dim, *laid)
        values = values.transpose([laid.index(d) for d in dims])
        return xr.DataArray(values, {d: self.coords(d) for d in dims}, dims, name)

    def active(self, c: Components) -> xr.DataArray:
        """Whether each unit is active at each build snapshot, (unit, snapshot)."""
        return self.memo(f"{c.name} active", lambda: self._active(c))

    def _active(self, c: Components) -> xr.DataArray:
        mask = c.get_activity_mask(self.network_index)
        if isinstance(mask.columns, pd.MultiIndex):
            mask = mask.T.groupby(level="name").any().T
        names = self.names(c)
        mask = mask.reindex(columns=names)
        return xr.DataArray(
            mask.to_numpy(dtype=bool),
            coords={"snapshot": self.snapshots, c._unit_dim: names},
            dims=["snapshot", c._unit_dim],
        )

    def active_by_period(self, c: Components) -> xr.DataArray:
        """Whether each unit is active in each period, (unit, period)."""
        return self.memo(f"{c.name} active by period", lambda: self._by_period(c))

    def _by_period(self, c: Components) -> xr.DataArray:
        active = self.active(c)
        if not len(self.period_of):
            return active.isel(snapshot=0, drop=True).expand_dims(period=self.periods)
        by_period = active.assign_coords(period=("snapshot", self.period_of))
        by_period = by_period.groupby("period").any("snapshot")
        return by_period.reindex(period=self.periods, fill_value=False)

    def cycle_matrices(self, period: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the cycle matrix of a period, plain and weighted, branches by cycles."""
        return self.n._cycle_matrices(period, self.cycle_basis_method)

    def outage_factors(
        self, period: int | None, outages: pd.MultiIndex
    ) -> pd.DataFrame:
        """Return the outage factors of a period's sub-networks, branches by outage."""
        n = self.n
        topology = self.memo(
            "outage topology",
            lambda: n.get_scenario(n.scenarios[0]) if n.has_scenarios else n,
        )
        topology.determine_network_topology(
            investment_period=period, skip_isolated_buses=True
        )
        frames = []
        for sub_network in topology.c.sub_networks.static.obj:
            branches = sub_network.branches_i()
            outaged = branches.intersection(outages)
            if outaged.empty:
                continue
            sub_network.calculate_BODF()
            frame = pd.DataFrame(sub_network.BODF, index=branches, columns=branches)
            frames.append(frame[outaged].droplevel(0, axis=1))
        return pd.concat(frames) if frames else pd.DataFrame()

    def over_snapshots(self, values: np.ndarray) -> xr.DataArray:
        """Return `values` laid over the build's snapshots."""
        return xr.DataArray(
            values, coords={"snapshot": self.snapshots}, dims="snapshot"
        )


def of_scenario(static: pd.DataFrame, n: Network, scenario: Any = None) -> pd.DataFrame:
    """Return the static data of one scenario, by default the first, which every scenario shares."""
    if not n.has_scenarios:
        return static
    if static.empty:
        return static.droplevel("scenario")
    return static.xs(n.scenarios[0] if scenario is None else scenario, level="scenario")


def _blank(labels: pd.Series) -> pd.Series:
    """Whether each label is blank: missing, or empty but for whitespace."""
    return labels.isna() | (labels.astype(str).str.strip() == "")


def _ports_long(s: NetworkSources, c: _Multiport) -> pd.DataFrame:
    dim = c._unit_dim
    static = s.static(c)
    frames = []
    for k in sorted(map(int, c.ports)):
        col = f"bus{k}"
        present = ~_blank(static[col])
        frames.append(
            pd.DataFrame(
                {
                    dim: static.index[present].to_numpy(dtype=object),
                    "port": k,
                    "bus": static.loc[present, col].to_numpy(),
                }
            )
        )
    columns = [dim, "port", "bus"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


# -- labels ---------------------------------------------------------------


@source("scenario", "snapshot", "period")
def _labels(s: NetworkSources, name: str) -> pd.Index:
    return getattr(s, f"{name}s")


def network_labels(labels: pd.Index) -> pd.Index:
    """Return component labels in their network dtype, as objects where there are none."""
    return pd.Index(labels if len(labels) else labels.astype(object), name=None)


@source("bus")
def _buses(s: NetworkSources, name: str) -> pd.Index:
    return network_labels(s.n.components.buses.names)


@source("carrier")
def _carriers(s: NetworkSources, name: str) -> pd.Index:
    """Every carrier a unit names, defined or not."""
    named = [
        c.static["carrier"].unique() for c in s.units.values() if "carrier" in c.static
    ]
    labels = pd.Index([], dtype=object)
    for carriers in [s.n.components.carriers.names, *named]:
        labels = labels.append(pd.Index(carriers).difference(labels))
    return pd.Index(labels.unique(), dtype=object)


@source("UNIT")
def _units(s: NetworkSources, name: str, c: Components) -> pd.Index:
    """Return a component's active assets, as PyPSA builds them.

    A one-port unit at a bus the network lacks is left out, as PyPSA's
    nodal balance never reaches it.
    """
    names = network_labels(c.active_assets)
    if "bus" not in c.static:
        return names
    bus = of_scenario(c.static, s.n)["bus"].reindex(names)
    return names[bus.isin(s.n.c.buses.names).to_numpy()]


@source("port")
def _port_numbers(s: NetworkSources, name: str) -> pd.Index:
    """Every port number a multiport unit of the build uses."""
    numbers = [
        s.ports(c)["port"] for c in s.units.values() if hasattr(c, "_output_ports")
    ]
    return pd.Index(sorted(set().union(*numbers)), dtype=int)


# -- weightings and globals -----------------------------------------------


@source(
    "snapshot_weighting_objective",
    "snapshot_weighting_generators",
    "snapshot_weighting_stores",
)
def _snapshot_weighting(s: NetworkSources, name: str) -> xr.DataArray:
    weighting = s.n.snapshot_weightings[name.removeprefix("snapshot_weighting_")]
    return s.over_snapshots(weighting.iloc[s.positions].to_numpy(dtype=float))


@source("scenario_weight")
def _scenario_weight(s: NetworkSources, name: str) -> xr.DataArray:
    values = np.ones(1)
    if s.n.has_scenarios:
        values = s.n.scenario_weightings["weight"].reindex(s.scenarios).to_numpy(float)
    return xr.DataArray(values, coords={"scenario": s.scenarios}, dims="scenario")


@source("period_weighting_objective", "period_weighting_years")
def _period_weighting(s: NetworkSources, name: str) -> xr.DataArray:
    """Return a column of `investment_period_weightings`, one under a single period."""
    values = np.ones(len(s.periods))
    if s.multi_invest:
        weights = s.n.investment_period_weightings[
            name.removeprefix("period_weighting_")
        ]
        values = weights.reindex(s.periods).to_numpy(dtype=float)
    return xr.DataArray(values, coords={"period": s.periods}, dims="period")


@source("snapshot_period")
def _snapshot_period(s: NetworkSources, name: str) -> pd.DataFrame:
    return pd.DataFrame({"snapshot": s.snapshots, "period": s.period_of})


@source("transmission_losses")
def _transmission_losses(s: NetworkSources, name: str) -> bool:
    return s.transmission_losses is not None


@source("CVaR_alpha", "CVaR_omega")
def _cvar(s: NetworkSources, name: str) -> float | xr.DataArray:
    preference = getattr(s.n, "risk_preference", None) if s.n.has_scenarios else None
    if not preference:
        return xr.DataArray(np.nan)
    return float(preference[name.removeprefix("CVaR_")])


@source("Bus_attached")
def _bus_attached(s: NetworkSources, name: str) -> xr.DataArray:
    """Whether any active unit has a port at each bus, (bus, snapshot)."""
    buses = s.coords("bus")
    attached = np.zeros((len(buses), len(s.snapshots)), dtype=bool)
    for c in s.units.values():
        active = s.active(c).transpose(c._unit_dim, "snapshot").to_numpy()
        static = s.static(c)
        for port in c.ports:
            at = buses.get_indexer(static[f"bus{port}"].to_numpy(dtype=object))
            found = at != -1
            np.logical_or.at(attached, at[found], active[found])
    coords = {"bus": buses, "snapshot": s.coords("snapshot")}
    return xr.DataArray(attached, coords, ("bus", "snapshot"))


def relation(s: NetworkSources, name: str, c: Components) -> pd.DataFrame:
    """Map each unit to the label of its column, a bus or a carrier; no unit where the component lacks it."""
    column = name.removeprefix(f"{c.name}_")
    static = s.static(c)
    if column not in static:
        static = static.iloc[:0].reindex(columns=[column])
    return pd.DataFrame(
        {
            c._unit_dim: static.index.to_numpy(dtype=object),
            column.rstrip("01"): static[column].to_numpy(dtype=object),
        }
    )


def optional(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """Return the attribute, NaN where the component has none."""
    attr = name.removeprefix(f"{c.name}_")
    if attr in c.static or attr in c.dynamic:
        return s.attribute(c, name)
    return s.filled(name, np.nan)


def finite(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """Return the attribute, absent where infinite, so a unit without a limit has no row."""
    limit = s.attribute(c, name)
    return limit.where(np.isfinite(limit))
