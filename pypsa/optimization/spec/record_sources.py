# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Experiment: a datarecord record read as the sources the spec declares.

`RecordSources` gives `linopy.Model.from_spec` the keys `NetworkSources`
gives it, read off a datarecord record with DuckDB. The derivations of
`sources` run unchanged: they read a component's data through the methods
`RecordSources` answers from the record. What a derivation reads off a
`pypsa.Network` besides is read off the network `PyPSA.build` makes of the
record, and `RecordSources.touched` records it per key. With `strict` such
a read raises, so a strict build proves that no network was needed.

See `dev/datarecord-sources.md` for the findings.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from datarecord import Flags, settle
from datarecord.schema.categories import matching
from datarecord.schema.schema import unpacked
from datarecord.tools.pypsa import (
    _AXIS_TYPES,
    _PORT_SUFFIX,
    _WEIGHTINGS,
    BUS,
    CARRIER,
    CONNECTION,
    CTYPE,
    ENTITY,
    PERIOD,
    PORT,
    SCENARIO,
    SNAPSHOT,
    NetworkShape,
    PyPSA,
    _port_table,
    _scenario_weights,
    _tuples_of,
    _weightings,
)

from pypsa.components._types.stores import needs_split_dispatch
from pypsa.components.components import (
    committable_big_m_scale,
    committable_big_m_values,
)
from pypsa.constants import HOURS_PER_YEAR
from pypsa.costs import periodized_cost
from pypsa.network.graph import _cycle_matrix, _incidence_matrix, _sub_network_labels
from pypsa.network.power_flow import _bodf, _ptdf, _susceptance_matrices
from pypsa.optimization.spec._render import formulated
from pypsa.optimization.spec.sources import (
    _SOURCES,
    BASE_SCENARIO,
    NetworkSources,
    _providers,
    of_scenario,
    source,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from datarecord import RecordLike
    from duckdb import DuckDBPyRelation
    from mathspec.spec import Spec

    from pypsa import Network
    from pypsa.components import Components


def _sub_networks(
    buses: pd.Series, branches: pd.DataFrame
) -> Iterator[tuple[pd.Series, pd.DataFrame]]:
    """Yield the buses and branches of each sub-network of more than one bus.

    Parameters
    ----------
    buses : pandas.Series
        The carrier of each bus.
    branches : pandas.DataFrame
        The passive branches by `(type, name)`, types sorted, with `bus0`
        and `bus1`, whether each `connects` its buses and whether it is a
        `member` of the sub-network of its `bus0`.

    """
    connecting = branches[branches["connects"].to_numpy(dtype=bool)]
    labels = _sub_network_labels(buses.index, connecting["bus0"], connecting["bus1"])
    of_branch = branches["bus0"].map(labels).where(branches["member"]).to_numpy()
    sizes = labels.value_counts().sort_index()
    for i in sizes.index[sizes > 1]:
        yield buses[labels.to_numpy() == i], branches[of_branch == i]


def _cycle_matrices(
    buses: pd.Series, branches: pd.DataFrame, method: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the cycle matrix of a network stated as tables, plain and weighted.

    A cycle weighs a branch by its `x_pu_eff` in a sub-network of carrier
    `AC` and by its `r_pu_eff` in any other.

    Parameters
    ----------
    buses, branches
        As `_sub_networks` takes them; `branches` also holds `active`,
        `x_pu`, `x_pu_eff` and `r_pu_eff`.
    method : str
        The cycle basis method.

    """
    branches = branches.rename_axis(index=["type", "name"])
    cycles = []
    weights = branches["r_pu_eff"].copy()
    for sub_buses, rows in _sub_networks(buses, branches):
        edges = rows["active"].astype(bool) & ~np.isinf(rows["x_pu"])
        C = _cycle_matrix(sub_buses.index, rows["bus0"], rows["bus1"], edges, method)
        if sub_buses.iat[0] == "AC":
            weights[rows.index] = rows["x_pu_eff"]
        if C.size:
            cycles.append(pd.DataFrame(C.todense(), index=rows.index))
    if not cycles:
        return pd.DataFrame(), pd.DataFrame()
    plain = pd.concat(cycles, axis=1, ignore_index=True).fillna(0)
    weighted = plain.where(plain == 0, plain.multiply(weights[plain.index], axis=0))
    branches_i = branches.loc[plain.index.unique("type")].index
    return tuple(
        df.reindex(branches_i, fill_value=0).rename_axis(columns="cycle")
        for df in (plain, weighted)
    )


def _outage_factors(
    buses: pd.Series, branches: pd.DataFrame, outages: pd.MultiIndex
) -> pd.DataFrame:
    """Return the outage factors of a network stated as tables, branches by outaged branch names.

    The first bus of a sub-network is its slack. A branch has the
    impedance `r_pu_eff` in a sub-network of carrier `DC` and `x_pu_eff`
    in any other.

    Parameters
    ----------
    buses, branches
        As `_cycle_matrices` takes them.
    outages : pandas.MultiIndex
        The outaged branches, `(type, name)`.

    """
    branches = branches.rename_axis(index=["type", "name"])
    frames = []
    for sub_buses, rows in _sub_networks(buses, branches):
        outaged = rows.index.intersection(outages)
        if outaged.empty:
            continue
        live = rows[rows["active"].to_numpy(dtype=bool)]
        K = _incidence_matrix(sub_buses.index, live["bus0"], live["bus1"])
        z = live["r_pu_eff" if sub_buses.iat[0] == "DC" else "x_pu_eff"]
        _, H, B = _susceptance_matrices(z.to_numpy(dtype=float), K)
        frame = pd.DataFrame(
            _bodf(_ptdf(B, H), K), index=rows.index, columns=rows.index
        )
        frames.append(frame[outaged].droplevel(0, axis=1))
    return pd.concat(frames) if frames else pd.DataFrame()


#: What a component states of its type, whatever units it holds.
COMPONENT_META = frozenset(
    {
        "name",
        "list_name",
        "category",
        "_unit_dim",
        "_formulation",
        "_operational_attrs",
        "_power_variable",
        "_level_attrs",
        "_coefficient_attr",
        "_port_suffix",
        "_port_coefficient_attr",
        "_unsuffixed_attrs",
        "get_delay_source_indexer",
    }
)
#: What a network states of its component types, whatever it holds.
NETWORK_META = frozenset(
    {
        "one_port_components",
        "branch_components",
        "passive_branch_components",
        "controllable_branch_components",
        "passive_one_port_components",
        "controllable_one_port_components",
    }
)
#: The numpy dtype a record dtype is laid out in; an int may be absent, so it is a float.
DTYPES: dict[str, type] = {"bool": bool, "str": object}

_RECORD: dict[str, Callable[..., Any]] = {}
record_source = functools.partial(source, registry=_RECORD)


class NetworkReadError(RuntimeError):
    """A strict build read data off a `pypsa.Network`."""


class _Component:
    """A component type; data of its units besides is read off the built network, which is recorded."""

    def __init__(self, s: RecordSources, meta: Components) -> None:
        self._s = s
        self._meta = meta

    @property
    def additional_ports(self) -> list[str]:
        """The ports of the record's units beyond those the type declares."""
        declared = {int(k or 0) for k in self._meta.ports}
        ports = set(self._s.connections(self._meta.name)[PORT]) - declared
        return [str(k) for k in sorted(ports)]

    @property
    def ports(self) -> list[str]:
        """The ports of the type and of the record's units."""
        return [*self._meta.ports, *self.additional_ports]

    @property
    def static(self) -> pd.DataFrame:
        """PyPSA's static frame of the type, read off the record."""
        return self._s.static_frame(self._meta)

    @property
    def capital_cost(self) -> pd.Series:
        """The cost of capacity over the horizon without fixed operation and maintenance, by member."""
        static = self.static
        inputs = ("capital_cost", "overnight_cost", "discount_rate", "lifetime")
        return periodized_cost(
            **{k: static[k] for k in inputs}, fom_cost=None, nyears=self._s.nyears
        )

    @property
    def names(self) -> pd.Index:
        """The labels of the record's members of the type."""
        return self._s.members(self._meta.name).rename("name")

    def _active_names(self, investment_period: Any = None) -> pd.Index:
        """Return the members active in any scenario, and in any of the periods given."""
        ctype = self._meta.name
        active = self._s.grid(ctype, "active").any(SCENARIO)
        if investment_period is not None and self._s.takes(self._meta, "build_year"):
            periods = xr.DataArray(np.atleast_1d(investment_period), dims=PERIOD)
            built = self._s.grid(ctype, "build_year").isel(scenario=0)
            retired = built + self._s.grid(ctype, "lifetime").isel(scenario=0)
            active = active & ((built <= periods) & (periods < retired)).any(PERIOD)
        return self.names[active.to_numpy()]

    def __getattr__(self, attr: str) -> Any:
        if attr in COMPONENT_META:
            return getattr(self._meta, attr)
        if attr == "_output_ports":
            return getattr(type(self._meta), attr).fget(self)
        network = self._s.touch(f"{self._meta.name}.{attr}")
        return getattr(network.c[self._meta.name], attr)


class _Registry:
    """The component types of a network, each a `_Component`."""

    def __init__(self, s: RecordSources) -> None:
        self._s = s

    def __getitem__(self, name: str) -> _Component:
        return _Component(self._s, self._s.meta.components[name])

    def __getattr__(self, name: str) -> _Component:
        return _Component(self._s, getattr(self._s.meta.components, name))


class _Network:
    """A network read off the record; what the record does not answer is read off the built network, which is recorded."""

    def __init__(self, s: RecordSources) -> None:
        self._s = s

    @property
    def has_scenarios(self) -> bool:
        return self._s.stochastic

    @property
    def scenarios(self) -> pd.Index:
        return self._s.scenarios

    @property
    def components(self) -> _Registry:
        return _Registry(self._s)

    c = components

    @property
    def snapshot_weightings(self) -> pd.DataFrame:
        return self._s.weightings(SNAPSHOT, self._s.network_index)

    @property
    def investment_period_weightings(self) -> pd.DataFrame:
        return self._s.weightings(PERIOD, self._s.periods)

    def __getattr__(self, attr: str) -> Any:
        if attr in NETWORK_META:
            return getattr(self._s.meta, attr)
        return getattr(self._s.touch(f"n.{attr}"), attr)


class RecordSources(NetworkSources):
    """The sources the spec reads, derived from a datarecord record on access.

    Parameters
    ----------
    record : datarecord.RecordLike
        The record to build, of the schema `PyPSA.to_datarecord` writes.
    spec : mathspec.spec.Spec
        The spec the sources are for; its declarations are the keys.
    strict : bool
        Whether a read off a `pypsa.Network` raises `NetworkReadError`.
    **options
        The build options of `NetworkSources`.

    """

    def __init__(
        self, record: RecordLike, spec: Spec, *, strict: bool = False, **options: Any
    ) -> None:
        """Initialise the sources of one build over every snapshot of the record."""
        import pypsa  # noqa: PLC0415

        self.record = record
        self.strict = strict
        self.touched: dict[str, set[str]] = {}
        self._deriving: list[str] = []
        self.shape = NetworkShape(record.schema, record.dims)
        levels = self.shape.snapshot_levels
        index = (
            pd.MultiIndex.from_frame(levels)
            if self.shape.multiperiod
            else pd.Index(levels[SNAPSHOT], name=SNAPSHOT)
        )
        super().__init__(pypsa.Network(), spec, index, index, **options)
        tracked: Any = {c._unit_dim: _Component(self, c) for c in formulated(self.meta)}
        self.units = tracked
        self._providers = _providers(self.units.values(), _SOURCES | _RECORD)
        if not any(len(self.members(k)) for k in self.meta.passive_branch_components):
            self._cache["cycles"] = {}
        if "flow_based" not in set(self.entities["type"]):
            self._cache["flow_based"] = None

    # -- the network --------------------------------------------------------

    @property
    def n(self) -> Any:
        """The network the derivations read, answered from the record where it can be."""
        return _Network(self)

    @n.setter
    def n(self, n: Network) -> None:
        self.meta = n

    @functools.cached_property
    def network(self) -> Network:
        """The network `PyPSA.build` makes of the record."""
        n = PyPSA.build(self.record)
        n.calculate_dependent_values()
        return n

    def touch(self, what: str) -> Network:
        """Return the built network, recording that the key being derived read `what` off it.

        Raises:
            NetworkReadError: The build is strict.

        """
        key = self._deriving[-1] if self._deriving else ""
        if self.strict:
            msg = f"{key} reads {what} off a pypsa.Network"
            raise NetworkReadError(msg)
        self.touched.setdefault(key, set()).add(what)
        return self.network

    def _derive(self, name: str) -> Any:
        self._deriving.append(name)
        try:
            return super()._derive(name)
        finally:
            self._deriving.pop()

    # -- labels -------------------------------------------------------------

    @property
    def stochastic(self) -> bool:
        """Whether the record has scenarios."""
        return self.shape.stochastic

    @functools.cached_property
    def scenarios(self) -> pd.Index:
        """The scenario labels, one placeholder for a record without scenarios."""
        labels = self.axis(SCENARIO).astype(str) if self.stochastic else [BASE_SCENARIO]
        return pd.Index(labels, dtype=object, name=SCENARIO)

    @functools.cached_property
    def positions(self) -> np.ndarray:
        """The positions of the build's snapshots in the record's."""
        return np.arange(len(self.network_index))

    every_snapshot = True

    def axis(self, dim: str) -> pd.Index:
        """Return the labels of a record dim in axis order."""
        rel = self.record.dims.get(dim)
        labels = [] if rel is None else [label for (label,) in rel.fetchall()]
        return pd.Index(labels, dtype=object)

    def properties(self, dim: str) -> pd.DataFrame:
        """Return the properties of a record dim by label, in axis order."""
        rel = self.record.properties[dim]
        return self.memo(f"{dim} properties", lambda: rel.df().set_index(dim))

    @property
    def entities(self) -> pd.DataFrame:
        """Every entity with its properties, its type among them, in axis order."""
        return self.properties(ENTITY)

    def members(self, ctype: str) -> pd.Index:
        """Return the labels of a type in axis order: its entities, or the labels of its own dim."""
        if ctype in _AXIS_TYPES:
            return self.axis(_AXIS_TYPES[ctype])
        entities = self.entities
        names = entities.index[entities[CTYPE] == ctype]
        return pd.Index(names.to_numpy(), dtype=object)

    def connections(self, ctype: str) -> pd.DataFrame:
        """Return the ports of a type's entities, one row per (entity, port) with its bus."""
        return self.memo(f"{ctype} connections", lambda: self._connections(ctype))

    def _connections(self, ctype: str) -> pd.DataFrame:
        rel = self.record.properties.get(CONNECTION)
        if rel is None:
            return pd.DataFrame(columns=[ENTITY, PORT, BUS])
        every = self.memo(
            "connections", lambda: unpacked(self.record.schema, CONNECTION, rel).df()
        )
        return every[every[ENTITY].isin(self.members(ctype))]

    # -- arrays -------------------------------------------------------------

    def takes(self, c: Components, attr: str) -> bool:
        """Whether a component type has the PyPSA attribute `attr`."""
        meta = getattr(c, "_meta", c)
        return attr in meta.static or attr in meta.dynamic

    def attr(self, c: Components, name: str) -> str:
        """Return the PyPSA attribute a parameter `<Component>_<attr>` names."""
        return super().attr(getattr(c, "_meta", c), name)

    def numeric(self, c: Components, attr: str) -> np.ndarray | None:
        """Return None: a record is read through `read`."""
        return None

    def read(self, c: Components, attr: str) -> xr.DataArray:
        """Return a PyPSA attribute of a component's units in the spec's dims, read off the record."""
        dim = c._unit_dim
        da = self.grid(c.name, attr).sel(name=self.names(c).to_numpy())
        return da.rename(name=dim).assign_coords({dim: self.names(c)})

    def grid(self, ctype: str, attr: str) -> xr.DataArray:
        """Return a PyPSA attribute of every entity of a type, (scenario, [snapshot,] name).

        The record's rows are settled by the broadcast rule; a point no row
        reaches takes the default the record's schema declares for the type.
        """
        return self.memo(f"{ctype} {attr} grid", lambda: self._grid(ctype, attr))

    def _grid(self, ctype: str, attr: str) -> xr.DataArray:
        stem, port = self.port(ctype, attr)
        schema = self.record.schema
        names = self.members(ctype)
        key = _AXIS_TYPES.get(ctype, ENTITY)
        if stem in schema.properties[key]:
            default = schema.default_for(f"{key}.{stem}", **{CTYPE: ctype})
            stated = self.properties(key)[stem].reindex(names)
            default = "" if default is None else default
            values = stated.where(stated.notna(), default).to_numpy()[np.newaxis]
            coords = {SCENARIO: self.scenarios, "name": names}
            return xr.DataArray(np.repeat(values, len(self.scenarios), 0), coords)
        declared = schema.attributes[stem]
        coords = {SCENARIO: self.scenarios}
        if SNAPSHOT in declared.dims:
            coords[SNAPSHOT] = self.snapshots
        coords["name"] = names
        default = schema.default_for(stem, **{CTYPE: ctype})
        dtype = DTYPES.get(declared.dtype, float)
        if default is None:
            default = "" if dtype is object else np.nan
        rows = self.rows(ctype, stem, port)
        by_scenario = self.stochastic and SCENARIO in rows
        shape = [len(v) for v in coords.values()]
        values = np.full([shape[0] if by_scenario else 1, *shape[1:]], default, dtype)
        value = rows["value"].to_numpy()
        scenario = np.zeros(len(rows), dtype=int)
        if by_scenario:
            scenario = self.scenarios.get_indexer(rows[SCENARIO])
        name = names.get_indexer(rows[key])
        if SNAPSHOT in declared.dims:
            steps = pd.MultiIndex.from_frame(rows[self.network_index.names])
            if steps.nlevels == 1:
                steps = steps.get_level_values(0)
            time = self.network_index.get_indexer(steps)
            static = time == -1
            values[scenario[static], :, name[static]] = value[static, np.newaxis]
            values[scenario[~static], time[~static], name[~static]] = value[~static]
        else:
            values[scenario, name] = value
        values = np.broadcast_to(values, shape)
        return xr.DataArray(values, coords, list(coords), attr)

    def port(self, ctype: str, attr: str) -> tuple[str, int | None]:
        """Return the record attribute and port a PyPSA attribute of a type names.

        `efficiency2` of a Link names `efficiency` at port 2, its `efficiency`
        port 1, and `efficiency` of a Generator port 0.
        """
        attributes = self.record.schema.attributes

        def per_port(stem: str) -> bool:
            return stem in attributes and CONNECTION in attributes[stem].dims

        if per_port(attr):
            return attr, int(attr in _port_table(ctype).unsuffixed)
        match = _PORT_SUFFIX.fullmatch(attr)
        if match and per_port(match[1]):
            return match[1], int(match[2])
        return attr, None

    def rows(self, ctype: str, attr: str, port: int | None) -> pd.DataFrame:
        """Return the settled rows of a record attribute at a type's entities, at one port for an attribute over connections."""
        rel = self.record.attributes.get(attr)
        key = _AXIS_TYPES.get(ctype, ENTITY)
        if rel is None:
            columns = [SCENARIO, *self.network_index.names, key, "value"]
            return pd.DataFrame(columns=columns)
        members = self.record.dims[key]
        if key == ENTITY:
            members = matching(self.record.properties[ENTITY], {CTYPE: (ctype,)})
        labels: dict[str, DuckDBPyRelation] = {key: members}
        if port is not None:
            labels = {CONNECTION: _tuples_of(self.record.dims[CONNECTION], members)}
        if self.stochastic and SCENARIO in rel.columns:
            labels[SCENARIO] = self.record.dims[SCENARIO]
        settled = settle(rel, labels, Flags(frozenset(labels), frozenset(labels)))
        schema = self.record.schema
        rows = unpacked(schema, CONNECTION, unpacked(schema, SNAPSHOT, settled)).df()
        return rows if port is None else rows[rows[PORT] == port]

    def static_frame(self, c: Components) -> pd.DataFrame:
        """Return PyPSA's static frame of a type, read off the record.

        It holds every member, by scenario where the record has scenarios,
        with its buses and each attribute the record holds static. An
        attribute that may vary by snapshot is left out.
        """
        return self.memo(f"{c.name} frame", lambda: self._frame(c))

    def _frame(self, c: Components) -> pd.DataFrame:
        names = self.members(c.name).rename("name")
        schema = self.record.schema
        key = _AXIS_TYPES.get(c.name, ENTITY)
        columns = {}
        for attr in c.static.columns:
            stem = self.port(c.name, attr)[0]
            declared = schema.attributes.get(stem)
            static = (
                declared is not None
                and schema.addresses(stem, key)
                and SNAPSHOT not in declared.dims
            )
            if (static or stem in schema.properties[key]) and stem != BUS:
                columns[attr] = self.grid(c.name, attr).to_numpy()
        frames = {}
        for i, scenario in enumerate(self.scenarios):
            frame = pd.DataFrame({k: v[i] for k, v in columns.items()}, index=names)
            if key == ENTITY:
                table = _port_table(c.name)
                for port, rows in self.connections(c.name).groupby(PORT):
                    bus = rows.set_index(ENTITY)[BUS].reindex(names).fillna("")
                    frame[table.column(BUS, port)] = bus.to_numpy(dtype=object)
            frames[scenario] = frame
        if not self.stochastic:
            return frames[self.scenarios[0]]
        return pd.concat(frames, names=[SCENARIO, "name"])

    def _static(self, c: Components) -> pd.DataFrame:
        """Return the buses, carrier and activity of a component's units, and the cyclic delay of each port."""
        names = self.names(c)
        table = _port_table(c.name)
        buses = [table.column(BUS, int(k or 0)) for k in c.ports]
        static = pd.DataFrame("", index=names, columns=buses, dtype=object)
        for port, rows in self.connections(c.name).groupby(PORT):
            bus = rows.set_index(ENTITY)[BUS].reindex(names)
            static[table.column(BUS, port)] = bus.fillna("").to_numpy(dtype=object)
            if "cyclic_delay" in table.unsuffixed | table.stems:
                column = table.column("cyclic_delay", port)
                cyclic = self.grid(c.name, column).isel(scenario=0)
                static[column] = cyclic.sel(name=names.to_numpy()).to_numpy()
        static[CARRIER] = self.entities[CARRIER].reindex(names).fillna("").to_numpy()
        active = self.grid(c.name, "active").isel(scenario=0)
        static["active"] = active.sel(name=names.to_numpy()).to_numpy()
        return static

    def _active(self, c: Components) -> xr.DataArray:
        """Whether each unit is active at each snapshot: `active`, and built and not retired in the snapshot's period."""
        active = self.read(c, "active").any(SCENARIO)
        mask = active.expand_dims(snapshot=self.snapshots)
        if self.shape.multiperiod:
            period = self.over_snapshots(
                self.network_index.get_level_values(PERIOD).to_numpy(dtype=float)
            )
            built = self.read(c, "build_year").isel(scenario=0)
            retired = built + self.read(c, "lifetime").isel(scenario=0)
            mask = mask & (built <= period) & (period < retired)
        return mask.transpose(SNAPSHOT, c._unit_dim)

    @functools.cached_property
    def nyears(self) -> float | pd.Series:
        """The years the record's snapshots span, by period where it has periods."""
        weights = self.weightings(SNAPSHOT, self.network_index)["objective"]
        if self.shape.multiperiod:
            return weights.groupby(level=PERIOD).sum() / HOURS_PER_YEAR
        return weights.sum() / HOURS_PER_YEAR

    def passive(self, period: int | None) -> tuple[pd.Series, pd.DataFrame]:
        """Return the passive network as tables: the carrier of each bus, and the passive branches.

        A branch connects its buses where it is active and, for a period,
        built and not retired in it. For a period only such a branch is a
        member of a sub-network.
        """
        buses = of_scenario(self.static_frame(self.meta.c["Bus"]), self.n)[CARRIER]
        columns = ["bus0", "bus1", "active", "x_pu", "x_pu_eff", "r_pu_eff"]
        frames = {}
        for name in sorted(self.meta.passive_branch_components):
            c = _Component(self, self.meta.c[name])
            if not len(c.names):
                continue
            static = of_scenario(c.static, self.n)[columns]
            connects = static.index.isin(c._active_names(period))
            member = connects if period is not None else True
            frames[name] = static.assign(connects=connects, member=member)
        if not frames:
            empty = pd.MultiIndex.from_arrays([[], []])
            return buses, pd.DataFrame(
                columns=[*columns, "connects", "member"], index=empty
            )
        return buses, pd.concat(frames)

    def cycle_matrices(self, period: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the cycle matrix of a period, plain and weighted, from the record's tables."""
        return _cycle_matrices(*self.passive(period), self.cycle_basis_method)

    def outage_factors(
        self, period: int | None, outages: pd.MultiIndex
    ) -> pd.DataFrame:
        """Return the outage factors of a period, branches by outage, from the record's tables."""
        return _outage_factors(*self.passive(period), outages)

    def weightings(self, dim: str, labels: pd.Index) -> pd.DataFrame:
        """Return the weightings of a record dim by label, one where the record states none."""
        columns = [w.column for w in _WEIGHTINGS.values() if w.dim == dim]
        table = pd.DataFrame(1.0, index=labels, columns=columns)
        return self.memo(
            f"{dim} weightings", lambda: _weightings(self.record, dim, table)
        )


# -- labels ---------------------------------------------------------------


@record_source("bus")
def _buses(s: RecordSources, name: str) -> pd.Index:
    return s.axis(BUS)


@record_source("carrier")
def _carriers(s: RecordSources, name: str) -> pd.Index:
    """Every carrier of the record, and the empty one where a unit names none."""
    labels = s.axis(CARRIER)
    units = s.entities[s.entities[CTYPE].isin(c.name for c in s.units.values())]
    return labels.append(pd.Index([""])) if units[CARRIER].isna().any() else labels


@record_source("UNIT")
def _units(s: RecordSources, name: str, c: Components) -> pd.Index:
    """Return a type's entities that are active, a one-port one only at a bus the record has."""
    names = s.members(c.name)
    keep = s.grid(c.name, "active").any(SCENARIO).to_numpy()
    table = _port_table(c.name)
    if table.one_port:
        bus = s.connections(c.name).set_index(ENTITY)[BUS].reindex(names)
        keep &= bus.isin(s.axis(BUS)).to_numpy()
    return pd.Index(names[keep], dtype=object, name=None)


# -- weightings -----------------------------------------------------------


@record_source(
    "snapshot_weightings_objective",
    "snapshot_weightings_generators",
    "snapshot_weightings_stores",
)
def _snapshot_weighting(s: RecordSources, name: str) -> xr.DataArray:
    column = name.removeprefix("snapshot_weightings_")
    table = s.weightings(SNAPSHOT, s.network_index)
    return s.over_snapshots(table[column].to_numpy(dtype=float))


@record_source("scenario_weight")
def _scenario_weight(s: RecordSources, name: str) -> xr.DataArray:
    values = np.ones(1)
    if s.stochastic:
        labels = s.shape.scenarios[SCENARIO].astype(str)
        values = _scenario_weights(s.record, labels).to_numpy(dtype=float)
    return xr.DataArray(values, coords={SCENARIO: s.scenarios}, dims=SCENARIO)


@record_source("period_weight_objective", "period_weight_years")
def _period_weight(s: RecordSources, name: str) -> xr.DataArray:
    values = np.ones(len(s.periods))
    if s.multi_invest:
        column = name.removeprefix("period_weight_")
        values = s.weightings(PERIOD, s.periods)[column].to_numpy(dtype=float)
    return xr.DataArray(values, coords={PERIOD: s.periods}, dims=PERIOD)


# -- costs ----------------------------------------------------------------


@record_source("COMPONENT_capital_cost")
def _capital_cost(s: RecordSources, name: str, c: Components) -> xr.DataArray:
    """Return the cost of capacity over the horizon, as `pypsa.costs.periodized_cost` states it."""
    inputs = ("capital_cost", "overnight_cost", "discount_rate", "lifetime", "fom_cost")
    kwargs = {k: s.da(c, k).to_series() for k in inputs if s.takes(c, k)}
    cost = periodized_cost(**kwargs, nyears=s.nyears)
    labels = {SCENARIO: s.scenarios, c._unit_dim: s.names(c)}
    return cost.to_xarray().reindex(labels)  # type: ignore[union-attr]


# -- activity stated in the spec ------------------------------------------

#: The year of a snapshot's period, as a condition reads it.
YEAR = "at(period_year, by=snapshot_period[period])"


def activity_patch(spec: Spec) -> dict[str, dict[str, Any]]:
    """Return the spec patch that states `<Component>_active` as a condition, not as data.

    A unit is active where it is switched on and, in a run over investment
    periods, built and not retired in the snapshot's period. The patch
    removes the parameter and declares what the condition reads: the raw
    `build_year` and `lifetime`, the year of each period and whether the
    run is over periods.
    """
    import pypsa  # noqa: PLC0415

    parameters: dict[str, Any] = {
        "multi_period": {"dims": [], "dtype": "bool", "description": "over periods"},
        "period_year": {"dims": ["period"], "description": "the year of a period"},
    }
    conditions = {}
    for c in formulated(pypsa.Network()):
        if f"{c.name}_active" not in spec.parameters:
            continue
        unit, on = c._unit_dim, f"{c.name}_active_asset"
        built, life = f"{c.name}_build_year", f"{c.name}_lifetime"
        parameters[f"{c.name}_active"] = None
        if on not in spec.parameters:
            parameters[on] = {"dims": [unit], "dtype": "bool", "description": "on"}
        conditions[f"{c.name}_active"] = on
        if "build_year" not in c.static:
            continue
        parameters[built] = {"dims": [unit], "description": "the year it is built"}
        parameters[life] = {"dims": [unit], "description": "the years it stands"}
        standing = f"{built} <= {YEAR} AND {YEAR} < {built} + {life}"
        conditions[f"{c.name}_active"] = f"{on} AND (NOT multi_period OR ({standing}))"
    return {"parameters": parameters, "conditions": conditions}


@record_source("multi_period")
def _multi_period(s: RecordSources, name: str) -> bool:
    return s.multi_invest


@record_source("period_year")
def _period_year(s: RecordSources, name: str) -> xr.DataArray:
    years = s.periods.to_numpy(dtype=float)
    return xr.DataArray(years, coords={PERIOD: s.periods}, dims=PERIOD)


# -- absent in a record ---------------------------------------------------


@record_source("breakpoint")
def _breakpoint(s: RecordSources, name: str) -> pd.Index:
    """No breakpoint: `PyPSA.to_datarecord` writes no curve of `c.piecewise`."""
    return pd.Index([], dtype=int)


@record_source("CVaR_alpha", "CVaR_omega")
def _cvar(s: RecordSources, name: str) -> xr.DataArray:
    """No risk preference: `PyPSA.to_datarecord` does not write `n.risk_preference`."""
    return xr.DataArray(np.nan)


# -- derivations over the record's values ---------------------------------


@record_source("COMPONENT_big_m")
def _big_m(s: RecordSources, name: str, c: Components) -> xr.DataArray:
    """Return PyPSA's big-M of each committable and extendable unit, NaN for any other."""
    nom = c._operational_attrs["nom"]
    both = s.da(c, "committable").astype(bool) & s.da(c, f"{nom}_extendable").astype(
        bool
    )
    if not both.any():
        return s.filled(name, np.nan)
    default = s.committable_big_m
    if default is None:
        load = s.grid("Load", "p_set").sum([SCENARIO, "name"])
        sizes = (s.grid(c.name, k).to_numpy() for k in (nom, f"{nom}_max"))
        default = committable_big_m_scale(float(abs(load).max()), *sizes)
    max_pu = s.da(c, "p_max_pu").max(SNAPSHOT)
    return committable_big_m_values(s.da(c, f"{nom}_max"), max_pu, default).where(both)


@record_source("Store_split_dispatch")
def _store_split_dispatch(s: RecordSources, name: str) -> xr.DataArray:
    c = s.units["Store"]
    inputs = ("efficiency_store", "efficiency_dispatch", "marginal_cost_dispatch")
    values = (s.da(c, k) for k in (*inputs, "marginal_cost_store", "cycles_max"))
    return needs_split_dispatch(*values).astype(bool)
