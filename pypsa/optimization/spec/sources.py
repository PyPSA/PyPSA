# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep: a `pypsa.Network` read as the sources the spec declares.

Every dimension, relation and parameter of the spec is one key of
`NetworkSources`. linopy reads each key once, when the build needs it. A
parameter named `<Component>_<attr>` after a PyPSA attribute is read off
`c.da`; every other one is derived by the one function registered for it
with `source`, as its description in the spec states.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.optimization.spec._render import PLACEHOLDERS, WORD, formulated, renamer

if TYPE_CHECKING:
    from collections.abc import Collection

    from mathspec.spec import Spec

    from pypsa import Network
    from pypsa.components import Components
    from pypsa.components._types.mixin.multiports import _Multiport

#: The scenario label of a network that has none.
NO_SCENARIO = "-"
#: The period label of a run without investment periods.
NO_PERIOD = 0
_PORT_SEP = "::"

_SOURCES: dict[str, Callable[..., Any]] = {}


def source[F: Callable[..., Any]](*keys: str) -> Callable[[F], F]:
    """Register `f` as the one derivation of each key, called as `f(s, name)`.

    A key with the placeholders of a trait, as `Unit_active`, stands for that
    source of every component with a formulation; `f` takes the component as
    `c` then.

    Raises:
        ValueError: A key has a derivation already.

    """

    def register(f: F) -> F:
        for key in keys:
            if key in _SOURCES:
                msg = f"{key} is derived by {_SOURCES[key].__qualname__} already."
                raise ValueError(msg)
            _SOURCES[key] = f
        return f

    return register


def _providers(components: Collection[Components]) -> dict[str, Callable[..., Any]]:
    """Return the derivation of each source name, a placeholder key stated per component."""
    general = {
        k: f for k, f in _SOURCES.items() if PLACEHOLDERS.isdisjoint(WORD.findall(k))
    }
    stated = {k: f for k, f in _SOURCES.items() if k not in general}
    providers = dict(general)
    for c in components:
        rename = renamer(c)
        for key, f in stated.items():
            name = rename(key)
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
        self._providers = _providers(self.units.values())
        self._keys = dict.fromkeys(
            [*spec.dimensions, *spec.relations, *spec.parameters]
        )
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
        if name in self._providers:
            value = self._providers[name](self, name)
        elif name in self.spec.parameters and (c := self.component(name)) is not None:
            value = self.attribute(c, name)
        else:
            msg = f"no data prep for {name}"
            raise NotImplementedError(msg)
        return self.frame(name, value) if isinstance(value, xr.DataArray) else value

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
        return pd.Index([NO_SCENARIO], name="scenario")

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
        """Return the output ports of a multiport read long: one row per unit and set `bus<k>`."""
        return self.memo(
            f"{c.name} ports", lambda: _ports_long(self, cast("_Multiport", c))
        )

    # -- arrays -----------------------------------------------------------

    def declared(self, name: str) -> tuple[str, ...]:
        """Return the declared dims of a parameter."""
        return tuple(self.spec.parameters[name].dims)

    def coords(self, dim: str) -> pd.Index:
        """Return the labels of a dimension, as passed."""
        return pd.Index(self[dim], name=None)

    def frame(self, name: str, da: xr.DataArray) -> xr.DataArray:
        """`da` broadcast over and laid out in the declared dims of `name`."""
        dims = self.declared(name)
        for dim in dims:
            if dim not in da.dims:
                da = da.expand_dims({dim: self.coords(dim)})
        extra = [d for d in da.dims if d not in dims]
        if extra:
            msg = f"{name}: dims {extra} are not declared ({dims})"
            raise ValueError(msg)
        return da.transpose(*dims).rename(name)

    def filled(self, name: str, value: Any) -> xr.DataArray:
        """Return a grid over the declared dims of `name`, every cell `value`."""
        dims = self.declared(name)
        coords = {d: self.coords(d).to_numpy() for d in dims}
        nan = isinstance(value, float) and np.isnan(value)
        dtype = {"bool": bool, "int": float if nan else int}.get(
            self.spec.parameters[name].dtype,
            object if isinstance(value, str) else float,
        )
        shape = [len(c) for c in coords.values()]
        return xr.DataArray(np.full(shape, value, dtype=dtype), coords, dims, name)

    def static(self, c: Components) -> pd.DataFrame:
        """Return the static data of a component's units, of the first scenario."""
        static = of_scenario(c.static, self.n)
        return static[static.index.isin(self.names(c))]

    def column(self, c: Components, attr: str, dtype: Any) -> xr.DataArray:
        """Return a static column of a component's units over its unit dim."""
        column = self.static(c)[attr]
        index = {c._unit_dim: column.index.to_numpy(dtype=object)}
        return xr.DataArray(column.to_numpy(dtype=dtype), index, c._unit_dim)

    def unit_array(self, c: Components, da: xr.DataArray) -> xr.DataArray:
        """Return a `c.da` array relabelled into the spec's dims."""
        dim = c._unit_dim
        if "name" in da.dims:
            da = da.sel(name=da.indexes["name"].isin(self.names(c)))
            labels = da.indexes["name"].to_numpy(dtype=object)
            da = da.rename(name=dim).assign_coords({dim: labels})
        if "snapshot" in da.dims:
            if da.sizes["snapshot"] == len(self.n.snapshots):
                da = da.isel(snapshot=self.positions)
            tied = [
                k
                for k in da.coords
                if k == "snapshot" or (k not in da.dims and "snapshot" in da[k].dims)
            ]
            da = da.drop_vars(tied).assign_coords(snapshot=self.snapshots)
        if "scenario" not in da.dims and self.n.has_scenarios:
            da = da.expand_dims(scenario=self.scenarios)
        elif "scenario" in da.dims and not self.n.has_scenarios:
            da = da.squeeze("scenario", drop=True)
        if not self.n.has_scenarios:
            da = da.expand_dims(scenario=self.scenarios)
        return da

    def da(self, c: Components, attr: str) -> xr.DataArray:
        """Return a PyPSA attribute of a component in the spec's dims."""
        raw = self.unit_array(c, c.da[attr])
        return raw.drop_vars([k for k in raw.coords if k not in raw.dims])

    def attribute(self, c: Components, name: str) -> xr.DataArray:
        """Return a parameter `<Component>_<attr>` named after a PyPSA attribute, read off `c.da`."""
        attr = name.removeprefix(f"{c.name}_")
        if attr not in c.static and attr not in c.dynamic:
            msg = f"no data prep for {name}: {c.name} has no attribute {attr}"
            raise NotImplementedError(msg)
        da = self.da(c, attr)
        dtype = self.spec.parameters[name].dtype
        if dtype == "bool":
            da = da.astype(bool)
        elif dtype == "float":
            da = da.astype(float)
        elif dtype == "int" and not da.isnull().any():
            da = da.astype(int)
        dims = self.declared(name)
        if "scenario" in da.dims and "scenario" not in dims:
            da = da.isel(scenario=0, drop=True)
        if "snapshot" in da.dims and "snapshot" not in dims:
            msg = f"{name} varies by snapshot in the network, the spec holds it static"
            raise ValueError(msg)
        return self.frame(name, da)

    def active(self, c: Components) -> xr.DataArray:
        """Whether each unit is active at each build snapshot, (unit, snapshot)."""
        mask = c.get_activity_mask(self.network_index)
        if isinstance(mask.columns, pd.MultiIndex):
            mask = mask.T.groupby(level="name").any().T
        names = self.names(c)
        mask = mask.reindex(columns=names)
        return xr.DataArray(
            mask.to_numpy(dtype=bool),
            coords={
                "snapshot": self.snapshots,
                c._unit_dim: names.to_numpy(dtype=object),
            },
            dims=["snapshot", c._unit_dim],
        )

    def active_by_period(self, c: Components) -> xr.DataArray:
        """Whether each unit is active in each period, (unit, period)."""
        active = self.active(c)
        if not len(self.period_of):
            return active.isel(snapshot=0, drop=True).expand_dims(period=self.periods)
        by_period = active.assign_coords(period=("snapshot", self.period_of))
        by_period = by_period.groupby("period").any("snapshot")
        return by_period.reindex(period=self.periods, fill_value=False)

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


def _ports_long(s: NetworkSources, c: _Multiport) -> pd.DataFrame:
    dim = c._unit_dim
    static = s.static(c)
    frames = []
    for k in sorted(map(int, c._output_ports)):
        col = f"bus{k}"
        present = static[col].notna() & (static[col].astype(str).str.strip() != "")
        units = static.index[present]
        frames.append(
            pd.DataFrame(
                {
                    "port": [f"{u}{_PORT_SEP}{k}" for u in units],
                    dim: units.to_numpy(dtype=object),
                    "k": k,
                    "bus": static.loc[present, col].to_numpy(),
                }
            )
        )
    columns = ["port", dim, "k", "bus"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


# -- labels ---------------------------------------------------------------


@source("scenario", "snapshot", "period")
def _labels(s: NetworkSources, name: str) -> pd.Index:
    return getattr(s, f"{name}s")


@source("bus")
def _buses(s: NetworkSources, name: str) -> pd.Index:
    return pd.Index(s.n.components.buses.names, dtype=object)


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
    names = pd.Index(c.active_assets, dtype=object, name=None)
    if "bus" not in c.static:
        return names
    bus = of_scenario(c.static, s.n)["bus"].reindex(names)
    return names[bus.isin(s.n.c.buses.names).to_numpy()]


@source("UNIT_output")
def _output_ports(s: NetworkSources, name: str, c: Components) -> pd.Index:
    return pd.Index(s.ports(c)["port"], dtype=object)


# -- weightings and globals -----------------------------------------------


@source(
    "snapshot_weightings_objective",
    "snapshot_weightings_generators",
    "snapshot_weightings_stores",
)
def _snapshot_weighting(s: NetworkSources, name: str) -> xr.DataArray:
    weighting = s.n.snapshot_weightings[name.removeprefix("snapshot_weightings_")]
    return s.over_snapshots(weighting.iloc[s.positions].to_numpy(dtype=float))


@source("scenario_weight")
def _scenario_weight(s: NetworkSources, name: str) -> xr.DataArray:
    values = np.ones(1)
    if s.n.has_scenarios:
        values = s.n.scenario_weightings["weight"].reindex(s.scenarios).to_numpy(float)
    return xr.DataArray(values, coords={"scenario": s.scenarios}, dims="scenario")


@source("period_weight_objective", "period_weight_years")
def _period_weight(s: NetworkSources, name: str) -> xr.DataArray:
    """Return a column of `investment_period_weightings`, one under a single period."""
    values = np.ones(len(s.periods))
    if s.multi_invest:
        weights = s.n.investment_period_weightings[name.removeprefix("period_weight_")]
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


# -- relations ------------------------------------------------------------


@source("Unit_bus", "Unit_bus0", "Unit_bus1", "Unit_carrier")
def _relation(s: NetworkSources, name: str, c: Components) -> pd.DataFrame:
    """Map each unit to the label of its column: a bus, or a carrier."""
    column = name.removeprefix(f"{c.name}_")
    static = s.static(c)
    return pd.DataFrame(
        {
            c._unit_dim: static.index.to_numpy(dtype=object),
            column.rstrip("01"): static[column].to_numpy(dtype=object),
        }
    )


@source("Unit_output_bus", "Unit_output_UNIT")
def _output(s: NetworkSources, name: str, c: Components) -> pd.DataFrame:
    ports = s.ports(c)
    column = name.removeprefix(f"{c.name}_output_")
    output = {f"{c._unit_dim}_output": ports["port"], column: ports[column]}
    return pd.DataFrame(output).reset_index(drop=True)


# -- activity -------------------------------------------------------------


@source("Unit_active")
def _active(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    if "snapshot" in s.declared(name):
        return s.active(c)
    return s.column(c, "active", bool)


@source("Unit_active_asset")
def _active_asset(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    return s.column(c, "active", bool)


@source("Unit_capital_weight")
def _capital_weight(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    by_period = s.active_by_period(c)
    if not s.multi_invest:
        return by_period.any("period").astype(float)
    return (by_period * s["period_weight_objective"]).sum("period")


@source("Unit_first_active")
def _first_active(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    by_period = s.active_by_period(c).astype(int)
    first = (by_period == 1) & (by_period.cumsum("period") == 1)
    return first.astype(float)


@source("Unit_capital_cost")
def _capital_cost(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    return s.unit_array(c, c.periodized_cost)


@source("Unit_NOM_mod")
def _nom_mod(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    da = s.attribute(c, name)
    return da.where(da > 0)


@source("Unit_NOM_set", "Unit_s_set")
def _optional(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """Return the attribute, NaN where the component has none."""
    attr = name.removeprefix(f"{c.name}_")
    if attr in c.static or attr in c.dynamic:
        return s.attribute(c, name)
    return s.filled(name, np.nan)


@source("Unit_p_min_pu_nonneg")
def _p_min_pu_nonneg(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    pu = s.da(c, "p_min_pu")
    return (pu >= 0).all([d for d in pu.dims if d != c._unit_dim])


@source("Unit_modules_installed")
def _modules_installed(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    nom = c._operational_attrs["nom"]
    mod = s.da(c, f"{nom}_mod")
    modular = (mod > 0) & ~s.da(c, f"{nom}_extendable").astype(bool)
    return xr.where(modular, s.da(c, nom) / mod.where(mod > 0, 1), 1.0)


@source("Unit_min_up_time", "Unit_min_down_time")
def _min_time(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """Return the minimum time, at most the horizon: a longer window reads nothing more."""
    return s.attribute(c, name).clip(max=len(s.snapshots))


@source("Unit_v_ang_max", "Unit_cycles_max")
def _finite(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """Return the attribute, absent where infinite, so a unit without a limit has no row."""
    limit = s.attribute(c, name)
    return limit.where(np.isfinite(limit))


# -- ports ----------------------------------------------------------------


def _port_attribute(
    s: NetworkSources, c: _Multiport, name: str, prefix: str
) -> xr.DataArray:
    """Return the attribute `<prefix><port suffix>` of each port, over the component's port dimension."""
    owner = c._unit_dim
    dim = f"{owner}_output"
    parts = []
    for k, group in s.ports(c).groupby("k", sort=True):
        da = s.da(c, f"{prefix}{c._port_suffix(k)}").sel(
            {owner: group[owner].to_numpy()}
        )
        parts.append(
            da.rename({owner: dim}).assign_coords({dim: group["port"].to_numpy()})
        )
    if not parts:
        return s.filled(name, np.nan)
    da = xr.concat(parts, dim=dim).reindex({dim: s.coords(dim).to_numpy()})
    if "scenario" in da.dims and "scenario" not in s.declared(name):
        da = da.isel(scenario=0, drop=True)
    return da


@source("Link_efficiency", "Process_rate")
def _coefficient(s: NetworkSources, name: str) -> xr.DataArray:
    c = s.n.components[name.partition("_")[0]]
    return _port_attribute(s, c, name, c._coefficient_attr).astype(float)


@source("Unit_output_delay")
def _output_delay(s: NetworkSources, name: str, c: _Multiport) -> xr.DataArray:
    try:
        delay = _port_attribute(s, c, name, "delay")
    except KeyError:
        return s.filled(name, 0)
    return delay.astype(float).fillna(0).astype(int)


@source("Unit_output_delay_source")
def _output_delay_source(s: NetworkSources, name: str, c: _Multiport) -> pd.DataFrame:
    """Each delayed port's source snapshot, as PyPSA's own indexer picks it."""
    owner = c._unit_dim
    static = s.static(c)
    ports = s.ports(c).set_index("port")
    cyclic = pd.Series(True, index=ports.index)
    for k, group in ports.groupby("k"):
        attr = f"cyclic_delay{c._port_suffix(k)}"
        if attr in static:
            cyclic[group.index] = static[attr].reindex(group[owner]).to_numpy()
    delay = s[f"{c.name}_output_delay"].to_series().reindex(ports.index)
    delayed = pd.DataFrame({"delay": delay, "cyclic": cyclic})[delay > 0]
    weightings = s.n.snapshot_weightings["generators"].iloc[s.positions].to_numpy()
    frames = [pd.DataFrame(columns=[f"{owner}_output", "snapshot", "source"])]
    for (d, cyc), group in delayed.groupby(["delay", "cyclic"]):
        src, valid = c.get_delay_source_indexer(
            s.network_index, weightings, int(d), bool(cyc)
        )
        targets = np.flatnonzero(valid)
        frames.append(
            pd.DataFrame(
                {
                    f"{owner}_output": np.repeat(
                        group.index.to_numpy(dtype=object), len(targets)
                    ),
                    "snapshot": np.tile(s.snapshots[targets], len(group)),
                    "source": np.tile(s.snapshots[src[targets]], len(group)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True).astype(object)


# -- commitment -----------------------------------------------------------


def _solved_before(s: NetworkSources, c: Components, attr: str) -> xr.DataArray | None:
    """`attr` as solved over the snapshots before the build, or None in the first window."""
    start = int(s.positions[0])
    if not start or c.dynamic[attr].empty:
        return None
    raw = c.da[attr].isel(snapshot=slice(0, start))
    raw = raw.drop_vars([k for k in raw.coords if k not in raw.dims or k == "snapshot"])
    raw = raw.assign_coords(snapshot=np.arange(start))
    dim = c._unit_dim
    raw = raw.sel(name=raw.indexes["name"].isin(s.names(c)))
    raw = raw.rename(name=dim).assign_coords(
        {dim: raw.indexes["name"].to_numpy(dtype=object)}
    )
    if "scenario" not in raw.dims:
        raw = raw.expand_dims(scenario=s.scenarios)
    return raw


def _history(s: NetworkSources, c: Components) -> xr.DataArray:
    """Return the status of each unit at the `lag` snapshots before the build, newest first.

    Solved snapshots of earlier windows are read as solved, fractions included;
    the time before the network's first snapshot is rebuilt from
    `up_time_before` and `down_time_before`, the latter winning. A unit that is
    not committable is on.
    """
    minimum = [s.da(c, f"min_{k}_time") for k in ("up", "down")]
    length = int(max([1, *(m.max().item() for m in minimum if m.size)])) + 1
    solved = _solved_before(s, c, "status")
    count = 0 if solved is None else min(solved.sizes["snapshot"], length)
    lag = xr.DataArray(np.arange(1, length - count + 1), dims="lag")
    up = s.da(c, "up_time_before")
    down = s.da(c, "down_time_before")
    before = xr.where(down > 0, lag > down, lag <= up).astype(float)
    parts = [before.assign_coords(lag=lag + count)]
    if solved is not None and count:
        recent = solved.isel(snapshot=slice(-count, None))
        recent = recent.isel(snapshot=slice(None, None, -1)).rename(snapshot="lag")
        parts.insert(0, recent.assign_coords(lag=np.arange(1, count + 1)))
    history = xr.concat(parts, dim="lag")
    return history.where(s.da(c, "committable").astype(bool), 1.0)


@source("Unit_up_time_brought_in", "Unit_down_time_brought_in")
def _brought_in(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """How much of the start-ups (`up`) or shut-downs (`down`) before the build still bind.

    A switch at lag `l` binds a snapshot at position `p` while `l + p` is
    below the minimum up or down time (`constraints.py:677-678`).
    """
    kind = name.removeprefix(f"{c.name}_").partition("_")[0]
    history = _history(s, c)
    switches = history - history.shift(lag=-1)
    switches = switches.isel(lag=slice(None, -1))
    past = (switches if kind == "up" else -switches).clip(min=0)
    minimum = s.da(c, f"min_{kind}_time")
    position = s.over_snapshots(np.arange(len(s.snapshots)))
    return (past * (past.lag + position < minimum)).sum("lag")


@source("Unit_p_init")
def _p_init(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """Return the output carried into the build.

    It is the solved output of an earlier window, else `p_init` where the
    unit came in running and zero where it came in off.
    """
    solved = _solved_before(s, c, "p0" if c.name == "Link" else "p")
    if solved is None:
        status = _history(s, c).sel(lag=1, drop=True)
        return s.da(c, "p_init").where(status > 0, 0.0)
    last = solved.isel(snapshot=-1, drop=True)
    return last.reindex({c._unit_dim: s.names(c).to_numpy()})


@source("Unit_status_initial")
def _status_initial(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    return _history(s, c).sel(lag=1, drop=True)


@source("Unit_big_m")
def _big_m(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    names = c.committables.intersection(c.extendables)
    if names.empty:
        return s.filled(name, np.nan)
    values = c.get_committable_big_m_values(
        names, committable_big_m=s.committable_big_m
    )
    return s.unit_array(c, values).reindex({c._unit_dim: s.names(c).to_numpy()})


# -- storage --------------------------------------------------------------


@source("Unit_inactive_snapshots")
def _inactive_snapshots(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    return (~s.active(c)).sum("snapshot").astype(int)


@source("Unit_opens_late")
def _opens_late(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    active = s.active(c)
    first = (active.cumsum("snapshot") == 1) & active
    return first & (s.over_snapshots(np.arange(len(s.snapshots))) > 0)


@source("Unit_retention")
def _retention(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    return (1 - s.da(c, "standing_loss")) ** s["snapshot_weightings_stores"]


@source("Store_split_dispatch")
def _store_split_dispatch(s: NetworkSources, name: str) -> xr.DataArray:
    c = s.n.components.stores
    return s.unit_array(c, xr.DataArray(c.split_dispatch())).astype(bool)


@source("Bus_attached")
def _bus_attached(s: NetworkSources, name: str) -> xr.DataArray:
    """Whether any active unit has a port at each bus, (bus, snapshot)."""
    buses = s.coords("bus")
    attached = xr.DataArray(
        np.zeros((len(buses), len(s.snapshots)), dtype=bool),
        coords={"bus": buses.to_numpy(), "snapshot": s.snapshots},
        dims=["bus", "snapshot"],
    )
    for c in s.units.values():
        active = s.active(c)
        for port in c.ports:
            at = s.column(c, f"bus{port}", object)
            attached = attached | _any_by_bus(active, at, buses)
    return attached


def _any_by_bus(
    active: xr.DataArray, bus: xr.DataArray, buses: pd.Index
) -> xr.DataArray:
    """Whether any active unit sits at each bus, (bus, snapshot)."""
    unit = bus.dims[0]
    if not active.sizes[unit]:
        return xr.DataArray(False)
    frame = pd.DataFrame(
        active.transpose(unit, "snapshot").to_numpy(), index=bus.to_numpy()
    )
    grouped = frame.groupby(level=0).any().reindex(buses, fill_value=False)
    return xr.DataArray(
        grouped.to_numpy(dtype=bool),
        coords={"bus": buses.to_numpy(), "snapshot": active.snapshot.values},
        dims=["bus", "snapshot"],
    )
