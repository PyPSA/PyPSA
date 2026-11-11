# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep: a `pypsa.Network` read as the sources `pypsa.yaml` declares.

Every dimension, relation and parameter of the spec is one key of
`NetworkSources`. linopy reads each key once, when the build needs it. A
parameter named `<Component>_<attr>` after a PyPSA attribute is read off
`c.da`; every other one is derived here, as its description in `pypsa.yaml`
states.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from mathspec.spec import Spec

    from pypsa import Network
    from pypsa.components import Components

#: The spec's dimension of each component's units.
UNITS = {
    "Generator": "generator",
    "Load": "load",
    "Link": "link",
    "Process": "process",
    "StorageUnit": "storage_unit",
    "Store": "store",
    "Line": "line",
    "Transformer": "transformer",
}
#: The scenario label of a network that has none.
NO_SCENARIO = "-"
#: The period label of a run without investment periods.
NO_PERIOD = 0
#: The nominal attribute of each component.
NOMINAL = {
    "Generator": "p_nom",
    "Link": "p_nom",
    "Process": "p_nom",
    "StorageUnit": "p_nom",
    "Store": "e_nom",
    "Line": "s_nom",
    "Transformer": "s_nom",
}

_FAMILIES: dict[str, Callable[..., Any]] = {}
_SINGLES: dict[str, Callable[..., Any]] = {}
_DIMENSIONS: dict[str, Callable[..., Any]] = {}


def dimension(*names: str) -> Callable:
    """Register a derivation of the labels of a dimension."""

    def register(f: Callable) -> Callable:
        for name in names:
            _DIMENSIONS[name] = f
        return f

    return register


def family(*suffixes: str) -> Callable:
    """Register a derivation of `<Component>_<suffix>` for every component."""

    def register(f: Callable) -> Callable:
        for suffix in suffixes:
            _FAMILIES[suffix] = f
        return f

    return register


def single(*names: str) -> Callable:
    """Register a derivation of one source by its full name."""

    def register(f: Callable) -> Callable:
        for name in names:
            _SINGLES[name] = f
        return f

    return register


class NetworkSources(Mapping[str, Any]):
    """The sources `pypsa.yaml` reads, derived from a network on access.

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
        self._keys = [
            *spec.dimensions,
            *spec.relations,
            *spec.parameters,
        ]
        self._cache: dict[str, Any] = {}

    def __iter__(self) -> Iterator[str]:
        """Iterate the declared names."""
        return iter(self._keys)

    def __len__(self) -> int:
        """Count the declared names."""
        return len(self._keys)

    def __getitem__(self, name: str) -> Any:
        """Derive the source declared as `name`."""
        if name not in self._cache:
            self._cache[name] = self._derive(name)
        return self._cache[name]

    def _derive(self, name: str) -> Any:
        if name in self.spec.dimensions:
            return self.dimension(name)
        if name in _SINGLES:
            return _SINGLES[name](self, name)
        component, _, suffix = name.partition("_")
        if component in UNITS and suffix in _FAMILIES:
            return _FAMILIES[suffix](self, component, suffix)
        if name in self.spec.relations:
            msg = f"no data prep for relation {name}"
            raise NotImplementedError(msg)
        if component in UNITS:
            return self.attribute(component, suffix, name)
        msg = f"no data prep for {name}"
        raise NotImplementedError(msg)

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

    def names(self, component: str) -> pd.Index:
        """Return the labels of a component's units: its active assets, as PyPSA builds them.

        A one-port unit at a bus the network lacks is left out, as PyPSA's
        nodal balance never reaches it.
        """
        c = self.n.components[component]
        names = pd.Index(c.active_assets, dtype=object, name=None)
        if "bus" not in c.static or component == "Bus":
            return names
        bus = _first_scenario(c.static, self.n)["bus"].reindex(names)
        return names[bus.isin(self.n.c.buses.names).to_numpy()]

    def dimension(self, dim: str) -> Any:
        """Return the labels of one declared dimension."""
        if dim in _DIMENSIONS:
            return _DIMENSIONS[dim](self)
        units = {v: k for k, v in UNITS.items()}
        if dim in units:
            return self.names(units[dim])
        derive = getattr(self, f"dimension_{dim}", None)
        if derive is None:
            msg = f"no data prep for dimension {dim}"
            raise NotImplementedError(msg)
        return derive()

    def dimension_scenario(self) -> pd.Index:
        """Scenario labels."""
        return self.scenarios

    def dimension_snapshot(self) -> pd.Index:
        """Snapshot labels."""
        return self.snapshots

    def dimension_period(self) -> pd.Index:
        """Period labels."""
        return self.periods

    def dimension_bus(self) -> pd.Index:
        """Bus labels."""
        return pd.Index(self.n.components.buses.names, dtype=object)

    def dimension_carrier(self) -> pd.Index:
        """Every carrier a unit names, defined or not."""
        carriers = [pd.Index(self.n.components.carriers.names)]
        for component in UNITS:
            static = self.n.components[component].static
            if "carrier" in static:
                carriers.append(pd.Index(static["carrier"].unique()))
        labels = pd.Index([], dtype=object)
        for idx in carriers:
            labels = labels.append(idx.difference(labels))
        return pd.Index(labels.unique(), dtype=object)

    def dimension_global_constraint(self) -> pd.Index:
        """GlobalConstraint labels."""
        return pd.Index(self.n.components.global_constraints.names, dtype=object)

    def dimension_link_output(self) -> pd.Index:
        """One label per link output port."""
        return pd.Index(self.link_ports["port"], dtype=object)

    def dimension_process_output(self) -> pd.Index:
        """One label per process port."""
        return pd.Index(self.process_ports["port"], dtype=object)

    def dimension_breakpoint(self) -> pd.Index:
        """Breakpoint positions of the piecewise curves."""
        return pd.Index([], dtype=int)

    def dimension_segment(self) -> pd.Index:
        """Loss cuts, none in a lossless run."""
        return pd.Index([], dtype=int)

    def dimension_outage(self) -> pd.Index:
        """Outages of a security-constrained run, none otherwise."""
        return pd.Index([], dtype=object)

    def dimension_cycle(self) -> pd.Index:
        """Cycles of the passive network."""
        return pd.Index([], dtype=object)

    def dimension_flow_based_domain(self) -> pd.Index:
        """Flow-based domains, none without flow-based rows."""
        return pd.Index([], dtype=object)

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

    def unit_array(self, component: str, da: xr.DataArray) -> xr.DataArray:
        """Return a `c.da` array relabelled into the spec's dims."""
        dim = UNITS[component]
        if "name" in da.dims:
            da = da.sel(name=da.indexes["name"].isin(self.names(component)))
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

    def da(self, component: str, attr: str) -> xr.DataArray:
        """Return a PyPSA attribute of a component in the spec's dims."""
        c = self.n.components[component]
        raw = self.unit_array(component, c.da[attr])
        return raw.drop_vars(_aux(raw))

    def attribute(self, component: str, attr: str, name: str) -> xr.DataArray:
        """Return a parameter named after a PyPSA attribute, read off `c.da`."""
        c = self.n.components[component]
        if attr not in c.static and attr not in c.dynamic:
            msg = f"no data prep for {name}: {component} has no attribute {attr}"
            raise NotImplementedError(msg)
        da = self.da(component, attr)
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

    def component(self, name: str) -> Components:
        """Return the component of a component-prefixed source name."""
        return self.n.components[name.partition("_")[0]]

    def active(self, component: str) -> xr.DataArray:
        """Whether each unit is active at each build snapshot, (unit, snapshot)."""
        c = self.n.components[component]
        mask = c.get_activity_mask(self.network_index)
        if isinstance(mask.columns, pd.MultiIndex):
            mask = mask.T.groupby(level="name").any().T
        names = self.names(component)
        mask = mask.reindex(columns=names)
        return xr.DataArray(
            mask.to_numpy(dtype=bool),
            coords={
                "snapshot": self.snapshots,
                UNITS[component]: names.to_numpy(dtype=object),
            },
            dims=["snapshot", UNITS[component]],
        )

    def active_by_period(self, component: str) -> xr.DataArray:
        """Whether each unit is active in each period, (unit, period)."""
        active = self.active(component)
        by = xr.DataArray(self.period_of, dims="snapshot", name="period")
        return (
            (
                active.assign_coords(period=("snapshot", self.period_of))
                .groupby("period")
                .any("snapshot")
                .reindex(period=self.periods, fill_value=False)
            )
            if len(by)
            else active.isel(snapshot=0, drop=True).expand_dims(period=self.periods)
        )

    def period_weight(self, column: str) -> xr.DataArray:
        """Return a column of `investment_period_weightings`, one under a single period."""
        if not self.multi_invest:
            return xr.DataArray(
                np.ones(len(self.periods)),
                coords={"period": self.periods},
                dims="period",
            )
        weights = self.n.investment_period_weightings[column].reindex(self.periods)
        return xr.DataArray(
            weights.to_numpy(dtype=float),
            coords={"period": self.periods},
            dims="period",
        )

    # -- ports ------------------------------------------------------------

    @functools.cached_property
    def link_ports(self) -> pd.DataFrame:
        """Link output ports read long: one row per link and `bus<k>`, k >= 1."""
        return _ports_long(self, "Link", start=1)

    @functools.cached_property
    def process_ports(self) -> pd.DataFrame:
        """Process ports read long: one row per process and `bus<k>`, k >= 0."""
        return _ports_long(self, "Process", start=0)


_PORT_SEP = "::"


def _port_numbers(static: pd.DataFrame, start: int) -> list[int]:
    numbers = sorted(
        int(col[3:]) if col[3:] else 0
        for col in static.columns
        if col.startswith("bus") and (col[3:].isdigit() or col == "bus")
    )
    return [k for k in numbers if k >= start]


def _ports_long(s: NetworkSources, component: str, start: int) -> pd.DataFrame:
    """Return the ports of a multiport read long: one row per unit and set `bus<k>`, k >= start."""
    dim = UNITS[component]
    static = _static(s, component)
    frames = []
    for k in _port_numbers(static, start):
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
    return (
        pd.concat(frames, ignore_index=True)[columns]
        if frames
        else pd.DataFrame(columns=columns)
    )


def _aux(da: xr.DataArray) -> list[str]:
    return [str(c) for c in da.coords if c not in da.dims]


def _first_scenario(static: pd.DataFrame, n: Network) -> pd.DataFrame:
    """Return the static data of the first scenario, the one every scenario shares."""
    if not n.has_scenarios:
        return static
    if static.empty:
        return static.droplevel("scenario")
    return static.xs(n.scenarios[0], level="scenario")


def _static(sources: NetworkSources, component: str) -> pd.DataFrame:
    static = _first_scenario(sources.n.components[component].static, sources.n)
    return static[static.index.isin(sources.names(component))]


def _relation(key: str, value: str, frame: pd.DataFrame) -> pd.DataFrame:
    return frame[[key, value]].reset_index(drop=True)


# -- weightings and globals -----------------------------------------------


@single(
    "snapshot_weightings_objective",
    "snapshot_weightings_generators",
    "snapshot_weightings_stores",
)
def _snapshot_weighting(s: NetworkSources, name: str) -> xr.DataArray:
    column = name.removeprefix("snapshot_weightings_")
    values = s.n.snapshot_weightings[column].iloc[s.positions].to_numpy(dtype=float)
    return xr.DataArray(values, coords={"snapshot": s.snapshots}, dims="snapshot")


@single("scenario_weight")
def _scenario_weight(s: NetworkSources, name: str) -> xr.DataArray:
    if s.n.has_scenarios:
        weight = s.n.scenario_weightings["weight"].reindex(s.scenarios)
        values = weight.to_numpy(dtype=float)
    else:
        values = np.ones(1)
    return xr.DataArray(values, coords={"scenario": s.scenarios}, dims="scenario")


@single("period_weight_objective", "period_weight_years")
def _period_weight(s: NetworkSources, name: str) -> xr.DataArray:
    return s.period_weight(name.removeprefix("period_weight_"))


@single("snapshot_period")
def _snapshot_period(s: NetworkSources, name: str) -> pd.DataFrame:
    return pd.DataFrame({"snapshot": s.snapshots, "period": s.period_of})


@single("transmission_losses")
def _transmission_losses(s: NetworkSources, name: str) -> bool:
    return s.transmission_losses is not None


@single("CVaR_alpha", "CVaR_omega")
def _cvar(s: NetworkSources, name: str) -> float | xr.DataArray:
    preference = getattr(s.n, "risk_preference", None) if s.n.has_scenarios else None
    if not preference:
        return xr.DataArray(np.nan)
    return float(preference[name.removeprefix("CVaR_")])


# -- relations ------------------------------------------------------------


@family("bus", "bus0", "bus1")
def _bus(s: NetworkSources, component: str, suffix: str) -> pd.DataFrame:
    dim = UNITS[component]
    static = _static(s, component)
    return pd.DataFrame(
        {dim: static.index.to_numpy(dtype=object), "bus": static[suffix].to_numpy()}
    )


@family("carrier")
def _carrier(s: NetworkSources, component: str, suffix: str) -> pd.DataFrame:
    dim = UNITS[component]
    static = _static(s, component)
    return pd.DataFrame(
        {
            dim: static.index.to_numpy(dtype=object),
            "carrier": static["carrier"].to_numpy(dtype=object),
        }
    )


@single("Link_output_bus", "Link_output_link")
def _link_output(s: NetworkSources, name: str) -> pd.DataFrame:
    ports = s.link_ports
    column = name.removeprefix("Link_output_")
    return pd.DataFrame(
        {"link_output": ports["port"], column: ports[column]}
    ).reset_index(drop=True)


@single("Process_output_bus", "Process_output_process")
def _process_output(s: NetworkSources, name: str) -> pd.DataFrame:
    ports = s.process_ports
    column = name.removeprefix("Process_output_")
    return pd.DataFrame(
        {"process_output": ports["port"], column: ports[column]}
    ).reset_index(drop=True)


# -- activity -------------------------------------------------------------


@family("active")
def _active(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    name = f"{component}_active"
    if "snapshot" not in s.declared(name):
        active = _static(s, component)["active"]
        da = xr.DataArray(
            active.to_numpy(dtype=bool),
            coords={UNITS[component]: active.index.to_numpy(dtype=object)},
            dims=UNITS[component],
        )
        return s.frame(name, da)
    return s.frame(name, s.active(component))


@family("active_asset")
def _active_asset(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    active = _static(s, component)["active"]
    da = xr.DataArray(
        active.to_numpy(dtype=bool),
        coords={UNITS[component]: active.index.to_numpy(dtype=object)},
        dims=UNITS[component],
    )
    return s.frame(f"{component}_{suffix}", da)


@family("capital_weight")
def _capital_weight(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    by_period = s.active_by_period(component)
    weight = (by_period * s.period_weight("objective")).sum("period")
    if not s.multi_invest:
        weight = by_period.any("period").astype(float)
    return s.frame(f"{component}_{suffix}", weight)


@family("first_active")
def _first_active(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    by_period = s.active_by_period(component).astype(int)
    first = (by_period == 1) & (by_period.cumsum("period") == 1)
    return s.frame(f"{component}_{suffix}", first.astype(float))


@family("capital_cost")
def _capital_cost(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    cost = s.n.components[component].periodized_cost
    return s.frame(f"{component}_{suffix}", s.unit_array(component, cost))


@family("sign")
def _sign(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    sign = _static(s, component)["sign"]
    da = xr.DataArray(
        sign.to_numpy(dtype=float),
        coords={UNITS[component]: sign.index.to_numpy(dtype=object)},
        dims=UNITS[component],
    )
    return s.frame(f"{component}_{suffix}", da)


def _absent_where(da: xr.DataArray, holds: xr.DataArray) -> xr.DataArray:
    return da.where(holds)


@family("p_nom_mod", "e_nom_mod", "s_nom_mod")
def _nom_mod(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    name = f"{component}_{suffix}"
    da = s.attribute(component, suffix, name)
    return da.where(da > 0)


@family("p_nom_set", "e_nom_set", "s_nom_set")
def _nom_set(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    name = f"{component}_{suffix}"
    if suffix not in s.n.components[component].static:
        return _filled(s, name, np.nan)
    return s.attribute(component, suffix, name)


@family("p_min_pu_nonneg")
def _p_min_pu_nonneg(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    pu = s.da(component, "p_min_pu")
    reduce = [d for d in pu.dims if d != UNITS[component]]
    return s.frame(f"{component}_{suffix}", (pu >= 0).all(reduce))


@family("modules_installed")
def _modules_installed(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    nom = s.da(component, NOMINAL[component])
    mod = s.da(component, f"{NOMINAL[component]}_mod")
    ext = s.da(component, f"{NOMINAL[component]}_extendable").astype(bool)
    modular = (mod > 0) & ~ext
    installed = xr.where(modular, nom / mod.where(mod > 0, 1), 1.0)
    return s.frame(f"{component}_{suffix}", installed)


# -- features a network may not use ---------------------------------------


@single("Bus_flow_based_domain")
def _bus_flow_based_domain(s: NetworkSources, name: str) -> pd.DataFrame:
    return pd.DataFrame({"bus": [], "flow_based_domain": []}, dtype=object)


@single("Outage_line", "Outage_transformer")
def _outage_branch(s: NetworkSources, name: str) -> pd.DataFrame:
    column = name.removeprefix("Outage_")
    return pd.DataFrame({"outage": [], column: []}, dtype=object)


@family("maintenance_cover")
def _maintenance_cover(s: NetworkSources, component: str, suffix: str) -> pd.DataFrame:
    dim = UNITS[component]
    return pd.DataFrame(
        {"start": [], "covered": [], dim: [], "scenario": []}, dtype=object
    )


# -- helpers for grids ----------------------------------------------------


def _filled(s: NetworkSources, name: str, value: Any) -> xr.DataArray:
    """Return a grid over the declared dims of `name`, every cell `value`."""
    dims = s.declared(name)
    coords = {d: s.coords(d) for d in dims}
    shape = [len(c) for c in coords.values()]
    dtype = {
        "bool": bool,
        "int": float if isinstance(value, float) and np.isnan(value) else int,
    }.get(
        s.spec.parameters[name].dtype, float if not isinstance(value, str) else object
    )
    return xr.DataArray(
        np.full(shape, value, dtype=dtype),
        coords={d: c.to_numpy() for d, c in coords.items()},
        dims=dims,
        name=name,
    )


def _ports(s: NetworkSources, component: str) -> pd.DataFrame:
    return s.link_ports if component == "Link" else s.process_ports


def _port_attribute(
    s: NetworkSources, component: str, name: str, attr: Callable[[int], str]
) -> xr.DataArray:
    """Return a per-port attribute read long over the component's port dimension."""
    ports = _ports(s, component)
    owner = "link" if component == "Link" else "process"
    dim = f"{owner}_output"
    parts = []
    for k, group in ports.groupby("k", sort=True):
        da = s.da(component, attr(int(k))).sel({owner: group[owner].to_numpy()})
        parts.append(
            da.rename({owner: dim}).assign_coords({dim: group["port"].to_numpy()})
        )
    if not parts:
        return _filled(s, name, np.nan)
    da = xr.concat(parts, dim=dim).reindex({dim: s.coords(dim).to_numpy()})
    dims = s.declared(name)
    if "scenario" in da.dims and "scenario" not in dims:
        da = da.isel(scenario=0, drop=True)
    return s.frame(name, da)


def _suffix(k: int) -> str:
    return "" if k == 1 else str(k)


@single("Link_efficiency")
def _link_efficiency(s: NetworkSources, name: str) -> xr.DataArray:
    return _port_attribute(s, "Link", name, lambda k: f"efficiency{_suffix(k)}").astype(
        float
    )


@single("Process_rate")
def _process_rate(s: NetworkSources, name: str) -> xr.DataArray:
    return _port_attribute(s, "Process", name, lambda k: f"rate{k}").astype(float)


def _port_column(component: str, attr: str, k: int) -> str:
    return f"{attr}{_suffix(k)}" if component == "Link" else f"{attr}{k}"


@single("Link_output_delay", "Process_output_delay")
def _output_delay(s: NetworkSources, name: str) -> xr.DataArray:
    component = name.partition("_")[0]
    try:
        delay = _port_attribute(
            s, component, name, lambda k: _port_column(component, "delay", k)
        )
    except KeyError:
        return _filled(s, name, 0)
    return delay.astype(float).fillna(0).astype(int)


@single("Link_output_delay_source", "Process_output_delay_source")
def _output_delay_source(s: NetworkSources, name: str) -> pd.DataFrame:
    """Each delayed port's source snapshot, as PyPSA's own indexer picks it."""
    component = name.partition("_")[0]
    owner = UNITS[component]
    static = _static(s, component)
    ports = _ports(s, component).set_index("port")
    cyclic = pd.Series(True, index=ports.index)
    for k, group in ports.groupby("k"):
        attr = _port_column(component, "cyclic_delay", int(k))
        if attr in static:
            cyclic[group.index] = static[attr].reindex(group[owner]).to_numpy()
    delay = s[f"{component}_output_delay"].to_series().reindex(ports.index)
    delayed = pd.DataFrame({"delay": delay, "cyclic": cyclic})[delay > 0]
    weightings = s.n.snapshot_weightings["generators"].iloc[s.positions].to_numpy()
    indexer = s.n.components[component].get_delay_source_indexer
    frames = [pd.DataFrame(columns=[f"{owner}_output", "snapshot", "source"])]
    for (d, cyc), group in delayed.groupby(["delay", "cyclic"]):
        src, valid = indexer(s.network_index, weightings, int(d), bool(cyc))
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


def _solved_before(s: NetworkSources, component: str, attr: str) -> xr.DataArray | None:
    """`attr` as solved over the snapshots before the build, or None in the first window."""
    start = int(s.positions[0])
    c = s.n.components[component]
    if not start or c.dynamic[attr].empty:
        return None
    raw = c.da[attr].isel(snapshot=slice(0, start))
    raw = raw.drop_vars([k for k in raw.coords if k not in raw.dims or k == "snapshot"])
    raw = raw.assign_coords(snapshot=np.arange(start))
    dim = UNITS[component]
    raw = raw.sel(name=raw.indexes["name"].isin(s.names(component)))
    raw = raw.rename(name=dim).assign_coords(
        {dim: raw.indexes["name"].to_numpy(dtype=object)}
    )
    if "scenario" not in raw.dims:
        raw = raw.expand_dims(scenario=s.scenarios)
    return raw


def _history(s: NetworkSources, component: str) -> xr.DataArray:
    """Return the status of each unit at the `lag` snapshots before the build, newest first.

    Solved snapshots of earlier windows are read as solved, fractions included;
    the time before the network's first snapshot is rebuilt from
    `up_time_before` and `down_time_before`, the latter winning. A unit that is
    not committable is on.
    """
    minimum = [s.da(component, f"min_{k}_time") for k in ("up", "down")]
    length = int(max([1, *(m.max().item() for m in minimum if m.size)])) + 1
    solved = _solved_before(s, component, "status")
    count = 0 if solved is None else min(solved.sizes["snapshot"], length)
    lag = xr.DataArray(np.arange(1, length - count + 1), dims="lag")
    up = s.da(component, "up_time_before")
    down = s.da(component, "down_time_before")
    before = xr.where(down > 0, lag > down, lag <= up).astype(float)
    parts = [before.assign_coords(lag=lag + count)]
    if solved is not None and count:
        recent = solved.isel(snapshot=slice(-count, None))
        recent = recent.isel(snapshot=slice(None, None, -1)).rename(snapshot="lag")
        parts.insert(0, recent.assign_coords(lag=np.arange(1, count + 1)))
    history = xr.concat(parts, dim="lag")
    committable = s.da(component, "committable").astype(bool)
    return history.where(committable, 1.0)


def _brought_in(s: NetworkSources, component: str, kind: str) -> xr.DataArray:
    """How much of the start-ups (`up`) or shut-downs (`down`) before the build still bind.

    A switch at lag `l` binds a snapshot at position `p` while `l + p` is
    below the minimum up or down time (`constraints.py:677-678`).
    """
    history = _history(s, component)
    switches = history - history.shift(lag=-1)
    switches = switches.isel(lag=slice(None, -1))
    past = (switches if kind == "up" else -switches).clip(min=0)
    minimum = s.da(component, f"min_{kind}_time")
    position = xr.DataArray(
        np.arange(len(s.snapshots)), coords={"snapshot": s.snapshots}, dims="snapshot"
    )
    return (past * (past.lag + position < minimum)).sum("lag")


@family("p_init")
def _p_init(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    """Return the output carried into the build.

    It is the solved output of an earlier window, else `p_init` where the
    unit came in running and zero where it came in off.
    """
    name = f"{component}_{suffix}"
    attr = "p0" if component in ("Link",) else "p"
    solved = _solved_before(s, component, attr)
    if solved is None:
        status = _history(s, component).sel(lag=1, drop=True)
        return s.frame(name, s.da(component, suffix).where(status > 0, 0.0))
    last = solved.isel(snapshot=-1, drop=True)
    return s.frame(
        name, last.reindex({UNITS[component]: s.coords(UNITS[component]).to_numpy()})
    )


@family("status_initial")
def _status_initial(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    status = _history(s, component).sel(lag=1, drop=True)
    return s.frame(f"{component}_{suffix}", status)


@family("up_time_brought_in")
def _up_time_brought_in(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    return s.frame(f"{component}_{suffix}", _brought_in(s, component, "up"))


@family("down_time_brought_in")
def _down_time_brought_in(
    s: NetworkSources, component: str, suffix: str
) -> xr.DataArray:
    return s.frame(f"{component}_{suffix}", _brought_in(s, component, "down"))


@family("big_m")
def _big_m(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    name = f"{component}_{suffix}"
    c = s.n.components[component]
    names = c.committables.intersection(c.extendables)
    if names.empty:
        return _filled(s, name, np.nan)
    values = c.get_committable_big_m_values(
        names, committable_big_m=s.committable_big_m
    )
    da = s.unit_array(component, values).reindex(
        {UNITS[component]: s.coords(UNITS[component]).to_numpy()}
    )
    return s.frame(name, da)


@family("s_set")
def _s_set(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    name = f"{component}_{suffix}"
    c = s.n.components[component]
    if "s_set" in c.static or "s_set" in c.dynamic:
        return s.attribute(component, "s_set", name)
    return _filled(s, name, np.nan)


@family("inactive_snapshots")
def _inactive_snapshots(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    inactive = (~s.active(component)).sum("snapshot")
    return s.frame(f"{component}_{suffix}", inactive.astype(int))


@family("opens_late")
def _opens_late(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    active = s.active(component)
    first = (active.cumsum("snapshot") == 1) & active
    position = xr.DataArray(
        np.arange(len(s.snapshots)), coords={"snapshot": s.snapshots}, dims="snapshot"
    )
    return s.frame(f"{component}_{suffix}", first & (position > 0))


@family("retention")
def _retention(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    standing = s.da(component, "standing_loss")
    hours = _snapshot_weighting(s, "snapshot_weightings_stores")
    return s.frame(f"{component}_{suffix}", (1 - standing) ** hours)


@single("Store_split_dispatch")
def _store_split_dispatch(s: NetworkSources, name: str) -> xr.DataArray:
    c = s.n.components.stores
    split = c.split_dispatch() if callable(getattr(c, "split_dispatch", None)) else None
    if split is None:
        return _filled(s, name, False)
    return s.frame(name, s.unit_array("Store", xr.DataArray(split)).astype(bool))


# -- not yet derived: placeholders for networks that do not use the feature --


def _unused(value: Any, *, empty_dims: tuple[str, ...] = ()) -> Callable:
    """Return a source a network without the feature fills with `value`.

    Raises NotImplementedError where the network uses the feature, which is
    where the dimension in `empty_dims` has labels.
    """

    def derive(s: NetworkSources, name: str) -> xr.DataArray:
        used = [d for d in empty_dims if len(s.coords(d))]
        if used:
            msg = f"no data prep for {name} on a network with {used}"
            raise NotImplementedError(msg)
        return _filled(s, name, value)

    return derive


for _component in ("Generator", "Link", "Process"):
    _SINGLES[f"{_component}_maintenance_start_blocked"] = _unused(False)
for _component in UNITS:
    _SINGLES[f"{_component}_tech_capacity_weight"] = _unused(
        np.nan, empty_dims=("global_constraint",)
    )
    _SINGLES[f"{_component}_operational_limit_weight"] = _unused(
        np.nan, empty_dims=("global_constraint",)
    )
    _SINGLES[f"{_component}_primary_energy_weight"] = _unused(
        np.nan, empty_dims=("global_constraint",)
    )
for _component in ("Line", "Link"):
    _SINGLES[f"{_component}_volume_weight"] = _unused(
        np.nan, empty_dims=("global_constraint",)
    )
    _SINGLES[f"{_component}_expansion_cost_weight"] = _unused(
        np.nan, empty_dims=("global_constraint",)
    )
for _name in (
    "GlobalConstraint_constant",
    "GlobalConstraint_ram",
):
    _SINGLES[_name] = _unused(np.nan, empty_dims=("global_constraint",))
_SINGLES["GlobalConstraint_counts_snapshot"] = _unused(
    False, empty_dims=("global_constraint",)
)
_SINGLES["GlobalConstraint_sense"] = _unused("==", empty_dims=("global_constraint",))
_SINGLES["GlobalConstraint_type"] = _unused("", empty_dims=("global_constraint",))
_SINGLES["GlobalConstraint_link_ptdf"] = _unused(
    np.nan, empty_dims=("global_constraint",)
)
_SINGLES["GlobalConstraint_zonal_ptdf"] = _unused(
    np.nan, empty_dims=("global_constraint",)
)
_SINGLES["Bus_flow_based_zone"] = _unused(False)
_SINGLES["Carrier_max_growth"] = _unused(np.nan)
_SINGLES["Carrier_max_relative_growth"] = _unused(0.0)
_SINGLES["Generator_marginal_cost_curve_cost"] = _unused(np.nan)
_SINGLES["Generator_marginal_cost_curve_p"] = _unused(np.nan)
for _component in ("Line", "Transformer"):
    _SINGLES[f"{_component}_BODF"] = _unused(np.nan, empty_dims=("outage",))
    _SINGLES[f"{_component}_cycle_weight"] = _unused(np.nan, empty_dims=("cycle",))
    _SINGLES[f"{_component}_loss_max"] = _unused(0.0)
    _SINGLES[f"{_component}_loss_offset"] = _unused(0.0, empty_dims=("segment",))
    _SINGLES[f"{_component}_loss_slope"] = _unused(0.0, empty_dims=("segment",))
_SINGLES["Transformer_phase_shift_cycle_weight"] = _unused(0.0, empty_dims=("cycle",))
_SINGLES["Transformer_phase_shift_weight"] = _unused(np.nan, empty_dims=("cycle",))
_SINGLES["Transformer_phase_shift_fixed"] = _unused(0.0, empty_dims=("transformer",))
_SINGLES["Transformer_phase_shift_varying"] = _unused(
    False, empty_dims=("transformer",)
)


@single("Bus_attached")
def _bus_attached(s: NetworkSources, name: str) -> xr.DataArray:
    buses = s.coords("bus")
    attached = xr.DataArray(
        np.zeros((len(buses), len(s.snapshots)), dtype=bool),
        coords={"bus": buses.to_numpy(), "snapshot": s.snapshots},
        dims=["bus", "snapshot"],
    )
    ends = {
        "Generator": ["bus"],
        "Load": ["bus"],
        "StorageUnit": ["bus"],
        "Store": ["bus"],
        "Line": ["bus0", "bus1"],
        "Transformer": ["bus0", "bus1"],
        "Link": ["bus0"],
    }
    for component, columns in ends.items():
        static = _static(s, component)
        active = s.active(component)
        for column in columns:
            at = xr.DataArray(
                static[column].to_numpy(dtype=object), dims=UNITS[component]
            )
            attached = attached | _any_by_bus(active, at, buses)
    for component, owner in (("Link", "link"), ("Process", "process")):
        ports = _ports(s, component)
        if ports.empty:
            continue
        active = s.active(component).sel({owner: ports[owner].to_numpy()})
        at = xr.DataArray(ports["bus"].to_numpy(dtype=object), dims=owner)
        attached = attached | _any_by_bus(active, at, buses)
    return s.frame(name, attached)


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


@family("min_up_time", "min_down_time")
def _min_time(s: NetworkSources, component: str, suffix: str) -> xr.DataArray:
    """Return the minimum time, at most the horizon: a longer window reads nothing more."""
    name = f"{component}_{suffix}"
    return s.attribute(component, suffix, name).clip(max=len(s.snapshots))


@single("Store_cycles_max")
def _cycles_max(s: NetworkSources, name: str) -> xr.DataArray:
    """`cycles_max`, absent where infinite, so a scenario without a limit has no row."""
    cycles = s.attribute("Store", "cycles_max", name)
    return cycles.where(np.isfinite(cycles))
