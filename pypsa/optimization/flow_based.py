# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Define flow-based market coupling domains for optimisation with Linopy.

A flow-based domain is a set of `GlobalConstraint` rows with
`type="flow_based"`, one per critical network element (CNEC). Their
`ptdf_<name>` columns hold the zonal PTDF, where `<name>` is a zone bus or a
corridor Link. Each zone bus gets a net position variable that enters its nodal
balance, and every CNEC limits
`sum_z PTDF[c, z] * NP[z] + sum_l PTDF[c, l] * p[l]` to its `ram`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.components._types.global_constraints import PTDF_PREFIX
from pypsa.components.array import _from_xarray
from pypsa.components.common import invariant
from pypsa.optimization.common import get_bus_counts
from pypsa.optimization.window import snapshot_array

if TYPE_CHECKING:
    from linopy import LinearExpression

    from pypsa import Network

NET_POSITION = "FlowBased-net_position"
BALANCE = "FlowBased-balance"
RAM_CONSTRAINT = "GlobalConstraint-ram"


@dataclass
class _FlowBasedDomain:
    """Validated flow-based domain of one model build."""

    rows: pd.Index
    zones: pd.Index
    links: pd.Index
    ptdf: xr.DataArray
    ram: xr.DataArray
    mask: xr.DataArray
    zone_domain: pd.Series | None


def _over_snapshots(da: xr.DataArray, sns: pd.Index) -> xr.DataArray:
    """Select `sns` from a dynamic array, or broadcast a static one over them."""
    if "snapshot" in da.dims:
        return da.sel(snapshot=sns)
    return da * snapshot_array(np.ones(len(sns)), sns)


def _endpoints(n: Network, component: str) -> pd.DataFrame:
    """`bus0` and `bus1` of all assets of a branch component, indexed by name."""
    return invariant(n.components[component].static[["bus0", "bus1"]])


def _zone_domains(n: Network, rows: pd.Index, zones: pd.Index) -> pd.Series | None:
    """Map each zone bus to its named domain, None when no row names one.

    A zone belongs to the domain whose rows set its `ptdf_<zone>` column.
    """
    c = n.c.global_constraints
    names = invariant(c.static["domain"]).loc[rows]
    unnamed = names == ""
    if unnamed.all():
        return None
    if unnamed.any():
        msg = (
            f"Flow-based rows {rows[unnamed].tolist()} have no `domain`, but others do. "
            "Name the domain of every flow-based row or of none."
        )
        raise ValueError(msg)
    set_by_row = {}
    for zone in zones:
        da = c.da[f"{PTDF_PREFIX}{zone}"].notnull()
        set_by_row[zone] = da.any([d for d in da.dims if d != "name"]).to_pandas()
    member = pd.DataFrame(set_by_row).loc[rows].groupby(names).any()
    ambiguous = member.columns[member.sum() != 1]
    if not ambiguous.empty:
        msg = (
            f"Zone buses {ambiguous.tolist()} must belong to exactly one flow-based "
            "domain. Set their `ptdf_<zone>` columns in the rows of one domain only."
        )
        raise ValueError(msg)
    return member.idxmax().rename_axis("bus").rename("domain")


def _build_domain(n: Network, sns: pd.Index) -> _FlowBasedDomain | None:
    """Collect and validate the flow-based domain, None when there is none."""
    c = n.c.global_constraints
    rows = c._flow_based_names()
    if rows.empty:
        return None
    if "ram" in c.names.difference(rows):
        msg = (
            "A global constraint named `ram` clashes with the flow-based "
            f"constraint `{RAM_CONSTRAINT}`. Rename it."
        )
        raise ValueError(msg)
    sense = invariant(c.static["sense"]).loc[rows]
    if (sense != "<=").any():
        msg = (
            f"Flow-based rows {sense.index[sense != '<='].tolist()} need "
            f'`sense="<="`; got {sorted(set(sense))}.'
        )
        raise ValueError(msg)

    columns = c._ptdf_columns()
    buses = n.c.buses.static.index.unique("name")
    link_names = n.c.links.static.index.unique("name")
    clash = columns[columns.isin(buses) & columns.isin(link_names)]
    if not clash.empty:
        msg = f"PTDF columns {clash.tolist()} name both a Bus and a Link."
        raise ValueError(msg)
    unknown = columns[~columns.isin(buses) & ~columns.isin(link_names)]
    if not unknown.empty:
        msg = (
            f"Unknown PTDF columns {unknown.tolist()}: each `ptdf_<name>` column "
            "must name a Bus (a zone) or a Link (a corridor)."
        )
        raise ValueError(msg)
    zones = columns[columns.isin(buses)]
    links = columns[columns.isin(link_names)]
    inactive = links.difference(n.c.links.active_assets)
    if not inactive.empty:
        msg = (
            f"Corridor Links {inactive.tolist()} of the flow-based domain are "
            "inactive, so they have no flow variable."
        )
        raise ValueError(msg)
    isolated = zones[get_bus_counts(n).reindex(zones, fill_value=0).to_numpy() == 0]
    if not isolated.empty:
        msg = (
            f"Zone buses {isolated.tolist()} of the flow-based domain have no "
            "attached components, so their net position would be unbalanced."
        )
        raise ValueError(msg)

    for component in ("Line", "Transformer", "Link"):
        ends = _endpoints(n, component).loc[n.c[component].active_assets]
        cross = ends.bus0.isin(zones) & ends.bus1.isin(zones)
        if component == "Link":
            cross &= ~ends.index.isin(links)
        if cross.any():
            msg = (
                f"{component} {ends.index[cross].tolist()} connect two zone buses "
                "of the flow-based domain. Remove them, or declare a Link as a "
                "corridor with a `ptdf_<name>` column."
            )
            raise ValueError(msg)
    link_ends = _endpoints(n, "Link").loc[links]
    detached = ~link_ends.bus0.isin(zones) & ~link_ends.bus1.isin(zones)
    if detached.any():
        msg = (
            f"Link PTDF columns {link_ends.index[detached].tolist()} touch no zone "
            "bus of the flow-based domain."
        )
        raise ValueError(msg)

    ptdf = c.da.zonal_ptdf.sel(name=rows, bus=zones.append(links))
    ptdf = _over_snapshots(ptdf, sns)
    ram = _over_snapshots(c.da.ram.sel(name=rows), sns)

    period = invariant(c.static["investment_period"]).loc[rows]
    if not n._multi_invest:
        if period.notna().any():
            msg = (
                f"Flow-based rows {period.index[period.notna()].tolist()} set "
                "investment_period but the model has no investment periods."
            )
            raise ValueError(msg)
        mask = xr.DataArray(np.ones(len(rows), dtype=bool), coords=[rows], dims="name")
    else:
        bad = period.notna() & ~period.isin(n.investment_periods)
        if bad.any():
            msg = (
                f"Flow-based rows {period.index[bad].tolist()} set an "
                f"investment_period outside {n.investment_periods.tolist()}."
            )
            raise ValueError(msg)
        period_da = xr.DataArray(period.to_numpy(), coords={"name": rows}, dims="name")
        period_of = n.optimize._window.subset(sns).period_of.to_numpy()
        mask = period_da.isnull() | (snapshot_array(period_of, sns) == period_da)
    mask = (mask & snapshot_array(np.ones(len(sns), dtype=bool), sns)).transpose(
        "snapshot", "name"
    )

    ram_all = c.da.ram.sel(name=rows).notnull()
    no_ram = ~ram_all.any([d for d in ram_all.dims if d != "name"])
    if no_ram.any():
        msg = f"Flow-based rows {rows[no_ram.to_numpy()].tolist()} have no `ram`."
        raise ValueError(msg)
    mask = mask & ram.notnull()

    zone_domain = _zone_domains(n, rows, zones)
    return _FlowBasedDomain(rows, zones, links, ptdf, ram, mask, zone_domain)


def define_flow_based_variables(n: Network, sns: pd.Index) -> _FlowBasedDomain | None:
    """Define the zonal net positions of the flow-based domain.

    Validates the domain and adds the unbounded variable `FlowBased-net_position`
    over snapshots and zone buses. A no-op without `flow_based` rows.

    Parameters
    ----------
    n : pypsa.Network
        Network to optimise.
    sns : pandas.Index
        Snapshots of the model.

    Returns
    -------
    _FlowBasedDomain or None
        The validated domain for `define_flow_based_constraints`, None without
        `flow_based` rows.

    """
    domain = _build_domain(n, sns)
    if domain is None:
        return None
    coords = [sns, pd.Index(domain.zones, name="bus")]
    if n.has_scenarios:
        coords = [n.scenarios, *coords]
    n.model.add_variables(coords=coords, name=NET_POSITION)
    return domain


def flow_based_balance_terms(n: Network, buses: pd.Index) -> list[LinearExpression]:
    """Nodal balance terms `-NP` of the zone buses among `buses`.

    Parameters
    ----------
    n : pypsa.Network
        Network to optimise.
    buses : pandas.Index
        Buses of the nodal balance constraint being built.

    Returns
    -------
    list[linopy.LinearExpression]
        One expression over `name`, empty when no zone bus is in `buses`.

    """
    if n.c.global_constraints._flow_based_names().empty:
        return []
    net_position = n.model[NET_POSITION]
    sel = net_position.indexes["bus"].intersection(buses)
    if sel.empty:
        return []
    return [(-1 * net_position.sel(bus=sel)).rename({"bus": "name"})]


def define_flow_based_constraints(n: Network, domain: _FlowBasedDomain | None) -> None:
    """Define the flow-based domain constraints.

    Adds `GlobalConstraint-ram`, the PTDF-weighted net positions and corridor
    flows of every CNEC limited by its `ram`, and `FlowBased-balance`, the zone
    net positions summing to zero, per named `domain` if rows name one. A row
    with an `investment_period` binds only in that period's snapshots, and a
    row with a NaN `ram` is dropped in that snapshot. `constant` is not read.

    Parameters
    ----------
    n : pypsa.Network
        Network to optimise.
    domain : _FlowBasedDomain or None
        Domain returned by `define_flow_based_variables`, None is a no-op.

    """
    if domain is None:
        return
    m = n.model
    net_position = m[NET_POSITION]
    lhs = (net_position * domain.ptdf.sel(bus=domain.zones)).sum("bus")
    if not domain.links.empty:
        flows = (1 * m["Link-p"].sel(name=domain.links)).rename({"name": "bus"})
        lhs += (flows * domain.ptdf.sel(bus=domain.links)).sum("bus")
    mask = xr.broadcast(domain.mask, domain.ram)[0]
    m.add_constraints(lhs <= domain.ram, name=RAM_CONSTRAINT, mask=mask)
    if domain.zone_domain is None:
        balance = net_position.sum("bus")
    else:
        balance = net_position.groupby(domain.zone_domain.to_xarray()).sum()
    m.add_constraints(balance == 0, name=BALANCE)


def flow_based_net_position(n: Network) -> pd.DataFrame | None:
    """Return generation minus load of each zone bus, or ``None`` without a domain.

    This is the solved net position without the flows of the zone's corridor links
    (the link columns of the domain), i.e. the net position of the virtual-hub
    convention. Read from `n.buses_t.net_position` after solving.
    """
    if NET_POSITION not in n.model.variables:
        return None
    solution = n.model[NET_POSITION].solution.rename({"bus": "name"})
    net_position = _from_xarray(solution, n.c.buses)
    links = n.c.links
    columns = n.c.global_constraints._ptdf_columns()
    corridors = columns[columns.isin(links.static.index.unique("name"))].rename("name")
    for i in links.ports:
        flow = links.dynamic[f"p{i}"].reindex(columns=corridors, level="name")
        flow = flow.rename(columns=invariant(links.static[f"bus{i}"]), level="name")
        withdrawn = flow.T.groupby(level=list(range(flow.columns.nlevels))).sum().T
        net_position += withdrawn.reindex(columns=net_position.columns, fill_value=0.0)
    return net_position
