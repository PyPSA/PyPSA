# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Flow-based market-coupling constraints.

A flow-based domain constrains the *net positions* of market zones (buses) by a set of
linear inequalities ``zonal_ptdf . NP <= RAM``. Each row is a critical network element
(CNEC). The zonal PTDF sensitivities are stored in the dedicated
``n.c.flow_based_domains.zonal_ptdf`` frame (cnec x zone); the ``ram`` attribute (static
or time-varying) is the right-hand side.

The net position of a zone is added as a variable directly inside the nodal balance
(``generation - load - net_position = 0``), so no auxiliary buses or links are needed and
the zonal prices remain the duals of the nodal balance. A single ``sum(NP) = 0`` closes the
copper-plate balance across zones.

A domain column may also name a ``Link`` instead of a zone bus. Such columns are the
controllable HVDC corridors of advanced hybrid coupling (to an external hub) and evolved
flow-based coupling (between two zones); both are handled identically as
``zonal_ptdf . Link-p`` terms, so no distinction and no auxiliary variables are required.
The link capacity is the link's own ``p_nom``.

This module is deliberately abstract: it implements linear constraints on grouped bus net
positions and controllable link flows, and is not specific to any region or framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import xarray as xr

    from pypsa import Network

NP_VAR = "FlowBasedDomain-net_position"


def _has_flow_based(n: Network) -> bool:
    """Whether the network carries at least one active flow-based constraint."""
    static = n.c.flow_based_domains.static
    return not static.empty and static["active"].any()


def _active(n: Network) -> pd.DataFrame:
    """Active flow-based constraints (static frame)."""
    static = n.c.flow_based_domains.static
    return static[static["active"]]


def _classify_columns(n: Network) -> tuple[list, list]:
    """Split the domain columns into zone buses and (controllable) links.

    A column is a **zone** if it names a bus and a **link** if it names a link (buses
    take priority on a name clash). Zone columns multiply the net-position variable;
    link columns multiply the link flow ``Link-p`` (advanced hybrid coupling and evolved
    flow-based corridors are both just link-flow terms, with no distinction needed).

    Raises
    ------
    ValueError
        If a column is neither a bus nor a link.

    """
    cols = n.c.flow_based_domains.zonal_ptdf.columns
    buses = set(n.c.buses.static.index)
    links = set(n.c.links.static.index)
    zone_cols = [c for c in cols if c in buses]
    link_cols = [c for c in cols if c not in buses and c in links]
    unknown = [c for c in cols if c not in buses and c not in links]
    if unknown:
        msg = (
            f"Flow-based domain columns must be buses (zones) or links, but "
            f"{unknown} are neither. Add them as buses/links or pass a `buses` mapping."
        )
        raise ValueError(msg)
    return zone_cols, link_cols


def _zonal_ptdf(n: Network) -> xr.DataArray:
    """Active zonal PTDF as a DataArray with dims (name, bus)."""
    return n.c.flow_based_domains.da.zonal_ptdf.sel(name=_active(n).index)


def validate_flow_based(n: Network) -> None:
    """Check the network is compatible with a flow-based domain.

    The domain replaces the electrical grid *between* the market zones, so no branch may
    directly connect two distinct zone buses, with one exception: a controllable ``Link``
    that is a declared domain column (an AHC/EvFB HVDC corridor) is kept, since its flow
    enters the constraint explicitly. Concretely this rejects any cross-zone ``Line`` or
    ``Transformer`` (passive AC the zonal PTDF already represents) and any cross-zone
    ``Link`` that is not a domain column.

    Raises
    ------
    ValueError
        If a forbidden cross-zone branch is found.

    """
    zone_cols, link_cols = _classify_columns(n)
    zones, fb_links = set(zone_cols), set(link_cols)

    def _cross_zone(name: str) -> pd.Index:
        c = n.c[name].static
        if c.empty:
            return c.index[:0]
        crossing = c[c["bus0"].isin(zones) & c["bus1"].isin(zones) & (c["bus0"] != c["bus1"])]
        return crossing.index

    forbidden = {
        "Line": list(_cross_zone("Line")),
        "Transformer": list(_cross_zone("Transformer")),
        "Link": [i for i in _cross_zone("Link") if i not in fb_links],
    }
    offenders = {k: v for k, v in forbidden.items() if v}
    if offenders:
        msg = (
            "Flow-based domain requires the grid between zones to be represented only by "
            f"the domain; found branches connecting two zone buses: {offenders}. Remove "
            "them (the domain replaces inter-zonal exchange), or, for a controllable HVDC "
            "corridor, add it as a domain column."
        )
        raise ValueError(msg)


def define_flow_based_variables(n: Network, sns: pd.Index) -> None:
    """Define the zonal net-position variables of the flow-based domain."""
    if not _has_flow_based(n):
        return
    validate_flow_based(n)
    zones = pd.Index(_classify_columns(n)[0], name="bus")
    n.model.add_variables(coords=[sns, zones], name=NP_VAR)


def define_flow_based_constraints(n: Network, sns: pd.Index) -> None:
    """Define the flow-based constraints ``zonal_ptdf . NP <= RAM`` and ``sum(NP) = 0``.

    The net-position variables are injected into the nodal balance elsewhere; here we add
    the domain half-spaces and the global zero-sum balance. Link columns (AHC/EvFB
    corridors) contribute ``zonal_ptdf . Link-p`` terms, using the link flow in its
    ``bus0 -> bus1`` direction (the domain column must follow that sign convention). The
    constraint is indexed by the CNEC (component ``name``), so its dual is mapped to
    ``mu_domain`` by the standard dual assignment.
    """
    if not _has_flow_based(n):
        return
    m = n.model
    zone_cols, link_cols = _classify_columns(n)
    ptdf = _zonal_ptdf(n)

    lhs = (m[NP_VAR] * ptdf.sel(bus=zone_cols)).sum("bus")  # dims (snapshot, name)
    if link_cols:
        link_p = m["Link-p"].sel(name=link_cols).rename(name="link")
        lhs = lhs + (link_p * ptdf.sel(bus=link_cols).rename(bus="link")).sum("link")

    ram = n.c.flow_based_domains.da.ram.sel(name=_active(n).index)
    m.add_constraints(lhs <= ram, name="FlowBasedDomain-domain")
    m.add_constraints(m[NP_VAR].sum("bus") == 0, name="FlowBasedDomain-balance")
