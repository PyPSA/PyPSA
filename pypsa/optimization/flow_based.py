# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Flow-based market-coupling constraints.

A flow-based domain constrains the *net positions* of market zones (buses) by a set of
linear inequalities ``zonal_ptdf . NP <= RAM``. Each row is a critical network element
(CNEC). The zonal PTDF sensitivities are stored in the dedicated
``n.c.flow_based_constraints.zonal_ptdf`` frame (cnec x zone, or ``(snapshot, cnec) x zone``
when time-varying); the ``ram`` attribute (static or time-varying) is the right-hand
side. Both the PTDF and the RAM may vary by snapshot; the constraint broadcasts either.

The net position of a zone is added as a variable directly inside the nodal balance
(``generation - load - net_position = 0``), so no auxiliary buses or links are needed and
the zonal prices remain the duals of the nodal balance. A single zero-sum balance
``sum(NP) = 0`` closes the copper plate across zones (AHC virtual hubs join it, below).

A domain column may also name a ``Link`` instead of a zone bus: the controllable HVDC
corridors of advanced hybrid coupling (AHC, a border to an external hub) and evolved
flow-based coupling (EvFB, between two zones). Each such column loads the CNECs through a
``zonal_ptdf . Link-p`` term, using the existing interconnector (its capacity is the link's
``p_nom``). To keep every zone's net position equal to ``generation - load``, the corridor
link's Core-side contribution is cancelled in the nodal balance (EvFB at both ends, AHC at
its Core end); an AHC border additionally routes its flow onto the ``sum(NP) = 0`` balance,
where it is the net position of the external virtual hub.

This module is deliberately abstract: it implements linear constraints on grouped bus net
positions and controllable link flows, and is not specific to any region or framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from pypsa import Network

NP_VAR = "FlowBasedConstraint-net_position"


def _has_flow_based(n: Network) -> bool:
    """Whether the network carries at least one active flow-based constraint."""
    static = n.c.flow_based_constraints.static
    return not static.empty and static["active"].any()


def _active(n: Network) -> pd.DataFrame:
    """Active flow-based constraints (static frame)."""
    static = n.c.flow_based_constraints.static
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
    cols = n.c.flow_based_constraints.zonal_ptdf.columns
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


def _corridor_cut(n: Network) -> Any:
    """Per-zone linear expression cancelling each corridor link's Core-side balance term.

    The nodal balance adds ``-Link-p`` at ``bus0`` and ``+efficiency . Link-p`` at ``bus1``.
    Cancelling those terms at a corridor's zone end(s) keeps every zone net position equal to
    ``generation - load``, so the corridor loads the CNECs only through its own column. A
    fully external link (neither end a zone) gets a zero column and drops out. Returns a
    ``(snapshot, name)`` expression over zone buses, or ``None`` if no corridor has a zone
    end. Its negated bus-sum is the AHC virtual hubs' net exchange with Core (see the plate
    balance in :func:`define_flow_based_constraints`); internal lossless EvFB nets to zero.
    """
    zones = set(_classify_columns(n)[0])
    links = n.c.links.static
    eff = links["efficiency"]
    coeff = pd.DataFrame(
        0.0, index=pd.Index(sorted(zones), name="name"), columns=_classify_columns(n)[1]
    )
    for link in coeff.columns:
        b0, b1 = links.at[link, "bus0"], links.at[link, "bus1"]
        if b0 in zones:
            coeff.at[b0, link] += 1.0
        if b1 in zones:
            coeff.at[b1, link] += -eff[link]
    coeff = coeff.loc[:, (coeff != 0).any()]
    if coeff.columns.empty:
        return None
    da = xr.DataArray(
        coeff.values, dims=["name", "link"],
        coords={"name": coeff.index, "link": coeff.columns},
    )
    link_p = n.model["Link-p"].sel(name=list(coeff.columns)).rename(name="link")
    return (link_p * da).sum("link")


def flow_based_balance_terms(n: Network, buses: pd.Index) -> Any:
    """Terms the flow-based domain contributes to the nodal balance, or ``None``.

    Each zone bus gets ``-net_position`` (its balance reads ``generation - load - NP = 0``)
    plus the cancellation of any corridor link's Core-side term (see :func:`_corridor_cut`),
    so net positions stay ``generation - load`` and corridors are not double-counted.
    """
    if NP_VAR not in n.model.variables:
        return None
    np_var = n.model[NP_VAR].rename(bus="name")
    fb_buses = np_var.indexes["name"].intersection(buses)
    if fb_buses.empty:
        return None
    cut = _corridor_cut(n)
    expr = -1 * np_var
    if cut is not None:
        expr = expr + cut
    return expr.sel(name=fb_buses)


def _zonal_ptdf(n: Network) -> xr.DataArray:
    """Active zonal PTDF as a DataArray with dims (name, bus)."""
    return n.c.flow_based_constraints.da.zonal_ptdf.sel(name=_active(n).index)


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
    the domain half-spaces and the global balance. Link columns (AHC/EvFB corridors)
    contribute ``zonal_ptdf . Link-p`` terms, using the link flow in its ``bus0 -> bus1``
    direction (the domain column must follow that sign convention). The global balance is
    ``sum(NP) + sum(AHC hub net positions) = 0``: Core zones plus the AHC virtual hubs sum
    to zero, so Core need not be internally balanced when it exchanges over AHC borders. The
    domain constraint is indexed by the CNEC (component ``name``), so its dual is mapped to
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

    ram = n.c.flow_based_constraints.da.ram.sel(name=_active(n).index)
    m.add_constraints(lhs <= ram, name="FlowBasedConstraint-domain")

    plate = m[NP_VAR].sum("bus")
    cut = _corridor_cut(n)
    if cut is not None:
        plate = plate - cut.sum("name")  # AHC hubs' net exchange with Core
    m.add_constraints(plate == 0, name="FlowBasedConstraint-balance")
