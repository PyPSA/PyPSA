# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Flow-based market-coupling constraints.

Bounds the net positions of market zones (buses) by ``zonal_ptdf . NP <= RAM``, one row
per critical network element (CNEC), plus a zero-sum balance ``sum(NP) = 0``. Each net
position is a variable inside the nodal balance. A domain column may instead name a ``Link``
(an AHC or EvFB HVDC corridor): its flow loads the CNECs through its own domain column.
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

    A column is a zone if it names a bus and a link if it names a link (buses
    take priority on a name clash). Zone columns multiply the net-position variable;
    link columns multiply the link flow ``Link-p``.

    Raises
    ------
    ValueError
        If a column is neither a bus nor a link.

    """
    cols = n.c.flow_based_constraints.zonal_ptdf.columns
    is_bus = cols.isin(n.c.buses.static.index)
    is_link = cols.isin(n.c.links.static.index) & ~is_bus
    if unknown := cols[~is_bus & ~is_link].tolist():
        msg = (
            f"Flow-based domain columns must be buses or links, but "
            f"{unknown} are neither."
        )
        raise ValueError(msg)
    return cols[is_bus].tolist(), cols[is_link].tolist()


def _corridor_cut(n: Network) -> Any:
    """Per-zone expression cancelling each corridor link's flow-based-side balance term.

    The nodal balance adds ``-Link-p`` at ``bus0`` and ``+efficiency . Link-p`` at ``bus1``;
    cancelling those at a corridor's zone end(s) preserves every zone net position.
    Returns a ``(snapshot, name)`` expression over zone buses, or
    ``None``. Its negated bus-sum is the AHC hubs' net
    exchange with flow-based region; EvFBs net to zero.
    """
    zone_cols, link_cols = _classify_columns(n)
    if not link_cols:
        return None
    links = n.c.links.static.loc[link_cols]
    zones = pd.Index(sorted(zone_cols), name="name")
    plus = pd.get_dummies(links["bus0"]).reindex(columns=zones, fill_value=0)
    minus = pd.get_dummies(links["bus1"]).reindex(columns=zones, fill_value=0)
    coeff = (plus - minus.mul(links["efficiency"], axis=0)).T
    coeff = coeff.loc[:, (coeff != 0).any()]
    if coeff.columns.empty:
        return None
    coeff.columns.name = "link"
    link_p = n.model["Link-p"].sel(name=list(coeff.columns)).rename(name="link")
    return (link_p * xr.DataArray(coeff)).sum("link")


def flow_based_balance_terms(n: Network, buses: pd.Index) -> Any:
    """Terms the flow-based domain contributes to the nodal balance, or ``None``.

    Each zone bus gets ``-net_position`` appended plus the cancellation of any
    corridor link's flow-based-side term to preserve net positions and ensure
    corridors are not double-counted.
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
    """Reject branches directly connecting two zone buses.

    The domain replaces the grid *between* zones, so a cross-zone ``Line``, ``Transformer``
    or ``Link`` is forbidden, except a ``Link`` that is a declared domain column (an AHC/EvFB
    corridor), whose flow enters the constraint explicitly.

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
        crossing = c[
            c["bus0"].isin(zones) & c["bus1"].isin(zones) & (c["bus0"] != c["bus1"])
        ]
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
            "them, or, for a controllable HVDC corridor, add it as a domain column."
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
    """Define the domain half-spaces ``zonal_ptdf . NP <= RAM`` and the balance ``sum(NP) = 0``.

    Link columns (AHC/EvFB corridors) add ``zonal_ptdf . Link-p`` terms in the link's
    ``bus0 -> bus1`` direction.
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
        plate = plate - cut.sum("name")  # AHC hubs' net exchange
    m.add_constraints(plate == 0, name="FlowBasedConstraint-balance")
