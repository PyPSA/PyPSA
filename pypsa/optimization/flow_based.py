# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Flow-based market-coupling constraints.

Bounds the net positions of market zones (buses) by ``zonal_ptdf . NP <= RAM``, one row
per critical network element (CNEC), plus a zero-sum balance ``sum(NP) = 0``. Each net
position is a variable inside the nodal balance, so it equals the zone bus's net injection
into the flow-based region. A domain column may instead name a ``Link`` (an AHC or EvFB
HVDC corridor): its flow loads the CNECs through its own column, the sensitivity to the
corridor flow with the zone net positions held fixed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from pypsa import Network

NP_VAR = "Bus-net_position"
FB_TYPE = "flow_based"


def _active(n: Network) -> pd.DataFrame:
    """Flow-based global constraints (static frame); each must bound with ``<=``."""
    static = n.c.global_constraints.static
    rows = static[static["type"] == FB_TYPE]
    if (rows["sense"] != "<=").any():
        msg = (
            f"Flow-based global constraints need sense '<='; got {set(rows['sense'])}."
        )
        raise ValueError(msg)
    return rows


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
    cols = n.c.global_constraints.zonal_ptdf.columns
    is_bus = cols.isin(n.c.buses.static.index)
    is_link = cols.isin(n.c.links.static.index) & ~is_bus
    if unknown := cols[~is_bus & ~is_link].tolist():
        msg = (
            f"Flow-based domain columns must be buses or links, but "
            f"{unknown} are neither."
        )
        raise ValueError(msg)
    return cols[is_bus].tolist(), cols[is_link].tolist()


def flow_based_balance_terms(n: Network, buses: pd.Index) -> Any:
    """``-net_position`` for every zone bus of the flow-based domain, or ``None``.

    Corridor links stay in the nodal balance, so a zone's net position is its generation
    minus load plus the inflow of its corridor links.
    """
    if NP_VAR not in n.model.variables:
        return None
    np_var = n.model[NP_VAR]
    return -1 * np_var.sel(name=np_var.indexes["name"].intersection(buses))


def validate_flow_based(n: Network) -> None:
    """Reject branches directly connecting two zone buses and links outside the region.

    The domain replaces the grid *between* zones, so a cross-zone ``Line``, ``Transformer``
    or ``Link`` is forbidden, except a ``Link`` that is a declared domain column (an AHC/EvFB
    corridor), whose flow enters the constraint explicitly. Such a link column must have
    at least one zone end.

    Raises
    ------
    ValueError
        If a forbidden branch or link column is found.

    """
    zone_cols, link_cols = _classify_columns(n)
    zones = set(zone_cols)

    def _cross_zone(comp: str) -> pd.Index:
        c = n.c[comp].static
        return c.index[c.bus0.isin(zones) & c.bus1.isin(zones) & (c.bus0 != c.bus1)]

    crossing = {comp: _cross_zone(comp) for comp in ("Line", "Transformer", "Link")}
    crossing["Link"] = crossing["Link"].difference(link_cols)
    if offenders := {k: v.tolist() for k, v in crossing.items() if not v.empty}:
        msg = (
            "Flow-based domain requires the grid between zones to be represented only by "
            f"the domain; found branches connecting two zone buses: {offenders}. Remove "
            "them, or, for a controllable HVDC corridor, add it as a domain column."
        )
        raise ValueError(msg)

    links = n.c.links.static.loc[link_cols]
    if outside := links.index[
        ~links.bus0.isin(zones) & ~links.bus1.isin(zones)
    ].tolist():
        msg = f"Link columns {outside} have no flow-based zone at either end."
        raise ValueError(msg)


def validate_constant(n: Network) -> None:
    """Allow a time-varying global constraint ``constant`` only for flow-based rows.

    Raises
    ------
    ValueError
        If a global constraint of another type has a time-varying ``constant``.

    """
    c = n.c.global_constraints
    names = c.dynamic["constant"].columns
    if bad := names[c.static["type"].reindex(names) != FB_TYPE].tolist():
        msg = f"Global constraints {bad} have a time-varying `constant`; only type {FB_TYPE!r} may."
        raise ValueError(msg)


def define_flow_based_variables(n: Network, sns: pd.Index) -> None:
    """Define the zonal net-position variables of the flow-based domain."""
    validate_constant(n)
    if _active(n).empty:
        return
    validate_flow_based(n)
    zones = pd.Index(_classify_columns(n)[0], name="name")
    n.model.add_variables(coords=[sns, zones], name=NP_VAR)


def _period_mask(n: Network, sns: pd.Index, snapshot: xr.DataArray) -> Any:
    """Where each row applies: in its investment period's snapshots, or everywhere."""
    period = _active(n)["investment_period"]
    if not n._multi_invest:
        return None
    if unknown := sorted(set(period.dropna()) - set(n.investment_periods)):
        msg = f"Flow-based rows refer to {unknown}, which are not investment periods."
        raise ValueError(msg)
    if period.isna().all():
        return None
    period_of = n.optimize._window.subset(sns).period_of
    of = xr.DataArray(period_of.values, coords={"snapshot": snapshot})
    row = xr.DataArray(period.rename_axis("name"))
    return (of == row) | row.isnull()


def define_flow_based_constraints(n: Network, sns: pd.Index) -> None:
    """Define the domain half-spaces ``zonal_ptdf . NP <= RAM`` and the balance ``sum(NP) = 0``.

    Link columns (AHC/EvFB corridors) add ``zonal_ptdf . Link-p`` terms in the link's
    ``bus0 -> bus1`` direction. Corridor imports are part of the zone net positions, so the
    balance closes over the zones alone.
    """
    if _active(n).empty:
        return
    m = n.model
    zone_cols, link_cols = _classify_columns(n)
    ptdf = n.c.global_constraints.da.zonal_ptdf

    net_pos = m[NP_VAR].rename(name="bus")
    lhs = (net_pos * ptdf.sel(bus=zone_cols)).sum("bus")  # dims (snapshot, name)
    if link_cols:
        link_p = m["Link-p"].sel(name=link_cols).rename(name="link")
        lhs = lhs + (link_p * ptdf.sel(bus=link_cols).rename(bus="link")).sum("link")

    ram = n.c.global_constraints.da.constant.sel(name=ptdf.indexes["name"])
    mask = _period_mask(n, sns, lhs.coords["snapshot"])
    m.add_constraints(lhs <= ram, name="GlobalConstraint-flow_based", mask=mask)

    m.add_constraints(
        net_pos.sum("bus") == 0, name="GlobalConstraint-flow_based_balance"
    )
