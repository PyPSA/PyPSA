# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Flow-based market-coupling constraints.

A flow-based domain constrains the *net positions* of market zones (buses) by a set of
linear inequalities ``PTDF . NP <= RAM``. Each row is a critical network element (CNEC).
The zonal PTDF sensitivities are stored in the dedicated ``n.c.flow_based_domains.ptdf``
frame (cnec x zone); the ``ram`` attribute (static or time-varying) is the right-hand
side.

The net position of a zone is added as a variable directly inside the nodal balance
(``generation - load - net_position = 0``), so no auxiliary buses or links are needed and
the zonal prices remain the duals of the nodal balance. A single ``sum(NP) = 0`` closes the
copper-plate balance across zones.

This module is deliberately abstract: it implements linear constraints on grouped bus net
positions and is not specific to any particular region or regulatory framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from pypsa.components.array import _from_xarray
from pypsa.optimization.common import _set_dynamic_data

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


def _zones(n: Network) -> list:
    """Zone buses referenced by the domain (PTDF columns)."""
    return list(n.c.flow_based_domains.ptdf.columns)


def _ptdf(n: Network) -> xr.DataArray:
    """Active zonal PTDF as a DataArray with dims (cnec, bus)."""
    da = n.c.flow_based_domains.da.ptdf.sel(name=_active(n).index)
    return da.rename(name="cnec")


def validate_flow_based(n: Network) -> None:
    """Check the network is compatible with a flow-based domain.

    The domain replaces the electrical exchange between zones, so there must be no
    electrical links directly connecting two different zone buses. Non-electrical links
    (e.g. gas pipelines, electrolysers) are ignored: only links whose both ends are zone
    buses are flagged.

    Raises
    ------
    ValueError
        If any link connects two distinct zone buses referenced by the domain.

    """
    zones = set(_zones(n))
    links = n.c.links.static
    if links.empty:
        return
    crossing = links[
        links["bus0"].isin(zones)
        & links["bus1"].isin(zones)
        & (links["bus0"] != links["bus1"])
    ]
    if not crossing.empty:
        msg = (
            "Flow-based domain requires cross-zone electrical links to be removed, but "
            f"found {len(crossing)} link(s) connecting two zone buses: "
            f"{list(crossing.index)}. Remove them (the flow-based domain replaces the "
            "inter-zonal exchange) or point them at non-zone buses."
        )
        raise ValueError(msg)


def define_flow_based_variables(n: Network, sns: pd.Index) -> None:
    """Define the zonal net-position variables of the flow-based domain."""
    if not _has_flow_based(n):
        return
    validate_flow_based(n)
    zones = pd.Index(_zones(n), name="bus")
    n.model.add_variables(coords=[sns, zones], name=NP_VAR)


def define_flow_based_constraints(n: Network, sns: pd.Index) -> None:
    """Define the flow-based domain constraints ``PTDF . NP <= RAM`` and ``sum(NP) = 0``.

    The net-position variables are injected into the nodal balance elsewhere; here we add
    the domain half-spaces and the global zero-sum balance.
    """
    if not _has_flow_based(n):
        return
    m = n.model
    np_var = m[NP_VAR]
    ptdf = _ptdf(n)

    lhs = (np_var * ptdf).sum("bus")  # dims (snapshot, cnec)
    ram = n.c.flow_based_domains.da.ram.sel(name=_active(n).index).rename(name="cnec")
    m.add_constraints(lhs <= ram, name="FlowBasedDomain-domain")

    m.add_constraints(np_var.sum("bus") == 0, name="FlowBasedDomain-balance")


def assign_flow_based_duals(n: Network) -> None:
    """Write the domain shadow prices into the CNEC-indexed ``mu`` series.

    The flow-based variables and constraints are indexed by zone bus / CNEC rather than
    by the component name, so the generic dual assignment cannot map them. This writes
    the ``FlowBasedDomain-domain`` dual (per snapshot and CNEC) into ``mu`` directly.
    """
    if not _has_flow_based(n):
        return
    con = n.model.constraints["FlowBasedDomain-domain"]
    if "dual" not in con:
        return
    c = n.c.flow_based_domains
    mu = _from_xarray(con.dual.rename(cnec="name"), c)
    _set_dynamic_data(n, "FlowBasedDomain", "mu", mu)
