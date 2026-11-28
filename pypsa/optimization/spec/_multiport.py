# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep of the spec: the ports of multiport units, their coefficients and delays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.optimization.spec.sources import NetworkSources, source

if TYPE_CHECKING:
    from pypsa.components import Components
    from pypsa.components._types.mixin.multiports import _Multiport


@source("COMPONENT_port_bus", "COMPONENT_port_UNIT")
def _port_field(s: NetworkSources, name: str, c: Components) -> pd.DataFrame:
    """Map each port, (unit, number), to its bus, or to its unit."""
    ports = s.ports(c)
    if name.endswith("_bus"):
        return ports.astype(object)
    return ports.assign(owner=ports[c._unit_dim])[[c._unit_dim, "port", "owner"]]


def _port_attribute(
    s: NetworkSources, c: _Multiport, name: str, prefix: str, unset: float = np.nan
) -> xr.DataArray:
    """Return the attribute `<prefix><port suffix>` over (unit, port), NaN where a unit has no such port.

    A port that has no such attribute, a Link's port 0, takes `unset`.
    """
    unit = c._unit_dim
    parts = []
    for k, group in s.ports(c).groupby("port", sort=True):
        attr = f"{prefix}{c._port_suffix(k)}"
        if attr in c.static or attr in c.dynamic:
            da = s.da(c, attr)
        else:
            names = s.names(c).to_numpy()
            da = xr.DataArray(np.full(len(names), unset), {unit: names}, unit)
        has = da.indexes[unit].isin(group[unit])
        mask = xr.DataArray(has, coords={unit: da.indexes[unit]}, dims=unit)
        parts.append(da.where(mask).expand_dims(port=[k]))
    if not parts:
        return s.filled(name, np.nan)
    da = xr.concat(parts, dim="port").reindex(port=s.coords("port"))
    if "scenario" in da.dims and "scenario" not in s.declared(name):
        da = da.isel(scenario=0, drop=True)
    return s.frame(name, da)


@source("Link_efficiency", "Process_rate")
def _coefficient(s: NetworkSources, name: str) -> xr.DataArray:
    c = s.n.components[name.partition("_")[0]]
    return _port_attribute(s, c, name, c._coefficient_attr, unset=-1).astype(float)


@source("COMPONENT_port_delay")
def _port_delay(s: NetworkSources, name: str, c: _Multiport) -> xr.DataArray:
    try:
        delay = _port_attribute(s, c, name, "delay", unset=0)
    except KeyError:
        return s.filled(name, 0)
    return delay.astype(float).fillna(0).astype(int)


@source("COMPONENT_port_delay_source")
def _port_delay_source(s: NetworkSources, name: str, c: _Multiport) -> pd.DataFrame:
    """Each delayed port's source snapshot, as PyPSA's own indexer picks it."""
    unit = c._unit_dim
    static = s.static(c)
    ports = s.ports(c)[[unit, "port"]]
    delay = s[f"{c.name}_port_delay"]
    at = (
        delay.indexes[unit].get_indexer(ports[unit]),
        delay.indexes["port"].get_indexer(ports["port"]),
    )
    ports = ports.assign(delay=delay.transpose(unit, "port").values[at], cyclic=True)
    for k, group in ports.groupby("port"):
        attr = f"cyclic_delay{c._port_suffix(k)}"
        if attr in static:
            cyclic = static[attr].reindex(group[unit]).to_numpy()
            ports.loc[group.index, "cyclic"] = cyclic
    delayed = ports[ports["delay"] > 0]
    weightings = s.n.snapshot_weightings["generators"].iloc[s.positions].to_numpy()
    frames = [pd.DataFrame(columns=[unit, "port", "snapshot", "source"])]
    for (d, cyc), group in delayed.groupby(["delay", "cyclic"]):
        src, valid = c.get_delay_source_indexer(
            s.network_index, weightings, int(d), bool(cyc)
        )
        targets = np.flatnonzero(valid)
        frames.append(
            pd.DataFrame(
                {
                    unit: np.repeat(group[unit].to_numpy(dtype=object), len(targets)),
                    "port": np.repeat(group["port"].to_numpy(), len(targets)),
                    "snapshot": np.tile(s.snapshots[targets], len(group)),
                    "source": np.tile(s.snapshots[src[targets]], len(group)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True).astype(object)
