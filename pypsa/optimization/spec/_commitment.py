# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep of the spec: unit commitment and the status carried into a build."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from pypsa.optimization.spec.sources import NetworkSources, source

if TYPE_CHECKING:
    from pypsa.components import Components


def solved_before(s: NetworkSources, c: Components, attr: str) -> xr.DataArray | None:
    """`attr` as solved over the snapshots before the build, or None in the first window."""
    start = int(s.positions[0])
    if not start or c.dynamic[attr].empty:
        return None
    raw = c.da[attr].isel(snapshot=slice(0, start))
    solved = s.unit_array(c, raw, snapshots=np.arange(start))
    return solved.drop_vars([k for k in solved.coords if k not in solved.dims])


def history(s: NetworkSources, c: Components) -> xr.DataArray:
    """Return the status of each unit at the `lag` snapshots before the build, newest first.

    Solved snapshots of earlier windows are read as solved, fractions included;
    the time before the network's first snapshot is rebuilt from
    `up_time_before` and `down_time_before`, the latter winning. A unit that is
    not committable is on.
    """
    minimum = [s.da(c, f"min_{k}_time") for k in ("up", "down")]
    length = int(max([1, *(m.max().item() for m in minimum if m.size)])) + 1
    solved = solved_before(s, c, "status")
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
    stacked = xr.concat(parts, dim="lag")
    return stacked.where(s.da(c, "committable").astype(bool), 1.0)


@source("COMPONENT_p_min_pu_nonneg")
def _p_min_pu_nonneg(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    pu = s.da(c, "p_min_pu")
    return (pu >= 0).all([d for d in pu.dims if d != c._unit_dim])


@source("COMPONENT_modules_installed")
def _modules_installed(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    nom = c._operational_attrs["nom"]
    mod = s.da(c, f"{nom}_mod")
    modular = (mod > 0) & ~s.da(c, f"{nom}_extendable").astype(bool)
    return xr.where(modular, s.da(c, nom) / mod.where(mod > 0, 1), 1.0)


@source("COMPONENT_min_up_time", "COMPONENT_min_down_time")
def _min_time(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """Return the minimum time, at most the horizon: a longer window reads nothing more."""
    return s.attribute(c, name).clip(max=len(s.snapshots))


@source("COMPONENT_up_time_brought_in", "COMPONENT_down_time_brought_in")
def _brought_in(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """How much of the start-ups (`up`) or shut-downs (`down`) before the build still bind.

    A switch at lag `l` binds a snapshot at position `p` while `l + p` is
    below the minimum up or down time.
    """
    kind = name.removeprefix(f"{c.name}_").partition("_")[0]
    status = history(s, c)
    switches = status - status.shift(lag=-1)
    switches = switches.isel(lag=slice(None, -1))
    past = (switches if kind == "up" else -switches).clip(min=0)
    minimum = s.da(c, f"min_{kind}_time")
    position = s.over_snapshots(np.arange(len(s.snapshots)))
    return (past * (past.lag + position < minimum)).sum("lag")


@source("COMPONENT_status_solved")
def _status_solved(s: NetworkSources, name: str, c: Components) -> bool:
    return solved_before(s, c, "status") is not None


@source("COMPONENT_status_initial")
def _status_initial(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    return history(s, c).sel(lag=1, drop=True)


@source("COMPONENT_big_m")
def _big_m(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    names = c.committables.intersection(c.extendables)
    if names.empty:
        return s.filled(name, np.nan)
    values = c.get_committable_big_m_values(
        names, committable_big_m=s.committable_big_m
    )
    return s.unit_array(c, values).reindex({c._unit_dim: s.names(c).to_numpy()})
