# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep of the spec: unit features."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.optimization.spec.sources import NetworkSources, source

if TYPE_CHECKING:
    from pypsa.components import Components


@source("Unit_maintainable")
def _maintainable(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """Whether a unit is maintainable in any scenario."""
    return s.da(c, "maintainable").astype(bool).any("scenario")


def _maintenance_windows(
    s: NetworkSources, c: Components
) -> tuple[xr.DataArray, xr.DataArray]:
    """Find the last snapshot each event window covers and whether it may start.

    Both are (unit, scenario, snapshot). A window runs from its start until the
    generator weightings reach `maintenance_duration`.
    """
    weighting = s["snapshot_weightings_generators"].to_numpy()
    cum = weighting.cumsum()
    before = xr.DataArray(cum - weighting, coords={"snapshot": s.snapshots})
    target = s.da(c, "maintenance_duration") * (1 - 1e-6) + before
    dims = (c._unit_dim, "scenario", "snapshot")
    end = xr.apply_ufunc(lambda v: np.searchsorted(cum, v), target).transpose(*dims)
    active = s.active(c).expand_dims(scenario=s.scenarios).transpose(*dims)
    inactive = (~active).cumsum("snapshot")
    last = len(s.snapshots) - 1
    stop = end.clip(max=last)
    end_inactive = inactive.copy(
        data=np.take_along_axis(inactive.to_numpy(), stop.to_numpy(), axis=-1)
    )
    window_inactive = end_inactive - inactive + ~active
    valid = active & (end <= last) & (window_inactive == 0)
    return stop, valid


@source("Unit_maintenance_cover")
def _maintenance_cover(s: NetworkSources, name: str, c: Components) -> pd.DataFrame:
    dim = c._unit_dim
    columns = ["start", "covered", dim, "scenario"]
    maintainable = s[f"{c.name}_maintainable"]
    if not maintainable.any():
        return pd.DataFrame(columns=columns, dtype=object)
    end, _ = _maintenance_windows(s, c)
    end = end.sel({dim: maintainable[dim][maintainable.to_numpy()]})
    snapshots = s.snapshots.to_numpy()
    rows = [
        (snapshots[start], snapshots[covered], unit, scenario)
        for unit, by_scenario in zip(end[dim].to_numpy(), end.to_numpy(), strict=True)
        for scenario, by_start in zip(
            end["scenario"].to_numpy(), by_scenario, strict=True
        )
        for start, stop in enumerate(by_start)
        for covered in range(start, int(stop) + 1)
    ]
    return pd.DataFrame(rows, columns=columns, dtype=object)


@source("Unit_maintenance_start_blocked")
def _maintenance_start_blocked(
    s: NetworkSources, name: str, c: Components
) -> xr.DataArray:
    _, valid = _maintenance_windows(s, c)
    return s.active(c) & ~valid & s[f"{c.name}_maintainable"]


@source("Unit_maintenance_duration")
def _maintenance_duration(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    duration = s.da(c, "maintenance_duration").astype(float)
    return duration.where(s[f"{c.name}_maintainable"])


@source("Unit_NOM_max")
def _nom_max(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """Return the build cap, absent where no scenario extends the unit."""
    extendable = s.da(c, c._operational_attrs["nom_extendable"]).astype(bool)
    return s.attribute(c, name).where(extendable.any("scenario"))
