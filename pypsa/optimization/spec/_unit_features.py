# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep of `pypsa.yaml`: unit features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.optimization.spec.sources import (
    UNITS,
    NetworkSources,
    _snapshot_weighting,
    single,
)

MAINTAINABLE = ("Generator", "Link", "Process")


def _maintainable(s: NetworkSources, component: str) -> xr.DataArray:
    """Whether a unit is maintainable in any scenario."""
    return s.da(component, "maintainable").astype(bool).any("scenario")


def _maintenance_windows(
    s: NetworkSources, component: str
) -> tuple[xr.DataArray, xr.DataArray]:
    """Find the last snapshot each event window covers and whether it may start.

    Both are (unit, scenario, snapshot). A window runs from its start until the
    generator weightings reach `maintenance_duration`.
    """
    weighting = _snapshot_weighting(s, "snapshot_weightings_generators").to_numpy()
    cum = weighting.cumsum()
    before = xr.DataArray(cum - weighting, coords={"snapshot": s.snapshots})
    duration = s.da(component, "maintenance_duration")
    target = duration * (1 - 1e-6) + before
    dims = (UNITS[component], "scenario", "snapshot")
    end = xr.apply_ufunc(lambda v: np.searchsorted(cum, v), target).transpose(*dims)
    active = s.active(component).expand_dims(scenario=s.scenarios).transpose(*dims)
    inactive = (~active).cumsum("snapshot")
    last = len(s.snapshots) - 1
    stop = end.clip(max=last)
    end_inactive = inactive.copy(
        data=np.take_along_axis(inactive.to_numpy(), stop.to_numpy(), axis=-1)
    )
    window_inactive = end_inactive - inactive + ~active
    valid = active & (end <= last) & (window_inactive == 0)
    return stop, valid


def _maintenance_cover(s: NetworkSources, name: str) -> pd.DataFrame:
    component = name.partition("_")[0]
    dim = UNITS[component]
    maintainable = _maintainable(s, component)
    end, _ = _maintenance_windows(s, component)
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
    return pd.DataFrame(
        rows, columns=["start", "covered", dim, "scenario"], dtype=object
    )


def _maintenance_start_blocked(s: NetworkSources, name: str) -> xr.DataArray:
    component = name.partition("_")[0]
    _, valid = _maintenance_windows(s, component)
    blocked = s.active(component) & ~valid & _maintainable(s, component)
    return s.frame(name, blocked)


def _maintainable_source(s: NetworkSources, name: str) -> xr.DataArray:
    return s.frame(name, _maintainable(s, name.partition("_")[0]))


def _maintenance_duration(s: NetworkSources, name: str) -> xr.DataArray:
    component = name.partition("_")[0]
    duration = s.da(component, "maintenance_duration").astype(float)
    return s.frame(name, duration.where(_maintainable(s, component)))


def _nom_max(s: NetworkSources, name: str) -> xr.DataArray:
    component = name.partition("_")[0]
    cap = s.attribute(component, "p_nom_max", name)
    extendable = s.da(component, "p_nom_extendable").astype(bool).any("scenario")
    return cap.where(extendable)


for _component in MAINTAINABLE:
    single(f"{_component}_maintenance_cover")(_maintenance_cover)
    single(f"{_component}_maintenance_start_blocked")(_maintenance_start_blocked)
    single(f"{_component}_maintainable")(_maintainable_source)
    single(f"{_component}_maintenance_duration")(_maintenance_duration)
    single(f"{_component}_p_nom_max")(_nom_max)
