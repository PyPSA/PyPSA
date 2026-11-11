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
    _static,
    dimension,
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


def _cost_curve(s: NetworkSources) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the output and running cost at each breakpoint, (breakpoint, generator)."""
    c = s.n.components.generators
    curves = c.piecewise.get("marginal_cost")
    if curves is None or curves.empty:
        empty = pd.DataFrame(index=pd.RangeIndex(0, name="breakpoint"))
        return empty, empty
    curves = curves.dropna(how="all", axis=1)
    names = curves.columns.unique("name").intersection(s.coords("generator"))
    curves = curves.loc[:, names]
    p_pu = curves.xs("p_pu", level="attribute", axis=1)
    slope = curves.xs("marginal_cost", level="attribute", axis=1)
    static = _static(s, "Generator").reindex(p_pu.columns)
    nominal = static["p_nom"]
    bad = static["p_nom_extendable"] | ~np.isfinite(nominal) | (nominal <= 0)
    if bad.any():
        msg = (
            "Piecewise 'marginal_cost' curves need a fixed, finite and positive "
            f"p_nom; violated by {bad.index[bad].tolist()}."
        )
        raise ValueError(msg)
    p = (p_pu * nominal).where(p_pu.notna() & slope.notna())
    cost = (p.diff() * slope).fillna(0).cumsum().where(p.notna())
    return p, cost


@dimension("breakpoint")
def _breakpoint(s: NetworkSources) -> pd.Index:
    return pd.Index(_cost_curve(s)[0].index, dtype=int)


@single("Generator_marginal_cost_curve_p", "Generator_marginal_cost_curve_cost")
def _marginal_cost_curve(s: NetworkSources, name: str) -> xr.DataArray:
    p, cost = _cost_curve(s)
    frame = (p if name.endswith("_p") else cost).reindex(
        index=s.coords("breakpoint"), columns=s.coords("generator")
    )
    da = xr.DataArray(
        frame.to_numpy(dtype=float),
        coords={
            "breakpoint": frame.index.to_numpy(),
            "generator": frame.columns.to_numpy(dtype=object),
        },
        dims=("breakpoint", "generator"),
    )
    return s.frame(name, da)


for _component in MAINTAINABLE:
    single(f"{_component}_maintenance_cover")(_maintenance_cover)
    single(f"{_component}_maintenance_start_blocked")(_maintenance_start_blocked)
    single(f"{_component}_maintainable")(_maintainable_source)
    single(f"{_component}_maintenance_duration")(_maintenance_duration)
    single(f"{_component}_p_nom_max")(_nom_max)
