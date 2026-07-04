# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Helpers for optimisation-only scaling factors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network


def validate_scaling(value: Any, label: str = "scaling") -> Any:
    """Validate that scaling values are finite and strictly positive."""
    values = np.asarray(value, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        msg = f"{label} must contain only finite positive values."
        raise ValueError(msg)
    return value


def bus_scaling_by_name(n: Network) -> pd.Series:
    """Return bus scaling indexed by bus name."""
    scaling = n.c.buses.static["scaling"].astype(float)
    if isinstance(scaling.index, pd.MultiIndex):
        scaling = scaling.groupby(level="name").first()
    validate_scaling(scaling, "Bus scaling")
    return scaling


def bus_scaling(n: Network, buses: xr.DataArray) -> xr.DataArray:
    """Map bus names to optimisation scaling factors."""
    lookup = bus_scaling_by_name(n)
    flat = pd.Series(np.asarray(buses.values).ravel()).map(lookup).fillna(1.0)
    values = flat.to_numpy(dtype=float).reshape(buses.shape)
    return xr.DataArray(values, coords=buses.coords, dims=buses.dims)


def component_owner_buses(
    n: Network, component: str, names: Sequence | pd.Index | None = None
) -> xr.DataArray:
    """Return owner bus names for a component, using ``bus0`` for multi-ports."""
    c = n.c[component]
    bus_attr = "bus0" if "bus0" in c.static.columns else "bus"
    buses = c._as_xarray(bus_attr)
    if names is not None:
        names = pd.Index(names)
        if isinstance(names, pd.MultiIndex):
            names = names.get_level_values("name").unique()
        buses = buses.sel(name=names)
    return buses


def component_variable_scaling(
    n: Network, component: str, names: Sequence | pd.Index | None = None
) -> xr.DataArray:
    """Return Linopy variable scaling for variables owned by component buses."""
    return bus_scaling(n, component_owner_buses(n, component, names))


def component_nominal_variable_scaling(
    n: Network, component: str, names: Sequence | pd.Index | None = None
) -> xr.DataArray:
    """Return name-only scaling for investment variables."""
    scaling = component_variable_scaling(n, component, names)
    extra_dims = [dim for dim in scaling.dims if dim != "name"]
    if extra_dims:
        scaling = scaling.isel(dict.fromkeys(extra_dims, 0), drop=True)
    return scaling


def component_constraint_scaling(
    n: Network,
    component: str,
    names: Sequence | pd.Index | None = None,
    dim: str = "name",
) -> xr.DataArray:
    """Return Linopy row scaling for physical component constraints."""
    if dim != "name" and names is not None:
        names = pd.Index(names)
        if isinstance(names, pd.MultiIndex):
            lookup = component_constraint_scaling(
                n, component, names.get_level_values("name").unique()
            )
            if "scenario" in lookup.dims and "scenario" in names.names:
                index = pd.MultiIndex.from_arrays(
                    [
                        names.get_level_values("scenario"),
                        names.get_level_values("name"),
                    ],
                    names=["scenario", "name"],
                )
                values = lookup.to_series().reindex(index).to_numpy(dtype=float)
            else:
                values = (
                    names.get_level_values("name")
                    .map(lookup.to_series())
                    .to_numpy(dtype=float)
                )
            return xr.DataArray(values, coords={dim: names}, dims=[dim])

    scaling = 1 / component_variable_scaling(n, component, names)
    if dim != "name" and "name" in scaling.dims:
        scaling = scaling.rename(name=dim)
        if names is not None:
            scaling = scaling.assign_coords({dim: pd.Index(names)})
    return scaling


def bus_constraint_scaling(n: Network, buses: xr.DataArray) -> xr.DataArray:
    """Return Linopy row scaling for bus balance constraints."""
    return 1 / bus_scaling(n, buses)


def global_constraint_scaling(glc: Any) -> Any:
    """Return direct Linopy row scaling for a global constraint row/group."""
    scaling = getattr(glc, "scaling", 1.0)
    return validate_scaling(scaling, "GlobalConstraint scaling")
