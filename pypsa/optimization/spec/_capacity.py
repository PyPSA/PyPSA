# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep of the spec: capacity, its activity and its cost."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pypsa.optimization.spec.sources import NetworkSources, optional, relation, source

if TYPE_CHECKING:
    import xarray as xr

    from pypsa.components import Components

source("COMPONENT_carrier")(relation)
source("COMPONENT_NOM_set")(optional)


@source("COMPONENT_activity_mask")
def _activity_mask(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    return s.active(c)


@source("COMPONENT_active")
def _active(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    return s.column(c, "active", bool)


@source("COMPONENT_capital_weight")
def _capital_weight(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    by_period = s.active_by_period(c)
    if not s.multi_invest:
        return by_period.any("period").astype(float)
    return (by_period * s["period_weighting_objective"]).sum("period")


@source("COMPONENT_first_active")
def _first_active(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    by_period = s.active_by_period(c).astype(int)
    first = (by_period == 1) & (by_period.cumsum("period") == 1)
    return first.astype(float)


@source("COMPONENT_capital_cost")
def _capital_cost(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    return s.unit_array(c, c.periodized_cost)


@source("COMPONENT_NOM_mod")
def _nom_mod(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    da = s.attribute(c, name)
    return da.where(da > 0)


@source("COMPONENT_NOM_min", "COMPONENT_NOM_max")
def _nom_bound(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """Return the tightest bound of the scenarios on the build they share.

    The cap is absent where no scenario extends the unit.
    """
    bound = s.da(c, s.attr(c, name))
    if name.endswith("_min"):
        return bound.max("scenario")
    extendable = s.da(c, c._operational_attrs["nom_extendable"]).astype(bool)
    return bound.min("scenario").where(extendable.any("scenario"))
