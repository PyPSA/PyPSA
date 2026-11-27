# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep of the spec: storage levels, their carry-over and the limits of stores."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from pypsa.optimization.spec.sources import NetworkSources, finite, source

if TYPE_CHECKING:
    from pypsa.components import Components

source("COMPONENT_cycles_max")(finite)


@source("COMPONENT_LEVEL_initial_per_period", "COMPONENT_CYCLIC_per_period")
def _per_period(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """Return the flag where the build has investment periods, false otherwise."""
    return s.attribute(c, name) & s.multi_invest


@source("COMPONENT_inactive_snapshots")
def _inactive_snapshots(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    return (~s.active(c)).sum("snapshot").astype(int)


@source("COMPONENT_opens_late")
def _opens_late(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    active = s.active(c)
    first = (active.cumsum("snapshot") == 1) & active
    return first & (s.over_snapshots(np.arange(len(s.snapshots))) > 0)


@source("COMPONENT_retention")
def _retention(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    return (1 - s.da(c, "standing_loss")) ** s["snapshot_weighting_stores"]


@source("Store_split_dispatch")
def _store_split_dispatch(s: NetworkSources, name: str) -> xr.DataArray:
    c = s.n.components.stores
    return s.unit_array(c, xr.DataArray(c.split_dispatch())).astype(bool)
