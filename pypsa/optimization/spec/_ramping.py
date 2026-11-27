# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep of the spec: ramping and the output carried into a build."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pypsa.optimization.spec._commitment import history, solved_before
from pypsa.optimization.spec.sources import NetworkSources, source

if TYPE_CHECKING:
    import xarray as xr

    from pypsa.components import Components


@source("COMPONENT_p_init")
def _p_init(s: NetworkSources, name: str, c: Components) -> xr.DataArray:
    """Return the output carried into the build.

    It is the solved output of an earlier window, else `p_init` where the
    unit came in running and zero where it came in off.
    """
    solved = solved_before(s, c, "p0" if c.name == "Link" else "p")
    if solved is None:
        status = history(s, c).sel(lag=1, drop=True)
        return s.da(c, "p_init").where(status > 0, 0.0)
    last = solved.isel(snapshot=-1, drop=True)
    return last.reindex({c._unit_dim: s.names(c).to_numpy()})
