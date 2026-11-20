# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Numerical scaling of optimisation models via linopy.

Factors are derived from `Bus.scaling`, `GlobalConstraint.scaling` and the
`objective_scaling` argument and passed to linopy where variables and
constraints are created. linopy transforms solutions, duals and the objective
back to original units.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from pypsa.components.common import invariant

if TYPE_CHECKING:
    import pandas as pd

    from pypsa import Network

KVL = "Kirchhoff-Voltage-Law"


def global_constraint_factors(n: Network, names: pd.Index) -> xr.DataArray | float:
    """Return the row factors of the global constraints `names`.

    Parameters
    ----------
    n : pypsa.Network
        Network with `GlobalConstraint.scaling` defined.
    names : pd.Index
        Names of the global constraints.

    Returns
    -------
    xarray.DataArray | float
        `GlobalConstraint.scaling` over `name`, or `1.0` if all factors are `1`
        so that unscaled models stay unchanged.

    Raises
    ------
    ValueError
        If a factor is not finite and positive, or differs across scenarios.

    """
    factors = invariant(n.c.global_constraints.static.scaling).reindex(names)
    invalid = ~np.isfinite(factors) | (factors <= 0)
    if invalid.any():
        msg = f"GlobalConstraint.scaling must be finite and positive, got:\n{factors[invalid]}"
        raise ValueError(msg)
    if (factors == 1).all():
        return 1.0
    return xr.DataArray(factors.rename("scaling").rename_axis("name"))


def check_sub_networks(n: Network) -> None:
    """Raise if buses of one sub-network carry different `Bus.scaling` factors.

    Parameters
    ----------
    n : pypsa.Network
        Network with determined topology.

    Raises
    ------
    ValueError
        If buses of one sub-network carry different factors.

    """
    scaling = invariant(n.c.buses.static.scaling)
    sub_network = invariant(n.c.buses.static.sub_network)
    mixed = scaling.groupby(sub_network).nunique().loc[lambda s: s > 1]
    if not mixed.empty:
        msg = (
            "Buses of one sub-network must share the same `Bus.scaling`, since "
            f"Kirchhoff voltage law rows mix their flows. Mixed: {list(mixed.index)}"
        )
        raise ValueError(msg)
