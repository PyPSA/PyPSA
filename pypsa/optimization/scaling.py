# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Numerical scaling of optimisation models via linopy.

Factors are derived from `Bus.scaling` (see `Components.scaling`),
`GlobalConstraint.scaling` and the `objective_scaling` argument and passed to
linopy where variables and constraints are created. linopy transforms
solutions, duals and the objective back to original units.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pypsa.components.common import invariant

if TYPE_CHECKING:
    import pandas as pd
    from xarray import DataArray

    from pypsa import Network
    from pypsa.components.components import Components

KVL = "Kirchhoff-Voltage-Law"

KVL_SCALING = 1e5


def factor(scaling: Components | float, names: pd.Index) -> DataArray | float:
    """Return scaling factors for a model element defined over `names`.

    Parameters
    ----------
    scaling : pypsa.Components | float
        Source of the factors: a component (uses `Components.scaling`) or a
        scalar.
    names : pd.Index
        Names the element is defined over.

    Returns
    -------
    xr.DataArray | float
        Factors over `name`, or `1.0` if all factors are `1` so that unscaled
        models stay unchanged.

    """
    if isinstance(scaling, int | float):
        return float(scaling)
    da = scaling.scaling.sel(name=names)
    if (da == 1).all():
        return 1.0
    return da


def global_constraint_factor(n: Network, name: str) -> float:
    """Return the row factor of the global constraint `name`.

    Parameters
    ----------
    n : pypsa.Network
        Network with `GlobalConstraint.scaling` defined.
    name : str
        Name of the global constraint.

    Returns
    -------
    float
        `GlobalConstraint.scaling` of `name`.

    Raises
    ------
    ValueError
        If the factor is not finite and positive, or differs across scenarios.

    """
    value = float(invariant(n.c.global_constraints.static.scaling)[name])
    if not np.isfinite(value) or value <= 0:
        msg = f"GlobalConstraint.scaling must be finite and positive, got {value} for '{name}'."
        raise ValueError(msg)
    return value


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
    scaling = n.c.buses.scaling.to_series()
    sub_network = invariant(n.c.buses.static.sub_network)
    mixed = scaling.groupby(sub_network).nunique().loc[lambda s: s > 1]
    if not mixed.empty:
        msg = (
            "Buses of one sub-network must share the same `Bus.scaling`, since "
            f"Kirchhoff voltage law rows mix their flows. Mixed: {list(mixed.index)}"
        )
        raise ValueError(msg)
