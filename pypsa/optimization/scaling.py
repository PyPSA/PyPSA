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

from pypsa.components.common import invariant

if TYPE_CHECKING:
    from pypsa import Network

KVL = "Kirchhoff-Voltage-Law"


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
    scaling = invariant(n.c.buses.static.scaling)
    sub_network = invariant(n.c.buses.static.sub_network)
    mixed = scaling.groupby(sub_network).nunique().loc[lambda s: s > 1]
    if not mixed.empty:
        msg = (
            "Buses of one sub-network must share the same `Bus.scaling`, since "
            f"Kirchhoff voltage law rows mix their flows. Mixed: {list(mixed.index)}"
        )
        raise ValueError(msg)
