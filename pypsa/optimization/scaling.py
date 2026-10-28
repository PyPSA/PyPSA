# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Numerical scaling of optimisation models via linopy.

Factors are derived from `Bus.scaling`, `GlobalConstraint.scaling` and the
`objective_scaling` argument and handed to linopy, which transforms solutions,
duals and the objective back to original units.
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import TYPE_CHECKING

import numpy as np
from linopy.constraints import CSRConstraint
from xarray import DataArray

from pypsa.components.common import invariant

if TYPE_CHECKING:
    import pandas as pd
    from linopy import Variable
    from linopy.constraints import ConstraintBase

    from pypsa import Components, Network

KVL = "Kirchhoff-Voltage-Law"

KVL_SCALING = 1e5

OBJECTIVE_CONSTANT = "objective_constant"

SCALED_VARIABLES = (
    "p",
    "p_nom",
    "s",
    "s_nom",
    "e",
    "e_nom",
    "p_store",
    "p_dispatch",
    "state_of_charge",
    "spill",
    "loss",
    "maintenance_capacity",
    "*_piecewise",
)

SCALED_ROWS = (
    "fix-*",
    "ext-*",
    "com-ext-p-*",
    "com-p-*",
    "com-mod-p-*",
    "com-partly-*",
    "maintcap_*",
    "*-ramp_limit_*",
    "*_set",
    "*_modularity",
    "energy_balance",
    "e_sum_min",
    "e_sum_max",
    "loss_upper",
    "loss_tangents-*",
    "loss_secants-*",
    "v_ang-*",
    "*_piecewise*_link",
    "*_piecewise*_chord",
    "*_piecewise*_domain_lo",
    "*_piecewise*_domain_hi",
)

UNSCALED = (
    "*-status",
    "*-start_up",
    "*-shut_down",
    "*-maintenance",
    "*-maintenance_start",
    "*-maintenance_status",
    "*-n_mod",
    "*_piecewise*_lambda",
    "*_piecewise*_delta",
    "*_piecewise*_order_binary",
    "*_piecewise*_segment_binary",
    "*_piecewise*_convex",
    "*_piecewise*_select",
    "*_piecewise*_fill_order",
    "*_piecewise*_delta_bound",
    "*_piecewise*_binary_order",
    "*_piecewise*_active_bound",
    "*-com-transition-*",
    "*-com-up-time",
    "*-com-down-time",
    "*-maint-*",
    "*-status-*-upper",
    "*-start_up-*-upper",
    "*-shut_down-*-upper",
    "Transformer-phase_shift",
    "Carrier-growth_limit",
    "CVaR",
    "CVaR-*",
)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch(name, p) for p in patterns)


def _validated(scaling: pd.Series, label: str) -> pd.Series:
    invalid = ~np.isfinite(scaling) | (scaling <= 0)
    if invalid.any():
        msg = f"{label} must be finite and positive, got:\n{scaling[invalid]}"
        raise ValueError(msg)
    return scaling


def bus_scaling(n: Network) -> pd.Series:
    """Return the column scaling factor of each bus.

    Parameters
    ----------
    n : pypsa.Network
        Network with `Bus.scaling` defined.

    Returns
    -------
    pd.Series
        Factor per bus name.

    Raises
    ------
    ValueError
        If a factor is non-finite or not positive, or differs across scenarios.

    """
    return _validated(invariant(n.c.buses.static.scaling), "Bus.scaling")


def component_scaling(n: Network, c: Components) -> DataArray:
    """Return the column scaling factor of each asset of a component.

    One-port components take the factor of `bus`, branch components the
    factor of `bus0`.

    Parameters
    ----------
    n : pypsa.Network
        Network with `Bus.scaling` defined.
    c : pypsa.Components
        Component, e.g. `n.c.generators`.

    Returns
    -------
    xr.DataArray
        Factor over dimension `name`.

    """
    col = "bus" if c.name in n.one_port_components else "bus0"
    bus = invariant(c.static[col])
    values = bus_scaling(n).reindex(bus.values).values
    return DataArray(values, coords={"name": bus.index.values}, dims="name")


def _set_scaling(obj: Variable | ConstraintBase, factor: DataArray | float) -> None:
    if isinstance(factor, DataArray):
        dim = "name" if "name" in obj.indexes else f"{obj.name}_i"
        factor = factor.rename(name=dim).sel({dim: obj.indexes[dim]})
    if isinstance(obj, CSRConstraint):
        con = obj.mutable()
        con.scaling = factor
        obj.model.constraints.add(con.freeze())
    else:
        obj.scaling = factor


def _check_sub_networks(n: Network, scaling: pd.Series) -> None:
    sub_network = invariant(n.c.buses.static.sub_network)
    mixed = scaling.groupby(sub_network).nunique().loc[lambda s: s > 1]
    if not mixed.empty:
        msg = (
            "Buses of one sub-network must share the same `Bus.scaling`, since "
            f"Kirchhoff voltage law rows mix their flows. Mixed: {list(mixed.index)}"
        )
        raise ValueError(msg)


def apply_scaling(n: Network, objective_scaling: float = 1.0) -> None:
    """Pass numerical scaling factors of the model at `n.model` to linopy.

    Columns of a component take the `Bus.scaling` of its bus (`bus0` for
    branches), rows of a component the same factor, nodal balance rows the
    factor of their bus and global constraint rows `GlobalConstraint.scaling`.
    Kirchhoff voltage law rows keep the factor `1e5` times the factor of their
    sub-network set on creation.
    The fixed `objective_constant` column takes `objective_scaling`, so its
    bound is expressed in objective units.
    Binary, integer, angle and auxiliary columns are not scaled.

    Parameters
    ----------
    n : pypsa.Network
        Network with a built model at `n.model`.
    objective_scaling : float, default 1.0
        Factor applied to all objective coefficients in the solver.

    Raises
    ------
    ValueError
        If the model contains a variable or constraint without a scaling rule,
        if a factor is invalid, or if buses of one sub-network carry different
        factors.

    """
    m = n.model
    components = {c.name for c in n.components}
    rows = set(m.constraints)
    special = {
        r
        for r in rows
        if r == KVL
        or fnmatch(r, "Bus*-nodal_balance")
        or r.startswith(("GlobalConstraint-", "Bus-nom_"))
    }
    regular = {
        name: name.partition("-")
        for name in [*m.variables, *m.constraints]
        if name not in {*special, OBJECTIVE_CONSTANT} and not _matches(name, UNSCALED)
    }
    unknown = [
        name
        for name, (c, _, suffix) in regular.items()
        if c not in components
        or not _matches(suffix, SCALED_ROWS if name in rows else SCALED_VARIABLES)
    ]
    if unknown:
        msg = f"No scaling rule for model elements {unknown}. Register them in `pypsa.optimization.scaling`."
        raise ValueError(msg)

    m.objective.scaling = objective_scaling
    if OBJECTIVE_CONSTANT in m.variables:
        _set_scaling(m.variables[OBJECTIVE_CONSTANT], objective_scaling)
    buses = bus_scaling(n)
    glcs = _validated(
        invariant(n.c.global_constraints.static.scaling), "GlobalConstraint.scaling"
    )
    if (buses == 1).all() and (glcs == 1).all():
        return

    if KVL in m.constraints:
        _check_sub_networks(n, buses)

    factors = {c: component_scaling(n, n.c[c]) for c, _, _ in regular.values()}
    for name, (c, _, _) in regular.items():
        obj = m.constraints[name] if name in rows else m.variables[name]
        _set_scaling(obj, factors[c])

    bus_factor = DataArray(
        buses.values, coords={"name": buses.index.values}, dims="name"
    )
    for name in special:
        con = m.constraints[name]
        if name.startswith("GlobalConstraint-"):
            _set_scaling(con, glcs[name.removeprefix("GlobalConstraint-")])
        elif name != KVL:
            _set_scaling(con, bus_factor)
