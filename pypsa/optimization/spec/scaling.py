# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Numerical scaling of the built model.

It owns the factor of each column and row, which PyPSA derives from
`Bus.scaling`, `GlobalConstraint.scaling` and the `objective_scaling`
argument by the scaling rule of each variable and constraint. A spec entry
states its rule in `scaling:`; one without takes the rule its kind gives.
linopy transforms solutions, duals and the objective back to original units.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import xarray as xr

from pypsa.components.common import invariant
from pypsa.optimization.spec._passive_network import cycle_scaling
from pypsa.optimization.spec._render import formulated, renamer

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from linopy import Model
    from linopy.constraints import Constraint
    from linopy.variables import Variable

    from pypsa import Network
    from pypsa.optimization.spec._render import Scalings
    from pypsa.optimization.spec.sources import NetworkSources

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


#: The rows of a piecewise curve that hold only its weights and binaries.
WEIGHT_ROWS = {"active_bound", "binary_order", "delta_bound", "fill_order"}
#: The rows of a cost curve that hold the cost: its chords and its `cost` link, gated or not.
COST_ROWS = ("chord", "cost", "cost_ungated")


class Rules(NamedTuple):
    """The scaling rule of each variable and constraint that states one, by PyPSA name."""

    columns: dict[str, str]
    rows: dict[str, str]


def rules(scalings: Scalings, rename: Callable[[str], str]) -> Rules:
    """Return the scaling rules the spec states by the PyPSA name of what they scale.

    The rule of a piecewise curve goes to the rows of `COST_ROWS` it expands into.
    """
    found = Rules({}, {})
    for (section, key), rule in scalings.items():
        if section == "variables":
            found.columns[rename(key)] = rule
        elif section == "constraints":
            found.rows[rename(key)] = rule
        else:
            found.rows.update({rename(f"{key}_{kind}"): rule for kind in COST_ROWS})
    return found


def _column_rule(name: str, var: Variable, n: Network, rules: Rules) -> str:
    """Return the scaling rule of the variable `name`: its `scaling:`, else one by its kind.

    A continuous column over the units of a component on a bus takes
    `bus`; any other, as the weights and binaries of a piecewise curve,
    `none`.
    """
    if name in rules.columns:
        return rules.columns[name]
    component, _, attr = name.partition("-")
    attached = n.one_port_components | n.branch_components
    discrete = var.attrs.get("binary") or var.attrs.get("integer")
    on_bus = component in attached and "name" in var.labels.dims
    return "bus" if on_bus and "_piecewise_" not in attr and not discrete else "none"


def _scale_columns(m: Model, n: Network, rules: Rules, objective: float) -> None:
    """Give every column the factor its scaling rule takes from the network data.

    `bus` takes `Bus.scaling` of each unit's bus, `objective` the objective
    factor, and `none` keeps `1`.
    """
    for name, var in m.variables.items():
        match _column_rule(name, var, n, rules):
            case "bus":
                component = name.partition("-")[0]
                var.scaling = n.c[component]._scaling(var.labels.indexes["name"])
            case "objective":
                var.scaling = objective


def _rule(name: str, con: Constraint, n: Network, rules: Rules) -> str:
    """Return the scaling rule of the constraint `name`: its `scaling:` or its curve's, else one by its kind.

    A row of a piecewise curve that holds only its weights and binaries
    takes `none`. A row over the units of a component on a bus takes
    `bus`, any other `none`.
    """
    if name in rules.rows:
        return rules.rows[name]
    _, piecewise, kind = name.rpartition("_piecewise_")
    if piecewise and kind in WEIGHT_ROWS:
        return "none"
    component = name.partition("-")[0]
    attached = n.one_port_components | n.branch_components | {"Bus"}
    return "bus" if component in attached and "name" in con.labels.dims else "none"


def _carrier_factors(
    n: Network, sources: NetworkSources, carriers: pd.Index
) -> xr.DataArray:
    """Return per carrier the geometric mean of the smallest and largest factor of its expandable assets."""
    low = np.full(len(carriers), np.nan)
    high = np.full(len(carriers), np.nan)
    for c in formulated(n, "capacity"):
        name = renamer(c)
        built = (
            sources[name("COMPONENT_NOM_extendable")]
            & sources[name("COMPONENT_active")]
            & (sources[name("COMPONENT_first_active")] != 0).any("period")
        )
        others = [d for d in built.dims if d != c._unit_dim]
        assets = built.indexes[c._unit_dim][built.any(others).values]
        carrier = sources[name("COMPONENT_carrier")].set_index(c._unit_dim)["carrier"]
        factors = np.broadcast_to(np.asarray(c._scaling(assets)), len(assets))
        at = carriers.get_indexer(carrier.reindex(assets).to_numpy())
        found = at != -1
        np.fmin.at(low, at[found], factors[found])
        np.fmax.at(high, at[found], factors[found])
    return xr.DataArray(np.sqrt(low * high), coords={"Carrier": carriers}).fillna(1.0)


def _cycle_factors(sources: NetworkSources, con: Constraint) -> xr.DataArray:
    """Return per row of Kirchhoff's voltage law the largest factor of the branches in its cycle."""
    grid = cycle_scaling(sources).reindex(cycle=con.labels.indexes["cycle"])
    period = sources.periods.get_indexer(sources.period_of)
    rows = xr.DataArray(grid.values[period], dims=("snapshot", "cycle"))
    rows = rows.broadcast_like(con.labels).transpose(*con.labels.dims)
    return con.labels.copy(data=rows.values).fillna(1.0)


def _scale_rows(
    m: Model,
    n: Network,
    sources: NetworkSources,
    rules: Rules,
    objective: float,
) -> None:
    """Give every constraint the factor its scaling rule takes from the network data.

    `bus` takes `Bus.scaling` of each unit's bus, `objective` the objective
    factor, `global` `GlobalConstraint.scaling`, `cycle` per row the largest
    factor of its cycle's branches, `carrier` per carrier the geometric mean
    of the smallest and largest factor of its expandable assets, and
    `none` keeps `1`.
    """
    for name, con in m.constraints.items():
        match _rule(name, con, n, rules):
            case "bus":
                component = name.partition("-")[0]
                con.scaling = n.c[component]._scaling(con.labels.indexes["name"])
            case "objective":
                con.scaling = objective
            case "global":
                con.scaling = global_constraint_factors(n, con.labels.indexes["name"])
            case "cycle":
                con.scaling = _cycle_factors(sources, con)
            case "carrier":
                con.scaling = _carrier_factors(
                    n, sources, con.labels.indexes["Carrier"]
                )


def _unscaled(n: Network, objective: float) -> bool:
    """Return whether the objective factor, `Bus.scaling` and `GlobalConstraint.scaling` are all `1`."""
    buses = invariant(n.c.buses.static.scaling)
    limits = invariant(n.c.global_constraints.static.scaling)
    return objective == 1 and bool((buses == 1).all() and (limits == 1).all())


def scale(
    m: Model,
    n: Network,
    sources: NetworkSources,
    rules: Rules,
    objective: float,
) -> None:
    """Apply PyPSA's numerical scaling to the built model.

    A network whose factors are all `1` keeps linopy's default factors.
    """
    if _unscaled(n, objective):
        return
    m.objective.scaling = objective
    if KVL in m.constraints:
        check_sub_networks(n)
    _scale_columns(m, n, rules, objective)
    _scale_rows(m, n, sources, rules, objective)
