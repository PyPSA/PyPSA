# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Define global constraints for optimisation problems with Linopy."""

from __future__ import annotations

import logging
import re
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from linopy import LinearExpression, merge
from xarray import DataArray

from pypsa.components.common import invariant
from pypsa.descriptors import nominal_attrs
from pypsa.optimization.piecewise import define_piecewise
from pypsa.optimization.scaling import global_constraint_factor
from pypsa.optimization.window import apply_period_weighting

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pypsa import Components, Network
    from pypsa.optimization.piecewise import PiecewiseOptions
logger = logging.getLogger(__name__)


STORAGE_ATTRS = {
    "StorageUnit": (
        "state_of_charge",
        "state_of_charge_initial",
        "cyclic_state_of_charge",
        "cyclic_state_of_charge_per_period",
        "state_of_charge_initial_per_period",
    ),
    "Store": (
        "e",
        "e_initial",
        "e_cyclic",
        "e_cyclic_per_period",
        "e_initial_per_period",
    ),
}


def _component_withdrawal(
    n: Network,
    c: Components,
    factor: pd.Series,
    sns: pd.Index,
    counted: np.ndarray,
    *,
    scenario: Any,
) -> tuple[list[LinearExpression], float]:
    """Net energy withdrawn from the storages of one component.

    Each storage is weighted by its `factor`. See `_storage_withdrawal`.

    In each counted investment period, a storage ends at its last active level and
    starts from the level that the energy balance hands over to the period. A start
    position of -1 stands for the initial level.
    """
    m = n.model
    level_attr, initial_attr, cyclic_attr, cyclic_pp_attr, initial_pp_attr = (
        STORAGE_ATTRS[c.name]
    )
    static = c.static.loc[scenario].loc[factor.index]
    window = n.optimize._window.subset(sns)
    n_sns = len(sns)

    def select(obj: Any, names: pd.Index = factor.index) -> Any:
        obj = obj.sel(name=names, snapshot=sns)
        return obj.sel(scenario=scenario, drop=True) if n.has_scenarios else obj

    active = select(c.da.active).transpose("snapshot", "name")
    is_active = active.to_numpy()
    weight = factor.to_numpy()

    period_of = np.zeros(n_sns)
    years = np.ones(n_sns)
    cyclic = static[cyclic_attr].to_numpy(bool)
    cyclic_pp = initial_pp = np.zeros(len(static), bool)
    if n._multi_invest:
        period_of = window.period_of.to_numpy()
        years = n.investment_period_weightings.years.loc[period_of].to_numpy()
        cyclic_pp = static[cyclic_pp_attr].to_numpy(bool)
        initial_pp = static[initial_pp_attr].to_numpy(bool)

    positions = np.flatnonzero(counted)
    breaks = np.flatnonzero(np.diff(period_of[positions])) + 1
    firsts = positions[np.r_[0, breaks]]
    lasts = positions[np.r_[breaks - 1, -1]]

    last_active = np.maximum.accumulate(
        np.where(is_active, np.arange(n_sns)[:, None], -1), axis=0
    )
    end = last_active[lasts]
    start = np.where(firsts[:, None] > 0, last_active[firsts - 1], -1)
    start = np.where((start < 0) & cyclic, last_active[-1], start)
    start = np.where(initial_pp, -1, start)
    from_initial = start < 0
    counts = (end >= firsts[:, None]) & ~cyclic_pp
    carried = ~(cyclic | cyclic_pp | initial_pp)
    if (carried & counts.any(axis=0)).any() and (years[counted] != 1).any():
        msg = (
            f"Found non-cyclic {c.list_name} of a limited carrier which carry their "
            "energy level over investment periods with year weightings != 1. The "
            "global constraint would be inconsistent. Set "
            f"`{initial_pp_attr}` to True, use equal period weightings or "
            "cyclic storage instead."
        )
        raise NotImplementedError(msg)

    level_weight = np.where(counts, years[lasts, None] * weight, 0)

    column = np.broadcast_to(np.arange(len(static)), end.shape)
    coeff = np.zeros(is_active.shape)
    np.add.at(coeff, (end, column), -level_weight)
    np.add.at(coeff, (start, column), np.where(from_initial, 0, level_weight))
    constant = (level_weight * from_initial * static[initial_attr].to_numpy()).sum()

    lhs = []
    rows = np.flatnonzero(coeff.any(axis=1))
    if rows.size:
        coeff_da = active.isel(snapshot=rows).copy(data=coeff[rows])
        level = select(m[f"{c.name}-{level_attr}"]).isel(snapshot=rows)
        lhs.append((level.where(coeff_da != 0) * coeff_da).sum())

    hours = window.snapshot_weightings("stores").to_numpy() * years
    flow_weight = active.copy(data=is_active * hours[:, None] * weight)
    flow_weight = flow_weight.isel(snapshot=positions)
    inflow = select(c.da.inflow).isel(snapshot=positions)
    constant += (inflow * flow_weight).sum().item()

    spill_name = f"{c.name}-spill"
    if spill_name in m.variables:
        spilling = factor.index.intersection(m[spill_name].indexes["name"])
        spill = select(m[spill_name], spilling).isel(snapshot=positions)
        if (spill.labels != -1).any():
            lhs.append((spill * -flow_weight.sel(name=spilling)).sum())

    return lhs, constant


def _storage_withdrawal(
    n: Network,
    sns: pd.Index,
    counted: np.ndarray,
    carrier_factor: pd.Series,
    *,
    scenario: Any,
) -> tuple[list[LinearExpression], float]:
    """Net energy withdrawn from storage units and stores in the counted snapshots.

    Summing the storage energy balance over the snapshots of an investment period
    shows that the net energy withdrawn from a storage, including standing losses, is
    its level before the period minus its level at the end of the period plus its
    weighted net inflow `inflow - spill`. Each period is weighted by its years and
    each storage by the `carrier_factor` of its carrier.

    Returns the variable terms and the constant of the summed withdrawal.
    """
    lhs: list[LinearExpression] = []
    constant = 0.0
    for c in (n.c.storage_units, n.c.stores):
        if c.active_assets.empty:
            continue
        carrier = c.static.loc[scenario].carrier.loc[c.active_assets]
        factor = carrier.map(carrier_factor).dropna()
        if factor.empty:
            continue
        terms, component_constant = _component_withdrawal(
            n, c, factor, sns, counted, scenario=scenario
        )
        lhs += terms
        constant += component_constant
    return lhs, constant


def define_tech_capacity_expansion_limit(n: Network, sns: Sequence) -> None:
    """Define per-carrier and potentially per-bus capacity expansion limits.

    Parameters
    ----------
    n : pypsa.Network
        The network to apply constraints to.
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    """
    m = n.model
    glcs = n.c.global_constraints.static.loc[
        lambda df: df.type == "tech_capacity_expansion_limit"
    ]

    if n.has_scenarios and not glcs.empty:
        msg = "Technology expansion limits for stochastic networks are not implemented."
        raise NotImplementedError(msg)

    for (carrier, sense, period), glcs_group in glcs.groupby(
        ["carrier_attribute", "sense", "investment_period"], dropna=False
    ):
        period = None if np.isnan(period) else int(period)
        sign = "=" if sense == "==" else sense
        busdim = f"Bus-{carrier}-{period}"
        lhs_per_bus_list = []

        for c, attr in nominal_attrs.items():
            var = f"{c}-{attr}"
            static = n.c[c].static

            if "carrier" not in static:
                continue

            ext_i = n.c[c].extendables.intersection(
                static.index[static.carrier == carrier]
            )
            ext_i = n.c[c].filter_by_active_assets(ext_i, period)

            if ext_i.empty:
                continue

            bus = "bus0" if c in n.branch_components else "bus"
            busmap = static.loc[ext_i, bus].rename(busdim).to_xarray()
            expr = m[var].loc[ext_i].groupby(busmap).sum()
            lhs_per_bus_list.append(expr)

        if not lhs_per_bus_list:
            continue

        lhs_per_bus = merge(lhs_per_bus_list)

        for name, glc in glcs_group.iterrows():
            bus_glc = glc.get("bus") or None
            if bus_glc is None:
                lhs = lhs_per_bus.sum(busdim)
            else:
                lhs = lhs_per_bus.sel(**{busdim: str(bus_glc)}, drop=True)

            n.model.add_constraints(
                lhs,
                sign,
                glc.constant,
                name=f"GlobalConstraint-{name}",
                scaling=global_constraint_factor(n, name),
            )


def define_nominal_constraints_per_bus_carrier(n: Network, sns: pd.Index) -> None:
    """Set an capacity expansion limit for assets of the same carrier at the same bus.

    The function searches for columns in the `buses` dataframe matching the pattern
    "nom_{min/max}_{carrier}". In case the constraint should only be defined for one
    investment period, the column name can be constructed according to
    "nom_{min/max}_{carrier}_{period}" where period must be in `n.investment_periods`.

    Parameters
    ----------
    n : pypsa.Network
        The network to apply constraints to.
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    """
    cols = n.c.buses.static.columns[n.c.buses.static.columns.str.startswith("nom_")]
    if cols.empty:
        return

    m = n.model
    warnings.warn(
        "Nominal constraints per bus carrier are deprecated and will be removed in the future. "
        "Use global constraint of type 'define_tech_capacity_expansion_limit' instead."
        "Deprecated in PyPSA 1.0 and will be removed in PyPSA 2.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    buses = n.c.buses.static.index[n.c.buses.static[cols].notnull().any(axis=1)]
    if n.has_scenarios and not buses.empty:
        msg = "Nominal constraints per bus carrier are not implemented for stochastic networks."
        raise NotImplementedError(msg)

    multi_invest = isinstance(n.snapshots, pd.MultiIndex)
    periods = n.optimize._window.subset(sns).periods if multi_invest else pd.Index([])

    for col in cols:
        msg = (
            f"Bus column '{col}' has invalid specification and cannot be "
            "interpreted as constraint, must match the pattern "
            "`nom_{min/max}_{carrier}` or `nom_{min/max}_{carrier}_{period}`"
        )
        if col.startswith("nom_min_"):
            sign = ">="
        elif col.startswith("nom_max_"):
            sign = "<="
        else:
            logger.warning(msg)
            continue
        remainder = col[len("nom_max_") :]
        if remainder in n.c.carriers.static.index:
            carrier = remainder
            period = None
        elif multi_invest:
            carrier, period = remainder.rsplit("_", 1)
            period = int(period)
            if carrier not in n.c.carriers.static.index or period not in periods:
                logger.warning(msg)
                continue
        else:
            logger.warning(msg)
            continue

        lhs = []

        for c, attr in nominal_attrs.items():
            var = f"{c}-{attr}"
            static = n.c[c].static

            if c not in n.one_port_components or "carrier" not in static:
                continue

            ext_i = n.c[c].extendables.intersection(
                static.index[static.carrier == carrier]
            )
            ext_i = n.c[c].filter_by_active_assets(ext_i, period)

            if ext_i.empty:
                continue

            busmap = static.loc[ext_i, "bus"].rename(buses.name).to_xarray()
            expr = m[var].loc[ext_i].groupby(busmap).sum().reindex({buses.name: buses})
            lhs.append(expr)

        if not lhs:
            continue

        lhs = merge(lhs)
        rhs = n.c.buses.static.loc[buses, col]
        mask = rhs.notnull()
        n.model.add_constraints(
            lhs,
            sign,
            rhs,
            name=f"Bus-{col}",
            mask=mask,
            scaling=n.c.buses._scaling(buses),
        )


def define_growth_limit(n: Network, sns: pd.Index) -> None:
    """Constraint new installed capacity per investment period.

    Parameters
    ----------
    n : pypsa.Network
        The network to apply constraints to.
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    """
    if not n._multi_invest:
        return

    m = n.model
    periods = n.optimize._window.periods

    # Handle stochastic optimization: find strictest (minimum) growth limit across scenarios
    if n.has_scenarios:
        max_growth = n.c.carriers.static.groupby(level="name")["max_growth"].min()
        max_relative_growth = n.c.carriers.static.groupby(level="name")[
            "max_relative_growth"
        ].min()
    else:
        max_growth = n.c.carriers.static["max_growth"]
        max_relative_growth = n.c.carriers.static["max_relative_growth"]

    carrier_i = max_growth[max_growth != float("inf")].index.rename("Carrier")
    max_absolute_growth = DataArray(max_growth.loc[carrier_i].rename_axis("Carrier"))
    max_relative_growth = DataArray(
        max_relative_growth.loc[carrier_i].rename_axis("Carrier")
    ).clip(min=0)

    if carrier_i.empty:
        return

    lhs_list = []
    factors = []
    for c, attr in nominal_attrs.items():
        var = f"{c}-{attr}"
        static = n.c[c].static

        if "carrier" not in static:
            continue

        carrier_map = invariant(static["carrier"])
        carriers_match = carrier_map.index[carrier_map.isin(carrier_i)]
        limited_names = carriers_match.intersection(
            n.c[c].filter_by_active_assets(n.c[c].extendables)
        )

        if limited_names.empty:
            continue

        # Get active assets for the limited components
        active = pd.concat(
            {p: n.components[c].get_active_assets(p) for p in periods}, axis=1
        )
        active = invariant(active).loc[limited_names].rename_axis(columns="periods").T
        first_active = DataArray(active & (active.cumsum() == 1))
        carriers = carrier_map.loc[limited_names].rename("Carrier")

        vars = m[var].sel(name=limited_names).where(first_active)
        expr = vars.groupby(carriers.to_xarray()).sum()

        if (max_relative_growth.loc[carriers.unique()] > 0).any():
            expr = expr - expr.shift(periods=1) * max_relative_growth

        lhs_list.append(expr)
        asset_factors = n.c[c]._scaling(limited_names)
        factors.append(pd.Series(asset_factors, index=carriers.values))

    if not lhs_list:
        return

    lhs = merge(lhs_list, join="outer")
    rhs = max_absolute_growth.reindex(lhs.indexes)
    grouped = pd.concat(factors).groupby(level=0)
    row_factor = np.sqrt(grouped.min() * grouped.max()).rename_axis("Carrier")
    scaling = 1.0 if (row_factor == 1).all() else DataArray(row_factor)

    m.add_constraints(lhs, "<=", rhs, name="Carrier-growth_limit", scaling=scaling)


def define_primary_energy_limit(
    n: Network, sns: pd.Index, piecewise_options: list[PiecewiseOptions]
) -> None:
    """Define primary energy constraints.

    It limits the byproducts of primary energy sources (defined by carriers) such
    as CO2.

    Parameters
    ----------
    n : pypsa.Network
        The network to apply constraints to.
    sns : list-like
        Set of snapshots to which the constraint should be applied.
    piecewise_options : list[PiecewiseOptions]
        Options to override defaults in piecewise constraint formulation.
        List is of the form ``[PiecewiseOptions(...), ...]``.

    """
    m = n.model
    glcs = n.c.global_constraints.static.query('type == "primary_energy"')
    if glcs.empty:
        return

    window = n.optimize._window.subset(sns)
    weight = window.snapshot_weightings("generators")
    if n._multi_invest:
        period_of = window.period_of
        periods = window.periods
        period_weighting = n.investment_period_weightings.years[periods]
        weight = apply_period_weighting(weight, period_weighting)

    unique_names = glcs.index.unique("name")

    gen_c = n.c.generators
    var_name = "Generator-p"
    pw_attr_eff = gen_c._piecewise_schema("efficiency")
    primary_energy_pw_var = None
    if not unique_names.empty and not pw_attr_eff.empty and var_name in m.variables:
        extra_options = filter(
            lambda opt: opt.component == gen_c.name and opt.attribute == "efficiency",
            piecewise_options,
        )
        status = (
            None
            if gen_c.committables.intersection(gen_c.active_assets).empty
            else m[f"{gen_c.name}-status"]
        )
        primary_energy_pw_var = define_piecewise(
            m,
            gen_c,
            x_var=m[var_name],
            pw_attr="efficiency",
            aux_var_name=f"{gen_c.name}-{pw_attr_eff.aux_variable}",
            active_names=gen_c.active_assets,
            sign="=",
            cumulative_attr=False,
            extra_options=extra_options,
            invert_attr=True,
            scaling=gen_c,
            status=status,
        )

    for name in unique_names:
        if n.has_scenarios:
            glc_group = glcs.xs(name, level="name")
            scenarios = glc_group.index.get_level_values("scenario")
        else:
            glc_group = glcs.loc[name]
            scenarios = [slice(None)]

        expressions = []
        has_variables = False
        for scenario in scenarios:
            glc = glc_group.loc[scenario]

            if np.isnan(glc.investment_period):
                in_period = slice(None)
            elif not n._multi_invest:
                msg = (
                    f"GlobalConstraint '{name}' sets `investment_period`, but the "
                    "network has no investment periods."
                )
                raise ValueError(msg)
            elif glc.investment_period in periods:
                in_period = period_of == glc.investment_period
            else:
                continue
            period_sns = sns[in_period]
            counted = np.zeros(len(sns), dtype=bool)
            counted[in_period] = True

            lhs = []
            emissions = n.c.carriers.static[glc.carrier_attribute][
                lambda ds: ds != 0
            ].loc[scenario]

            if emissions.empty:
                continue

            # Determine investment period for active asset filtering
            period = glc.investment_period if n._multi_invest else None

            # generators
            gens = n.c.generators.static[
                n.c.generators.static.carrier.isin(emissions.index)
            ]
            gens = n.c.generators.filter_by_active_assets(gens, period)

            if not gens.empty:
                gens = gens.loc[scenario]
                p = m[var_name].sel(name=gens.index, snapshot=period_sns)

                if n.has_scenarios:
                    p = p.sel(scenario=scenario, drop=True)

                linear_names = gens.index
                to_sum = []
                if primary_energy_pw_var is not None:
                    pw_names = gens.index.intersection(
                        primary_energy_pw_var.indexes["name"]
                    )
                    if not pw_names.empty:
                        pw_var = primary_energy_pw_var.sel(
                            name=pw_names, snapshot=period_sns
                        )
                        to_sum.append(pw_var.to_linexpr())
                        linear_names = linear_names.difference(pw_names)

                if not linear_names.empty:
                    efficiency = n.c.generators.da.efficiency.sel(
                        name=linear_names, snapshot=period_sns
                    )
                    if n.has_scenarios:
                        efficiency = efficiency.sel(scenario=scenario, drop=True)
                    to_sum.append(p.sel(name=linear_names) / efficiency)
                dispatch = to_sum[0]
                for term in to_sum[1:]:
                    dispatch = dispatch.add(term, join="outer")
                emission_rate = gens.carrier.map(emissions).reindex(
                    dispatch.indexes["name"]
                )
                expr = (
                    dispatch * weight.sel(snapshot=period_sns) * emission_rate
                ).sum()
                lhs.append(expr)

            storage_lhs, constant = _storage_withdrawal(
                n, sns, counted, emissions, scenario=scenario
            )
            lhs += storage_lhs

            has_variables |= bool(lhs)
            expression = merge(lhs) if lhs else LinearExpression(None, m)
            expressions.append(expression + constant)

        if not has_variables:
            continue

        if n.has_scenarios:
            expression = merge(expressions, dim="scenario").assign_coords(
                scenario=scenarios
            )
        else:
            expression = expressions[0]

        m.add_constraints(
            expression,
            glc_group.sense,
            glc_group.constant,
            name=f"GlobalConstraint-{name}",
            scaling=global_constraint_factor(n, name),
        )


def define_operational_limit(n: Network, sns: pd.Index) -> None:
    """Define operational limit constraints.

    It limits the net production of a carrier taking into account generator, storage
    units and stores.

    Parameters
    ----------
    n : pypsa.Network
        The network to apply constraints to.
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    """
    m = n.model
    glcs = n.c.global_constraints.static.query('type == "operational_limit"')
    if glcs.empty:
        return

    window = n.optimize._window.subset(sns)
    weight = window.snapshot_weightings("generators")
    if n._multi_invest:
        period_of = window.period_of
        periods = window.periods
    unique_names = glcs.index.unique("name")

    for name in unique_names:
        if n.has_scenarios:
            glc_group = glcs.xs(name, level="name")
            scenarios = glc_group.index.get_level_values("scenario")
        else:
            glc_group = glcs.loc[name]
            scenarios = [slice(None)]

        expressions = []
        has_variables = False
        for scenario in scenarios:
            glc = glc_group.loc[scenario]

            if np.isnan(glc.investment_period):
                in_period = slice(None)
            elif not n._multi_invest:
                msg = (
                    f"GlobalConstraint '{name}' sets `investment_period`, but the "
                    "network has no investment periods."
                )
                raise ValueError(msg)
            elif glc.investment_period in periods:
                in_period = period_of == glc.investment_period
            else:
                continue
            period_sns = sns[in_period]
            counted = np.zeros(len(sns), dtype=bool)
            counted[in_period] = True

            # Filter weightings and calculate period-specific values
            w = weight.sel(snapshot=period_sns)
            if n._multi_invest:
                period_weighting = n.investment_period_weightings.years[
                    period_of[in_period].unique()
                ]
                w = apply_period_weighting(w, period_weighting)

            lhs = []

            # generators
            gens = n.c.generators.static.query(
                "carrier == @glc.carrier_attribute and active"
            )
            if not gens.empty:
                gens = gens.loc[scenario]
                p = m["Generator-p"].sel(name=gens.index, snapshot=period_sns)
                if n.has_scenarios:
                    p = p.sel(scenario=scenario, drop=True)

                expr = (p * w).sum()
                lhs.append(expr)

            carrier_factor = pd.Series({glc.carrier_attribute: 1.0})
            storage_lhs, constant = _storage_withdrawal(
                n, sns, counted, carrier_factor, scenario=scenario
            )
            lhs += storage_lhs

            has_variables |= bool(lhs)
            expression = merge(lhs) if lhs else LinearExpression(None, m)
            expressions.append(expression + constant)

        if not has_variables:
            continue

        if n.has_scenarios:
            expression = merge(expressions, dim="scenario").assign_coords(
                scenario=scenarios
            )
        else:
            expression = expressions[0]

        m.add_constraints(
            expression,
            glc_group.sense,
            glc_group.constant,
            name=f"GlobalConstraint-{name}",
            scaling=global_constraint_factor(n, name),
        )


def _define_transmission_expansion_limit(
    n: Network,
    sns: Sequence,
    kind: str,
    coefficient: Callable[[Components, pd.Index, Any, Any], pd.Series],
) -> None:
    """Add per-scenario limits on weighted Line and Link nominal capacities.

    Parameters
    ----------
    n : pypsa.Network
        The network to apply constraints to.
    sns : list-like
        Set of snapshots to which the constraint should be applied.
    kind : str
        Global constraint type.
    coefficient : callable
        Returns the coefficients of the extendable assets for a component, asset
        names, scenario selector and investment period filter.

    """
    m = n.model
    glcs = n.c.global_constraints.static.query(f"type == '{kind}'")
    if glcs.empty:
        return

    window = n.optimize._window.subset(sns)

    for name in glcs.index.unique("name"):
        if n.has_scenarios:
            group = glcs.xs(name, level="name")
            rows = list(group.iterrows())
        else:
            rows = [(slice(None), glcs.loc[name])]

        scenarios, expressions = [], []
        for scenario, glc in rows:
            carrier_attribute = re.sub(r"[\[\]\(\)]", "", glc.carrier_attribute)
            car = [s.strip() for s in carrier_attribute.split(",")]
            if not np.isnan(glc.investment_period):
                period_filter = glc.investment_period
            elif window.has_periods:
                period_filter = list(window.periods)
            else:
                period_filter = None

            lhs = []
            for c in n.components[["Line", "Link"]]:
                carriers = invariant(c.static.carrier)
                ext_i = c.extendables.intersection(carriers.index[carriers.isin(car)])
                ext_i = c.filter_by_active_assets(ext_i, period_filter)
                if ext_i.empty:
                    continue
                coeff = coefficient(c, ext_i, scenario, period_filter)
                vars = m[f"{c.name}-{nominal_attrs[c.name]}"].loc[ext_i]
                lhs.append(m.linexpr((coeff, vars)).sum())

            if lhs:
                scenarios.append(scenario)
                expressions.append(merge(lhs))

        if not expressions:
            continue

        if n.has_scenarios:
            scenario_i = pd.Index(scenarios, name="scenario")
            expression = merge(expressions, dim="scenario")
            expression = expression.assign_coords(scenario=scenario_i)
            glc = group.loc[scenario_i]
        else:
            expression = expressions[0]
            glc = rows[0][1]

        m.add_constraints(
            expression,
            glc.sense,
            glc.constant,
            name=f"GlobalConstraint-{name}",
            scaling=global_constraint_factor(n, name),
        )


def define_transmission_volume_expansion_limit(n: Network, sns: Sequence) -> None:
    """Set a limit for line volume expansion.

    For the capacity expansion only the carriers 'AC' and 'DC' are considered.

    Parameters
    ----------
    n : pypsa.Network
        The network to apply constraints to.
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    """

    def length(c: Components, ext_i: pd.Index, scenario: Any, _: Any) -> pd.Series:
        return c.static.length.loc[scenario].reindex(ext_i)

    _define_transmission_expansion_limit(
        n, sns, "transmission_volume_expansion_limit", length
    )


def define_transmission_expansion_cost_limit(n: Network, sns: pd.Index) -> None:
    """Set a limit for line expansion costs.

    For the capacity expansion only the carriers 'AC' and 'DC' are considered.

    Parameters
    ----------
    n : pypsa.Network
        The network to apply constraints to.
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    """
    weightings = n.investment_period_weightings.objective

    def cost(
        c: Components, ext_i: pd.Index, scenario: Any, period_filter: Any
    ) -> pd.Series:
        capital_cost = c.capital_cost.loc[scenario].reindex(ext_i)
        if not isinstance(period_filter, list):
            return capital_cost
        active = pd.DataFrame(
            {p: ext_i.isin(c._active_names(p)) for p in period_filter}, index=ext_i
        )
        return capital_cost * (active @ weightings[period_filter])

    _define_transmission_expansion_limit(
        n, sns, "transmission_expansion_cost_limit", cost
    )
