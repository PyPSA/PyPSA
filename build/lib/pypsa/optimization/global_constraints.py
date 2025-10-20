# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Define global constraints for optimisation problems with Linopy."""

from __future__ import annotations

import logging
import re
import warnings
from typing import TYPE_CHECKING

import pandas as pd
from linopy.expressions import merge
from numpy import isnan
from xarray import DataArray

from pypsa.descriptors import nominal_attrs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network
logger = logging.getLogger(__name__)


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
        ["carrier_attribute", "sense", "investment_period"]
    ):
        period = None if isnan(period) else int(period)
        sign = "=" if sense == "==" else sense
        busdim = f"Bus-{carrier}-{period}"
        lhs_per_bus_list = []

        for c, attr in nominal_attrs.items():
            var = f"{c}-{attr}"
            static = n.c[c].static

            if "carrier" not in static:
                continue

            ext_i = n.components[c].extendables.difference(n.c[c].inactive_assets)
            ext_i = ext_i.intersection(static.index[static.carrier == carrier])
            if period is not None:
                ext_i = ext_i[n.components[c].get_active_assets(period)[ext_i]]

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
                lhs, sign, glc.constant, name=f"GlobalConstraint-{name}"
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
    m = n.model
    cols = n.c.buses.static.columns[n.c.buses.static.columns.str.startswith("nom_")]
    buses = n.c.buses.static.index[n.c.buses.static[cols].notnull().any(axis=1)]

    if not cols.empty:
        warnings.warn(
            "Nominal constraints per bus carrier are deprecated and will be removed in the future. "
            "Use global constraint of type 'define_tech_capacity_expansion_limit' instead."
            "Deprecated in PyPSA 1.0 and will be removed in PyPSA 2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    if n.has_scenarios and not buses.empty:
        msg = "Nominal constraints per bus carrier are not implemented for stochastic networks."
        raise NotImplementedError(msg)

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
        elif isinstance(n.snapshots, pd.MultiIndex):
            carrier, period = remainder.rsplit("_", 1)
            period = int(period)
            if carrier not in n.c.carriers.static.index or period not in sns.unique(
                "period"
            ):
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

            ext_i = n.c[c].extendables.difference(n.c[c].inactive_assets)
            ext_i = ext_i.intersection(static.index[static.carrier == carrier])
            if period is not None:
                ext_i = ext_i[n.components[c].get_active_assets(period)[ext_i]]

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
        n.model.add_constraints(lhs, sign, rhs, name=f"Bus-{col}", mask=mask)


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
    periods = sns.unique("period")

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
    max_absolute_growth = DataArray(max_growth.loc[carrier_i])
    max_relative_growth = DataArray(max_relative_growth.loc[carrier_i]).clip(min=0)

    if carrier_i.empty:
        return

    lhs_list = []
    for c, attr in nominal_attrs.items():
        var = f"{c}-{attr}"
        static = n.c[c].static

        if "carrier" not in static:
            continue

        component_carriers = static.loc[:, "carrier"]

        if n.has_scenarios:
            unique_component_names = n.components[c].names
            carrier_map = component_carriers.groupby(level="name").first()
        else:
            unique_component_names = static.index
            carrier_map = component_carriers

        carriers_match = unique_component_names[carrier_map.isin(carrier_i)]
        limited_names = carriers_match.intersection(
            n.c[c].extendables.difference(n.c[c].inactive_assets)
        )

        if limited_names.empty:
            continue

        # Get active assets for the limited components
        active = pd.concat(
            {p: n.components[c].get_active_assets(p) for p in periods}, axis=1
        )

        if n.has_scenarios:
            active = active.groupby(level="name").first()

        active = active.loc[limited_names].rename_axis(columns="periods").T
        first_active = DataArray(active.cumsum() == 1)
        carriers = carrier_map.loc[limited_names].rename("Carrier")

        vars = m[var].sel(name=limited_names).where(first_active)
        expr = vars.groupby(carriers.to_xarray()).sum()

        if (max_relative_growth.loc[carriers.unique()] > 0).any():
            expr = expr - expr.shift(periods=1) * max_relative_growth

        lhs_list.append(expr)

    if not lhs_list:
        return

    lhs = merge(lhs_list)
    rhs = max_absolute_growth.reindex_like(lhs.data)

    m.add_constraints(lhs, "<=", rhs, name="Carrier-growth_limit")


def define_primary_energy_limit(n: Network, sns: pd.Index) -> None:
    """Define primary energy constraints.

    It limits the byproducts of primary energy sources (defined by carriers) such
    as CO2.

    Parameters
    ----------
    n : pypsa.Network
        The network to apply constraints to.
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    """
    m = n.model
    weightings = n.snapshot_weightings.loc[sns]
    glcs = n.c.global_constraints.static.query('type == "primary_energy"')

    if n._multi_invest:
        period_weighting = n.investment_period_weightings.years[sns.unique("period")]
        weightings = weightings.mul(period_weighting, level=0, axis=0)
        period_last_sns = pd.MultiIndex.from_frame(
            sns.to_frame(index=False).groupby("period").timestep.last().reset_index()
        )
        storage_weightings = (
            pd.Series(1, n.snapshots).mul(period_weighting).loc[period_last_sns]
        )

    unique_names = glcs.index.unique("name")

    for name in unique_names:
        if n.has_scenarios:
            glc_group = glcs.xs(name, level="name")
            scenarios = glc_group.index.get_level_values("scenario")
        else:
            glc_group = glcs.loc[name]
            scenarios = [slice(None)]

        expressions = []
        for scenario in scenarios:
            glc = glc_group.loc[scenario]

            if isnan(glc.investment_period):
                sns_sel = slice(None)
            elif glc.investment_period in sns.unique("period"):
                sns_sel = sns.get_loc(glc.investment_period)
            else:
                continue

            lhs = []
            emissions = n.c.carriers.static[glc.carrier_attribute][
                lambda ds: ds != 0
            ].loc[scenario]

            if emissions.empty:
                continue

            # generators
            emission_carriers = emissions.index
            gens = n.c.generators.static[
                n.c.generators.static.carrier.isin(emission_carriers)
            ]

            if not gens.empty:
                gens = gens.loc[scenario]
                efficiency = (
                    n.c.generators._as_dynamic("efficiency")
                    .loc[:, scenario]
                    .loc[sns[sns_sel], gens.index]
                )
                em_pu = gens.carrier.map(emissions) / efficiency
                em_pu = em_pu.multiply(weightings.generators[sns_sel], axis=0)

                p = m["Generator-p"].sel(name=gens.index, snapshot=sns[sns_sel])

                if n.has_scenarios:
                    p = p.sel(scenario=scenario, drop=True)

                expr = (p * em_pu).sum()
                lhs.append(expr)

            # storage units
            cond = "carrier in @emissions.index and not cyclic_state_of_charge"
            sus = n.c.storage_units.static.query(cond)
            if not sus.empty:
                sus = sus.loc[scenario]
                em_pu = sus.carrier.map(emissions)
                soc = m["StorageUnit-state_of_charge"].sel(
                    name=sus.index, snapshot=sns[sns_sel]
                )

                if n._multi_invest:
                    sus_continuous = sus.query("not state_of_charge_initial_per_period")
                    if not sus_continuous.empty and period_weighting.ne(1).any():
                        msg = (
                            "Found non-cyclic storage units with associated carrier emissions "
                            "and continuous depletion over multiple investment periods "
                            "combined with investment period year weightings != 1. "
                            "The primary energy constraint will be inconsistent. "
                            "Please consider setting `state_of_charge_initial_per_period` to True, "
                            "using equal period weightings or a cyclic storage unit instead."
                        )
                        raise NotImplementedError(msg)

                    if not sus_continuous.empty and period_weighting.eq(1).all():
                        soc_final = (
                            soc.sel(name=sus_continuous.index)
                            .ffill("snapshot")
                            .isel(snapshot=-1)
                        )
                        if n.has_scenarios:
                            soc_final = soc_final.sel(scenario=scenario, drop=True)
                        lhs.append(
                            (soc_final * -em_pu).sum()
                            + em_pu @ sus_continuous.state_of_charge_initial
                        )

                    sus_per_period = sus.query("state_of_charge_initial_per_period")
                    if not sus_per_period.empty:
                        soc_final = soc.loc[period_last_sns, sus_per_period.index]
                        if n.has_scenarios:
                            soc_final = soc_final.sel(scenario=scenario, drop=True)
                        soc_delta = -soc_final + sus_per_period.state_of_charge_initial
                        lhs.append((soc_delta * storage_weightings * em_pu).sum())

                else:
                    soc_final = soc.ffill("snapshot").isel(snapshot=-1)
                    if n.has_scenarios:
                        soc_final = soc_final.sel(scenario=scenario, drop=True)
                    lhs.append(
                        (soc_final * -em_pu).sum() + em_pu @ sus.state_of_charge_initial
                    )

            # stores
            stores = n.c.stores.static.query(
                "carrier in @emissions.index and not e_cyclic"
            )
            if not stores.empty:
                stores = stores.loc[scenario]
                em_pu = stores.carrier.map(emissions)
                e = m["Store-e"].sel(name=stores.index, snapshot=sns[sns_sel])

                if n._multi_invest:
                    stores_continuous = stores.query("not e_initial_per_period")
                    if not stores_continuous.empty and period_weighting.ne(1).any():
                        msg = (
                            "Found non-cyclic stores with associated carrier emissions "
                            "and continuous depletion over multiple investment periods "
                            "combined with investment period year weightings != 1. "
                            "The primary energy constraint will be inconsistent. "
                            "Please consider setting `e_initial_per_period` to True, "
                            "using equal period weightings or a cyclic store instead."
                        )
                        raise NotImplementedError(msg)

                    if not stores_continuous.empty and period_weighting.eq(1).all():
                        e_final = (
                            e.sel(name=stores_continuous.index)
                            .ffill("snapshot")
                            .isel(snapshot=-1)
                        )
                        if n.has_scenarios:
                            e_final = e_final.sel(scenario=scenario, drop=True)
                        lhs.append(
                            (e_final * -em_pu).sum()
                            + em_pu @ stores_continuous.e_initial
                        )

                    stores_per_period = stores.query("e_initial_per_period")
                    if not stores_per_period.empty:
                        e_final = e.loc[period_last_sns, stores_per_period.index]
                        if n.has_scenarios:
                            e_final = e_final.sel(scenario=scenario, drop=True)
                        e_delta = -e_final + stores_per_period.e_initial
                        lhs.append((e_delta * storage_weightings * em_pu).sum())

                else:
                    e_final = e.ffill("snapshot").isel(snapshot=-1)
                    if n.has_scenarios:
                        e_final = e_final.sel(scenario=scenario, drop=True)
                    lhs.append((e_final * -em_pu).sum() + em_pu @ stores.e_initial)

            if not lhs:
                continue

            lhs = merge(lhs)
            expressions.append(lhs)

        if not expressions:
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
    weightings = n.snapshot_weightings.loc[sns]
    glcs = n.c.global_constraints.static.query('type == "operational_limit"')

    if n._multi_invest:
        period_weighting = n.investment_period_weightings.years[sns.unique("period")]
        weightings = weightings.mul(period_weighting, level=0, axis=0)
        period_last_sns = pd.MultiIndex.from_frame(
            sns.to_frame(index=False).groupby("period").timestep.last().reset_index()
        )
        storage_weightings = (
            pd.Series(1, n.snapshots).mul(period_weighting).loc[period_last_sns]
        )

    unique_names = glcs.index.unique("name")

    for name in unique_names:
        if n.has_scenarios:
            glc_group = glcs.xs(name, level="name")
            scenarios = glc_group.index.get_level_values("scenario")
        else:
            glc_group = glcs.loc[name]
            scenarios = [slice(None)]

        expressions = []
        for scenario in scenarios:
            glc = glc_group.loc[scenario]

            if isnan(glc.investment_period):
                sns_sel = slice(None)
            elif glc.investment_period in sns.unique("period"):
                sns_sel = sns.get_loc(glc.investment_period)
            else:
                continue

            lhs = []

            # generators
            gens = n.c.generators.static.query(
                "carrier == @glc.carrier_attribute and active"
            )
            if not gens.empty:
                gens = gens.loc[scenario]
                p = m["Generator-p"].sel(name=gens.index, snapshot=sns[sns_sel])
                if n.has_scenarios:
                    p = p.sel(scenario=scenario, drop=True)

                w = DataArray(weightings.generators[sns_sel])
                expr = (p * w).sum()
                lhs.append(expr)

            # storage units (non-cyclic): subtract end SoC and add initial SoC as constant
            cond = "carrier == @glc.carrier_attribute and not cyclic_state_of_charge and active"
            sus = n.c.storage_units.static.query(cond)

            if not sus.empty:
                sus = sus.loc[scenario]
                soc = m["StorageUnit-state_of_charge"].sel(
                    name=sus.index, snapshot=sns[sns_sel]
                )

                if n._multi_invest:
                    sus_continuous = sus.query("not state_of_charge_initial_per_period")
                    if not sus_continuous.empty and period_weighting.ne(1).any():
                        msg = (
                            "Found non-cyclic storage units with "
                            "continuous depletion over multiple investment periods "
                            "combined with investment period year weightings != 1. "
                            "The operational constraint will be inconsistent. "
                            "Please consider setting `state_of_charge_initial_per_period` to True, "
                            "using equal period weightings or a cyclic storage unit instead."
                        )
                        raise NotImplementedError(msg)

                    if not sus_continuous.empty and period_weighting.eq(1).all():
                        soc_final = (
                            soc.sel(name=sus_continuous.index)
                            .ffill("snapshot")
                            .isel(snapshot=-1)
                        )
                        if n.has_scenarios:
                            soc_final = soc_final.sel(scenario=scenario, drop=True)
                        lhs.append(
                            (-soc_final).sum()
                            + sus_continuous.state_of_charge_initial.sum()
                        )

                    sus_per_period = sus.query("state_of_charge_initial_per_period")
                    if not sus_per_period.empty:
                        soc_final = soc.loc[period_last_sns, sus_per_period.index]
                        if n.has_scenarios:
                            soc_final = soc_final.sel(scenario=scenario, drop=True)
                        soc_delta = -soc_final + sus_per_period.state_of_charge_initial
                        lhs.append((soc_delta * storage_weightings).sum())

                else:
                    soc_final = soc.ffill("snapshot").isel(snapshot=-1)
                    if n.has_scenarios:
                        soc_final = soc_final.sel(scenario=scenario, drop=True)
                    lhs.append((-soc_final).sum() + sus.state_of_charge_initial.sum())

            # stores (non-cyclic): subtract end e and add initial e as constant
            stores = n.c.stores.static.query(
                "carrier == @glc.carrier_attribute and not e_cyclic and active"
            )
            if not stores.empty:
                stores = stores.loc[scenario]
                e = m["Store-e"].sel(name=stores.index, snapshot=sns[sns_sel])

                if n._multi_invest:
                    stores_continuous = stores.query("not e_initial_per_period")
                    if not stores_continuous.empty and period_weighting.ne(1).any():
                        msg = (
                            "Found non-cyclic stores with "
                            "continuous depletion over multiple investment periods "
                            "combined with investment period year weightings != 1. "
                            "The primary energy constraint will be inconsistent. "
                            "Please consider setting `e_initial_per_period` to True, "
                            "using equal period weightings or a cyclic store instead."
                        )
                        raise NotImplementedError(msg)

                    if not stores_continuous.empty and period_weighting.eq(1).all():
                        e_final = (
                            e.sel(name=stores_continuous.index)
                            .ffill("snapshot")
                            .isel(snapshot=-1)
                        )
                        if n.has_scenarios:
                            e_final = e_final.sel(scenario=scenario, drop=True)
                        lhs.append((-e_final).sum() + stores_continuous.e_initial.sum())

                    stores_per_period = stores.query("e_initial_per_period")
                    if not stores_per_period.empty:
                        e_final = e.loc[period_last_sns, stores_per_period.index]
                        if n.has_scenarios:
                            e_final = e_final.sel(scenario=scenario, drop=True)
                        e_delta = -e_final + stores_per_period.e_initial
                        lhs.append((e_delta * storage_weightings).sum())

                else:
                    e_final = e.ffill("snapshot").isel(snapshot=-1)
                    if n.has_scenarios:
                        e_final = e_final.sel(scenario=scenario, drop=True)
                    lhs.append((-e_final).sum() + stores.e_initial.sum())

            if not lhs:
                continue

            lhs = merge(lhs)
            expressions.append(lhs)

        if not expressions:
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
    m = n.model
    glcs = n.c.global_constraints.static.query(
        "type == 'transmission_volume_expansion_limit'"
    )

    def substr(s: str) -> str:
        return re.sub("[\\[\\]\\(\\)]", "", s)

    # Create one constraint per name, optionally with a scenario dimension
    if glcs.empty:
        return

    unique_names = (
        glcs.index.unique("name")
        if isinstance(glcs.index, pd.MultiIndex)
        else glcs.index.unique()
    )

    for name in unique_names:
        if n.has_scenarios:
            glc_group = glcs.xs(name, level="name")
            scenarios = glc_group.index.get_level_values("scenario")
        else:
            glc_group = glcs.loc[name]
            scenarios = [slice(None)]

        expressions = []
        for scenario in scenarios:
            glc = glc_group.loc[scenario]

            lhs = []
            # fmt: off
            car = [substr(c.strip()) for c in  # noqa: F841
                   glc.carrier_attribute.split(",")]
            # fmt: on
            period = glc.investment_period

            for c in n.components[["Line", "Link"]]:
                attr = nominal_attrs[c.name]

                # Start from extendable components by name
                ext_all = c.extendables.difference(c.inactive_assets)
                if ext_all.empty:
                    continue

                # Filter by carrier, handling scenarios (MultiIndex) if present
                if n.has_scenarios and isinstance(c.static.index, pd.MultiIndex):
                    eligible_by_carrier = (
                        c.static.query("carrier in @car")
                        .groupby(level="name")
                        .first()
                        .index
                    )
                else:
                    eligible_by_carrier = c.static.query("carrier in @car").index

                ext_i = ext_all.intersection(eligible_by_carrier).rename(ext_all.name)
                if ext_i.empty:
                    continue

                # Filter by investment period activity
                if not isnan(period):
                    active = c.get_active_assets(investment_period=int(period))
                    ext_i = ext_i[active.loc[ext_i]].rename(ext_i.name)
                elif isinstance(sns, pd.MultiIndex):
                    # Active in any of the periods present in sns
                    periods = sns.unique("period")
                    active_df = pd.concat(
                        {
                            p: c.get_active_assets(investment_period=int(p))
                            for p in periods
                        },
                        axis=1,
                    )
                    active_any = active_df.any(axis=1)
                    ext_i = ext_i[active_any.loc[ext_i]].rename(ext_i.name)

                if ext_i.empty:
                    continue

                # Length per name (collapse scenario level if present)
                if n.has_scenarios and isinstance(c.static.index, pd.MultiIndex):
                    length = (
                        c.static.length.groupby(level="name").first().reindex(ext_i)
                    )
                else:
                    length = c.static.length.reindex(ext_i)

                vars = m[f"{c.name}-{attr}"].loc[ext_i]
                lhs.append(m.linexpr((length, vars)).sum())

            if not lhs:
                continue

            expr = merge(lhs)
            expressions.append(expr)

        if not expressions:
            continue

        if n.has_scenarios:
            expression = merge(expressions, dim="scenario").assign_coords(
                scenario=scenarios
            )
        else:
            expression = expressions[0]

        sign = glc_group.sense
        rhs = glc_group.constant

        m.add_constraints(expression, sign, rhs, name=f"GlobalConstraint-{name}")


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
    m = n.model
    glcs = n.c.global_constraints.static.query(
        "type == 'transmission_expansion_cost_limit'"
    )

    if n._multi_invest:
        periods = sns.unique("period")
        period_weighting = n.investment_period_weightings.objective[periods]

    def substr(s: str) -> str:
        return re.sub("[\\[\\]\\(\\)]", "", s)

    for name, glc in glcs.iterrows():
        lhs = []
        # fmt: off
        car = [substr(c.strip()) for c in  # noqa: F841
               glc.carrier_attribute.split(",")]
        # fmt: on
        period = glc.investment_period

        for c in n.components[["Line", "Link"]]:
            attr = nominal_attrs[c.name]

            ext_i = c.extendables.difference(c.inactive_assets)
            if ext_i.empty:
                continue

            ext_i = ext_i.intersection(c.static.query("carrier in @car").index).rename(
                ext_i.name
            )

            if not isnan(period):
                ext_i = ext_i[
                    c.get_active_assets(investment_period=period)[ext_i]
                ].rename(ext_i.name)
                weights = 1

            elif isinstance(sns, pd.MultiIndex):
                ext_i = ext_i[
                    c.get_active_assets(investment_period=sns.unique("period"))[ext_i]
                ].rename(ext_i.name)
                active = pd.concat(
                    {
                        period: c.get_active_assets(investment_period=period)[ext_i]
                        for period in sns.unique("period")
                    },
                    axis=1,
                )
                weights = active @ period_weighting
            else:
                weights = 1

            cost = c.static.capital_cost.reindex(ext_i) * weights
            vars = m[f"{c.name}-{attr}"].loc[ext_i]
            lhs.append(m.linexpr((cost, vars)).sum())

        if not lhs:
            continue

        lhs = merge(lhs)
        sign = "=" if glc.sense == "==" else glc.sense
        m.add_constraints(lhs, sign, glc.constant, name=f"GlobalConstraint-{name}")
