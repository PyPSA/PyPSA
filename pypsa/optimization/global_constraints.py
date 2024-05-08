#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define global constraints for optimisation problems with Linopy.
"""

import logging
import re

import pandas as pd
from linopy.expressions import merge
from numpy import isnan
from xarray import DataArray

from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.descriptors import nominal_attrs

logger = logging.getLogger(__name__)


def define_tech_capacity_expansion_limit(n, sns):
    """
    Defines per-carrier and potentially per-bus capacity expansion limits.

    Parameters
    ----------
    n : pypsa.Network
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    Returns
    -------
    None.
    """
    m = n.model
    glcs = n.global_constraints.loc[
        lambda df: df.type == "tech_capacity_expansion_limit"
    ]

    for (carrier, sense, period), glcs_group in glcs.groupby(
        ["carrier_attribute", "sense", "investment_period"]
    ):
        period = None if isnan(period) else int(period)
        sign = "=" if sense == "==" else sense
        busdim = f"Bus-{carrier}-{period}"
        lhs_per_bus = []

        for c, attr in nominal_attrs.items():
            var = f"{c}-{attr}"
            dim = f"{c}-ext"
            df = n.df(c)

            if "carrier" not in df:
                continue

            ext_i = (
                n.get_extendable_i(c)
                .intersection(df.index[df.carrier == carrier])
                .rename(dim)
            )
            if period is not None:
                ext_i = ext_i[n.get_active_assets(c, period)[ext_i]]

            if ext_i.empty:
                continue

            bus = "bus0" if c in n.branch_components else "bus"
            busmap = df.loc[ext_i, bus].rename(busdim).to_xarray()
            expr = m[var].loc[ext_i].groupby(busmap).sum()
            lhs_per_bus.append(expr)

        if not lhs_per_bus:
            continue

        lhs_per_bus = merge(lhs_per_bus)

        for name, glc in glcs_group.iterrows():
            bus = glc.get("bus")
            if bus is None:
                lhs = lhs_per_bus.sum(busdim)
            else:
                lhs = lhs_per_bus.sel(**{busdim: str(bus)}, drop=True)

            n.model.add_constraints(
                lhs, sign, glc.constant, name=f"GlobalConstraint-{name}"
            )


def define_nominal_constraints_per_bus_carrier(n, sns):
    """
    Set an capacity expansion limit for assets of the same carrier at the same
    bus (e.g. 'onwind' at bus '1'). The function searches for columns in the
    `buses` dataframe matching the pattern "nom_{min/max}_{carrier}". In case
    the constraint should only be defined for one investment period, the column
    name can be constructed according to "nom_{min/max}_{carrier}_{period}"
    where period must be in `n.investment_periods`.

    Parameters
    ----------
    n : pypsa.Network
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    Returns
    -------
    None.
    """
    m = n.model
    cols = n.buses.columns[n.buses.columns.str.startswith("nom_")]
    buses = n.buses.index[n.buses[cols].notnull().any(axis=1)].rename("Bus-nom_min_max")

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
        if remainder in n.carriers.index:
            carrier = remainder
            period = None
        elif isinstance(n.snapshots, pd.MultiIndex):
            carrier, period = remainder.rsplit("_", 1)
            period = int(period)
            if carrier not in n.carriers.index or period not in sns.unique("period"):
                logger.warning(msg)
                continue
        else:
            logger.warning(msg)
            continue

        lhs = []

        for c, attr in nominal_attrs.items():
            var = f"{c}-{attr}"
            dim = f"{c}-ext"
            df = n.df(c)

            if c not in n.one_port_components or "carrier" not in df:
                continue

            ext_i = (
                n.get_extendable_i(c)
                .intersection(df.index[df.carrier == carrier])
                .rename(dim)
            )
            if period is not None:
                ext_i = ext_i[n.get_active_assets(c, period)[ext_i]]

            if ext_i.empty:
                continue

            busmap = df.loc[ext_i, "bus"].rename(buses.name).to_xarray()
            expr = m[var].loc[ext_i].groupby(busmap).sum().reindex({buses.name: buses})
            lhs.append(expr)

        if not lhs:
            continue

        lhs = merge(lhs)
        rhs = n.buses.loc[buses, col]
        mask = rhs.notnull()
        n.model.add_constraints(lhs, sign, rhs, name=f"Bus-{col}", mask=mask)


def define_growth_limit(n, sns):
    """
    Constraint new installed capacity per investment period.

    Parameters
    ----------
    n : pypsa.Network
    sns : list-like
        Set of snapshots to which the constraint should be applied.
    """
    if not n._multi_invest:
        return

    m = n.model
    periods = sns.unique("period")
    carrier_i = n.carriers.query("max_growth != inf").index.rename("Carrier")
    max_absolute_growth = DataArray(n.carriers.loc[carrier_i, "max_growth"])
    max_relative_growth = DataArray(
        n.carriers.loc[carrier_i, "max_relative_growth"]
    ).clip(min=0)

    if carrier_i.empty:
        return

    lhs = []
    for c, attr in nominal_attrs.items():
        var = f"{c}-{attr}"
        dim = f"{c}-ext"
        df = n.df(c)

        if "carrier" not in df:
            continue

        limited_i = (
            df.index[df.carrier.isin(carrier_i)]
            .intersection(n.get_extendable_i(c))
            .rename(dim)
        )
        if limited_i.empty:
            continue

        active = pd.concat({p: n.get_active_assets(c, p) for p in periods}, axis=1)
        active = active.loc[limited_i].rename_axis(columns="periods").T
        first_active = DataArray(active.cumsum() == 1)
        carriers = df.loc[limited_i, "carrier"].rename("Carrier")

        vars = m[var].sel({dim: limited_i}).where(first_active)
        expr = vars.groupby(carriers.to_xarray()).sum()

        if (max_relative_growth.loc[carriers.unique()] > 0).any():
            expr = expr - expr.shift(periods=1) * max_relative_growth

        lhs.append(expr)

    if not lhs:
        return

    lhs = merge(lhs)
    rhs = max_absolute_growth.reindex_like(lhs.data)

    m.add_constraints(lhs, "<=", rhs, name="Carrier-growth_limit")


def define_primary_energy_limit(n, sns):
    """
    Defines primary energy constraints. It limits the byproducts of primary
    energy sources (defined by carriers) such as CO2.

    Parameters
    ----------
    n : pypsa.Network
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    Returns
    -------
    None.
    """
    m = n.model
    weightings = n.snapshot_weightings.loc[sns]
    glcs = n.global_constraints.query('type == "primary_energy"')

    if n._multi_invest:
        period_weighting = n.investment_period_weightings.years[sns.unique("period")]
        weightings = weightings.mul(period_weighting, level=0, axis=0)

    for name, glc in glcs.iterrows():
        if isnan(glc.investment_period):
            sns_sel = slice(None)
        elif glc.investment_period in sns.unique("period"):
            sns_sel = sns.get_loc(glc.investment_period)
        else:
            continue

        lhs = []
        rhs = glc.constant
        emissions = n.carriers[glc.carrier_attribute][lambda ds: ds != 0]

        if emissions.empty:
            continue

        # generators
        gens = n.generators.query("carrier in @emissions.index")
        if not gens.empty:
            efficiency = get_as_dense(
                n, "Generator", "efficiency", snapshots=sns[sns_sel], inds=gens.index
            )
            em_pu = gens.carrier.map(emissions) / efficiency
            em_pu = em_pu.multiply(weightings.generators[sns_sel], axis=0)
            p = m["Generator-p"].loc[sns[sns_sel], gens.index]
            expr = (p * em_pu).sum()
            lhs.append(expr)

        # storage units
        cond = "carrier in @emissions.index and not cyclic_state_of_charge"
        sus = n.storage_units.query(cond)
        if not sus.empty:
            em_pu = sus.carrier.map(emissions)
            sus_i = sus.index
            soc = m["StorageUnit-state_of_charge"].loc[sns[sns_sel], sus_i]
            soc = soc.ffill("snapshot").isel(snapshot=-1)
            lhs.append(m.linexpr((-em_pu, soc)).sum())
            rhs -= em_pu @ sus.state_of_charge_initial

        # stores
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            em_pu = stores.carrier.map(emissions)
            e = m["Store-e"].loc[sns[sns_sel], stores.index]
            e = e.ffill("snapshot").isel(snapshot=-1)
            lhs.append(m.linexpr((-em_pu, e)).sum())
            rhs -= em_pu @ stores.e_initial

        if not lhs:
            continue

        lhs = merge(lhs)
        sign = "=" if glc.sense == "==" else glc.sense
        m.add_constraints(lhs, sign, rhs, name=f"GlobalConstraint-{name}")


def define_operational_limit(n, sns):
    """
    Defines operational limit constraints. It limits the net production of a
    carrier taking into account generator, storage units and stores.

    Parameters
    ----------
    n : pypsa.Network
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    Returns
    -------
    None.
    """
    m = n.model
    weightings = n.snapshot_weightings.loc[sns]
    glcs = n.global_constraints.query('type == "operational_limit"')

    if n._multi_invest:
        period_weighting = n.investment_period_weightings.years[sns.unique("period")]
        weightings = weightings.mul(period_weighting, level=0, axis=0)

    # storage units
    cond = "carrier == @glc.carrier_attribute and not cyclic_state_of_charge"
    for name, glc in glcs.iterrows():
        snapshots = (
            sns
            if isnan(glc.investment_period)
            else sns[sns.get_loc(glc.investment_period)]
        )
        lhs = []
        rhs = glc.constant

        # generators
        gens = n.generators.query("carrier == @glc.carrier_attribute")
        if not gens.empty:
            p = m["Generator-p"].loc[snapshots, gens.index]
            w = DataArray(weightings.generators[snapshots])
            if "dim_0" in w.dims:
                w = w.rename({"dim_0": "snapshot"})
            expr = (p * w).sum()
            lhs.append(expr)

        sus = n.storage_units.query(cond)
        if not sus.empty:
            sus_i = sus.index
            soc = m["StorageUnit-state_of_charge"].loc[snapshots, sus_i]
            soc = soc.ffill("snapshot").isel(snapshot=-1)
            lhs.append(-1 * soc.sum())
            rhs -= sus.state_of_charge_initial.sum()

        # stores
        bus_carrier = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query(
            "@bus_carrier == @glc.carrier_attribute and not e_cyclic"
        )
        if not stores.empty:
            e = m["Store-e"].loc[snapshots, stores.index]
            e = e.ffill("snapshot").isel(snapshot=-1)
            lhs.append(-e.sum())
            rhs -= stores.e_initial.sum()

        if not lhs:
            continue

        lhs = merge(lhs)
        sign = "=" if glc.sense == "==" else glc.sense
        m.add_constraints(lhs, sign, rhs, name=f"GlobalConstraint-{name}")


def define_transmission_volume_expansion_limit(n, sns):
    """
    Set a limit for line volume expansion. For the capacity expansion only the
    carriers 'AC' and 'DC' are considered.

    Parameters
    ----------
    n : pypsa.Network
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    Returns
    -------
    None.
    """
    m = n.model
    glcs = n.global_constraints.query("type == 'transmission_volume_expansion_limit'")

    def substr(s):
        return re.sub("[\\[\\]\\(\\)]", "", s)

    for name, glc in glcs.iterrows():
        lhs = []
        car = [substr(c.strip()) for c in glc.carrier_attribute.split(",")]
        period = glc.investment_period

        for c in ["Line", "Link"]:
            attr = nominal_attrs[c]

            ext_i = n.get_extendable_i(c)
            if ext_i.empty:
                continue

            ext_i = ext_i.intersection(n.df(c).query("carrier in @car").index).rename(
                ext_i.name
            )

            if ext_i.empty:
                continue

            if not isnan(period):
                ext_i = ext_i[n.get_active_assets(c, period)[ext_i]].rename(ext_i.name)
            elif isinstance(sns, pd.MultiIndex):
                ext_i = ext_i[
                    n.get_active_assets(c, sns.unique("period"))[ext_i]
                ].rename(ext_i.name)

            length = n.df(c).length.reindex(ext_i)
            vars = m[f"{c}-{attr}"].loc[ext_i]
            lhs.append(m.linexpr((length, vars)).sum())

        if not lhs:
            continue

        lhs = merge(lhs)
        sign = "=" if glc.sense == "==" else glc.sense
        m.add_constraints(lhs, sign, glc.constant, name=f"GlobalConstraint-{name}")


def define_transmission_expansion_cost_limit(n, sns):
    """
    Set a limit for line expansion costs. For the capacity expansion only the
    carriers 'AC' and 'DC' are considered.

    Parameters
    ----------
    n : pypsa.Network
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    Returns
    -------
    None.
    """
    m = n.model
    glcs = n.global_constraints.query("type == 'transmission_expansion_cost_limit'")

    def substr(s):
        return re.sub("[\\[\\]\\(\\)]", "", s)

    for name, glc in glcs.iterrows():
        lhs = []
        car = [substr(c.strip()) for c in glc.carrier_attribute.split(",")]
        period = glc.investment_period

        for c in ["Line", "Link"]:
            attr = nominal_attrs[c]

            ext_i = n.get_extendable_i(c)
            if ext_i.empty:
                continue

            ext_i = ext_i.intersection(n.df(c).query("carrier in @car").index).rename(
                ext_i.name
            )

            if not isnan(period):
                ext_i = ext_i[n.get_active_assets(c, period)[ext_i]].rename(ext_i.name)
            elif isinstance(sns, pd.MultiIndex):
                ext_i = ext_i[
                    n.get_active_assets(c, sns.unique("period"))[ext_i]
                ].rename(ext_i.name)

            cost = n.df(c).capital_cost.reindex(ext_i)
            vars = m[f"{c}-{attr}"].loc[ext_i]
            lhs.append(m.linexpr((cost, vars)).sum())

        if not lhs:
            continue

        lhs = merge(lhs)
        sign = "=" if glc.sense == "==" else glc.sense
        m.add_constraints(lhs, sign, glc.constant, name=f"GlobalConstraint-{name}")
