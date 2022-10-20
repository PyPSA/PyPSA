#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define global constraints for optimisation problems with Linopy.
"""

import logging
import re

import pandas as pd
from linopy.expressions import merge
from numpy import isnan, nan
from xarray import DataArray

from pypsa.descriptors import nominal_attrs

logger = logging.getLogger(__name__)


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

    for col in cols:
        msg = (
            f"Bus column '{col}' has invalid specifaction and cannot be "
            "interpreted as constraint, must match the pattern "
            "`nom_{min/max}_{carrier}` or `nom_{min/max}_{carrier}_{period}`"
        )
        if col.startswith("nom_min_"):
            sign = ">="
        elif col.startswith("nom_max_"):
            sign = "<="
        else:
            logger.warn(msg)
        remainder = col[len("nom_max_") :]
        if remainder in n.carriers.index:
            carrier = remainder
            period = None
        elif not isinstance(n.snapshots, pd.MultiIndex):
            logger.warn(msg)
            continue
        else:
            carrier, period = remainder.rsplit("_", 1)
            if carrier not in n.carriers.index or period not in sns.unique("period"):
                logger.warn(msg)
                continue

        rhs = n.buses[col]
        lhs = []

        for c, attr in nominal_attrs.items():
            var = f"{c}-{attr}"
            dim = f"{c}-ext"
            ext_i = n.get_extendable_i(c)

            if c not in n.one_port_components or ext_i.empty:
                continue

            if period is not None:
                ext_i = ext_i[n.get_active_assets(c, 1)[ext_i]]

            buses = n.df(c)["bus"][ext_i].rename("Bus").rename_axis(dim).to_xarray()
            expr = m[var].loc[ext_i].groupby_sum(buses)
            lhs.append(expr)

        if not lhs:
            continue

        lhs = merge(lhs)
        rhs = rhs[lhs.Bus.data]
        mask = rhs.notnull()
        n.model.add_constraints(lhs, sign, rhs, f"Bus-{col}", mask=mask)


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
    carrier_i = n.carriers.query("max_growth != inf").index

    if carrier_i.empty:
        return

    lhs = []
    for c, attr in nominal_attrs.items():
        var = f"{c}-{attr}"
        dim = f"{c}-ext"

        if "carrier" not in n.df(c):
            continue

        limited_i = n.df(c).query("carrier in @carrier_i").index
        limited_i = limited_i.intersection(n.get_extendable_i(c)).rename(dim)

        if limited_i.empty:
            continue

        active = pd.concat({p: n.get_active_assets(c, p) for p in periods}, axis=1)
        active = active.reindex(limited_i).rename_axis(columns="periods").T
        first_active = DataArray(active.cumsum() == 1)
        assets = n.df(c).reindex(limited_i)

        carriers = assets.carrier.to_xarray().rename("Carrier")
        vars = m[var].sel({dim: limited_i}).where(first_active)
        expr = vars.groupby_sum(carriers)
        lhs.append(expr)

    if not lhs:
        return

    lhs = merge(lhs)
    rhs = n.carriers.max_growth[carrier_i].rename_axis("Carrier")

    m.add_constraints(lhs, "<=", rhs, "Carrier-growth_limit_{}".format(c))


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
            snapshots = sns
        else:
            snapshots = sns[sns.get_loc(glc.investment_period)]

        lhs = []
        rhs = glc.constant
        emissions = n.carriers[glc.carrier_attribute][lambda ds: ds != 0]

        if emissions.empty:
            continue

        # generators
        gens = n.generators.query("carrier in @emissions.index")
        if not gens.empty:
            w = weightings["generators"].to_frame("weight")
            em_pu = (gens.carrier.map(emissions) / gens.efficiency).to_frame("weight")
            em_pu = w @ em_pu.T
            p = m["Generator-p"].loc[snapshots, gens.index]
            expr = (p * em_pu).sum()
            lhs.append(expr)

        # storage units
        cond = "carrier in @emissions.index and not cyclic_state_of_charge"
        sus = n.storage_units.query(cond)
        sus_i = sus.index
        if not sus.empty:
            em_pu = sus.carrier.map(emissions)
            soc = m["StorageUnit-state_of_charge"].loc[snapshots, sus_i]
            soc = soc.where(soc != -1, nan).ffill("snapshot").isel(snapshot=-1)
            lhs.append(m.linexpr((-em_pu, soc)).sum())
            rhs -= em_pu @ sus.state_of_charge_initial

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            em_pu = stores.carrier.map(emissions)
            e = m["Store-e"].loc[snapshots, stores.index]
            e = e.where(e != -1, nan).ffill("snapshot").isel(snapshot=-1)
            lhs.append(m.linexpr((-em_pu, e)).sum())
            rhs -= em_pu @ stores.e_initial

        if not lhs:
            continue

        lhs = merge(lhs)
        sign = "=" if glc.sense == "==" else glc.sense
        m.add_constraints(lhs, sign, rhs, f"GlobalConstraint-{name}")


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

    substr = lambda s: re.sub(r"[\[\]\(\)]", "", s)

    lhs = []
    for name, glc in glcs.iterrows():
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
                ext_i = ext_i[n.get_active_assets(c, period)].rename(ext_i.name)
            elif isinstance(sns, pd.MultiIndex):
                ext_i = ext_i[n.get_active_assets(c, sns.unique("period"))].rename(
                    ext_i.name
                )

            length = n.df(c).length.reindex(ext_i)
            vars = m[f"{c}-{attr}"].loc[ext_i]
            lhs.append(m.linexpr((length, vars)).sum())

        if not lhs:
            continue

        lhs = merge(lhs)
        sign = "=" if glc.sense == "==" else glc.sense
        m.add_constraints(lhs, sign, glc.constant, f"GlobalConstraint-{name}")


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

    substr = lambda s: re.sub(r"[\[\]\(\)]", "", s)

    lhs = []
    for name, glc in glcs.iterrows():
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
                ext_i = ext_i[n.get_active_assets(c, period)].rename(ext_i.name)
            elif isinstance(sns, pd.MultiIndex):
                ext_i = ext_i[n.get_active_assets(c, sns.unique("period"))].rename(
                    ext_i.name
                )

            cost = n.df(c).capital_cost.reindex(ext_i)
            vars = m[f"{c}-{attr}"].loc[ext_i]
            lhs.append(m.linexpr((cost, vars)).sum())

        if not lhs:
            continue

        lhs = merge(lhs)
        sign = "=" if glc.sense == "==" else glc.sense
        m.add_constraints(lhs, sign, glc.constant, f"GlobalConstraint-{name}")
