#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:31:18 2021

@author: fabian
"""

import logging
import pandas as pd
from numpy import inf, nan, roll, cumsum, isnan
from xarray import DataArray, concat
from linopy import LinearExpression, Variable
from linopy.expressions import merge

from .common import reindex, get_var
from ..descriptors import (
    get_bounds_pu,
    get_activity_mask,
    get_switchable_as_dense as get_as_dense,
    expand_series,
    nominal_attrs,
    additional_linkports,
    Dict,
)

logger = logging.getLogger(__name__)


def define_nominal_constraints_per_bus_carrier(n, sns):
    m = n.model
    cols = n.buses.columns[n.buses.columns.str.startswith("nom_")]

    for col in cols:
        if col.startswith("nom_min"):
            sense = ">="
        elif col.startswith("nom_max"):
            sense = "<="
        else:
            continue

        carrier = col.split("_", 2)[-1]
        if carrier not in n.carriers.index:
            logger.warn(f"Column {col} has non-defined carrier.")

        rhs = n.buses[col]
        exprs = []

        for c, attr in nominal_attrs.items():
            var = f"{c}-{attr}"
            dim = f"{c}-ext"
            ext_i = n.get_extendable_i(c)

            if c not in n.one_port_components or ext_i.empty:
                continue

            buses = n.df(c)["bus"][ext_i].rename("Bus").rename_axis(dim).to_xarray()
            expr = m[var].group_terms(buses)
            exprs.append(expr)

        # a bit faster than sum
        lhs = merge(exprs)
        rhs = rhs[lhs.Bus.data]
        mask = rhs.notnull()
        n.model.add_constraints(lhs, sense, rhs, "Bus-{col}", mask)


def define_growth_limit(n, sns):
    """Constraint new installed capacity per investment period.

    Parameters
    ----------
    n : pypsa.Network
    sns : list-like
        Set of snapshots where the constraint should be applied.
    """
    if not n._multi_invest:
        return

    m = n.model
    periods = sns.unique("period")
    carrier_i = n.carriers.query("max_growth != inf").index

    exprs = []
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
        vars = Variable(m[var].sel({dim: limited_i}).where(first_active, -1))
        expr = vars.group_terms(carriers)
        exprs.append(expr)

    lhs = merge(exprs)
    rhs = n.carriers.max_growth[carrier_i].rename_axis("Carrier")

    m.add_constraints(lhs, "<=", rhs, "Carrier-growth_limit_{}".format(c))


def define_global_constraints(n, sns):
    """
    Defines global constraints for the optimization. Possible types are

        1. primary_energy
            Use this to constraint the byproducts of primary energy sources as
            CO2
        2. transmission_volume_expansion_limit
            Use this to set a limit for line volume expansion. Possible carriers
            are 'AC' and 'DC'
        3. transmission_expansion_cost_limit
            Use this to set a limit for line expansion costs. Possible carriers
            are 'AC' and 'DC'
        4. tech_capacity_expansion_limit
            Use this to se a limit for the summed capacitiy of a carrier (e.g.
            'onwind') for each investment period at choosen nodes. This limit
            could e.g. represent land resource/ building restrictions for a
            technology in a certain region. Currently, only the
            capacities of extendable generators have to be below the set limit.

    """

    if n._multi_invest:
        period_weighting = n.investment_period_weightings["years"]
        weightings = n.snapshot_weightings.mul(period_weighting, level=0, axis=0).loc[
            sns
        ]
    else:
        weightings = n.snapshot_weightings.loc[sns]

    # (1) primary_energy
    glcs = n.global_constraints.query('type == "primary_energy"')
    for name, glc in glcs.iterrows():
        rhs = glc.constant
        lhs = ""
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]
        period = get_period(n, glc, sns)

        if emissions.empty:
            continue

        # generators
        gens = n.generators.query("carrier in @emissions.index")
        if not gens.empty:
            em_pu = gens.carrier.map(emissions) / gens.efficiency
            em_pu = (
                weightings["generators"].to_frame("weightings")
                @ em_pu.to_frame("weightings").T
            ).loc[period]
            p = get_var(n, "Generator", "p").loc[sns, gens.index].loc[period]

            vals = linexpr((em_pu, p), as_pandas=False)
            lhs += join_exprs(vals)

        # storage units
        sus = n.storage_units.query(
            "carrier in @emissions.index and " "not cyclic_state_of_charge"
        )
        sus_i = sus.index
        if not sus.empty:
            em_pu = sus.carrier.map(emissions)
            soc = (
                get_var(n, "StorageUnit", "state_of_charge").loc[sns, sus_i].loc[period]
            )
            soc = soc.where(soc != -1).ffill().iloc[-1]
            vals = linexpr((-em_pu, soc), as_pandas=False)
            lhs = lhs + "\n" + join_exprs(vals)
            rhs -= em_pu @ sus.state_of_charge_initial

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            em_pu = stores.carrier.map(emissions)
            e = get_var(n, "Store", "e").loc[sns, stores.index].loc[period]
            e = e.where(e != -1).ffill().iloc[-1]
            vals = linexpr((-em_pu, e), as_pandas=False)
            lhs = lhs + "\n" + join_exprs(vals)
            rhs -= stores.carrier.map(emissions) @ stores.e_initial

        define_constraints(
            n,
            lhs,
            glc.sense,
            rhs,
            "GlobalConstraint",
            "mu",
            axes=pd.Index([name]),
            spec=name,
        )

    # (2) transmission_volume_expansion_limit
    glcs = n.global_constraints.query(
        "type == " '"transmission_volume_expansion_limit"'
    )
    substr = lambda s: re.sub(r"[\[\]\(\)]", "", s)
    for name, glc in glcs.iterrows():
        car = [substr(c.strip()) for c in glc.carrier_attribute.split(",")]
        lhs = ""
        period = get_period(n, glc, sns)
        for c, attr in (("Line", "s_nom"), ("Link", "p_nom")):
            if n.df(c).empty:
                continue
            ext_i = n.df(c).query(f"carrier in @car and {attr}_extendable").index
            ext_i = ext_i[get_activity_mask(n, c, sns)[ext_i].loc[period].any()]

            if ext_i.empty:
                continue
            v = linexpr(
                (n.df(c).length[ext_i], get_var(n, c, attr)[ext_i]), as_pandas=False
            )
            lhs += "\n" + join_exprs(v)
        if lhs == "":
            continue
        sense = glc.sense
        rhs = glc.constant
        define_constraints(
            n,
            lhs,
            sense,
            rhs,
            "GlobalConstraint",
            "mu",
            axes=pd.Index([name]),
            spec=name,
        )

    # (3) transmission_expansion_cost_limit
    glcs = n.global_constraints.query("type == " '"transmission_expansion_cost_limit"')
    for name, glc in glcs.iterrows():
        car = [substr(c.strip()) for c in glc.carrier_attribute.split(",")]
        lhs = ""
        period = get_period(n, glc, sns)
        for c, attr in (("Line", "s_nom"), ("Link", "p_nom")):
            ext_i = n.df(c).query(f"carrier in @car and {attr}_extendable").index
            ext_i = ext_i[get_activity_mask(n, c, sns)[ext_i].loc[period].any()]

            if ext_i.empty:
                continue

            v = linexpr(
                (n.df(c).capital_cost[ext_i], get_var(n, c, attr)[ext_i]),
                as_pandas=False,
            )
            lhs += "\n" + join_exprs(v)
        if lhs == "":
            continue
        sense = glc.sense
        rhs = glc.constant
        define_constraints(
            n,
            lhs,
            sense,
            rhs,
            "GlobalConstraint",
            "mu",
            axes=pd.Index([name]),
            spec=name,
        )

    # (4) tech_capacity_expansion_limit
    # TODO: Generalize to carrier capacity expansion limit (i.e. also for stores etc.)
    substr = lambda s: re.sub(r"[\[\]\(\)]", "", s)
    glcs = n.global_constraints.query("type == " '"tech_capacity_expansion_limit"')
    c, attr = "Generator", "p_nom"

    for name, glc in glcs.iterrows():
        period = get_period(n, glc, sns)
        car = glc["carrier_attribute"]
        bus = str(glc.get("bus", ""))  # in pypsa buses are always strings
        ext_i = n.df(c).query("carrier == @car and p_nom_extendable").index
        if bus:
            ext_i = n.df(c).loc[ext_i].query("bus == @bus").index
        ext_i = ext_i[get_activity_mask(n, c, sns)[ext_i].loc[period].any()]

        if ext_i.empty:
            continue

        cap_vars = get_var(n, c, attr)[ext_i]

        lhs = join_exprs(linexpr((1, cap_vars)))
        rhs = glc.constant
        sense = glc.sense

        define_constraints(
            n,
            lhs,
            sense,
            rhs,
            "GlobalConstraint",
            "mu",
            axes=pd.Index([name]),
            spec=name,
        )
        rhs = glc.constant
        define_constraints(
            n,
            lhs,
            sense,
            rhs,
            "GlobalConstraint",
            "mu",
            axes=pd.Index([name]),
            spec=name,
        )

    # (4) tech_capacity_expansion_limit
    # TODO: Generalize to carrier capacity expansion limit (i.e. also for stores etc.)
    substr = lambda s: re.sub(r"[\[\]\(\)]", "", s)
    glcs = n.global_constraints.query("type == " '"tech_capacity_expansion_limit"')
    c, attr = "Generator", "p_nom"

    for name, glc in glcs.iterrows():
        period = get_period(n, glc, sns)
        car = glc["carrier_attribute"]
        bus = str(glc.get("bus", ""))  # in pypsa buses are always strings
        ext_i = n.df(c).query("carrier == @car and p_nom_extendable").index
        if bus:
            ext_i = n.df(c).loc[ext_i].query("bus == @bus").index
        ext_i = ext_i[get_activity_mask(n, c, sns)[ext_i].loc[period].any()]

        if ext_i.empty:
            continue

        cap_vars = get_var(n, c, attr)[ext_i]

        lhs = join_exprs(linexpr((1, cap_vars)))
        rhs = glc.constant
        sense = glc.sense

        define_constraints(
            n,
            lhs,
            sense,
            rhs,
            "GlobalConstraint",
            "mu",
            axes=pd.Index([name]),
            spec=name,
        )
