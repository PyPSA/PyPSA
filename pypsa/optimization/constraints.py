#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:30:57 2021

@author: fabian
"""
import logging
import pandas as pd
from linopy import LinearExpression
from numpy import inf, nan, roll, cumsum
from xarray import DataArray, concat

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


def define_operational_constraints_for_non_extendables(n, sns, c, attr):
    """
    Sets power dispatch constraints for non-extendable and non-commitable
    assets for a given component and a given attribute.

    Parameters
    ----------
    n : pypsa.Network
    sns : pd.Index
        Snapshots of the constraint.
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'

    """
    fix_i = n.get_non_extendable_i(c)
    fix_i = fix_i.difference(n.get_committable_i(c)).rename(fix_i.name)

    if fix_i.empty:
        return

    nominal_fix = n.df(c)[nominal_attrs[c]].reindex(fix_i)
    min_pu, max_pu = get_bounds_pu(n, c, sns, fix_i, attr)
    lower = min_pu.mul(nominal_fix)
    upper = max_pu.mul(nominal_fix)

    active = get_activity_mask(n, c, sns, fix_i) if n._multi_invest else None

    dispatch = reindex(n.model[f"{c}-{attr}"], c, fix_i)
    n.model.add_constraints(dispatch, ">=", lower, f"{c}-fix-{attr}-lower", active)
    n.model.add_constraints(dispatch, "<=", upper, f"{c}-fix-{attr}-upper", active)


def define_operational_constraints_for_extendables(n, sns, c, attr):
    """
    Sets power dispatch constraints for extendable devices for a given
    component and a given attribute.

    Parameters
    ----------
    n : pypsa.Network
    sns : pd.Index
        Snapshots of the constraint.
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'

    """
    ext_i = n.get_extendable_i(c)

    if ext_i.empty:
        return

    min_pu, max_pu = map(DataArray, get_bounds_pu(n, c, sns, ext_i, attr))
    dispatch = reindex(n.model[f"{c}-{attr}"], c, ext_i)
    capacity = n.model[f"{c}-{nominal_attrs[c]}"]
    rhs = 0

    active = get_activity_mask(n, c, sns, ext_i) if n._multi_invest else None

    lhs = (max_pu, capacity), (-1, dispatch)
    n.model.add_constraints(lhs, ">=", rhs, f"{c}-ext-{attr}-upper", active)

    lhs = (min_pu, capacity), (-1, dispatch)
    n.model.add_constraints(lhs, "<=", rhs, f"{c}-ext-{attr}-lower", active)


def define_operational_constraints_for_committables(n, sns, c):
    """
    Sets power dispatch constraints for commitable devices for a given
    component and a given attribute.

    Parameters
    ----------
    n : pypsa.Network
    sns : pd.Index
        Snapshots of the constraint.
    c : str
        name of the network component

    """
    com_i = n.get_committable_i(c)

    if com_i.empty:
        return

    nominal = DataArray(n.df(c)[nominal_attrs[c]].reindex(com_i))
    min_pu, max_pu = map(DataArray, get_bounds_pu(n, c, sns, com_i, "p"))
    lower = min_pu * nominal
    upper = max_pu * nominal

    status = n.model[f"{c}-status"]
    p = reindex(n.model[f"{c}-p"], c, com_i)
    active = get_activity_mask(n, c, sns, com_i) if n._multi_invest else None

    lhs = (lower, status), (-1, p)
    n.model.add_constraints(lhs, "<=", 0, f"{c}-com-p-lower", active)

    lhs = (upper, status), (-1, p)
    n.model.add_constraints(lhs, ">=", 0, f"{c}-com-p-upper", active)


def define_nominal_constraints_for_extendables(n, c, attr):
    """
    Sets capacity expansion constraints for extendable
    assets for a given component and a given attribute.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'

    """
    ext_i = n.get_extendable_i(c)

    if ext_i.empty:
        return

    capacity = n.model[f"{c}-{attr}"]
    lower = n.df(c)[attr + "_min"].reindex(ext_i)
    upper = n.df(c)[attr + "_max"].reindex(ext_i)
    n.model.add_constraints(capacity, ">=", lower, f"{c}-fix-{attr}-lower")
    n.model.add_constraints(capacity, "<=", upper, f"{c}-fix-{attr}-upper")


def define_ramp_limit_constraints(n, sns, c):
    """
    Defines ramp limits for assets with valid ramplimit.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component

    """
    m = n.model

    if {"ramp_limit_up", "ramp_limit_down"}.isdisjoint(n.df(c)):
        return
    if n.df(c)[["ramp_limit_up", "ramp_limit_down"]].isnull().all().all():
        return

    p = lambda idx: reindex(m[f"{c}-p"].sel(snapshot=sns[1:]), c, idx)
    p_prev = lambda idx: reindex(
        m[f"{c}-p"].shift(snapshot=1).sel(snapshot=sns[1:]), c, idx
    )

    fix_i = n.get_non_extendable_i(c)
    assets = n.df(c).reindex(fix_i)
    active = get_activity_mask(n, c, sns[1:], fix_i)

    # fix up
    if not assets.ramp_limit_up.isnull().all():
        lhs = p(fix_i) - p_prev(fix_i)
        rhs = assets.eval("ramp_limit_up * p_nom")
        m.add_constraints(lhs, "<=", rhs, f"{c}-fix-p-ramp_limit_up", active)

    # fix down
    if not assets.ramp_limit_down.isnull().all():
        lhs = p(fix_i) - p_prev(fix_i)
        rhs = assets.eval("-1 * ramp_limit_down * p_nom")
        m.add_constraints(lhs, ">=", rhs, f"{c}-fix-p-ramp_limit_down", active)

    ext_i = n.get_extendable_i(c)
    assets = n.df(c).reindex(ext_i)
    active = get_activity_mask(n, c, sns[1:], ext_i)

    # ext up
    if not assets.ramp_limit_up.isnull().all():
        p_nom = m[f"{c}-p_nom"]
        limit_pu = assets.ramp_limit_up.to_xarray()
        lhs = (1, p(ext_i)), (-1, p_prev(ext_i)), (-limit_pu, p_nom)
        m.add_constraints(lhs, "<=", 0, f"{c}-ext-p-ramp_limit_up", active)

    # ext down
    if not assets.ramp_limit_down.isnull().all():
        p_nom = m[f"{c}-p_nom"]
        limit_pu = assets.ramp_limit_down.to_xarray()
        lhs = (1, p(ext_i)), (-1, p_prev(ext_i)), (limit_pu, p_nom)
        m.add_constraints(lhs, ">=", 0, f"{c}-ext-p-ramp_limit_down", active)

    com_i = n.get_committable_i(c)
    assets = n.df(c).reindex(com_i)
    active = get_activity_mask(n, c, sns[1:], com_i)

    # com up
    if not assets.ramp_limit_up.isnull().all():
        limit_start = assets.eval("ramp_limit_start_up * p_nom").to_xarray()
        limit_up = assets.eval("ramp_limit_up * p_nom").to_xarray()

        status = m[f"{c}-status"].sel(snapshot=sns[1:])
        status_prev = m[f"{c}-status"].shift(snapshot=1).sel(snapshot=sns[1:])

        lhs = (
            (1, p(com_i)),
            (-1, p_prev(com_i)),
            (limit_start - limit_up, status_prev),
            (-limit_start, status),
        )
        m.add_constraints(lhs, "<=", 0, f"{c}-com-p-ramp_limit_down", active)

    # com down
    if not assets.ramp_limit_down.isnull().all():
        limit_shut = assets.eval("ramp_limit_shut_down * p_nom").to_xarray()
        limit_down = assets.eval("ramp_limit_down * p_nom").to_xarray()

        status = m[f"{c}-status"].sel(snapshot=sns[1:])
        status_prev = m[f"{c}-status"].shift(snapshot=1).sel(snapshot=sns[1:])

        lhs = (
            (1, p(com_i)),
            (-1, p_prev(com_i)),
            (limit_down - limit_shut, status),
            (limit_shut, status_prev),
        )
        m.add_constraints(lhs, ">=", 0, f"{c}-com-p-ramp_limit_up", active)


def define_nodal_balance_constraints(n, sns):
    """
    Defines nodal balance constraints.

    """
    m = n.model

    args = [
        ["Generator", "p", "bus", 1],
        ["Store", "p", "bus", 1],
        ["StorageUnit", "p_dispatch", "bus", 1],
        ["StorageUnit", "p_store", "bus", -1],
        ["Line", "s", "bus0", -1],
        ["Line", "s", "bus1", 1],
        ["Transformer", "s", "bus0", -1],
        ["Transformer", "s", "bus1", 1],
        ["Link", "p", "bus0", -1],
        ["Link", "p", "bus1", get_as_dense(n, "Link", "efficiency", sns)],
    ]

    for i in additional_linkports(n):
        eff = get_as_dense(n, "Link", f"efficiency{i}", sns)
        args.append(["Link", "p", f"bus{i}", eff])

    exprs = []

    for arg in args:
        c, attr, column, sign = arg

        if n.df(c).empty:
            continue

        if "sign" in n.df(c):
            # additional sign only necessary for branches in reverse direction
            sign = sign * n.df(c).sign

        # TODO: drop empty bus2, bus3 if multiline link

        expr = DataArray(sign) * m[f"{c}-{attr}"]
        expr = expr.group_terms(n.df(c)[column].rename("Bus").to_xarray())
        exprs.append(expr)

    if not len(exprs):
        raise ValueError("Empty LHS in nodal balance constraint.")

    # a bit faster than sum
    fill_value = LinearExpression.fill_value
    lhs = LinearExpression(concat(exprs, "_term", fill_value=fill_value))
    sense = "="
    rhs = (
        (-get_as_dense(n, "Load", "p_set", sns) * n.loads.sign)
        .groupby(n.loads.bus, axis=1)
        .sum()
        .reindex(columns=n.buses.index, fill_value=0)
    )
    rhs.index.name = "snapshot"  # the name for multi-index is getting lost by groupby
    n.model.add_constraints(lhs, sense, rhs, "Bus-nodal_balance")


def define_kirchhoff_constraints(n, sns):
    """
    Defines Kirchhoff voltage constraints

    """
    m = n.model
    n.calculate_dependent_values()

    comps = [c for c in n.passive_branch_components if not n.df(c).empty]

    if len(comps) == 0:
        return

    lhs = []

    periods = sns.unique("period") if n._multi_invest else [None]

    for period in periods:
        n.determine_network_topology(investment_period=period)

        coeffs = {c: [] for c in comps}

        for sub in n.sub_networks.obj:
            branches = sub.branches()
            C = pd.DataFrame(sub.C.todense(), index=branches.index)

            if C.empty:
                continue

            carrier = n.sub_networks.carrier[sub.name]

            weightings = branches.x_pu_eff if carrier == "AC" else branches.r_pu_eff
            cycles = 1e5 * C.mul(weightings, axis=0).rename_axis(columns="cycle")

            for c in coeffs:
                coeffs[c].append(cycles.loc[c])

        snapshots = sns if period is None else sns[sns.get_loc(period)]

        lhs_period = []

        for c in comps:
            idx = n.df(c).index
            dfs = [df.reindex(idx, fill_value=0) for df in coeffs[c]]
            coeff = pd.concat(dfs, axis=1, ignore_index=True)
            coeff = DataArray(coeff.rename_axis(columns="cycle"))
            s = m[c + "-s"].sel(snapshot=snapshots)
            lhs_period.append((coeff * s).sum(c, drop_zeros=True))

        lhs.append(sum(lhs_period))

    fill_value = LinearExpression.fill_value
    lhs = LinearExpression(concat(lhs, "snapshot", fill_value=fill_value))
    m.add_constraints(lhs, "=", 0, name="Kirchhoff-Voltage-Law")


def define_fixed_nominal_constraints(n, c, attr):
    """
    Sets constraints for fixing static variables of a given component and attribute
    to the corresponding values in `n.df(c)[attr + '_set']`.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'

    """

    if attr + "_set" not in n.df(c):
        return

    dim = f"{c}-{attr}_set_i"
    fix = n.df(c)[attr + "_set"].dropna().rename_axis(dim)

    if fix.empty:
        return

    var = n.model[f"{c}-{attr}"]
    var = reindex(var, var.dims[0], fix.index)
    n.model.add_constraints(var, "=", fix, f"{c}-{attr}_set")


def define_fixed_operation_constraints(n, sns, c, attr):
    """
    Sets constraints for fixing time-dependent variables of a given component
    and attribute to the corresponding values in `n.pnl(c)[attr + '_set']`.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'

    """

    if attr + "_set" not in n.pnl(c):
        return

    dim = f"{c}-{attr}_set_i"
    fix = n.pnl(c)[attr + "_set"].reindex(index=sns).rename_axis(columns=dim)
    fix.index.name = "snapshot"  # still necessary: reindex looses the index name

    if fix.empty:
        return

    if n._multi_invest:
        active = get_activity_mask(n, c, sns, index=fix.columns)
    else:
        active = None

    var = reindex(n.model[f"{c}-{attr}"], c, fix.columns)
    n.model.add_constraints(var, "=", fix, f"{c}-{attr}_set", active)


def define_storage_unit_constraints(n, sns):
    """
    Defines energy balance constraints for storage units. In principal
    the constraints states:

        previous_soc + p_store - p_dispatch + inflow - spill == soc

    """

    m = n.model
    c = "StorageUnit"
    dim = "snapshot"
    assets = n.df(c)
    active = DataArray(get_activity_mask(n, c, sns))

    if assets.empty:
        return

    # elapsed hours
    eh = expand_series(n.snapshot_weightings.stores[sns], assets.index)
    # efficiencies
    eff_stand = expand_series(1 - assets.standing_loss, sns).T.pow(eh)
    eff_dispatch = expand_series(assets.efficiency_dispatch, sns).T
    eff_store = expand_series(assets.efficiency_store, sns).T

    soc = m[f"{c}-state_of_charge"]

    lhs = [
        (-1, soc),
        (-1 / eff_dispatch * eh, m[c + "-p_dispatch"]),
        (eff_store * eh, m[c + "-p_store"]),
    ]

    if f"{c}-spill" in m.variables:
        lhs += [(-eh, m[c + "-spill"])]

    # We create a mask `include_previous_soc` which exludes the first snapshot
    # for non-cyclic assets.
    noncyclic_b = ~assets.cyclic_state_of_charge.to_xarray()
    include_previous_soc = (active.cumsum(dim) != 1).where(noncyclic_b, True)

    kwargs = dict(snapshot=1, roll_coords=False)
    previous_soc = soc.where(active).ffill(dim).roll(**kwargs).ffill(dim)
    previous_soc = previous_soc.where(include_previous_soc, -1)

    # We add inflow and initial soc for for noncyclic assets to rhs
    soc_init = assets.state_of_charge_initial.to_xarray()
    rhs = DataArray(-get_as_dense(n, c, "inflow", sns).mul(eh))
    rhs = rhs.where(include_previous_soc, rhs - soc_init)

    if isinstance(sns, pd.MultiIndex):
        # If multi-horizon optimizing, we update the previous_soc and the rhs
        # for all assets which are cyclid/non-cyclid per period.
        periods = soc.period
        per_period = (
            assets.cyclic_state_of_charge_per_period.to_xarray()
            | assets.state_of_charge_initial_per_period.to_xarray()
        )

        # We calculate the previous soc per period while cycling within a period
        kwargs = dict(shortcut=True, shift=1, axis=list(soc.dims).index(dim))
        previous_soc_pp = soc.groupby(periods).map(roll, **kwargs)

        # We create a mask `include_previous_soc_pp` which exludes the first
        # snapshot of each period for non-cyclic assets.
        kwargs = dict(shortcut=True, axis=list(active.dims).index(dim))
        include_previous_soc_pp = active.groupby(periods).map(cumsum, **kwargs) != 1
        include_previous_soc_pp = include_previous_soc_pp.where(noncyclic_b, True)
        previous_soc_pp = previous_soc_pp.where(include_previous_soc_pp, -1)

        # update the previous_soc variables and right hand side
        previous_soc = previous_soc_pp.where(per_period, previous_soc)
        rhs = (rhs - soc_init).where(per_period, rhs)

    lhs += [(eff_stand, previous_soc)]
    m.add_constraints(lhs, "=", rhs, f"{c}-energy-balance", mask=active)


def define_store_constraints(n, sns):
    """
    Defines energy balance constraints for stores. In principal
    the constraints states:

        previous_e - p == e

    """

    m = n.model
    c = "Store"
    dim = "snapshot"
    assets = n.df(c)
    active = DataArray(get_activity_mask(n, c, sns))

    if assets.empty:
        return

    # elapsed hours
    eh = expand_series(n.snapshot_weightings.stores[sns], assets.index)
    # efficiencies
    eff_stand = expand_series(1 - assets.standing_loss, sns).T.pow(eh)

    e = m[f"{c}-e"]
    p = m[c + "-p_dispatch"]

    lhs = [(-1, e), (-eh, p)]

    # We create a mask `include_previous_e` which exludes the first snapshot
    # for non-cyclic assets.
    noncyclic_b = ~assets.cyclic_e.to_xarray()
    include_previous_e = (active.cumsum(dim) != 1).where(noncyclic_b, True)

    kwargs = dict(snapshot=1, roll_coords=False)
    previous_e = e.where(active).ffill(dim).roll(**kwargs).ffill(dim)
    previous_e = previous_e.where(include_previous_e, -1)

    # We add inflow and initial e for for noncyclic assets to rhs
    e_init = assets.e_initial.to_xarray()
    rhs = DataArray(-get_as_dense(n, c, "inflow", sns).mul(eh))
    rhs = rhs.where(include_previous_e, rhs - e_init)

    if isinstance(sns, pd.MultiIndex):
        # If multi-horizon optimizing, we update the previous_e and the rhs
        # for all assets which are cyclid/non-cyclid per period.
        periods = e.period
        per_period = (
            assets.cyclic_e_per_period.to_xarray()
            | assets.e_initial_per_period.to_xarray()
        )

        # We calculate the previous e per period while cycling within a period
        kwargs = dict(shortcut=True, shift=1, axis=list(e.dims).index(dim))
        previous_e_pp = e.groupby(periods).map(roll, **kwargs)

        # We create a mask `include_previous_e_pp` which exludes the first
        # snapshot of each period for non-cyclic assets.
        kwargs = dict(shortcut=True, axis=list(active.dims).index(dim))
        include_previous_e_pp = active.groupby(periods).map(cumsum, **kwargs) != 1
        include_previous_e_pp = include_previous_e_pp.where(noncyclic_b, True)
        previous_e_pp = previous_e_pp.where(include_previous_e_pp, -1)

        # update the previous_e variables and right hand side
        previous_e = previous_e_pp.where(per_period, previous_e)
        rhs = (rhs - e_init).where(per_period, rhs)

    lhs += [(eff_stand, previous_e)]
    m.add_constraints(lhs, "=", rhs, f"{c}-energy-balance", mask=active)
