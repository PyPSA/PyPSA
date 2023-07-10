#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define optimisation constraints from PyPSA networks with Linopy.
"""
import logging

import pandas as pd
from linopy import LinearExpression, Variable, merge
from numpy import arange, cumsum, inf, isfinite, nan, roll
from scipy import sparse
from xarray import DataArray, Dataset, concat

from pypsa.descriptors import (
    additional_linkports,
    expand_series,
    get_activity_mask,
    get_bounds_pu,
)
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.descriptors import nominal_attrs
from pypsa.optimization.common import reindex
from pypsa.optimization.compat import define_constraints, get_var, linexpr

logger = logging.getLogger(__name__)


def define_operational_constraints_for_non_extendables(
    n, sns, c, attr, transmission_losses
):
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

    dispatch_lower = reindex(n.model[f"{c}-{attr}"], c, fix_i)
    dispatch_upper = reindex(n.model[f"{c}-{attr}"], c, fix_i)
    if c in n.passive_branch_components and transmission_losses:
        loss = reindex(n.model[f"{c}-loss"], c, fix_i)
        dispatch_lower = (1, dispatch_lower), (-1, loss)
        dispatch_upper = (1, dispatch_upper), (1, loss)
    n.model.add_constraints(
        dispatch_lower, ">=", lower, f"{c}-fix-{attr}-lower", active
    )
    n.model.add_constraints(
        dispatch_upper, "<=", upper, f"{c}-fix-{attr}-upper", active
    )


def define_operational_constraints_for_extendables(
    n, sns, c, attr, transmission_losses
):
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

    active = get_activity_mask(n, c, sns, ext_i) if n._multi_invest else None

    lhs_lower = (1, dispatch), (-min_pu, capacity)
    lhs_upper = (1, dispatch), (-max_pu, capacity)
    if c in n.passive_branch_components and transmission_losses:
        loss = reindex(n.model[f"{c}-loss"], c, ext_i)
        lhs_upper += ((1, loss),)
        lhs_lower += ((1, loss),)

    n.model.add_constraints(lhs_lower, ">=", 0, f"{c}-ext-{attr}-lower", active)
    n.model.add_constraints(lhs_upper, "<=", 0, f"{c}-ext-{attr}-upper", active)


def define_operational_constraints_for_committables(n, sns, c):
    """
    Sets power dispatch constraints for committable devices for a given
    component and a given attribute. The linearized approximation of the unit
    commitment problem is inspired by Hua et al. (2017) DOI:
    10.1109/TPWRS.2017.2735026.

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

    # variables
    status = n.model[f"{c}-status"]
    start_up = n.model[f"{c}-start_up"]
    shut_down = n.model[f"{c}-shut_down"]
    status_diff = status - status.shift(snapshot=1)
    p = reindex(n.model[f"{c}-p"], c, com_i)
    active = get_activity_mask(n, c, sns, com_i) if n._multi_invest else None

    # parameters
    nominal = DataArray(n.df(c)[nominal_attrs[c]].reindex(com_i))
    min_pu, max_pu = map(DataArray, get_bounds_pu(n, c, sns, com_i, "p"))
    lower_p = min_pu * nominal
    upper_p = max_pu * nominal
    min_up_time_set = n.df(c).min_up_time[com_i]
    min_down_time_set = n.df(c).min_down_time[com_i]
    ramp_up_limit = nominal * n.df(c).ramp_limit_up[com_i].fillna(1)
    ramp_down_limit = nominal * n.df(c).ramp_limit_down[com_i].fillna(1)
    ramp_start_up = nominal * n.df(c).ramp_limit_start_up[com_i]
    ramp_shut_down = nominal * n.df(c).ramp_limit_shut_down[com_i]
    up_time_before_set = n.df(c)["up_time_before"].reindex(com_i)
    down_time_before_set = n.df(c)["down_time_before"].reindex(com_i)
    initially_up = up_time_before_set.astype(bool)
    initially_down = down_time_before_set.astype(bool)

    # check if there are status calculated/fixed before given sns interval
    if sns[0] != n.snapshots[0]:
        start_i = n.snapshots.get_loc(sns[0])
        # get generators which are online until the first regarded snapshot
        until_start_up = n.pnl(c).status.iloc[:start_i][::-1].reindex(columns=com_i)
        ref = range(1, len(until_start_up) + 1)
        up_time_before = until_start_up[until_start_up.cumsum().eq(ref, axis=0)].sum()
        up_time_before_set = up_time_before.clip(
            upper=min_up_time_set, lower=up_time_before_set
        )
        # get number of snapshots for generators which are offline before the first regarded snapshot
        until_start_down = ~until_start_up.astype(bool)
        ref = range(1, len(until_start_down) + 1)
        down_time_before = until_start_down[
            until_start_down.cumsum().eq(ref, axis=0)
        ].sum()
        down_time_before_set = down_time_before.clip(
            upper=min_down_time_set, lower=down_time_before_set
        )

    # lower dispatch level limit
    lhs = (1, p), (-lower_p, status)
    n.model.add_constraints(lhs, ">=", 0, f"{c}-com-p-lower", mask=active)

    # upper dispatch level limit
    lhs = (1, p), (-upper_p, status)
    n.model.add_constraints(lhs, "<=", 0, f"{c}-com-p-upper", mask=active)

    # state-transition constraint
    rhs = pd.DataFrame(0, sns, com_i)
    rhs.loc[sns[0], initially_up] = -1
    lhs = start_up - status_diff
    n.model.add_constraints(lhs, ">=", rhs, f"{c}-com-transition-start-up", mask=active)

    rhs = pd.DataFrame(0, sns, com_i)
    rhs.loc[sns[0], initially_up] = 1
    lhs = shut_down + status_diff
    n.model.add_constraints(
        lhs, ">=", rhs, f"{c}-com-transition-shut-down", mask=active
    )

    # min up time
    expr = []
    min_up_time_i = com_i[min_up_time_set.astype(bool)]
    if not min_up_time_i.empty:
        for g in min_up_time_i:
            su = start_up.loc[:, g]
            expr.append(su.rolling(snapshot=min_up_time_set[g]).sum())
        lhs = -status.loc[:, min_up_time_i] + merge(expr, dim=com_i.name)
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(lhs, "<=", 0, f"{c}-com-up-time", mask=active)

    # min down time
    expr = []
    min_down_time_i = com_i[min_down_time_set.astype(bool)]
    if not min_down_time_i.empty:
        for g in min_down_time_i:
            su = shut_down.loc[:, g]
            expr.append(su.rolling(snapshot=min_down_time_set[g]).sum())
        lhs = status.loc[:, min_down_time_i] + merge(expr, dim=com_i.name)
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(lhs, "<=", 1, f"{c}-com-down-time", mask=active)

    # up time before
    timesteps = pd.DataFrame([range(1, len(sns) + 1)] * len(com_i), com_i, sns).T
    if initially_up.any():
        must_stay_up = (min_up_time_set - up_time_before_set).clip(lower=0)
        mask = (must_stay_up >= timesteps) & initially_up
        name = f"{c}-com-status-min_up_time_must_stay_up"
        mask = mask & active if active is not None else mask
        n.model.add_constraints(status, "=", 1, name, mask=mask)

    # down time before
    if initially_down.any():
        must_stay_down = (min_down_time_set - down_time_before_set).clip(lower=0)
        mask = (must_stay_down >= timesteps) & initially_down
        name = f"{c}-com-status-min_down_time_must_stay_up"
        mask = mask & active if active is not None else mask
        n.model.add_constraints(status, "=", 0, name, mask=mask)

    # linearized approximation because committable can partly start up and shut down
    if n._linearized_uc:
        # dispatch limit for partly start up/shut down for t-1
        lhs = (
            p.shift(snapshot=1)
            - ramp_shut_down * status.shift(snapshot=1)
            - (upper_p - ramp_shut_down) * (status - start_up)
        )
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(lhs, "<=", 0, f"{c}-com-p-before", active)

        # dispatch limit for partly start up/shut down for t
        lhs = p - upper_p * status + (upper_p - ramp_start_up) * start_up
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(lhs, "<=", 0, f"{c}-com-p-current", active)

        # ramp up if committable is only partly active and some capacity is starting up
        lhs = (
            p
            - p.shift(snapshot=1)
            - (lower_p + ramp_up_limit) * status
            + lower_p * status.shift(snapshot=1)
            + (lower_p + ramp_up_limit - ramp_start_up) * start_up
        )
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(lhs, "<=", 0, f"{c}-com-partly-start-up", active)

        # ramp down if committable is only partly active and some capacity is shutting up
        lhs = (
            p.shift(snapshot=1)
            - p
            - ramp_shut_down * status.shift(snapshot=1)
            + (ramp_shut_down - ramp_down_limit) * status
            - (lower_p + ramp_down_limit - ramp_shut_down) * start_up
        )
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(lhs, "<=", 0, f"{c}-com-partly-shut-down", active)


def define_nominal_constraints_for_extendables(n, c, attr):
    """
    Sets capacity expansion constraints for extendable assets for a given
    component and a given attribute.

    Note: As GLPK does not like inf values on the right-hand-side we as masking these out.

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
    mask = upper != inf
    n.model.add_constraints(capacity, ">=", lower, f"{c}-ext-{attr}-lower")
    n.model.add_constraints(capacity, "<=", upper, f"{c}-ext-{attr}-upper", mask=mask)


def define_ramp_limit_constraints(n, sns, c, attr):
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
    if n.df(c)[["ramp_limit_up", "ramp_limit_down"]].eq(1).all().all():
        return

    # ---------------- Check if ramping is at start of n.snapshots --------------- #

    pnl = n.pnl(c)
    attr = {"p", "p0"}.intersection(pnl).pop()  # dispatch for either one or two ports
    start_i = n.snapshots.get_loc(sns[0]) - 1
    p_start = pnl[attr].iloc[start_i]

    is_rolling_horizon = (sns[0] != n.snapshots[0]) and not p_start.empty
    p = m[f"{c}-p"]

    if is_rolling_horizon:
        active = get_activity_mask(n, c, sns)
        rhs_start = pd.DataFrame(0, index=sns, columns=n.df(c).index)
        rhs_start.loc[sns[0]] = p_start

        def p_actual(idx):
            return reindex(p, c, idx)

        def p_previous(idx):
            return reindex(p, c, idx).shift(snapshot=1)

    else:
        active = get_activity_mask(n, c, sns[1:])
        rhs_start = pd.DataFrame(0, index=sns[1:], columns=n.df(c).index)
        rhs_start.index.name = "snapshot"

        def p_actual(idx):
            return reindex(p, c, idx).sel(snapshot=sns[1:])

        def p_previous(idx):
            return reindex(p, c, idx).shift(snapshot=1).sel(snapshot=sns[1:])

    # ----------------------------- Fixed Generators ----------------------------- #

    fix_i = n.get_non_extendable_i(c)
    assets = n.df(c).reindex(fix_i)

    # fix up
    if not assets.ramp_limit_up.isnull().all():
        lhs = p_actual(fix_i) - p_previous(fix_i)
        rhs = assets.eval("ramp_limit_up * p_nom") + rhs_start.reindex(columns=fix_i)
        mask = active.reindex(columns=fix_i) & assets.ramp_limit_up.notnull()
        m.add_constraints(lhs, "<=", rhs, f"{c}-fix-{attr}-ramp_limit_up", mask=mask)

    # fix down
    if not assets.ramp_limit_down.isnull().all():
        lhs = p_actual(fix_i) - p_previous(fix_i)
        rhs = assets.eval("- ramp_limit_down * p_nom") + rhs_start.reindex(
            columns=fix_i
        )
        mask = active.reindex(columns=fix_i) & assets.ramp_limit_down.notnull()
        m.add_constraints(lhs, ">=", rhs, f"{c}-fix-{attr}-ramp_limit_down", mask=mask)

    # ----------------------------- Extendable Generators ----------------------------- #

    ext_i = n.get_extendable_i(c)
    assets = n.df(c).reindex(ext_i)

    # ext up
    if not assets.ramp_limit_up.isnull().all():
        p_nom = m[f"{c}-p_nom"]
        limit_pu = assets.ramp_limit_up.to_xarray()
        lhs = p_actual(ext_i) - p_previous(ext_i) - limit_pu * p_nom
        rhs = rhs_start.reindex(columns=ext_i)
        mask = active.reindex(columns=ext_i) & assets.ramp_limit_up.notnull()
        m.add_constraints(lhs, "<=", rhs, f"{c}-ext-{attr}-ramp_limit_up", mask=mask)

    # ext down
    if not assets.ramp_limit_down.isnull().all():
        p_nom = m[f"{c}-p_nom"]
        limit_pu = assets.ramp_limit_down.to_xarray()
        lhs = (1, p_actual(ext_i)), (-1, p_previous(ext_i)), (limit_pu, p_nom)
        rhs = rhs_start.reindex(columns=ext_i)
        mask = active.reindex(columns=ext_i) & assets.ramp_limit_down.notnull()
        m.add_constraints(lhs, ">=", rhs, f"{c}-ext-{attr}-ramp_limit_down", mask=mask)

    # ----------------------------- Committable Generators ----------------------------- #

    com_i = n.get_committable_i(c)
    assets = n.df(c).reindex(com_i)

    # com up
    if not assets.ramp_limit_up.isnull().all():
        limit_start = assets.eval("ramp_limit_start_up * p_nom").to_xarray()
        limit_up = assets.eval("ramp_limit_up * p_nom").to_xarray()

        status = m[f"{c}-status"].sel(snapshot=active.index)
        status_prev = m[f"{c}-status"].shift(snapshot=1).sel(snapshot=active.index)

        lhs = (
            (1, p_actual(com_i)),
            (-1, p_previous(com_i)),
            (limit_start - limit_up, status_prev),
            (-limit_start, status),
        )

        rhs = rhs_start.reindex(columns=com_i)
        if is_rolling_horizon:
            status_start = n.pnl(c)["status"][com_i].iloc[start_i]
            rhs.loc[sns[0]] += (limit_up - limit_start) * status_start

        mask = active.reindex(columns=com_i) & assets.ramp_limit_up.notnull()
        m.add_constraints(lhs, "<=", rhs, f"{c}-com-{attr}-ramp_limit_up", mask=mask)

    # com down
    if not assets.ramp_limit_down.isnull().all():
        limit_shut = assets.eval("ramp_limit_shut_down * p_nom").to_xarray()
        limit_down = assets.eval("ramp_limit_down * p_nom").to_xarray()

        status = m[f"{c}-status"].sel(snapshot=active.index)
        status_prev = m[f"{c}-status"].shift(snapshot=1).sel(snapshot=active.index)

        lhs = (
            (1, p_actual(com_i)),
            (-1, p_previous(com_i)),
            (limit_down - limit_shut, status),
            (limit_shut, status_prev),
        )

        rhs = rhs_start.reindex(columns=com_i)
        if is_rolling_horizon:
            status_start = n.pnl(c)["status"][com_i].iloc[start_i]
            rhs.loc[sns[0]] += -limit_shut * status_start

        mask = active.reindex(columns=com_i) & assets.ramp_limit_down.notnull()

        m.add_constraints(lhs, ">=", rhs, f"{c}-com-{attr}-ramp_limit_down", mask=mask)


def define_nodal_balance_constraints(
    n, sns, transmission_losses=0, buses=None, suffix=""
):
    """
    Defines nodal balance constraints.
    """
    m = n.model
    if buses is None:
        buses = n.buses.index

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

    if not n.links.empty:
        for i in additional_linkports(n):
            eff = get_as_dense(n, "Link", f"efficiency{i}", sns)
            args.append(["Link", "p", f"bus{i}", eff])

    if transmission_losses:
        args.extend(
            [
                ["Line", "loss", "bus0", -0.5],
                ["Line", "loss", "bus1", -0.5],
                ["Transformer", "loss", "bus0", -0.5],
                ["Transformer", "loss", "bus1", -0.5],
            ]
        )

    exprs = []

    for arg in args:
        c, attr, column, sign = arg

        if n.df(c).empty:
            continue

        if "sign" in n.df(c):
            # additional sign necessary for branches in reverse direction
            sign = sign * n.df(c).sign

        expr = DataArray(sign) * m[f"{c}-{attr}"]
        cbuses = n.df(c)[column][lambda ds: ds.isin(buses)].rename("Bus")

        #  drop non-existent multiport buses which are ''
        if column in ["bus" + i for i in additional_linkports(n)]:
            cbuses = cbuses[cbuses != ""]

        expr = expr.sel({c: cbuses.index})

        if expr.size:
            exprs.append(expr.groupby(cbuses).sum())

    lhs = merge(exprs, join="outer").reindex(
        Bus=buses, fill_value=LinearExpression.fill_value
    )
    rhs = (
        (-get_as_dense(n, "Load", "p_set", sns) * n.loads.sign)
        .groupby(n.loads.bus, axis=1)
        .sum()
        .reindex(columns=buses, fill_value=0)
    )
    # the name for multi-index is getting lost by groupby before pandas 1.4.0
    # TODO remove once we bump the required pandas version to >= 1.4.0
    rhs.index.name = "snapshot"

    empty_nodal_balance = (lhs.vars == -1).all("_term")
    rhs = DataArray(rhs)
    if empty_nodal_balance.any():
        if (empty_nodal_balance & (rhs != 0)).any().item():
            raise ValueError("Empty LHS with non-zero RHS in nodal balance constraint.")

        mask = ~empty_nodal_balance
    else:
        mask = None

    if suffix:
        lhs = lhs.rename(Bus=f"Bus{suffix}")
        rhs = rhs.rename(Bus=f"Bus{suffix}")
    n.model.add_constraints(lhs, "=", rhs, f"Bus{suffix}-nodal_balance", mask=mask)


def define_kirchhoff_voltage_constraints(n, sns):
    """
    Defines Kirchhoff voltage constraints.
    """
    m = n.model
    n.calculate_dependent_values()

    comps = [c for c in n.passive_branch_components if not n.df(c).empty]

    if len(comps) == 0:
        return

    names = ["component", "name"]
    s = pd.concat({c: m[f"{c}-s"].to_pandas() for c in comps}, axis=1, names=names)

    lhs = []

    periods = sns.unique("period") if n._multi_invest else [None]

    for period in periods:
        n.determine_network_topology(investment_period=period, skip_isolated_buses=True)

        snapshots = sns if period is None else sns[sns.get_loc(period)]

        exprs = []
        for sub in n.sub_networks.obj:
            branches = sub.branches()

            if not sub.C.size:
                continue

            carrier = n.sub_networks.carrier[sub.name]
            weightings = branches.x_pu_eff if carrier == "AC" else branches.r_pu_eff
            C = 1e5 * sparse.diags(weightings) * sub.C
            ssub = s.loc[snapshots, branches.index].values

            ncycles = C.shape[1]

            for j in range(ncycles):
                c = C.getcol(j).tocoo()
                coeffs = DataArray(c.data, dims="_term")
                vars = DataArray(
                    ssub[:, c.row],
                    dims=("snapshot", "_term"),
                    coords={"snapshot": snapshots},
                )
                ds = Dataset({"coeffs": coeffs, "vars": vars})
                exprs.append(LinearExpression(ds, m))

        if len(exprs):
            exprs = merge(exprs, dim="cycles")
            exprs = exprs.assign_coords(cycles=range(len(exprs.data.cycles)))
            lhs.append(exprs)

    if len(lhs):
        lhs = merge(lhs, dim="snapshot")
        m.add_constraints(lhs, "=", 0, name="Kirchhoff-Voltage-Law")


def define_fixed_nominal_constraints(n, c, attr):
    """
    Sets constraints for fixing static variables of a given component and
    attribute to the corresponding values in `n.df(c)[attr + '_set']`.

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
    fix.index.name = "snapshot"  # still necessary: reindex loses the index name

    if fix.empty:
        return

    if n._multi_invest:
        active = get_activity_mask(n, c, sns, index=fix.columns)
        mask = fix.notna() & active
    else:
        active = None
        mask = fix.notna()

    var = reindex(n.model[f"{c}-{attr}"], c, fix.columns)
    n.model.add_constraints(var, "=", fix, f"{c}-{attr}_set", mask=mask)


def define_storage_unit_constraints(n, sns):
    """
    Defines energy balance constraints for storage units. In principal the
    constraints states:

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
    eff_stand = (1 - get_as_dense(n, c, "standing_loss", sns)).pow(eh)
    eff_dispatch = get_as_dense(n, c, "efficiency_dispatch", sns)
    eff_store = get_as_dense(n, c, "efficiency_store", sns)

    soc = m[f"{c}-state_of_charge"]

    lhs = [
        (-1, soc),
        (-1 / eff_dispatch * eh, m[c + "-p_dispatch"]),
        (eff_store * eh, m[c + "-p_store"]),
    ]

    if f"{c}-spill" in m.variables:
        lhs += [(-eh, m[c + "-spill"])]

    # We create a mask `include_previous_soc` which excludes the first snapshot
    # for non-cyclic assets.
    noncyclic_b = ~assets.cyclic_state_of_charge.to_xarray()
    include_previous_soc = (active.cumsum(dim) != 1).where(noncyclic_b, True)

    previous_soc = (
        soc.where(active)
        .ffill(dim)
        .roll(snapshot=1)
        .ffill(dim)
        .where(include_previous_soc)
    )

    # We add inflow and initial soc for noncyclic assets to rhs
    soc_init = assets.state_of_charge_initial.to_xarray()
    rhs = DataArray(-get_as_dense(n, c, "inflow", sns).mul(eh))

    if isinstance(sns, pd.MultiIndex):
        # If multi-horizon optimizing, we update the previous_soc and the rhs
        # for all assets which are cyclid/non-cyclid per period.
        periods = soc.coords["period"]
        per_period = (
            assets.cyclic_state_of_charge_per_period.to_xarray()
            | assets.state_of_charge_initial_per_period.to_xarray()
        )

        # We calculate the previous soc per period while cycling within a period
        # Normally, we should use groupby, but is broken for multi-index
        # see https://github.com/pydata/xarray/issues/6836
        ps = n.investment_periods.rename("period")
        sl = slice(None, None)
        previous_soc_pp = [soc.data.sel(snapshot=(p, sl)).roll(snapshot=1) for p in ps]
        previous_soc_pp = concat(previous_soc_pp, dim="snapshot")

        # We create a mask `include_previous_soc_pp` which excludes the first
        # snapshot of each period for non-cyclic assets.
        include_previous_soc_pp = active & (periods == periods.shift(snapshot=1))
        include_previous_soc_pp = include_previous_soc_pp.where(noncyclic_b, True)
        # We take values still to handle internal xarray multi-index difficulties
        previous_soc_pp = previous_soc_pp.where(
            include_previous_soc_pp.values, Variable.fill_value
        )

        # update the previous_soc variables and right hand side
        previous_soc = previous_soc.where(~per_period, previous_soc_pp)
        include_previous_soc = include_previous_soc_pp.where(
            per_period, include_previous_soc
        )
    lhs += [(eff_stand, previous_soc)]
    rhs = rhs.where(include_previous_soc, rhs - soc_init)
    m.add_constraints(lhs, "=", rhs, f"{c}-energy_balance", mask=active)


def define_store_constraints(n, sns):
    """
    Defines energy balance constraints for stores. In principal the constraints
    states:

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
    eff_stand = (1 - get_as_dense(n, c, "standing_loss", sns)).pow(eh)

    e = m[c + "-e"]
    p = m[c + "-p"]

    lhs = [(-1, e), (-eh, p)]

    # We create a mask `include_previous_e` which excludes the first snapshot
    # for non-cyclic assets.
    noncyclic_b = ~assets.e_cyclic.to_xarray()
    include_previous_e = (active.cumsum(dim) != 1).where(noncyclic_b, True)

    previous_e = (
        e.where(active).ffill(dim).roll(snapshot=1).ffill(dim).where(include_previous_e)
    )

    # We add inflow and initial e for for noncyclic assets to rhs
    e_init = assets.e_initial.to_xarray()

    if isinstance(sns, pd.MultiIndex):
        # If multi-horizon optimizing, we update the previous_e and the rhs
        # for all assets which are cyclid/non-cyclid per period.
        periods = e.coords["period"]
        per_period = (
            assets.e_cyclic_per_period.to_xarray()
            | assets.e_initial_per_period.to_xarray()
        )

        # We calculate the previous e per period while cycling within a period
        # Normally, we should use groupby, but is broken for multi-index
        # see https://github.com/pydata/xarray/issues/6836
        ps = n.investment_periods.rename("period")
        sl = slice(None, None)
        previous_e_pp = [e.data.sel(snapshot=(p, sl)).roll(snapshot=1) for p in ps]
        previous_e_pp = concat(previous_e_pp, dim="snapshot")

        # We create a mask `include_previous_e_pp` which excludes the first
        # snapshot of each period for non-cyclic assets.
        include_previous_e_pp = active & (periods == periods.shift(snapshot=1))
        include_previous_e_pp = include_previous_e_pp.where(noncyclic_b, True)
        # We take values still to handle internal xarray multi-index difficulties
        previous_e_pp = previous_e_pp.where(
            include_previous_e_pp.values, Variable.fill_value
        )

        # update the previous_e variables and right hand side
        previous_e = previous_e.where(~per_period, previous_e_pp)
        include_previous_e = include_previous_e_pp.where(per_period, include_previous_e)

    lhs += [(eff_stand, previous_e)]
    rhs = -e_init.where(~include_previous_e, 0)

    m.add_constraints(lhs, "=", rhs, f"{c}-energy_balance", mask=active)


def define_loss_constraints(n, sns, c, transmission_losses):
    if n.df(c).empty or c not in n.passive_branch_components:
        return

    tangents = transmission_losses
    active = get_activity_mask(n, c, sns) if n._multi_invest else None

    s_max_pu = get_as_dense(n, c, "s_max_pu").loc[sns]

    s_nom_max = n.df(c)["s_nom_max"].where(
        n.df(c)["s_nom_extendable"], n.df(c)["s_nom"]
    )

    assert isfinite(
        s_nom_max
    ).all(), f"Loss approximation requires finite 's_nom_max' for extendable branches:\n {s_nom_max[~isfinite(s_nom_max)]}"

    r_pu_eff = n.df(c)["r_pu_eff"]

    upper_limit = r_pu_eff * (s_max_pu * s_nom_max) ** 2

    loss = n.model[f"{c}-loss"]
    flow = n.model[f"{c}-s"]

    n.model.add_constraints(loss <= upper_limit, name=f"{c}-loss_upper", mask=active)

    for k in range(1, tangents + 1):
        p_k = k / tangents * s_max_pu * s_nom_max
        loss_k = r_pu_eff * p_k**2
        slope_k = 2 * r_pu_eff * p_k
        offset_k = loss_k - slope_k * p_k

        for sign in [-1, 1]:
            lhs = n.model.linexpr((1, loss), (sign * slope_k, flow))

            n.model.add_constraints(
                lhs >= offset_k, name=f"{c}-loss_tangents-{k}-{sign}", mask=active
            )
