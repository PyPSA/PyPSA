#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:20:12 2022

@author: fabian
"""

import pytest
import pandas as pd
from pypsa.descriptors import (
    expand_series,
    get_switchable_as_dense as get_as_dense,
    nominal_attrs,
)


TOLERANCE = 1e-2


def describe_storage_unit_contraints(n):
    """
    Checks whether all storage units are balanced over time. This function
    requires the network to contain the separate variables p_store and
    p_dispatch, since they cannot be reconstructed from p. The latter results
    from times tau where p_store(tau) > 0 **and** p_dispatch(tau) > 0, which
    is allowed (even though not economic). Therefor p_store is necessarily
    equal to negative entries of p, vice versa for p_dispatch.
    """
    sus = n.storage_units
    sus_i = sus.index
    if sus_i.empty:
        return
    sns = n.snapshots
    c = "StorageUnit"
    pnl = n.pnl(c)

    description = {}

    eh = expand_series(n.snapshot_weightings.stores, sus_i)
    stand_eff = expand_series(1 - n.df(c).standing_loss, sns).T.pow(eh)
    dispatch_eff = expand_series(n.df(c).efficiency_dispatch, sns).T
    store_eff = expand_series(n.df(c).efficiency_store, sns).T
    inflow = get_as_dense(n, c, "inflow") * eh
    spill = eh[pnl.spill.columns] * pnl.spill

    description["Spillage Limit"] = pd.Series(
        {"min": (inflow[spill.columns] - spill).min().min()}
    )

    if "p_store" in pnl:
        soc = pnl.state_of_charge

        store = store_eff * eh * pnl.p_store  # .clip(upper=0)
        dispatch = 1 / dispatch_eff * eh * pnl.p_dispatch  # (lower=0)
        start = soc.iloc[-1].where(
            sus.cyclic_state_of_charge, sus.state_of_charge_initial
        )
        previous_soc = stand_eff * soc.shift().fillna(start)

        reconstructed = (
            previous_soc.add(store, fill_value=0)
            .add(inflow, fill_value=0)
            .add(-dispatch, fill_value=0)
            .add(-spill, fill_value=0)
        )
        description["SOC Balance StorageUnit"] = (
            (reconstructed - soc).unstack().describe()
        )
    return pd.concat(description, axis=1, sort=False)


def describe_nodal_balance_constraint(n):
    """
    Helper function to double check whether network flow is balanced
    """
    network_injection = (
        pd.concat(
            [
                n.pnl(c)[f"p{inout}"].rename(columns=n.df(c)[f"bus{inout}"])
                for inout in (0, 1)
                for c in ("Line", "Transformer")
            ],
            axis=1,
        )
        .groupby(level=0, axis=1)
        .sum()
    )
    return (
        (n.buses_t.p - network_injection)
        .unstack()
        .describe()
        .to_frame("Nodal Balance Constr.")
    )


def describe_upper_dispatch_constraints(n):
    """
    Recalculates the minimum gap between operational status and nominal capacity
    """
    description = {}
    key = " Upper Limit"
    for c, attr in nominal_attrs.items():
        dispatch_attr = "p0" if c in ["Line", "Transformer", "Link"] else attr[0]
        description[c + key] = pd.Series(
            {
                "min": (
                    n.df(c)[attr + "_opt"] * get_as_dense(n, c, attr[0] + "_max_pu")
                    - n.pnl(c)[dispatch_attr]
                )
                .min()
                .min()
            }
        )
    return pd.concat(description, axis=1)


def describe_lower_dispatch_constraints(n):
    description = {}
    key = " Lower Limit"
    for c, attr in nominal_attrs.items():
        if c in ["Line", "Transformer", "Link"]:
            dispatch_attr = "p0"
            description[c] = pd.Series(
                {
                    "min": (
                        n.df(c)[attr + "_opt"] * get_as_dense(n, c, attr[0] + "_max_pu")
                        + n.pnl(c)[dispatch_attr]
                    )
                    .min()
                    .min()
                }
            )
        else:
            dispatch_attr = attr[0]
            description[c + key] = pd.Series(
                {
                    "min": (
                        -n.df(c)[attr + "_opt"]
                        * get_as_dense(n, c, attr[0] + "_min_pu")
                        + n.pnl(c)[dispatch_attr]
                    )
                    .min()
                    .min()
                }
            )
    return pd.concat(description, axis=1)


def describe_store_contraints(n):
    """
    Checks whether all stores are balanced over time.
    """
    stores = n.stores
    stores_i = stores.index
    if stores_i.empty:
        return
    sns = n.snapshots
    c = "Store"
    pnl = n.pnl(c)

    eh = expand_series(n.snapshot_weightings.stores, stores_i)
    stand_eff = expand_series(1 - n.df(c).standing_loss, sns).T.pow(eh)

    start = pnl.e.iloc[-1].where(stores.e_cyclic, stores.e_initial)
    previous_e = stand_eff * pnl.e.shift().fillna(start)

    return (
        (previous_e - pnl.p - pnl.e).unstack().describe().to_frame("SOC Balance Store")
    )


def describe_cycle_constraints(n):
    weightings = n.lines.x_pu_eff.where(n.lines.carrier == "AC", n.lines.r_pu_eff)

    def cycle_flow(sub):
        C = pd.DataFrame(sub.C.todense(), index=sub.lines_i())
        if C.empty:
            return None
        C_weighted = 1e5 * C.mul(weightings[sub.lines_i()], axis=0)
        return C_weighted.apply(lambda ds: ds @ n.lines_t.p0[ds.index].T)

    return (
        pd.concat([cycle_flow(sub) for sub in n.sub_networks.obj], axis=0)
        .unstack()
        .describe()
        .to_frame("Cycle Constr.")
    )


funcs = (
    [
        describe_cycle_constraints,
        # describe_store_contraints,
        # describe_storage_unit_contraints,
        describe_nodal_balance_constraint,
        describe_lower_dispatch_constraints,
        describe_upper_dispatch_constraints,
    ],
)


@pytest.fixture(scope="module")
def solved_network(ac_dc_network):
    n = ac_dc_network
    n.lopf(pyomo=False)
    n.lines["carrier"] = n.lines.bus0.map(n.buses.carrier)
    return n


@pytest.mark.parametrize("func", *funcs)
def test_tolerance(solved_network, func):
    n = solved_network
    description = func(n).fillna(0)
    for col in description:
        assert abs(description[col]["min"]) < TOLERANCE
        if "max" in description:
            assert description[col]["max"] < TOLERANCE
