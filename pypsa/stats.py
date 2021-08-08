#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Copyright 2015-2021 PyPSA Developers

## You can find the list of PyPSA Developers at
## https://pypsa.readthedocs.io/en/latest/developers.html

## PyPSA is released under the open source MIT License, see
## https://github.com/PyPSA/PyPSA/blob/master/LICENSE.txt

"""
Post-solving statistics of network. This module contains functions to anaylize
an optimized network. Basic information of network can be summarized as well as
constraint gaps can be double-checked.
"""

__author__ = "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
__copyright__ = ("Copyright 2015-2021 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
                 "MIT License")

from .descriptors import (expand_series, get_switchable_as_dense as get_as_dense,
                          nominal_attrs)
import pandas as pd
import logging

idx = pd.IndexSlice


# =============================================================================
# Network summary
# =============================================================================

opt_name = {"Store": "e", "Line" : "s", "Transformer" : "s"}

def calculate_costs(n):
    raise NotImplementedError
    mc = {}
    for c in n.iterate_comonents():
        if 'marginal_cost' in c.df:

            mc[c] = c.df @ c.pnl['p']


def calculate_curtailment(n):
    max_pu = n.generators_t.p_max_pu
    avail = (max_pu.multiply(n.generators.p_nom_opt.loc[max_pu.columns]).sum()
             .groupby(n.generators.carrier).sum())
    used = (n.generators_t.p[max_pu.columns].sum()
            .groupby(n.generators.carrier).sum())
    return (((avail - used)/avail)*100).round(3)


# and others from pypsa-eur


# =============================================================================
# gap analysis
# =============================================================================


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
    if sus_i.empty: return
    sns = n.snapshots
    c = 'StorageUnit'
    pnl = n.pnl(c)

    description = {}

    eh = expand_series(n.snapshot_weightings.stores, sus_i)
    stand_eff = expand_series(1-n.df(c).standing_loss, sns).T.pow(eh)
    dispatch_eff = expand_series(n.df(c).efficiency_dispatch, sns).T
    store_eff = expand_series(n.df(c).efficiency_store, sns).T
    inflow = get_as_dense(n, c, 'inflow') * eh
    spill = eh[pnl.spill.columns] * pnl.spill

    description['Spillage Limit'] = pd.Series({'min':
                                (inflow[spill.columns] - spill).min().min()})

    if 'p_store' in pnl:
        soc = pnl.state_of_charge

        store = store_eff * eh * pnl.p_store#.clip(upper=0)
        dispatch = 1/dispatch_eff * eh * pnl.p_dispatch#(lower=0)
        start = soc.iloc[-1].where(sus.cyclic_state_of_charge,
                                   sus.state_of_charge_initial)
        previous_soc = stand_eff * soc.shift().fillna(start)


        reconstructed = (previous_soc.add(store, fill_value=0)
                        .add(inflow, fill_value=0)
                        .add(-dispatch, fill_value=0)
                        .add(-spill, fill_value=0))
        description['SOC Balance StorageUnit'] = ((reconstructed - soc)
                                                  .unstack().describe())
    else:
        logging.info('Storage Unit SOC balance not reconstructable as no '
                     'p_store and p_dispatch in n.storage_units_t.')
    return pd.concat(description, axis=1, sort=False)


def describe_nodal_balance_constraint(n):
    """
    Helper function to double check whether network flow is balanced
    """
    network_injection = pd.concat(
            [n.pnl(c)[f'p{inout}'].rename(columns=n.df(c)[f'bus{inout}'])
            for inout in (0, 1) for c in ('Line', 'Transformer')], axis=1)\
            .groupby(level=0, axis=1).sum()
    return (n.buses_t.p - network_injection).unstack().describe()\
            .to_frame('Nodal Balance Constr.')

def describe_upper_dispatch_constraints(n):
    '''
    Recalculates the minimum gap between operational status and nominal capacity
    '''
    description = {}
    key = ' Upper Limit'
    for c, attr in nominal_attrs.items():
        dispatch_attr = 'p0' if c in ['Line', 'Transformer', 'Link'] else attr[0]
        description[c + key] = pd.Series({'min':
                               (n.df(c)[attr + '_opt'] *
                               get_as_dense(n, c, attr[0] + '_max_pu') -
                               n.pnl(c)[dispatch_attr]).min().min()})
    return pd.concat(description, axis=1)


def describe_lower_dispatch_constraints(n):
    description = {}
    key = ' Lower Limit'
    for c, attr in nominal_attrs.items():
        if c in ['Line', 'Transformer', 'Link']:
            dispatch_attr = 'p0'
            description[c] = pd.Series({'min':
                              (n.df(c)[attr + '_opt'] *
                              get_as_dense(n, c, attr[0] + '_max_pu') +
                              n.pnl(c)[dispatch_attr]).min().min()})
        else:
            dispatch_attr = attr[0]
            description[c + key] = pd.Series({'min':
                                   (-n.df(c)[attr + '_opt'] *
                                   get_as_dense(n, c, attr[0] + '_min_pu') +
                                   n.pnl(c)[dispatch_attr]).min().min()})
    return pd.concat(description, axis=1)


def describe_store_contraints(n):
    """
    Checks whether all stores are balanced over time.
    """
    stores = n.stores
    stores_i = stores.index
    if stores_i.empty: return
    sns = n.snapshots
    c = 'Store'
    pnl = n.pnl(c)

    eh = expand_series(n.snapshot_weightings.stores, stores_i)
    stand_eff = expand_series(1-n.df(c).standing_loss, sns).T.pow(eh)

    start = pnl.e.iloc[-1].where(stores.e_cyclic, stores.e_initial)
    previous_e = stand_eff * pnl.e.shift().fillna(start)

    return (previous_e - pnl.p - pnl.e).unstack().describe()\
            .to_frame('SOC Balance Store')


def describe_cycle_constraints(n):
    weightings = n.lines.x_pu_eff.where(n.lines.carrier == 'AC', n.lines.r_pu_eff)

    def cycle_flow(sub):
        C = pd.DataFrame(sub.C.todense(), index=sub.lines_i())
        if C.empty:
            return None
        C_weighted = 1e5 * C.mul(weightings[sub.lines_i()], axis=0)
        return C_weighted.apply(lambda ds: ds @ n.lines_t.p0[ds.index].T)

    return pd.concat([cycle_flow(sub) for sub in n.sub_networks.obj], axis=0)\
             .unstack().describe().to_frame('Cycle Constr.')



def constraint_stats(n, round_digit=1e-30):
    """
    Post-optimization function to recalculate gap statistics of different
    constraints. For inequality constraints only the minimum of lhs - rhs, with
    lhs >= rhs is returned.
    """
    return pd.concat([describe_cycle_constraints(n),
                      describe_store_contraints(n),
                      describe_storage_unit_contraints(n),
                      describe_nodal_balance_constraint(n),
                      describe_lower_dispatch_constraints(n),
                      describe_upper_dispatch_constraints(n)],
                   axis=1, sort=False)

def check_constraints(n, tol=1e-3):
    """
    Post-optimization test function to double-check most of the lopf
    constraints. For relevant equaility constraints, it test whether the
    deviation between lhs and rhs is below the given tolerance. For inequality
    constraints, it test whether the inequality is violated with a higher
    value then the tolerance.

    Parameters
    ----------
    n : pypsa.Network
    tol : float
        Gap tolerance

    Returns AssertionError if tolerance is exceeded.

    """
    n.lines['carrier'] = n.lines.bus0.map(n.buses.carrier)
    stats = constraint_stats(n).rename(index=str.title)
    condition = stats.T[['Min', 'Max']].query('Min < -@tol | Max > @tol').T
    assert condition.empty, (f'The following constraint(s) are exceeding the '
                             f'given tolerance of {tol}: \n{condition}')
