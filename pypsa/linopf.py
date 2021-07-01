## Copyright 2019 Tom Brown (KIT), Fabian Hofmann (FIAS)

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Build optimisation problems from PyPSA networks without Pyomo.
Originally retrieved from nomopyomo ( -> 'no more Pyomo').
"""


from .pf import (_as_snapshots, get_switchable_as_dense as get_as_dense)
from .descriptors import (get_bounds_pu, get_extendable_i, get_non_extendable_i,
                          expand_series, nominal_attrs, additional_linkports,
                          Dict, get_active_assets, snapshot_consistency)

from .linopt import (linexpr, write_bound, write_constraint, write_objective,
                     set_conref, set_varref, get_con, get_var, join_exprs,
                     run_and_read_cbc, run_and_read_gurobi, run_and_read_glpk,
                     run_and_read_cplex, run_and_read_xpress,
                     define_constraints, define_variables, define_binaries,
                     align_with_static_component)


import pandas as pd
import numpy as np
from numpy import inf

import gc, time, os, re, shutil
from tempfile import mkstemp

import logging
logger = logging.getLogger(__name__)

lookup = pd.read_csv(os.path.join(os.path.dirname(__file__), 'variables.csv'),
                     index_col=['component', 'variable'])

def define_nominal_for_extendable_variables(n, c, attr):
    """
    Initializes variables for nominal capacities for a given component and a
    given attribute.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        network component of which the nominal capacity should be defined
    attr : str
        name of the variable, e.g. 'p_nom'

    """
    ext_i = get_extendable_i(n, c)
    if ext_i.empty: return
    lower = n.df(c)[attr+'_min'][ext_i]
    upper = n.df(c)[attr+'_max'][ext_i]
    define_variables(n, lower, upper, c, attr)


def define_dispatch_for_extendable_and_committable_variables(n, sns, c, attr):
    """
    Initializes variables for power dispatch for a given component and a
    given attribute.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'

    """
    ext_i = get_extendable_i(n, c)
    if c == 'Generator':
        ext_i = ext_i.union(n.generators.query('committable').index)
    if ext_i.empty: return
    define_variables(n, -inf, inf, c, attr, axes=[sns, ext_i], spec='ext')


def define_dispatch_for_non_extendable_variables(n, sns, c, attr):
    """
    Initializes variables for power dispatch for a given component and a
    given attribute.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'

    """
    fix_i = get_non_extendable_i(n, c)
    if c == 'Generator':
        fix_i = fix_i.difference(n.generators.query('committable').index)
    if fix_i.empty: return
    nominal_fix = n.df(c)[nominal_attrs[c]][fix_i]
    min_pu, max_pu = get_bounds_pu(n, c, sns, fix_i, attr)
    lower = min_pu.mul(nominal_fix)
    upper = max_pu.mul(nominal_fix)
    axes = [sns, fix_i]
    dispatch = define_variables(n, -inf, inf, c, attr, axes=axes, spec='non_ext')
    dispatch = linexpr((1, dispatch))
    define_constraints(n, dispatch, '>=', lower, c, 'mu_lower', spec='non_ext')
    define_constraints(n, dispatch, '<=', upper, c, 'mu_upper', spec='non_ext')


def define_dispatch_for_extendable_constraints(n, sns, c, attr):
    """
    Sets power dispatch constraints for extendable devices for a given
    component and a given attribute.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'

    """
    ext_i = get_extendable_i(n, c)
    if ext_i.empty: return
    min_pu, max_pu = get_bounds_pu(n, c, sns, ext_i, attr)
    operational_ext_v = get_var(n, c, attr)[ext_i]
    nominal_v = get_var(n, c, nominal_attrs[c])[ext_i]
    rhs = 0

    lhs, *axes = linexpr((max_pu, nominal_v), (-1, operational_ext_v),
                         return_axes=True)
    define_constraints(n, lhs, '>=', rhs, c, 'mu_upper', axes=axes, spec=attr)

    lhs, *axes = linexpr((min_pu, nominal_v), (-1, operational_ext_v),
                         return_axes=True)
    define_constraints(n, lhs, '<=', rhs, c, 'mu_lower', axes=axes, spec=attr)


def define_fixed_variable_constraints(n, sns, c, attr, pnl=True):
    """
    Sets constraints for fixing variables of a given component and attribute
    to the corresponding values in n.df(c)[attr + '_set'] if pnl is True, or
    n.pnl(c)[attr + '_set']

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'
    pnl : bool, default True
        Whether variable which should be fixed is time-dependent

    """

    if pnl:
        if attr + '_set' not in n.pnl(c): return
        fix = n.pnl(c)[attr + '_set'].stack()
        if fix.empty: return
        lhs = linexpr((1, get_var(n, c, attr).stack()[fix.index]),
                      as_pandas=False)
        constraints = write_constraint(n, lhs, '=', fix).unstack().T
    else:
        if attr + '_set' not in n.df(c): return
        fix = n.df(c)[attr + '_set'].dropna()
        if fix.empty: return
        lhs = linexpr((1, get_var(n, c, attr)[fix.index]), as_pandas=False)
        constraints = write_constraint(n, lhs, '=', fix)
    set_conref(n, constraints, c, f'mu_{attr}_set')


def define_generator_status_variables(n, snapshots):
    com_i = n.generators.query('committable').index
    ext_i = get_extendable_i(n, 'Generator')
    if not (ext_i.intersection(com_i)).empty:
        logger.warning("The following generators have both investment optimisation"
        f" and unit commitment:\n\n\t{', '.join((ext_i.intersection(com_i)))}\n\nCurrently PyPSA cannot "
        "do both these functions, so PyPSA is choosing investment optimisation "
        "for these generators.")
        com_i = com_i.difference(ext_i)
    if com_i.empty: return
    define_binaries(n, (snapshots, com_i), 'Generator', 'status')


def define_committable_generator_constraints(n, snapshots):
    c, attr = 'Generator', 'status'
    com_i = n.df(c).query('committable and not p_nom_extendable').index
    if com_i.empty: return
    nominal = n.df(c)[nominal_attrs[c]][com_i]
    min_pu, max_pu = get_bounds_pu(n, c, snapshots, com_i, 'p')
    lower = min_pu.mul(nominal)
    upper = max_pu.mul(nominal)

    status = get_var(n, c, attr)
    p = get_var(n, c, 'p')[com_i]

    lhs = linexpr((lower, status), (-1, p))
    define_constraints(n, lhs, '<=', 0, 'Generators', 'committable_lb')

    lhs = linexpr((upper, status), (-1, p))
    define_constraints(n, lhs, '>=', 0, 'Generators', 'committable_ub')



def define_ramp_limit_constraints(n, sns):
    """
    Defines ramp limits for generators with valid ramplimit

    """
    c = 'Generator'
    rup_i = n.df(c).query('ramp_limit_up == ramp_limit_up').index
    rdown_i = n.df(c).query('ramp_limit_down == ramp_limit_down').index
    if rup_i.empty & rdown_i.empty:
        return
    fix_i = get_non_extendable_i(n, c)
    ext_i = get_extendable_i(n, c)
    com_i = n.df(c).query('committable').index.difference(ext_i)
    p = get_var(n, c, 'p').loc[sns[1:]]
    p_prev = get_var(n, c, 'p').shift(1).loc[sns[1:]]

    # fix up
    gens_i = rup_i.intersection(fix_i)
    if not gens_i.empty:
        lhs = linexpr((1, p[gens_i]), (-1, p_prev[gens_i]))
        rhs = n.df(c).loc[gens_i].eval('ramp_limit_up * p_nom')
        define_constraints(n, lhs, '<=', rhs,  c, 'mu_ramp_limit_up', spec='nonext.')

    # ext up
    gens_i = rup_i.intersection(ext_i)
    if not gens_i.empty:
        limit_pu = n.df(c)['ramp_limit_up'][gens_i]
        p_nom = get_var(n, c, 'p_nom')[gens_i]
        lhs = linexpr((1, p[gens_i]), (-1, p_prev[gens_i]), (-limit_pu, p_nom))
        define_constraints(n, lhs, '<=', 0, c, 'mu_ramp_limit_up', spec='ext.')

    # com up
    gens_i = rup_i.intersection(com_i)
    if not gens_i.empty:
        limit_start = n.df(c).loc[gens_i].eval('ramp_limit_start_up * p_nom')
        limit_up = n.df(c).loc[gens_i].eval('ramp_limit_up * p_nom')
        status = get_var(n, c, 'status').loc[sns[1:], gens_i]
        status_prev = get_var(n, c, 'status').shift(1).loc[sns[1:], gens_i]
        lhs = linexpr((1, p[gens_i]), (-1, p_prev[gens_i]),
                      (limit_start - limit_up, status_prev),
                      (- limit_start, status))
        define_constraints(n, lhs, '<=', 0, c, 'mu_ramp_limit_up', spec='com.')

    # fix down
    gens_i = rdown_i.intersection(fix_i)
    if not gens_i.empty:
        lhs = linexpr((1, p[gens_i]), (-1, p_prev[gens_i]))
        rhs = n.df(c).loc[gens_i].eval('-1 * ramp_limit_down * p_nom')
        define_constraints(n, lhs, '>=', rhs, c, 'mu_ramp_limit_down', spec='nonext.')

    # ext down
    gens_i = rdown_i.intersection(ext_i)
    if not gens_i.empty:
        limit_pu = n.df(c)['ramp_limit_down'][gens_i]
        p_nom = get_var(n, c, 'p_nom')[gens_i]
        lhs = linexpr((1, p[gens_i]), (-1, p_prev[gens_i]), (limit_pu, p_nom))
        define_constraints(n, lhs, '>=', 0, c, 'mu_ramp_limit_down', spec='ext.')

    # com down
    gens_i = rdown_i.intersection(com_i)
    if not gens_i.empty:
        limit_shut = n.df(c).loc[gens_i].eval('ramp_limit_shut_down * p_nom')
        limit_down = n.df(c).loc[gens_i].eval('ramp_limit_down * p_nom')
        status = get_var(n, c, 'status').loc[sns[1:], gens_i]
        status_prev = get_var(n, c, 'status').shift(1).loc[sns[1:], gens_i]
        lhs = linexpr((1, p[gens_i]), (-1, p_prev[gens_i]),
                      (limit_down - limit_shut, status),
                      (limit_shut, status_prev))
        define_constraints(n, lhs, '>=', 0, c, 'mu_ramp_limit_down', spec='com.')


def define_nominal_constraints_per_bus_carrier(n, sns):
    for carrier in n.carriers.index:
        for bound, sense in [("max", "<="), ("min", ">=")]:

            col = f'nom_{bound}_{carrier}'
            if col not in n.buses.columns: continue
            rhs = n.buses[col].dropna()
            lhs = pd.Series('', rhs.index)

            for c, attr in nominal_attrs.items():
                if c not in n.one_port_components: continue
                attr = nominal_attrs[c]
                if (c, attr) not in n.variables.index: continue
                nominals = get_var(n, c, attr)[n.df(c).carrier == carrier]
                if nominals.empty: continue
                per_bus = linexpr((1, nominals)).groupby(n.df(c).bus).sum()
                lhs += per_bus.reindex(lhs.index, fill_value='')

            if bound == 'max':
                lhs = lhs[lhs != '']
                rhs = rhs.reindex(lhs.index)
            else:
                assert (lhs != '').all(), (
                    f'No extendable components of carrier {carrier} on bus '
                    f'{list(lhs[lhs == ""].index)}')
            define_constraints(n, lhs, sense, rhs, 'Bus', 'mu_' + col)


def define_nodal_balance_constraints(n, sns):
    """
    Defines nodal balance constraint.

    """

    def bus_injection(c, attr, groupcol='bus', sign=1):
        # additional sign only necessary for branches in reverse direction
        if 'sign' in n.df(c):
            sign = sign * n.df(c).sign
        expr = linexpr((sign, get_var(n, c, attr))).rename(columns=n.df(c)[groupcol])
        # drop empty bus2, bus3 if multiline link
        if c == 'Link':
            expr.drop(columns='', errors='ignore', inplace=True)
        return expr

    # one might reduce this a bit by using n.branches and lookup
    args = [['Generator', 'p'], ['Store', 'p'], ['StorageUnit', 'p_dispatch'],
            ['StorageUnit', 'p_store', 'bus', -1], ['Line', 's', 'bus0', -1],
            ['Line', 's', 'bus1', 1], ['Transformer', 's', 'bus0', -1],
            ['Transformer', 's', 'bus1', 1], ['Link', 'p', 'bus0', -1],
            ['Link', 'p', 'bus1', get_as_dense(n, 'Link', 'efficiency', sns)]]
    args = [arg for arg in args if not n.df(arg[0]).empty]

    for i in additional_linkports(n):
        eff = get_as_dense(n, 'Link', f'efficiency{i}', sns)
        args.append(['Link', 'p', f'bus{i}', eff])

    lhs = (pd.concat([bus_injection(*arg) for arg in args], axis=1)
           .groupby(axis=1, level=0)
           .sum()
           .reindex(columns=n.buses.index, fill_value=''))
    sense = '='
    rhs = ((- get_as_dense(n, 'Load', 'p_set', sns) * n.loads.sign)
           .groupby(n.loads.bus, axis=1).sum()
           .reindex(columns=n.buses.index, fill_value=0))
    define_constraints(n, lhs, sense, rhs, 'Bus', 'marginal_price')


def define_kirchhoff_constraints(n, sns):
    """
    Defines Kirchhoff voltage constraints

    """
    comps = n.passive_branch_components & set(n.variables.index.levels[0])
    if len(comps) == 0: return
    branch_vars = pd.concat({c:get_var(n, c, 's') for c in comps}, axis=1)

    def cycle_flow(ds, sns):
        ds = ds[lambda ds: ds!=0.].dropna()
        vals = linexpr((ds, branch_vars.loc[sns, ds.index]), as_pandas=False)
        return vals.sum(1)

    constraints = []
    investment_periods = sns.levels[0] if isinstance(sns, pd.MultiIndex) else slice(None)
    for period in investment_periods:
        n.determine_network_topology(investment_period=period)
        subconstraints = []
        for sub in n.sub_networks.obj:
            branches = sub.branches()
            C = pd.DataFrame(sub.C.todense(), index=branches.index)
            if C.empty:
                continue
            carrier = n.sub_networks.carrier[sub.name]
            weightings = branches.x_pu_eff if carrier == 'AC' else branches.r_pu_eff
            C_weighted = 1e5 * C.mul(weightings, axis=0)
            cycle_sum = C_weighted.apply(cycle_flow, sns=period)
            snapshots = sns if period == slice(None) else sns[sns.get_loc(period)]
            cycle_sum.set_index(snapshots, inplace=True)
            con = write_constraint(n, cycle_sum, '=', 0)
            subconstraints.append(con)
        if len(subconstraints) == 0:
            continue
        constraints.append(pd.concat(subconstraints, axis=1, ignore_index=True))
    constraints = pd.concat(constraints).rename_axis(columns='cycle')
    set_conref(n, constraints, 'SubNetwork', 'mu_kirchhoff_voltage_law')


def define_storage_unit_constraints(n, sns):
    """
    Defines state of charge (soc) constraints for storage units. In principal
    the constraints states:

        previous_soc + p_store - p_dispatch + inflow - spill == soc

    """
    sus_i = n.storage_units.index
    if sus_i.empty: return
    c = 'StorageUnit'
    # spillage
    upper = get_as_dense(n, c, 'inflow', sns).loc[:, lambda df: df.max() > 0]
    spill = write_bound(n, 0, upper)
    set_varref(n, spill, 'StorageUnit', 'spill')

    # create multiindex [time of investment, snapshots]
    if isinstance(sns, pd.MultiIndex):
        sns_multi = sns
    else:
        sns_multi = pd.MultiIndex.from_product([[0], sns])

    # elapsed hours
    eh = (expand_series(n.snapshot_weightings.loc[sns, "stores"], sus_i)
         .reindex(sns_multi, level=1))
    # efficiencies
    eff_stand = expand_series(1-n.df(c).standing_loss, sns_multi).T.pow(eh)
    eff_dispatch = expand_series(n.df(c).efficiency_dispatch, sns_multi).T
    eff_store = expand_series(n.df(c).efficiency_store, sns_multi).T

    soc = get_var(n, c, 'state_of_charge').reindex(sns_multi, level=1)

    # storage unit index
    cyclic_i = n.df(c).query('cyclic_state_of_charge').index
    noncyclic_i = n.df(c).query('~cyclic_state_of_charge & ~state_of_charge_initial_per_period').index
    periodically_i = n.df(c).query("~cyclic_state_of_charge & state_of_charge_initial_per_period").index

    # active assets
    _ = [get_active_assets(n, c, period) for period in sns_multi.levels[0]]
    active = pd.concat(_, axis=1).T.astype(int)

    # state of charge at the beginning of each investment period (soc_first_inv)
    # shoud be the same during the lifetime of the storage
    eff_stand_periodic = ((eff_stand.mul(active, level=0)).groupby(level=0)
                          .shift().dropna())
    previous_soc_periodic = soc.groupby(level=0).shift().dropna()

    # cyclic constraint for soc - shift each investment period
    previous_soc_cyclic = soc.groupby(level=0).shift()
    fill_with_last = soc.groupby(level=0).last()
    previous_soc_cyclic = previous_soc_cyclic.combine_first(
                          previous_soc_cyclic.add(fill_with_last, level=0, fill_value=0))

    coeff_var = [(-1, soc),
                 (-1/eff_dispatch * eh, get_var(n, c, 'p_dispatch')),
                 (eff_store * eh, get_var(n, c, 'p_store'))]

    lhs, *axes = linexpr(*coeff_var, return_axes=True)


    def masked_term(coeff, var, cols):
        return linexpr((coeff[cols], var[cols]))\
               .reindex(index=axes[0], columns=axes[1], fill_value='').values

    if ('StorageUnit', 'spill') in n.variables.index:
        lhs += masked_term(-eh,
                           get_var(n, c, 'spill').reindex(sns_multi, level=1),
                           spill.columns)

    lhs += masked_term(eff_stand, previous_soc_cyclic, cyclic_i)
    lhs += masked_term(eff_stand.loc[sns_multi[1:]], soc.shift().loc[sns_multi[1:]], noncyclic_i)
    lhs += masked_term(eff_stand_periodic, previous_soc_periodic, periodically_i)

    # rhs consider inflow and initial state of charge
    rhs = -get_as_dense(n, c, 'inflow', sns).reindex(sns_multi, level=1).mul(eh)
    rhs.loc[sns_multi[:1], noncyclic_i] -= n.df(c).state_of_charge_initial[noncyclic_i]

    # set state of charge at the beginning of each investment period to soc_initial
    soc_initial_period = expand_series(n.df(c).state_of_charge_initial, sns_multi).T
    exclude = ~(rhs.groupby(level=0).shift().isna() &
                active.reindex(sns_multi, level=0).astype(bool))
    soc_initial_period[exclude] = 0
    rhs[periodically_i] -= soc_initial_period[periodically_i]

    define_constraints(n, lhs, '==', rhs, c, 'mu_state_of_charge')


def define_store_constraints(n, sns):
    """
    Defines energy balance constraints for stores. In principal this states:

        previous_e - p == e

    """
    stores_i = n.stores.index
    if stores_i.empty: return
    c = 'Store'
    variables = write_bound(n, -inf, inf, axes=[sns, stores_i])
    set_varref(n, variables, c, 'p')

    if isinstance(sns, pd.MultiIndex):
        sns_multi = sns
    else:
        sns_multi = pd.MultiIndex.from_product([[0], sns])

    # elapsed hours
    eh = (expand_series(n.snapshot_weightings.loc[sns, "stores"], stores_i)
         .reindex(sns_multi, level=1))
    eff_stand = expand_series(1-n.df(c).standing_loss, sns_multi).T.pow(eh)

    e = get_var(n, c, 'e').reindex(sns_multi, level=1)
    cyclic_i = n.df(c).query('e_cyclic').index
    noncyclic_i = n.df(c).query('~e_cyclic & ~e_initial_per_period').index
    periodically_i = n.df(c).query("~e_cyclic & e_initial_per_period").index

    # active assets
    _ = [get_active_assets(n, c, period) for period in sns_multi.levels[0]]
    active = pd.concat(_, axis=1).T.astype(int)

    # e at the beginning of each investment period (e_first_inv) should be the
    # same during the lifetime of the storage for e_initial_per_period=True
    eff_stand_periodic = ((eff_stand.mul(active, level=0)).groupby(level=0).shift()
                          .apply(lambda x: x.dropna()))
    previous_e_periodic = e.groupby(level=0).shift().dropna()

    # cyclic constraint
    previous_e_cyclic = e.groupby(level=0).shift()
    fill_with_last = e.groupby(level=0).last()
    previous_e_cyclic = previous_e_cyclic.combine_first(
                          previous_e_cyclic.add(fill_with_last, level=0, fill_value=0))

    coeff_var = [(-eh, get_var(n, c, 'p')), (-1, e)]

    lhs, *axes = linexpr(*coeff_var, return_axes=True)

    def masked_term(coeff, var, cols):
        return linexpr((coeff[cols], var[cols]))\
               .reindex(index=axes[0], columns=axes[1], fill_value='').values

    lhs += masked_term(eff_stand, previous_e_cyclic, cyclic_i)
    lhs += masked_term(eff_stand.loc[sns_multi[1:]], e.shift().loc[sns_multi[1:]], noncyclic_i)
    lhs += masked_term(eff_stand_periodic, previous_e_periodic, periodically_i)

    rhs = pd.DataFrame(0, sns_multi, stores_i)
    rhs.loc[sns_multi[:1], noncyclic_i] -= n.df(c)['e_initial'][noncyclic_i]
    e_initial_period = expand_series(n.df(c).e_initial, sns_multi).T
    exclude = ~(rhs.groupby(level=0).shift().isna() &
                active.astype(bool).reindex(sns_multi, level=0))
    e_initial_period[exclude] = 0
    rhs[periodically_i] -= e_initial_period[periodically_i]

    define_constraints(n, lhs, '==', rhs, c, 'mu_state_of_charge')


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
    # (1)(a) primary_energy
    glcs = n.global_constraints.query('type == "primary_energy"')

    weightings = (n.snapshot_weightings
                  .mul(n.investment_period_weightings["time"]
                       .reindex(sns).fillna(method="bfill").fillna(1.), axis=0)
                  )


    for name, glc in glcs.iterrows():
        rhs = glc.constant
        lhs = ''
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f'{carattr} != 0')[carattr]

        if emissions.empty: continue

        # generators
        gens = n.generators.query('carrier in @emissions.index')
        if not gens.empty:
            em_pu = gens.carrier.map(emissions)/gens.efficiency
            em_pu = weightings["generators"].to_frame('weightings') @\
                    em_pu.to_frame('weightings').T
            vals = linexpr((em_pu, get_var(n, 'Generator', 'p')[gens.index]),
                           as_pandas=True).groupby(level=0).sum()
            # get time period to which constraint applies
            inv_valid = sns.levels[0] if isinstance(sns, pd.MultiIndex) else sns
            time_valid = (inv_valid if glc["investment_period"]=='' else
                          int(glc["investment_period"]))
            lhs +=  join_exprs(vals.loc[time_valid])

        # storage units
        sus = n.storage_units.query('carrier in @emissions.index and '
                                    'not cyclic_state_of_charge')
        sus_i = sus.index
        if not sus.empty:
            coeff_val = (-sus.carrier.map(emissions), get_var(n, 'StorageUnit',
                         'state_of_charge').loc[sns[-1], sus_i])
            vals = linexpr(coeff_val, as_pandas=False)
            lhs = lhs + '\n' + join_exprs(vals)
            rhs -= sus.carrier.map(emissions) @ sus.state_of_charge_initial

        # stores
        n.stores['carrier'] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query('carrier in @emissions.index and not e_cyclic')
        if not stores.empty:
            coeff_val = (-stores.carrier.map(emissions), get_var(n, 'Store', 'e')
                         .loc[sns[-1], stores.index])
            vals = linexpr(coeff_val, as_pandas=False)
            lhs = lhs + '\n' + join_exprs(vals)
            rhs -= stores.carrier.map(emissions) @ stores.e_initial

        con = write_constraint(n, lhs, glc.sense, rhs, axes=pd.Index([name]))
        set_conref(n, con, 'GlobalConstraint', 'mu', name)

    # for line expansion we need to add a line carrier
    if any(n.global_constraints.type.isin(["transmission_volume_expansion_limit",
                                           "transmission_expansion_cost_limit"])):
        n.lines['carrier'] = n.lines.bus0.map(n.buses.carrier)

     # (1)(b) Budget
    glcs = n.global_constraints.query('type == "Budget"')
    for name, glc in glcs.iterrows():
        rhs = glc.constant
        lhs = ''
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f'{carattr} != 0')[carattr]

        if emissions.empty: continue

        # generators
        gens = n.generators.query('carrier in @emissions.index')
        if not gens.empty:
            em_pu = gens.carrier.map(emissions)/gens.efficiency
            em_pu = weightings["generators"].to_frame('weightings') @\
                    em_pu.to_frame('weightings').T
            vals = linexpr((em_pu, get_var(n, 'Generator', 'p')[gens.index]),
                           as_pandas=False)
            lhs += join_exprs(vals)

        # storage units
        sus = n.storage_units.query('carrier in @emissions.index and '
                                    'not cyclic_state_of_charge')
        sus_i = sus.index
        if not sus.empty:
            coeff_val = (-sus.carrier.map(emissions), get_var(n, 'StorageUnit',
                         'state_of_charge').loc[sns[-1], sus_i])
            vals = linexpr(coeff_val, as_pandas=False)
            lhs = lhs + '\n' + join_exprs(vals)
            rhs -= sus.carrier.map(emissions) @ sus.state_of_charge_initial

        # stores
        n.stores['carrier'] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query('carrier in @emissions.index and not e_cyclic')
        if not stores.empty:
            coeff_val = (-stores.carrier.map(emissions), get_var(n, 'Store', 'e')
                         .loc[sns[-1], stores.index])
            vals = linexpr(coeff_val, as_pandas=False)
            lhs = lhs + '\n' + join_exprs(vals)
            rhs -= stores.carrier.map(emissions) @ stores.e_initial

        con = write_constraint(n, lhs, glc.sense, rhs, axes=pd.Index([name]))
        set_conref(n, con, 'GlobalConstraint', 'mu', name)


    # for line expansion we need to add a line carrier
    if any(n.global_constraints.type.isin(["transmission_volume_expansion_limit",
                                           "transmission_expansion_cost_limit"])):
        n.lines['carrier'] = n.lines.bus0.map(n.buses.carrier)

    # (2) transmission_volume_expansion_limit
    glcs = n.global_constraints.query('type == '
                                      '"transmission_volume_expansion_limit"')
    substr = lambda s: re.sub('[\[\]\(\)]', '', s)
    for name, glc in glcs.iterrows():
        car = [substr(c.strip()) for c in glc.carrier_attribute.split(',')]
        lhs = ''
        for c, attr in (('Line', 's_nom'), ('Link', 'p_nom')):
            if n.df(c).empty: continue
            ext_i = n.df(c).query(f'carrier in @car and {attr}_extendable').index
            if ext_i.empty: continue
            # get time period to which constraint applies
            inv_valid = sns.levels[0] if isinstance(sns, pd.MultiIndex) else [0.]
            time_valid = (inv_valid if glc["investment_period"]=='' else
                          pd.DatetimeIndex([glc["investment_period"]]))
            # get active assets during time valid
            active_i = pd.concat([get_active_assets(n, c, period)
                              for period in time_valid], axis=1).astype(int)
            ext_and_active = active_i.T[active_i.index.intersection(ext_i)]
            v = linexpr((n.df(c).loc[ext_and_active.columns, "length"].mul(ext_and_active),
                         get_var(n, c, attr)[ext_and_active.columns]),
                        as_pandas=False)
            lhs += v.sum(axis=1)
        if type(lhs) == str: continue
        sense = glc.sense
        rhs = glc.constant
        con = write_constraint(n, lhs, sense, rhs, axes=pd.Index([name]))
        set_conref(n, con, 'GlobalConstraint', 'mu', name)

    # (3) transmission_expansion_cost_limit
    glcs = n.global_constraints.query('type == '
                                      '"transmission_expansion_cost_limit"')
    for name, glc in glcs.iterrows():
        car = [substr(c.strip()) for c in glc.carrier_attribute.split(',')]
        lhs = ''
        for c, attr in (('Line', 's_nom'), ('Link', 'p_nom')):
            ext_i = n.df(c).query(f'carrier in @car and {attr}_extendable').index

            inv_valid = sns.levels[0] if isinstance(sns, pd.MultiIndex) else [0.]
            time_valid = (inv_valid if glc["investment_period"]=='' else
                          pd.DatetimeIndex([glc["investment_period"]]))

             # get active assets during time valid
            active_i = pd.concat([get_active_assets(n, c, period)
                              for period in time_valid], axis=1).astype(int)
            ext_and_active = active_i.T[active_i.index.intersection(ext_i)]

            if ext_and_active.empty: continue
            v = linexpr((n.df(c).loc[ext_and_active.columns, "capital_cost"].mul(ext_and_active),
                         get_var(n, c, attr)[ext_and_active.columns]),
                        as_pandas=False)
            lhs += v.sum(axis=1)
        if type(lhs) == str: continue
        sense = glc.sense
        rhs = glc.constant
        con = write_constraint(n, lhs, sense, rhs, axes=pd.Index([name]))
        set_conref(n, con, 'GlobalConstraint', 'mu', name)


    # (4) tech_capacity_expansion_limit
    substr = lambda s: re.sub('[\[\]\(\)]', '', s)
    glcs = n.global_constraints.query('type == '
                                      '"tech_capacity_expansion_limit"')
    c, attr = 'Generator', 'p_nom'

    for name, glc in glcs.iterrows():
        ext_i = n.df(c)[(n.df(c)["carrier"]==glc.loc["carrier_attribute"])
                        & (n.df(c)["bus"] ==glc.loc["bus"])
                        & (n.df(c)["p_nom_extendable"])].index
        inv_valid = sns.levels[0] if isinstance(sns, pd.MultiIndex) else [0.]
        time_valid = (inv_valid if glc["investment_period"]=='' else
              pd.DatetimeIndex([glc["investment_period"]]))
        active_i = pd.concat([get_active_assets(n,c,period)
                              for period in time_valid], axis=1).astype(int)

        ext_and_active = active_i.T[active_i.index.intersection(ext_i)]

        if ext_and_active.empty: continue

        cap_vars = get_var(n, c, attr)[ext_and_active.columns]

        lhs = linexpr((ext_and_active, cap_vars)).sum(axis=1)
        rhs = glc.constant
        sense = glc.sense

        con = write_constraint(n, lhs, sense, rhs, axes=pd.Index([name]))

        set_conref(n, con, 'GlobalConstraint', 'mu_cap_limit', name)


def define_objective(n, sns):
    """
    Defines and writes out the objective function

    """

    # TODO check if investment_periods can be taken here.
    investments = [0.] if n.investment_periods.empty else n.investment_periods

    objective_w_investment = (n.investment_period_weightings["objective"]
                             .reindex(investments).fillna(1.))

    objective = (n.snapshot_weightings
                            .mul(n.investment_period_weightings["objective"]
                                 .reindex(sns)
                            .fillna(method="bfill").fillna(1.), axis=0))

    # constant for already done investment
    nom_attr = nominal_attrs.items()
    constant = 0
    for c, attr in nom_attr:
        ext_i = get_extendable_i(n, c)
        active = pd.concat({period: get_active_assets(n, c, period)
                            for period in investments}, axis=1).astype(int).loc[ext_i]

        constant += (n.df(c)[attr][ext_i] @
                    (active.mul(n.df(c).capital_cost[ext_i], axis=0) @
                     objective_w_investment))
    object_const = write_bound(n, constant, constant)
    write_objective(n, linexpr((-1, object_const), as_pandas=False)[0])
    n.objective_constant = constant

    # marginal cost
    for c, attr in lookup.query('marginal_cost').index:
        cost = (get_as_dense(n, c, 'marginal_cost', sns)
                .loc[:, lambda ds: (ds != 0).all()]
                .mul(objective.loc[sns, "objective"], axis=0))
        if cost.empty: continue
        terms = linexpr((cost, get_var(n, c, attr).loc[sns, cost.columns]))
        write_objective(n, terms)

    # investment
    for c, attr in nominal_attrs.items():
        ext_i = get_extendable_i(n, c)
        cost = n.df(c)['capital_cost'][ext_i]
        if cost.empty: continue
        active = pd.concat({period: get_active_assets(n, c, period)
                            for period in investments}, axis=1).astype(int).loc[ext_i]

        caps = get_var(n, c, attr).loc[ext_i]
        cost_weighted = active.mul(cost, axis=0) @ objective_w_investment
        terms = linexpr((cost_weighted, caps))
        write_objective(n, terms)



def prepare_lopf(n, snapshots=None, keep_files=False, skip_objective=False,
                 extra_functionality=None, solver_dir=None):
    """
    Sets up the linear problem and writes it out to a lp file.

    Returns
    -------
    Tuple (fdp, problem_fn) indicating the file descriptor and the file name of
    the lp file

    """
    n._xCounter, n._cCounter = 1, 1
    n.vars, n.cons = Dict(), Dict()

    cols = ['component', 'name', 'pnl', 'specification']
    n.variables = pd.DataFrame(columns=cols).set_index(cols[:2])
    n.constraints = pd.DataFrame(columns=cols).set_index(cols[:2])

    snapshots = n.snapshots if snapshots is None else snapshots
    start = time.time()

    tmpkwargs = dict(text=True, dir=solver_dir)
    # mkstemp(suffix, prefix, **tmpkwargs)
    fdo, objective_fn = mkstemp('.txt', 'pypsa-objectve-', **tmpkwargs)
    fdc, constraints_fn = mkstemp('.txt', 'pypsa-constraints-', **tmpkwargs)
    fdb, bounds_fn = mkstemp('.txt', 'pypsa-bounds-', **tmpkwargs)
    fdi, binaries_fn = mkstemp('.txt', 'pypsa-binaries-', **tmpkwargs)
    fdp, problem_fn = mkstemp('.lp', 'pypsa-problem-', **tmpkwargs)

    n.objective_f = open(objective_fn, mode='w')
    n.constraints_f = open(constraints_fn, mode='w')
    n.bounds_f = open(bounds_fn, mode='w')
    n.binaries_f = open(binaries_fn, mode='w')

    n.objective_f.write('\* LOPF *\n\nmin\nobj:\n')
    n.constraints_f.write("\n\ns.t.\n\n")
    n.bounds_f.write("\nbounds\n")
    n.binaries_f.write("\nbinary\n")

    for c, attr in lookup.query('nominal and not handle_separately').index:
        define_nominal_for_extendable_variables(n, c, attr)
        # define_fixed_variable_constraints(n, snapshots, c, attr, pnl=False)
    for c, attr in lookup.query('not nominal and not handle_separately').index:
        define_dispatch_for_non_extendable_variables(n, snapshots, c, attr)
        define_dispatch_for_extendable_and_committable_variables(n, snapshots, c, attr)
        align_with_static_component(n, c, attr)
        define_dispatch_for_extendable_constraints(n, snapshots, c, attr)
        # define_fixed_variable_constraints(n, snapshots, c, attr)
    define_generator_status_variables(n, snapshots)
    define_nominal_constraints_per_bus_carrier(n, snapshots)

    # consider only state_of_charge_set for the moment
    define_fixed_variable_constraints(n, snapshots, 'StorageUnit', 'state_of_charge')
    define_fixed_variable_constraints(n, snapshots, 'Store', 'e')

    define_committable_generator_constraints(n, snapshots)
    define_ramp_limit_constraints(n, snapshots)
    define_storage_unit_constraints(n, snapshots)
    define_store_constraints(n, snapshots)
    define_kirchhoff_constraints(n, snapshots)
    define_nodal_balance_constraints(n, snapshots)
    define_global_constraints(n, snapshots)
    if skip_objective:
        logger.info("The argument `skip_objective` is set to True. Expecting a "
                    "custom objective to be build via `extra_functionality`.")
    else:
        define_objective(n, snapshots)

    if extra_functionality is not None:
        extra_functionality(n, snapshots)

    n.binaries_f.write("end\n")

    # explicit closing with file descriptor is necessary for windows machines
    for f, fd in (('bounds_f', fdb), ('constraints_f', fdc),
                  ('objective_f', fdo), ('binaries_f', fdi)):
        getattr(n, f).close(); delattr(n, f); os.close(fd)

    # concatenate files
    with open(problem_fn, 'wb') as wfd:
        for f in [objective_fn, constraints_fn, bounds_fn, binaries_fn]:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)
            if not keep_files:
                os.remove(f)

    logger.info(f'Total preparation time: {round(time.time()-start, 2)}s')
    return fdp, problem_fn


def assign_solution(n, sns, variables_sol, constraints_dual,
                    keep_references=False, keep_shadowprices=None):
    """Map solution to network components.

    Helper function. Assigns the solution of a succesful optimization to the
    network.

    """

    def set_from_frame(pnl, attr, df):
        if attr not in pnl:
            if isinstance(n.snapshots, pd.MultiIndex):
                pnl[attr] = df.reindex(sns, level=0)
            else:
                pnl[attr] = df.reindex(sns)
        elif pnl[attr].empty:
            pnl[attr] = df.reindex(sns)
        else:
            # for variables indexed with investment period not MultiIndex
            if not isinstance(df, pd.MultiIndex):
                df = df.reindex(sns, level=0)
            pnl[attr].loc[sns, :] = df.reindex(columns=pnl[attr].columns)

    pop = not keep_references
    def map_solution(c, attr):
        variables = get_var(n, c, attr, pop=pop)
        predefined = True
        if (c, attr) not in lookup.index:
            predefined = False
            n.sols[c] = n.sols[c] if c in n.sols else Dict(df=pd.DataFrame(), pnl={})
        n.solutions.at[(c, attr), 'in_comp'] = predefined
        if isinstance(variables, pd.DataFrame):
            # case that variables are timedependent
            n.solutions.at[(c, attr), 'pnl'] = True
            pnl = n.pnl(c) if predefined else n.sols[c].pnl
            values = variables.applymap(lambda x: variables_sol.loc[x])
            if c in n.passive_branch_components and attr == "s":
                set_from_frame(pnl, 'p0', values)
                set_from_frame(pnl, 'p1', - values)
            elif c == 'Link' and attr == "p":
                set_from_frame(pnl, 'p0', values)
                for i in ['1'] + additional_linkports(n):
                    i_eff = '' if i == '1' else i
                    eff = get_as_dense(n, 'Link', f'efficiency{i_eff}', sns)
                    set_from_frame(pnl, f'p{i}', - values * eff)
                    pnl[f'p{i}'].loc[sns, n.links.index[n.links[f'bus{i}'] == ""]] = \
                        n.component_attrs['Link'].loc[f'p{i}','default']
            else:
                set_from_frame(pnl, attr, values)
        else:
            # case that variables are static
            n.solutions.at[(c, attr), 'pnl'] = False
            sol = variables.map(variables_sol)
            if predefined:
                non_ext = n.df(c)[attr]
                n.df(c)[attr + '_opt'] = sol.reindex(non_ext.index).fillna(non_ext)
            else:
                n.sols[c].df[attr] = sol

    n.sols = Dict()
    n.solutions = pd.DataFrame(index=n.variables.index, columns=['in_comp', 'pnl'])
    for c, attr in n.variables.index:
        map_solution(c, attr)

    # if nominal capacity was no variable set optimal value to nominal
    for c, attr in lookup.query('nominal').index.difference(n.variables.index):
        n.df(c)[attr+'_opt'] = n.df(c)[attr]

    # recalculate storageunit net dispatch
    if not n.df('StorageUnit').empty:
        c = 'StorageUnit'
        n.pnl(c)['p'] = n.pnl(c)['p_dispatch'] - n.pnl(c)['p_store']

    # duals
    if keep_shadowprices == False:
        keep_shadowprices = []

    sp = n.constraints.index
    if isinstance(keep_shadowprices, list):
        sp = sp[sp.isin(keep_shadowprices, level=0)]

    def map_dual(c, attr):
        # If c is a pypsa component name the dual is stored at n.pnl(c)
        # or n.df(c). For the second case the index of the constraints have to
        # be a subset of n.df(c).index otherwise the dual is stored at
        # n.duals[c].df
        constraints = get_con(n, c, attr, pop=pop)
        is_pnl = isinstance(constraints, pd.DataFrame)
        # TODO: setting the sign is not very clear
        sign = 1 if 'upper' in attr or attr == 'marginal_price' else -1
        n.dualvalues.at[(c, attr), 'pnl'] = is_pnl
        to_component = c in n.all_components
        if is_pnl:
            n.dualvalues.at[(c, attr), 'in_comp'] = to_component
            duals = constraints.applymap(lambda x: sign * constraints_dual.loc[x]
                                         if x in constraints_dual.index else np.nan)
            if c not in n.duals and not to_component:
                n.duals[c] = Dict(df=pd.DataFrame(), pnl={})
            pnl = n.pnl(c) if to_component else n.duals[c].pnl
            set_from_frame(pnl, attr, duals)
        else:
            # here to_component can change
            duals = constraints.map(sign * constraints_dual)
            if to_component:
                to_component = (duals.index.isin(n.df(c).index).all())
            n.dualvalues.at[(c, attr), 'in_comp'] = to_component
            if c not in n.duals and not to_component:
                n.duals[c] = Dict(df=pd.DataFrame(), pnl={})
            df = n.df(c) if to_component else n.duals[c].df
            df[attr] = duals

    n.duals = Dict()
    n.dualvalues = pd.DataFrame(index=sp, columns=['in_comp', 'pnl'])
    # extract shadow prices attached to components
    for c, attr in sp:
        map_dual(c, attr)

    # correct prices for snapshot weightings
    n.buses_t.marginal_price.loc[sns] = (
        n.buses_t.marginal_price.loc[sns].divide(
            n.snapshot_weightings.objective.loc[sns],axis=0))

    # discard remaining if wanted
    if not keep_references:
        for c, attr in n.constraints.index.difference(sp):
            get_con(n, c, attr, pop)

    # load
    if len(n.loads):
        set_from_frame(n.pnl('Load'), 'p', get_as_dense(n, 'Load', 'p_set', sns))

    # clean up vars and cons
    for c in list(n.vars):
        if n.vars[c].df.empty and n.vars[c].pnl == {}: n.vars.pop(c)
    for c in list(n.cons):
        if n.cons[c].df.empty and n.cons[c].pnl == {}: n.cons.pop(c)

    # recalculate injection
    ca = [('Generator', 'p', 'bus' ), ('Store', 'p', 'bus'),
          ('Load', 'p', 'bus'), ('StorageUnit', 'p', 'bus'),
          ('Link', 'p0', 'bus0'), ('Link', 'p1', 'bus1')]
    for i in additional_linkports(n):
        ca.append(('Link', f'p{i}', f'bus{i}'))

    sign = lambda c: n.df(c).sign if 'sign' in n.df(c) else -1 #sign for 'Link'
    n.buses_t.p = pd.concat(
            [n.pnl(c)[attr].mul(sign(c)).rename(columns=n.df(c)[group])
             for c, attr, group in ca], axis=1).groupby(level=0, axis=1).sum()\
            .reindex(columns=n.buses.index, fill_value=0)

    def v_ang_for_(sub):
        buses_i = sub.buses_o
        if len(buses_i) == 1:
            return pd.DataFrame(0, index=sns, columns=buses_i)
        sub.calculate_B_H(skip_pre=True)
        Z = pd.DataFrame(np.linalg.pinv((sub.B).todense()), buses_i, buses_i)
        Z -= Z[sub.slack_bus]
        return n.buses_t.p.reindex(columns=buses_i) @ Z
    n.buses_t.v_ang = (pd.concat([v_ang_for_(sub) for sub in n.sub_networks.obj],
                                  axis=1)
                      .reindex(columns=n.buses.index, fill_value=0))


def network_lopf(n, snapshots=None, solver_name="cbc",
         solver_logfile=None, extra_functionality=None,
         multi_investment_periods=False, skip_objective=False,
         skip_pre=False, extra_postprocessing=None, formulation="kirchhoff",
         keep_references=False, keep_files=False,
         keep_shadowprices=['Bus', 'Line', 'Transformer', 'Link', 'GlobalConstraint'],
         solver_options=None, warmstart=False, store_basis=False,
         solver_dir=None):
    """
    Linear optimal power flow for a group of snapshots.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    solver_name : string
        Must be a solver name that pyomo recognises and that is
        installed, e.g. "glpk", "gurobi"
    solver_logfile : None|string
        If not None, sets the logfile option of the solver.
    solver_options : dictionary
        A dictionary with additional options that get passed to the solver.
        (e.g. {'threads':2} tells gurobi to use only 2 cpus)
    solver_dir : str, default None
        Path to directory where necessary files are written, default None leads
        to the default temporary directory used by tempfile.mkstemp().
    keep_files : bool, default False
        Keep the files that pyomo constructs from OPF problem
        construction, e.g. .lp file - useful for debugging
    formulation : string
        Formulation of the linear power flow equations to use; must be
        one of ["angles","cycles","kirchhoff","ptdf"]
    extra_functionality : callable function
        This function must take two arguments
        `extra_functionality(network,snapshots)` and is called after
        the model building is complete, but before it is sent to the
        solver. It allows the user to
        add/change constraints and add/change the objective function.
    skip_pre : bool, default False
        Skip the preliminary steps of computing topology.
    skip_objective : bool, default False
        Skip writing the default objective function. If False, a custom
        objective has to be defined via extra_functionality.
    extra_postprocessing : callable function
        This function must take three arguments
        `extra_postprocessing(network,snapshots,duals)` and is called after
        the model has solved and the results are extracted. It allows the user
        to extract further information about the solution, such as additional
        shadow prices.
    warmstart : bool or string, default False
        Use this to warmstart the optimization. Pass a string which gives
        the path to the basis file. If set to True, a path to
        a basis file must be given in network.basis_fn.
    store_basis : bool, default False
        Whether to store the basis of the optimization results. If True,
        the path to the basis file is saved in network.basis_fn. Note that
        a basis can only be stored if simplex, dual-simplex, or barrier
        *with* crossover is used for solving.
    keep_references : bool, default False
        Keep the references of variable and constraint names withing the
        network. These can be looked up in `n.vars` and `n.cons` after solving.
    keep_shadowprices : bool or list of component names
        Keep shadow prices for all constraints, if set to True. If a list
        is passed the shadow prices will only be parsed for those constraint
        names. Defaults to ['Bus', 'Line', 'GlobalConstraint'].
        After solving, the shadow prices can be retrieved using
        :func:`pypsa.linopt.get_dual` with corresponding name

    """
    supported_solvers = ["cbc", "gurobi", 'glpk', 'cplex', 'xpress']
    if solver_name not in supported_solvers:
        raise NotImplementedError(f"Solver {solver_name} not in "
                                  f"supported solvers: {supported_solvers}")

    if formulation != "kirchhoff":
        raise NotImplementedError("Only the kirchhoff formulation is supported")

    if n.generators.committable.any():
        logger.warning("Unit commitment is not yet completely implemented for "
        "optimising without pyomo. Thus minimum up time, minimum down time, "
        "start up costs, shut down costs will be ignored.")

    snapshots = _as_snapshots(n, snapshots)
    # check snapshots have right form for single or multiindex optimisation
    snapshots = snapshot_consistency(n, snapshots, multi_investment_periods)

    if not skip_pre:
        n.calculate_dependent_values()
        n.determine_network_topology()

    logger.info("Prepare linear problem")
    fdp, problem_fn = prepare_lopf(n, snapshots, keep_files, skip_objective,
                                   extra_functionality, solver_dir)
    fds, solution_fn = mkstemp(prefix='pypsa-solve', suffix='.sol', dir=solver_dir)

    if warmstart == True:
        warmstart = n.basis_fn
        logger.info("Solve linear problem using warmstart")
    else:
        logger.info(f"Solve linear problem using {solver_name.title()} solver")

    solve = eval(f'run_and_read_{solver_name}')
    res = solve(n, problem_fn, solution_fn, solver_logfile,
                solver_options, warmstart, store_basis)

    status, termination_condition, variables_sol, constraints_dual, obj = res

    if not keep_files:
        os.close(fdp); os.remove(problem_fn)
        os.close(fds); os.remove(solution_fn)

    if status == "ok" and termination_condition == "optimal":
        logger.info('Optimization successful. Objective value: {:.2e}'.format(obj))
    elif status == "warning" and termination_condition == "suboptimal":
        logger.warning('Optimization solution is sub-optimal. '
                       'Objective value: {:.2e}'.format(obj))
    else:
        logger.warning(f'Optimization failed with status {status} and '
                       f'termination condition {termination_condition}')
        return status, termination_condition

    n.objective = obj
    assign_solution(n, snapshots, variables_sol, constraints_dual,
                    keep_references=keep_references,
                    keep_shadowprices=keep_shadowprices)
    gc.collect()

    return status,termination_condition


def ilopf(n, snapshots=None, msq_threshold=0.05, min_iterations=1,
          max_iterations=100, track_iterations=False, **kwargs):
    '''
    Iterative linear optimization updating the line parameters for passive
    AC and DC lines. This is helpful when line expansion is enabled. After each
    sucessful solving, line impedances and line resistance are recalculated
    based on the optimization result. If warmstart is possible, it uses the
    result from the previous iteration to fasten the optimization.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    msq_threshold: float, default 0.05
        Maximal mean square difference between optimized line capacity of
        the current and the previous iteration. As soon as this threshold is
        undercut, and the number of iterations is bigger than 'min_iterations'
        the iterative optimization stops
    min_iterations : integer, default 1
        Minimal number of iteration to run regardless whether the msq_threshold
        is already undercut
    max_iterations : integer, default 100
        Maximal number of iterations to run regardless whether msq_threshold
        is already undercut
    track_iterations: bool, default False
        If True, the intermediate branch capacities and values of the
        objective function are recorded for each iteration. The values of
        iteration 0 represent the initial state.
    **kwargs
        Keyword arguments of the lopf function which runs at each iteration

    '''

    n.lines['carrier'] = n.lines.bus0.map(n.buses.carrier)
    ext_i = get_extendable_i(n, 'Line')
    typed_i = n.lines.query('type != ""').index
    ext_untyped_i = ext_i.difference(typed_i)
    ext_typed_i = ext_i.intersection(typed_i)
    base_s_nom = (np.sqrt(3) * n.lines['type'].map(n.line_types.i_nom) *
                  n.lines.bus0.map(n.buses.v_nom))
    n.lines.loc[ext_typed_i, 'num_parallel'] = (n.lines.s_nom/base_s_nom)[ext_typed_i]

    def update_line_params(n, s_nom_prev):
        factor = n.lines.s_nom_opt / s_nom_prev
        for attr, carrier in (('x', 'AC'), ('r', 'DC')):
            ln_i = (n.lines.query('carrier == @carrier').index.intersection(ext_untyped_i))
            n.lines.loc[ln_i, attr] /= factor[ln_i]
        ln_i = ext_i.intersection(typed_i)
        n.lines.loc[ln_i, 'num_parallel'] = (n.lines.s_nom_opt/base_s_nom)[ln_i]

    def msq_diff(n, s_nom_prev):
        lines_err = np.sqrt((s_nom_prev - n.lines.s_nom_opt).pow(2).mean()) / \
                        n.lines['s_nom_opt'].mean()
        logger.info(f"Mean square difference after iteration {iteration} is "
                    f"{lines_err}")
        return lines_err

    def save_optimal_capacities(n, iteration, status):
        for c, attr in pd.Series(nominal_attrs)[n.branch_components].items():
            n.df(c)[f'{attr}_opt_{iteration}'] = n.df(c)[f'{attr}_opt']
        setattr(n, f"status_{iteration}", status)
        setattr(n, f"objective_{iteration}", n.objective)
        n.iteration = iteration
        n.global_constraints = n.global_constraints.rename(columns={'mu': f'mu_{iteration}'})


    if track_iterations:
        for c, attr in pd.Series(nominal_attrs)[n.branch_components].items():
            n.df(c)[f'{attr}_opt_0'] = n.df(c)[f'{attr}']
    iteration = 1
    kwargs['store_basis'] = True
    diff = msq_threshold
    while diff >= msq_threshold or iteration < min_iterations:
        if iteration > max_iterations:
            logger.info(f'Iteration {iteration} beyond max_iterations '
                        f'{max_iterations}. Stopping ...')
            break

        s_nom_prev = n.lines.s_nom_opt.copy() if iteration else n.lines.s_nom.copy()
        kwargs['warmstart'] = bool(iteration and ('basis_fn' in n.__dir__()))
        status, termination_condition = network_lopf(n, snapshots, **kwargs)
        assert status == 'ok', (f'Optimization failed with status {status}'
                                f'and termination {termination_condition}')
        if track_iterations:
            save_optimal_capacities(n, iteration, status)
        update_line_params(n, s_nom_prev)
        diff = msq_diff(n, s_nom_prev)
        iteration += 1
    logger.info('Running last lopf with fixed branches (HVDC links and HVAC lines)')
    ext_dc_links_b = n.links.p_nom_extendable & (n.links.carrier == "DC")
    s_nom_orig = n.lines.s_nom.copy()
    p_nom_orig = n.links.p_nom.copy()
    n.lines.loc[ext_i, ['s_nom', 's_nom_extendable']] = n.lines.loc[ext_i, 's_nom_opt'], False
    n.links.loc[ext_dc_links_b, ["p_nom", "p_nom_extendable"]] = n.links.loc[ext_dc_links_b, "p_nom_opt"], False
    kwargs['warmstart'] = False
    network_lopf(n, snapshots, **kwargs)
    n.lines.loc[ext_i, ['s_nom', 's_nom_extendable']] = s_nom_orig.loc[ext_i], True
    n.links.loc[ext_dc_links_b, ['p_nom', 'p_nom_extendable']] = p_nom_orig.loc[ext_dc_links_b], True
    ## add costs of additional infrastructure to objective value of last iteration
    obj_links = n.links[ext_dc_links_b].eval("capital_cost * (p_nom_opt - p_nom_min)").sum()
    obj_lines = n.lines.eval("capital_cost * (s_nom_opt - s_nom_min)").sum()
    n.objective += obj_links + obj_lines
    n.objective_constant -= (obj_links + obj_lines)
