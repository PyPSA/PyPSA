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
                          expand_series, nominal_attrs)

from .linopt import (linexpr, write_bound, write_constraint, set_conref,
                     set_varref, get_con, get_var, reset_counter, join_exprs,
                     run_and_read_cbc, run_and_read_gurobi, run_and_read_glpk,
                     clear_references)


import pandas as pd
import numpy as np

import gc, string, random, time, os, re

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
    variables = write_bound(n, lower, upper)
    set_varref(n, variables, c, attr, pnl=False)


def define_dispatch_for_extendable_variables(n, sns, c, attr):
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
    if ext_i.empty: return
    variables = write_bound(n, -np.inf, np.inf, axes=[sns, ext_i])
    set_varref(n, variables, c, attr, pnl=True, spec='extendables')


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
    if fix_i.empty: return
    nominal_fix = n.df(c)[nominal_attrs[c]][fix_i]
    min_pu, max_pu = get_bounds_pu(n, c, sns, fix_i, attr)
    lower = min_pu.mul(nominal_fix)
    upper = max_pu.mul(nominal_fix)
    variables = write_bound(n, lower, upper)
    set_varref(n, variables, c, attr, pnl=True, spec='nonextendables')


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
    constraints = write_constraint(n, lhs, '>=', rhs, axes)
    set_conref(n, constraints, c, 'mu_upper', pnl=True, spec=attr)

    lhs, *axes = linexpr((min_pu, nominal_v), (-1, operational_ext_v),
                         return_axes=True)
    constraints = write_constraint(n, lhs, '<=', rhs, axes)
    set_conref(n, constraints, c, 'mu_lower', pnl=True, spec=attr)


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
        fix = n.pnl(c)[attr + '_set'].unstack().dropna()
        if fix.empty: return
        lhs = linexpr((1, get_var(n, c, attr).unstack()[fix.index]))
        constraints = write_constraint(n, lhs, '=', fix).unstack().T
    else:
        if attr + '_set' not in n.df(c): return
        fix = n.df(c)[attr + '_set'].dropna()
        if fix.empty: return
        lhs = linexpr((1, get_var(n, c, attr)[fix.index]))
        constraints = write_constraint(n, lhs, '=', fix)
    set_conref(n, constraints, c, f'mu_{attr}_set', pnl)


def define_ramp_limit_constraints(n, sns):
    """
    Defines ramp limits for generators wiht valid ramplimit

    """
    c = 'Generator'
    rup_i = n.df(c).query('ramp_limit_up == ramp_limit_up').index
    rdown_i = n.df(c).query('ramp_limit_down == ramp_limit_down').index
    if rup_i.empty & rdown_i.empty:
        return
    p = get_var(n, c, 'p').loc[sns[1:]]
    p_prev = get_var(n, c, 'p').shift(1).loc[sns[1:]]

    #fix up
    gens_i = rup_i & get_non_extendable_i(n, c)
    lhs = linexpr((1, p[gens_i]), (-1, p_prev[gens_i]), as_pandas=True)
    rhs = n.df(c).loc[gens_i].eval('ramp_limit_up * p_nom')
    constraints = write_constraint(n, lhs, '<=', rhs)
    set_conref(n, constraints, c, 'mu_ramp_limit_up', spec='nonextendables')

    #ext up
    gens_i = rup_i & get_extendable_i(n, c)
    limit_pu = n.df(c)['ramp_limit_up'][gens_i]
    p_nom = get_var(n, c, 'p_nom')[gens_i]
    lhs = linexpr((1, p[gens_i]), (-1, p_prev[gens_i]), (-limit_pu, p_nom),
                  as_pandas=True)
    constraints = write_constraint(n, lhs, '<=', 0)
    set_conref(n, constraints, c, 'mu_ramp_limit_up', spec='extendables')

    #fix down
    gens_i = rdown_i & get_non_extendable_i(n, c)
    lhs = linexpr((1, p[gens_i]), (-1, p_prev[gens_i]), as_pandas=True)
    rhs = n.df(c).loc[gens_i].eval('-1 * ramp_limit_down * p_nom')
    constraints = write_constraint(n, lhs, '>=', rhs)
    set_conref(n, constraints, c, 'mu_ramp_limit_down', spec='nonextendables')

    #ext down
    gens_i = rdown_i & get_extendable_i(n, c)
    limit_pu = n.df(c)['ramp_limit_down'][gens_i]
    p_nom = get_var(n, c, 'p_nom')[gens_i]
    lhs = linexpr((1, p[gens_i]), (-1, p_prev[gens_i]), (limit_pu, p_nom),
                  as_pandas=True)
    constraints = write_constraint(n, lhs, '>=', 0)
    set_conref(n, constraints, c, 'mu_ramp_limit_down', spec='extendables')


def define_nodal_balance_constraints(n, sns):
    """
    Defines nodal balance constraint.

    """

    def bus_injection(c, attr, groupcol='bus', sign=1):
        #additional sign only necessary for branches in reverse direction
        if 'sign' in n.df(c):
            sign = sign * n.df(c).sign
        return linexpr((sign, get_var(n, c, attr)), as_pandas=True)\
                       .rename(columns=n.df(c)[groupcol])

    # one might reduce this a bit by using n.branches and lookup
    args = [['Generator', 'p'], ['Store', 'p'], ['StorageUnit', 'p_dispatch'],
            ['StorageUnit', 'p_store', 'bus', -1], ['Line', 's', 'bus0', -1],
            ['Line', 's', 'bus1', 1], ['Transformer', 's', 'bus0', -1],
            ['Transformer', 's', 'bus1', 1], ['Link', 'p', 'bus0', -1],
            ['Link', 'p', 'bus1', n.links.efficiency]]
    args = [arg for arg in args if not n.df(arg[0]).empty]

    lhs = (pd.concat([bus_injection(*args) for args in args], axis=1)
           .groupby(axis=1, level=0)
           .agg(lambda x: ''.join(x.values))
           .reindex(columns=n.buses.index))
    sense = '='
    rhs = ((- n.loads_t.p_set.loc[sns] * n.loads.sign)
           .groupby(n.loads.bus, axis=1).sum()
           .reindex(columns=n.buses.index, fill_value=0))
    constraints = write_constraint(n, lhs, sense, rhs)
    set_conref(n, constraints, 'Bus', 'marginal_price')


def define_kirchhoff_constraints(n, sns):
    """
    Defines Kirchhoff voltage constraints

    """
    comps = n.passive_branch_components & set(n.variables.index.levels[0])
    branch_vars = pd.concat({c:get_var(n, c, 's') for c in comps}, axis=1)

    def cycle_flow(ds):
        ds = ds[lambda ds: ds!=0.].dropna()
        vals = linexpr((ds, branch_vars[ds.index])) + '\n'
        return vals.sum(1)

    constraints = []
    for sub in n.sub_networks.obj:
        branches = sub.branches()
        C = pd.DataFrame(sub.C.todense(), index=branches.index)
        if C.empty:
            continue
        carrier = n.sub_networks.carrier[sub.name]
        weightings = branches.x_pu_eff if carrier == 'AC' else branches.r_pu_eff
        C_weighted = 1e5 * C.mul(weightings, axis=0)
        cycle_sum = C_weighted.apply(cycle_flow)
        cycle_sum.index = sns
        con = write_constraint(n, cycle_sum, '=', 0)
        constraints.append(con)
    constraints = pd.concat(constraints, axis=1, ignore_index=True)
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
    #spillage
    upper = get_as_dense(n, c, 'inflow', sns).loc[:, lambda df: df.max() > 0]
    spill = write_bound(n, 0, upper)
    set_varref(n, spill, 'StorageUnit', 'spill')

    eh = expand_series(n.snapshot_weightings[sns], sus_i) #elapsed hours

    eff_stand = expand_series(1-n.df(c).standing_loss, sns).T.pow(eh)
    eff_dispatch = expand_series(n.df(c).efficiency_dispatch, sns).T
    eff_store = expand_series(n.df(c).efficiency_store, sns).T

    soc = get_var(n, c, 'state_of_charge')
    cyclic_i = n.df(c).query('cyclic_state_of_charge').index
    noncyclic_i = n.df(c).query('~cyclic_state_of_charge').index

    prev_soc_cyclic = soc.shift().fillna(soc.loc[sns[-1]])

    coeff_var = [(-1, soc),
                 (-1/eff_dispatch * eh, get_var(n, c, 'p_dispatch')),
                 (eff_store * eh, get_var(n, c, 'p_store'))]

    lhs, *axes = linexpr(*coeff_var, return_axes=True)

    def masked_term(coeff, var, cols):
        return linexpr((coeff[cols], var[cols]), as_pandas=True)\
               .reindex(index=axes[0], columns=axes[1], fill_value='').values

    if ('StorageUnit', 'spill') in n.variables.index:
        lhs += masked_term(-eh, get_var(n, c, 'spill'), spill.columns)
    lhs += masked_term(eff_stand, prev_soc_cyclic, cyclic_i)
    lhs += masked_term(eff_stand.loc[sns[1:]], soc.shift().loc[sns[1:]], noncyclic_i)

    rhs = -get_as_dense(n, c, 'inflow', sns).mul(eh)
    rhs.loc[sns[0], noncyclic_i] -= n.df(c).state_of_charge_initial[noncyclic_i]

    constraints = write_constraint(n, lhs, '==', rhs)
    set_conref(n, constraints, c, 'mu_state_of_charge')


def define_store_constraints(n, sns):
    """
    Defines energy balance constraints for stores. In principal this states:

        previous_e - p == e

    """
    stores_i = n.stores.index
    if stores_i.empty: return
    c = 'Store'
    variables = write_bound(n, -np.inf, np.inf, axes=[sns, stores_i])
    set_varref(n, variables, c, 'p')

    eh = expand_series(n.snapshot_weightings[sns], stores_i)  #elapsed hours
    eff_stand = expand_series(1-n.df(c).standing_loss, sns).T.pow(eh)

    e = get_var(n, c, 'e')
    cyclic_i = n.df(c).query('e_cyclic').index
    noncyclic_i = n.df(c).query('~e_cyclic').index

    previous_e_cyclic = e.shift().fillna(e.loc[sns[-1]])

    coeff_var = [(-eh, get_var(n, c, 'p')), (-1, e)]

    lhs, *axes = linexpr(*coeff_var, return_axes=True)

    def masked_term(coeff, var, cols):
        return linexpr((coeff[cols], var[cols]), as_pandas=True)\
               .reindex(index=axes[0], columns=axes[1], fill_value='').values

    lhs += masked_term(eff_stand, previous_e_cyclic, cyclic_i)
    lhs += masked_term(eff_stand.loc[sns[1:]], e.shift().loc[sns[1:]], noncyclic_i)

    rhs = pd.DataFrame(0, sns, stores_i)
    rhs.loc[sns[0], noncyclic_i] -= n.df(c)['e_initial'][noncyclic_i]

    constraints = write_constraint(n, lhs, '==', rhs)
    set_conref(n, constraints, c, 'mu_state_of_charge')


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

    """
    glcs = n.global_constraints.query('type == "primary_energy"')
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f'{carattr} != 0')[carattr]
        if emissions.empty: continue
        gens = n.generators.query('carrier in @emissions.index')
        em_pu = gens.carrier.map(emissions)/gens.efficiency
        em_pu = n.snapshot_weightings.to_frame() @ em_pu.to_frame('weightings').T
        vals = linexpr((em_pu, get_var(n, 'Generator', 'p')[gens.index]))
        lhs = join_exprs(vals)
        rhs = glc.constant

        #storage units
        sus = n.storage_units.query('carrier in @emissions.index and '
                                    'not cyclic_state_of_charge')
        sus_i = sus.index
        if not sus.empty:
            vals = linexpr((-sus.carrier.map(emissions),
                get_var(n, 'StorageUnit', 'state_of_charge').loc[sns[-1], sus_i]))
            lhs = lhs + '\n' + join_exprs(vals)
            rhs -= sus.carrier.map(emissions) @ sus.state_of_charge_initial

        #stores
        n.stores['carrier'] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query('carrier in @emissions.index and not e_cyclic')
        if not stores.empty:
            vals = linexpr((-stores.carrier.map(n.emissions),
                        get_var(n, 'Store', 'e').loc[sns[-1], stores.index]))
            lhs = lhs + '\n' + join_exprs(vals)
            rhs -= stores.carrier.map(emissions) @ stores.state_of_charge_initial


        con = write_constraint(n, lhs, glc.sense, rhs, axes=pd.Index([name]))
        set_conref(n, con, 'GlobalConstraint', 'mu', False, name)

    # for the next two to we need a line carrier
    if len(n.global_constraints) > len(glcs):
        n.lines['carrier'] = n.lines.bus0.map(n.buses.carrier)
    #expansion limits
    glcs = n.global_constraints.query('type == '
                                      '"transmission_volume_expansion_limit"')
    substr = lambda s: re.sub('[\[\]\(\)]', '', s)
    for name, glc in glcs.iterrows():
        car = [substr(c.strip()) for c in glc.carrier_attribute.split(',')]
        lhs = ''
        for c, attr in (('Line', 's_nom'), ('Link', 'p_nom')):
            ext_i = n.df(c).query(f'carrier in @car and {attr}_extendable').index
            if ext_i.empty: continue
            v = linexpr((n.df(c).length[ext_i], get_var(n, c, attr)[ext_i]))
            lhs += join_exprs(v) + '\n'
        if lhs == '': continue
        sense = glc.sense
        rhs = glc.constant
        con = write_constraint(n, lhs, sense, rhs, axes=pd.Index([name]))
        set_conref(n, con, 'GlobalConstraint', 'mu', False, name)

    #expansion cost limits
    glcs = n.global_constraints.query('type == '
                                      '"transmission_expansion_cost_limit"')
    for name, glc in glcs.iterrows():
        car = [substr(c.strip()) for c in glc.carrier_attribute.split(',')]
        lhs = ''
        for c, attr in (('Line', 's_nom'), ('Link', 'p_nom')):
            ext_i = n.df(c).query(f'carrier in @car and {attr}_extendable').index
            if ext_i.empty: continue
            v = linexpr((n.df(c).capital_cost[ext_i], get_var(n, c, attr)[ext_i]))
            lhs += join_exprs(v) + '\n'
        if lhs == '': continue
        sense = glc.sense
        rhs = glc.constant
        con = write_constraint(n, lhs, sense, rhs, axes=pd.Index([name]))
        set_conref(n, con, 'GlobalConstraint', 'mu', False, name)


def define_objective(n, sns):
    """
    Defines and writes out the objective function

    """
    for c, attr in lookup.query('marginal_cost').index:
        cost = (get_as_dense(n, c, 'marginal_cost', sns)
                .loc[:, lambda ds: (ds != 0).all()]
                .mul(n.snapshot_weightings[sns], axis=0))
        if cost.empty: continue
        terms = linexpr((cost, get_var(n, c, attr).loc[sns, cost.columns]))
        for t in terms.flatten():
            n.objective_f.write(t)
    #investment
    for c, attr in nominal_attrs.items():
        cost = n.df(c)['capital_cost'][get_extendable_i(n, c)]
        if cost.empty: continue
        terms = linexpr((cost, get_var(n, c, attr)[cost.index])) + '\n'
        for t in terms.flatten():
            n.objective_f.write(t)



def prepare_lopf(n, snapshots=None, keep_files=False,
                 extra_functionality=None):
    """
    Sets up the linear problem and writes it out to a lp file, stored at
    n.problem_fn

    """
    reset_counter()

    cols = ['component', 'name', 'pnl', 'specification']
    n.variables = pd.DataFrame(columns=cols).set_index(cols[:2])
    n.constraints = pd.DataFrame(columns=cols).set_index(cols[:2])

    snapshots = n.snapshots if snapshots is None else snapshots
    start = time.time()
    def time_info(message):
        logger.info(f'{message} {round(time.time()-start, 2)}s')

    n.identifier = ''.join(random.choice(string.ascii_lowercase)
                        for i in range(8))
    objective_fn = f"/tmp/objective-{n.identifier}.txt"
    constraints_fn = f"/tmp/constraints-{n.identifier}.txt"
    bounds_fn = f"/tmp/bounds-{n.identifier}.txt"
    n.problem_fn = f"/tmp/test-{n.identifier}.lp"

    n.objective_f = open(objective_fn, mode='w')
    n.constraints_f = open(constraints_fn, mode='w')
    n.bounds_f = open(bounds_fn, mode='w')

    n.objective_f.write('\* LOPF *\n\nmin\nobj:\n')
    n.constraints_f.write("\n\ns.t.\n\n")
    n.bounds_f.write("\nbounds\n")


    for c, attr in lookup.query('nominal and not handle_separately').index:
        define_nominal_for_extendable_variables(n, c, attr)
        # define_fixed_variable_constraints(n, snapshots, c, attr, pnl=False)
    for c, attr in lookup.query('not nominal and not handle_separately').index:
        define_dispatch_for_non_extendable_variables(n, snapshots, c, attr)
        define_dispatch_for_extendable_variables(n, snapshots, c, attr)
        define_dispatch_for_extendable_constraints(n, snapshots, c, attr)
        # define_fixed_variable_constraints(n, snapshots, c, attr)

    # consider only state_of_charge_set for the moment
    define_fixed_variable_constraints(n, snapshots, 'StorageUnit', 'state_of_charge')

    define_ramp_limit_constraints(n, snapshots)
    define_storage_unit_constraints(n, snapshots)
    define_store_constraints(n, snapshots)
    define_kirchhoff_constraints(n, snapshots)
    define_nodal_balance_constraints(n, snapshots)
    define_global_constraints(n, snapshots)
    define_objective(n, snapshots)

    if extra_functionality is not None:
        extra_functionality(n, snapshots)

    n.objective_f.close()
    n.constraints_f.close()
    n.bounds_f.write("end\n")
    n.bounds_f.close()

    del n.objective_f
    del n.constraints_f
    del n.bounds_f

    os.system(f"cat {objective_fn} {constraints_fn} {bounds_fn} "
              f"> {n.problem_fn}")

    time_info('Total preparation time:')

    if not keep_files:
        for fn in [objective_fn, constraints_fn, bounds_fn]:
            os.system("rm "+ fn)


def assign_solution(n, sns, variables_sol, constraints_dual,
                    keep_references=False, keep_shadowprices=None):
    """
    Helper function. Assigns the solution of a succesful optimization to the
    network.

    """
    def set_from_frame(c, attr, df):
        if n.pnl(c)[attr].empty:
            n.pnl(c)[attr] = df.reindex(n.snapshots)
        else:
            n.pnl(c)[attr].loc[sns, :] = df.reindex(columns=n.pnl(c)[attr].columns)

    pop = not keep_references
    #solutions, if nominal capcity was no variable set optimal value to nominal
    def map_solution(c, attr):
        if (c, attr) in n.variables.index:
            variables = get_var(n, c, attr, pop=pop)
            pnl = isinstance(variables, pd.DataFrame)
            if pnl:
                values = variables.stack().map(variables_sol).unstack()
                if c in n.passive_branch_components:
                    set_from_frame(c, 'p0', values)
                    set_from_frame(c, 'p1', - values)
                elif c == 'Link':
                    set_from_frame(c, 'p0', values)
                    set_from_frame(c, 'p1', - values * n.df(c).efficiency)
                else:
                    set_from_frame(c, attr, values)
            else:
                n.df(c)[attr+'_opt'] = variables.map(variables_sol)\
                                        .fillna(n.df(c)[attr])
        elif lookup.at[(c, attr), 'nominal']:
            n.df(c)[attr+'_opt'] = n.df(c)[attr]

    for c, attr in lookup.index:
        map_solution(c, attr)

    if not n.df('StorageUnit').empty:
        c = 'StorageUnit'
        n.pnl(c)['p'] = n.pnl(c)['p_dispatch'] - n.pnl(c)['p_store']

    #duals
    def map_dual(c, attr, pnl):
        sign = 1 if 'upper' in attr else -1
        if pnl:
            set_from_frame(c, attr, get_con(n, c, attr, pop=pop).stack()
                            .map(sign * constraints_dual).unstack())
        else:
            n.df(c)[attr] = get_con(n, c, attr, pop=pop).map(sign* constraints_dual)

    if keep_shadowprices == False:
        keep_shadowprices = []
    elif keep_shadowprices is None:
        keep_shadowprices = ['Bus', 'Line', 'GlobalConstraint']

    for (c, attr), pnl in n.constraints.pnl.items():
        if keep_shadowprices == True:
            map_dual(c, attr, pnl)
        elif c in keep_shadowprices:
            map_dual(c, attr, pnl)
        else:
            get_con(n, c, attr, pop=True)

    #load
    n.loads_t.p = n.loads_t.p_set

    # recalculate injection
    ca = [('Generator', 'p', 'bus' ), ('Store', 'p', 'bus'),
          ('Load', 'p', 'bus'), ('StorageUnit', 'p', 'bus'),
          ('Link', 'p0', 'bus0'), ('Link', 'p1', 'bus1')]
    sign = lambda c: n.df(c).sign if 'sign' in n.df(c) else -1 #sign for 'Link'
    n.buses_t.p = pd.concat(
            [n.pnl(c)[attr].mul(sign(c)).rename(columns=n.df(c)[group])
             for c, attr, group in ca], axis=1).groupby(level=0, axis=1).sum()\
            .reindex(columns=n.buses.index, fill_value=0)

    def v_ang_for_(sub):
        buses_i = sub.buses_o
        if len(buses_i) == 1: return
        sub.calculate_B_H(skip_pre=True)
        if len(sub.buses_i()) == 1: return
        Z = pd.DataFrame(np.linalg.pinv((sub.B).todense()), buses_i, buses_i)
        Z -= Z[sub.slack_bus]
        return n.buses_t.p.reindex(columns=buses_i) @ Z
    n.buses_t.v_ang = (pd.concat(
                       [v_ang_for_(sub) for sub in n.sub_networks.obj], axis=1)
                      .reindex(columns=n.buses.index, fill_value=0))


def network_lopf(n, snapshots=None, solver_name="cbc",
         solver_logfile=None, extra_functionality=None,
         extra_postprocessing=None, formulation="kirchhoff",
         keep_references=False, keep_files=False,
         keep_shadowprices=None, solver_options={},
         warmstart=False, store_basis=True):
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
    pyomo : bool, default True
        Whether to use pyomo for building and solving the model, setting
        this to False saves a lot of memory and time.
    solver_logfile : None|string
        If not None, sets the logfile option of the solver.
    solver_options : dictionary
        A dictionary with additional options that get passed to the solver.
        (e.g. {'threads':2} tells gurobi to use only 2 cpus)
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
    store_basis : bool, default True
        Whether to store the basis of the optimization results. If True,
        the path to the basis file is saved in network.basis_fn. Note that
        a basis can only be stored if simplex, dual-simplex, or barrier
        *with* crossover is used for solving.
    keep_references : bool, default False
        Keep the references of variable and constraint names withing the
        network, e.g. n.generators_t.p_varref - useful for constructing
        extra_functionality or debugging
    keep_shadowprices : bool or list of component names, default None
        Keep shadow prices for all constraints, if set to True.
        These are stored at e.g. n.generators_t.mu_upper for upper limit
        of p_nom. If a list of component names is passed, shadow
        prices of variables attached to those are extracted. If set to None,
        components default to ['Bus', 'Line', 'GlobalConstraint']

    """
    supported_solvers = ["cbc", "gurobi", 'glpk', 'scs']
    if solver_name not in supported_solvers:
        raise NotImplementedError(f"Solver {solver_name} not in "
                                  f"supported solvers: {supported_solvers}")

    if formulation != "kirchhoff":
        raise NotImplementedError("Only the kirchhoff formulation is supported")

    if n.generators.committable.any():
        logger.warn("Unit commitment is not yet implemented for optimsation "
           "without using pyomo. The following generators will be treated as "
          f"non-commitables:\n{list(n.generators.query('committable').index)}")

    #disable logging because multiple slack bus calculations, keep output clean
    snapshots = _as_snapshots(n, snapshots)
    n.calculate_dependent_values()
    n.determine_network_topology()
    clear_references(n)


    logger.info("Prepare linear problem")
    prepare_lopf(n, snapshots, keep_files, extra_functionality)
    gc.collect()
    solution_fn = f"/tmp/pypsa-solve-{n.identifier}.sol"
    if solver_logfile is None:
        solver_logfile = f"pypsa-solve-{n.identifier}.log"

    if warmstart == True:
        warmstart = n.basis_fn
        logger.info("Solve linear problem using warmstart")
    else:
        logger.info("Solve linear problem")

    solve = eval(f'run_and_read_{solver_name}')
    res = solve(n, n.problem_fn, solution_fn, solver_logfile,
                solver_options, keep_files, warmstart, store_basis)
    status, termination_condition, variables_sol, constraints_dual, obj = res
    del n.problem_fn

    if termination_condition != "optimal":
        return status,termination_condition

    #adjust objective value
    for c, attr in nominal_attrs.items():
        obj -= n.df(c)[attr] @ n.df(c).capital_cost
    n.objective = obj
    gc.collect()
    assign_solution(n, snapshots, variables_sol, constraints_dual,
                    keep_references=keep_references,
                    keep_shadowprices=keep_shadowprices)
    gc.collect()

    return status,termination_condition


def ilopf(n, snapshots=None, msq_threshold=0.05, min_iterations=1,
          max_iterations=100, **kwargs):
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
        Maximal numbder of iterations to run regardless whether msq_threshold
        is already undercut
    **kwargs
        Keyword arguments of the lopf function which runs at each iteration

    '''

    ext_i = get_extendable_i(n, 'Line')
    typed_i = n.lines.query('type != ""').index
    ext_untyped_i = ext_i.difference(typed_i)
    ext_typed_i = ext_i & typed_i
    base_s_nom = (np.sqrt(3) * n.lines['type'].map(n.line_types.i_nom) *
                  n.lines.bus0.map(n.buses.v_nom))
    n.lines.loc[ext_typed_i, 'num_parallel'] = (n.lines.s_nom/base_s_nom)[ext_typed_i]

    def update_line_params(n, s_nom_prev):
        factor = n.lines.s_nom_opt / s_nom_prev
        for attr, carrier in (('x', 'AC'), ('r', 'DC')):
            ln_i = (n.lines.query('carrier == @carrier').index & ext_untyped_i)
            n.lines.loc[ln_i, attr] /= factor[ln_i]
        ln_i = ext_i & typed_i
        n.lines.loc[ln_i, 'num_parallel'] = (n.lines.s_nom_opt/base_s_nom)[ln_i]

    def msq_diff(n, s_nom_prev):
        lines_err = np.sqrt((s_nom_prev - n.lines.s_nom_opt).pow(2).mean()) / \
                        n.lines['s_nom_opt'].mean()
        logger.info(f"Mean square difference after iteration {iteration} is "
                    f"{lines_err}")
        return lines_err

    iteration = 0
    diff = msq_threshold
    while diff >= msq_threshold or iteration < min_iterations:
        if iteration >= max_iterations:
            logger.info(f'Iteration {iteration} beyond max_iterations '
                        f'{max_iterations}. Stopping ...')
            break

        s_nom_prev = n.lines.s_nom_opt if iteration else n.lines.s_nom
        kwargs['warmstart'] = bool(iteration and ('basis_fn' in n.__dir__()))
        network_lopf(n, snapshots, **kwargs)
        update_line_params(n, s_nom_prev)
        diff = msq_diff(n, s_nom_prev)
        iteration += 1
    logger.info('Running last lopf with fixed branches, overwrite p_nom '
                'for links and s_nom for lines')
    ext_links_i = get_extendable_i(n, 'Link')
    n.lines[['s_nom', 's_nom_extendable']] = n.lines['s_nom_opt'], False
    n.links[['p_nom', 'p_nom_extendable']] = n.links['p_nom_opt'], False
    network_lopf(n, snapshots, **kwargs)
    n.lines.loc[ext_i, 's_nom_extendable'] = True
    n.links.loc[ext_links_i, 'p_nom_extendable'] = True


