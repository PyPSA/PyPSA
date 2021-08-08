
## Copyright 2015-2021 PyPSA Developers

## You can find the list of PyPSA Developers at
## https://pypsa.readthedocs.io/en/latest/developers.html

## PyPSA is released under the open source MIT License, see
## https://github.com/PyPSA/PyPSA/blob/master/LICENSE.txt

"""Optimal Power Flow functions.
"""

__author__ = "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
__copyright__ = ("Copyright 2015-2021 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
                 "MIT License")

import numpy as np
import pandas as pd
from scipy.sparse.linalg import spsolve
from pyomo.environ import (ConcreteModel, Var, NonNegativeReals, Constraint,
                           Reals, Suffix, Binary, SolverFactory)

try:
    from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
except ImportError:
    # Only used in conjunction with isinstance, so we mock it to be backwards compatible
    class PersistentSolver(): pass

import logging
logger = logging.getLogger(__name__)


from .pf import (calculate_dependent_values, find_slack_bus,
                 find_bus_controls, calculate_B_H, calculate_PTDF, find_tree,
                 find_cycles, _as_snapshots)
from .opt import (l_constraint, l_objective, LExpression, LConstraint,
                  patch_optsolver_record_memusage_before_solving,
                  empty_network, free_pyomo_initializers)
from .descriptors import (get_switchable_as_dense, get_switchable_as_iter,
                          allocate_series_dataframes, zsum)

pd.Series.zsum = zsum



def network_opf(network,snapshots=None):
    """Optimal power flow for snapshots."""

    raise NotImplementedError("Non-linear optimal power flow not supported yet")



def define_generator_variables_constraints(network,snapshots):

    extendable_gens_i = network.generators.index[network.generators.p_nom_extendable]
    fixed_gens_i = network.generators.index[~network.generators.p_nom_extendable & ~network.generators.committable]
    fixed_committable_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable]

    if (network.generators.p_nom_extendable & network.generators.committable).any():
        logger.warning("The following generators have both investment optimisation and unit commitment:\n{}\nCurrently PyPSA cannot do both these functions, so PyPSA is choosing investment optimisation for these generators.".format(network.generators.index[network.generators.p_nom_extendable & network.generators.committable]))

    bad_uc_gens = network.generators.index[network.generators.committable & (network.generators.min_up_time > 0) & (network.generators.min_down_time > 0) & (network.generators.up_time_before > 0) & (network.generators.down_time_before > 0)]
    if not bad_uc_gens.empty:
        logger.warning("The following committable generators were both up and down before the simulation: {}. This will cause an infeasibility.".format(bad_uc_gens))

    start_i = network.snapshots.get_loc(snapshots[0])

    p_min_pu = get_switchable_as_dense(network, 'Generator', 'p_min_pu', snapshots)
    p_max_pu = get_switchable_as_dense(network, 'Generator', 'p_max_pu', snapshots)

    ## Define generator dispatch variables ##

    gen_p_bounds = {(gen,sn) : (None,None)
                    for gen in extendable_gens_i.union(fixed_committable_gens_i)
                    for sn in snapshots}

    if len(fixed_gens_i):
        var_lower = p_min_pu.loc[:,fixed_gens_i].multiply(network.generators.loc[fixed_gens_i, 'p_nom'])
        var_upper = p_max_pu.loc[:,fixed_gens_i].multiply(network.generators.loc[fixed_gens_i, 'p_nom'])

        gen_p_bounds.update({(gen,sn) : (var_lower[gen][sn],var_upper[gen][sn])
                             for gen in fixed_gens_i
                             for sn in snapshots})

    def gen_p_bounds_f(model,gen_name,snapshot):
        return gen_p_bounds[gen_name,snapshot]

    network.model.generator_p = Var(list(network.generators.index), snapshots,
                                    domain=Reals, bounds=gen_p_bounds_f)
    free_pyomo_initializers(network.model.generator_p)

    ## Define generator capacity variables if generator is extendable ##

    def gen_p_nom_bounds(model, gen_name):
        return (network.generators.at[gen_name,"p_nom_min"],
                network.generators.at[gen_name,"p_nom_max"])

    network.model.generator_p_nom = Var(list(extendable_gens_i),
                                        domain=NonNegativeReals, bounds=gen_p_nom_bounds)
    free_pyomo_initializers(network.model.generator_p_nom)


    ## Define generator dispatch constraints for extendable generators ##

    gen_p_lower = {(gen,sn) :
                   [[(1,network.model.generator_p[gen,sn]),
                     (-p_min_pu.at[sn, gen],
                      network.model.generator_p_nom[gen])],">=",0.]
                   for gen in extendable_gens_i for sn in snapshots}
    l_constraint(network.model, "generator_p_lower", gen_p_lower,
                 list(extendable_gens_i), snapshots)

    gen_p_upper = {(gen,sn) :
                   [[(1,network.model.generator_p[gen,sn]),
                     (-p_max_pu.at[sn, gen],
                      network.model.generator_p_nom[gen])],"<=",0.]
                   for gen in extendable_gens_i for sn in snapshots}
    l_constraint(network.model, "generator_p_upper", gen_p_upper,
                 list(extendable_gens_i), snapshots)



    ## Define committable generator statuses ##

    network.model.generator_status = Var(list(fixed_committable_gens_i), snapshots,
                                         within=Binary)

    var_lower = p_min_pu.loc[:,fixed_committable_gens_i].multiply(network.generators.loc[fixed_committable_gens_i, 'p_nom'])
    var_upper = p_max_pu.loc[:,fixed_committable_gens_i].multiply(network.generators.loc[fixed_committable_gens_i, 'p_nom'])


    committable_gen_p_lower = {(gen,sn) : LConstraint(LExpression([(var_lower[gen][sn],network.model.generator_status[gen,sn]),(-1.,network.model.generator_p[gen,sn])]),"<=") for gen in fixed_committable_gens_i for sn in snapshots}

    l_constraint(network.model, "committable_gen_p_lower", committable_gen_p_lower,
                 list(fixed_committable_gens_i), snapshots)


    committable_gen_p_upper = {(gen,sn) : LConstraint(LExpression([(var_upper[gen][sn],network.model.generator_status[gen,sn]),(-1.,network.model.generator_p[gen,sn])]),">=") for gen in fixed_committable_gens_i for sn in snapshots}

    l_constraint(network.model, "committable_gen_p_upper", committable_gen_p_upper,
                 list(fixed_committable_gens_i), snapshots)


    ## Deal with minimum up time ##

    up_time_gens = fixed_committable_gens_i[network.generators.loc[fixed_committable_gens_i,"min_up_time"] > 0]
    must_stay_up_too_long = False

    for gen_i, gen in enumerate(up_time_gens):

        min_up_time = network.generators.at[gen, 'min_up_time']

        #find out how long the generator has been up before snapshots
        up_time_before = 0
        for i in range(1,min(min_up_time,start_i)+1):
            if network.generators_t.status.at[network.snapshots[start_i-i],gen] == 0:
                break
            else:
                up_time_before += 1

        if up_time_before == start_i:
            up_time_before = min(min_up_time,start_i+network.generators.at[gen,"up_time_before"])

        if up_time_before == 0:
            initial_status = 0
            must_stay_up = 0
        else:
            initial_status = 1
            must_stay_up = min_up_time - up_time_before
            if must_stay_up > len(snapshots):
                must_stay_up_too_long = True
                must_stay_up = len(snapshots)


        def force_up(model,i):
            return model.generator_status[gen,snapshots[i]] == 1
        network.model.add_component("gen_up_time_force_{}".format(gen_i),Constraint(range(must_stay_up),rule=force_up))

        blocks = range(must_stay_up,len(snapshots)-1)
        def gen_rule(model,i):
            period = min(min_up_time,len(snapshots)-i)
            lhs = sum(network.model.generator_status[gen,snapshots[j]] for j in range(i,i+period))
            if i == 0:
                rhs = period*network.model.generator_status[gen,snapshots[i]] - period*initial_status
            else:
                rhs = period*network.model.generator_status[gen,snapshots[i]] - period*network.model.generator_status[gen,snapshots[i-1]]
            return lhs >= rhs
        network.model.add_component("gen_up_time_{}".format(gen_i),Constraint(blocks,rule=gen_rule))

    if must_stay_up_too_long:
        logger.warning('At least one generator was set to an min_up_time longer '
                       'than possible. Setting it to the maximal possible value.')

    ## Deal with minimum down time ##

    down_time_gens = fixed_committable_gens_i[network.generators.loc[fixed_committable_gens_i,"min_down_time"] > 0]

    for gen_i, gen in enumerate(down_time_gens):

        min_down_time = network.generators.at[gen,"min_down_time"]

        #find out how long the generator has been down before snapshots
        down_time_before = 0
        for i in range(1,min(min_down_time,start_i)+1):
            if network.generators_t.status.at[network.snapshots[start_i-i],gen] == 1:
                break
            else:
                down_time_before += 1

        if down_time_before == start_i:
            down_time_before = min(min_down_time,start_i+network.generators.at[gen,"down_time_before"])

        if down_time_before == 0:
            initial_status = 1
            must_stay_down = 0
        else:
            initial_status = 0
            must_stay_down = min_down_time - down_time_before


        def force_down(model,i):
            return model.generator_status[gen,snapshots[i]] == 0
        network.model.add_component("gen_down_time_force_{}".format(gen_i),Constraint(range(must_stay_down),rule=force_down))

        blocks = range(must_stay_down,len(snapshots)-1)
        def gen_rule(model,i):
            period = min(min_down_time,len(snapshots)-i)
            lhs = period - sum(network.model.generator_status[gen,snapshots[j]] for j in range(i,i+period))
            if i == 0:
                rhs = -period*network.model.generator_status[gen,snapshots[i]] + period*initial_status
            else:
                rhs = -period*network.model.generator_status[gen,snapshots[i]] + period*network.model.generator_status[gen,snapshots[i-1]]
            return lhs >= rhs
        network.model.add_component("gen_down_time_{}".format(gen_i),Constraint(blocks,rule=gen_rule))


    ## Deal with start up costs ##

    suc_gens = fixed_committable_gens_i[network.generators.loc[fixed_committable_gens_i,"start_up_cost"] > 0]

    network.model.generator_start_up_cost = Var(list(suc_gens),snapshots,
                                                domain=NonNegativeReals)

    sucs = {}

    for gen in suc_gens:
        suc = network.generators.at[gen,"start_up_cost"]

        if start_i == 0:
            if network.generators.at[gen,"up_time_before"] > 0:
                initial_status = 1
            else:
                initial_status = 0
        else:
            initial_status = network.generators_t.status.at[network.snapshots[start_i-1],gen]

        for i,sn in enumerate(snapshots):

            if i == 0:
                rhs = LExpression([(suc, network.model.generator_status[gen,sn])],-suc*initial_status)
            else:
                rhs = LExpression([(suc, network.model.generator_status[gen,sn]),(-suc,network.model.generator_status[gen,snapshots[i-1]])])

            lhs = LExpression([(1,network.model.generator_start_up_cost[gen,sn])])

            sucs[gen,sn] = LConstraint(lhs,">=",rhs)

    l_constraint(network.model, "generator_start_up", sucs, list(suc_gens), snapshots)



    ## Deal with shut down costs ##

    sdc_gens = fixed_committable_gens_i[network.generators.loc[fixed_committable_gens_i,"shut_down_cost"] > 0]

    network.model.generator_shut_down_cost = Var(list(sdc_gens),snapshots,
                                                domain=NonNegativeReals)

    sdcs = {}

    for gen in sdc_gens:
        sdc = network.generators.loc[gen,"shut_down_cost"]

        if start_i == 0:
            if network.generators.at[gen,"down_time_before"] > 0:
                initial_status = 0
            else:
                initial_status = 1
        else:
            initial_status = network.generators_t.status.at[network.snapshots[start_i-1],gen]

        for i,sn in enumerate(snapshots):

            if i == 0:
                rhs = LExpression([(-sdc, network.model.generator_status[gen,sn])],sdc*initial_status)
            else:
                rhs = LExpression([(-sdc, network.model.generator_status[gen,sn]),(sdc,network.model.generator_status[gen,snapshots[i-1]])])

            lhs = LExpression([(1,network.model.generator_shut_down_cost[gen,sn])])

            sdcs[gen,sn] = LConstraint(lhs,">=",rhs)

    l_constraint(network.model, "generator_shut_down", sdcs, list(sdc_gens), snapshots)



    ## Deal with ramp limits without unit commitment ##

    ru_gens = network.generators.index[network.generators.ramp_limit_up.notnull()]

    ru = {}

    for gen in ru_gens:
        for sn, sn_prev in zip(snapshots[1:], snapshots[:-1]):
            if network.generators.at[gen, "p_nom_extendable"]:
                lhs = LExpression([(1, network.model.generator_p[gen,sn]),
                                   (-1, network.model.generator_p[gen,sn_prev]),
                                   (-network.generators.at[gen, "ramp_limit_up"],
                                    network.model.generator_p_nom[gen])])
            elif not network.generators.at[gen, "committable"]:
                lhs = LExpression([(1, network.model.generator_p[gen,sn]),
                                   (-1, network.model.generator_p[gen,sn_prev])],
                                  -network.generators.at[gen, "ramp_limit_up"]*network.generators.at[gen, "p_nom"])
            else:
                lhs = LExpression([(1, network.model.generator_p[gen,sn]),
                                   (-1, network.model.generator_p[gen,sn_prev]),
                                   ((network.generators.at[gen, "ramp_limit_start_up"] - network.generators.at[gen, "ramp_limit_up"])*network.generators.at[gen, "p_nom"],
                                    network.model.generator_status[gen,sn_prev]),
                                   (-network.generators.at[gen, "ramp_limit_start_up"]*network.generators.at[gen, "p_nom"],
                                    network.model.generator_status[gen,sn])])

            ru[gen,sn] = LConstraint(lhs,"<=")

    l_constraint(network.model, "ramp_up", ru, list(ru_gens), snapshots[1:])

    #case of ramping if not at start of network.snapshots
    if start_i > 0:
        ru_start = {}
        sn = snapshots[0]
        for gen in ru_gens:
            p_prev = network.generators_t.p.at[network.snapshots[start_i-1],gen]
            if network.generators.at[gen, "p_nom_extendable"]:
                lhs = LExpression([(1, network.model.generator_p[gen,sn]),
                                   (-network.generators.at[gen, "ramp_limit_up"],
                                    network.model.generator_p_nom[gen])],
                                  -p_prev)
            elif not network.generators.at[gen, "committable"]:
                lhs = LExpression([(1, network.model.generator_p[gen,sn])],
                                  -network.generators.at[gen, "ramp_limit_up"]*network.generators.at[gen, "p_nom"]-p_prev)
            else:
                status_prev = network.generators_t.status.at[network.snapshots[start_i-1],gen]
                lhs = LExpression([(1, network.model.generator_p[gen,sn]),
                                   (-network.generators.at[gen, "ramp_limit_start_up"]*network.generators.at[gen, "p_nom"],
                                    network.model.generator_status[gen,sn])],
                                  -p_prev + status_prev*(network.generators.at[gen, "ramp_limit_start_up"] - network.generators.at[gen, "ramp_limit_up"])*network.generators.at[gen, "p_nom"])
            ru_start[gen] = LConstraint(lhs,"<=")

        l_constraint(network.model, "ramp_up_start", ru_start, list(ru_gens))


    rd_gens = network.generators.index[network.generators.ramp_limit_down.notnull()]

    rd = {}


    for gen in rd_gens:
        for sn, sn_prev in zip(snapshots[1:], snapshots[:-1]):
            if network.generators.at[gen, "p_nom_extendable"]:
                lhs = LExpression([(1, network.model.generator_p[gen,sn]),
                                   (-1, network.model.generator_p[gen,sn_prev]),
                                   (network.generators.at[gen, "ramp_limit_down"],
                                    network.model.generator_p_nom[gen])])
            elif not network.generators.at[gen, "committable"]:
                lhs = LExpression([(1, network.model.generator_p[gen,sn]),
                                   (-1, network.model.generator_p[gen,sn_prev])],
                                  network.generators.loc[gen, "ramp_limit_down"]*network.generators.at[gen, "p_nom"])
            else:
                lhs = LExpression([(1, network.model.generator_p[gen,sn]),
                                   (-1, network.model.generator_p[gen,sn_prev]),
                                   ((network.generators.at[gen, "ramp_limit_down"] - network.generators.at[gen, "ramp_limit_shut_down"])*network.generators.at[gen, "p_nom"],
                                    network.model.generator_status[gen,sn]),
                                   (network.generators.at[gen, "ramp_limit_shut_down"]*network.generators.at[gen, "p_nom"],
                                    network.model.generator_status[gen,sn_prev])])

            rd[gen,sn] = LConstraint(lhs,">=")

    l_constraint(network.model, "ramp_down", rd, list(rd_gens), snapshots[1:])


    #case of ramping if not at start of network.snapshots
    if start_i > 0:
        rd_start = {}
        sn = snapshots[0]
        for gen in rd_gens:
            p_prev = network.generators_t.p.at[network.snapshots[start_i-1],gen]
            if network.generators.at[gen, "p_nom_extendable"]:
                lhs = LExpression([(1, network.model.generator_p[gen,sn]),
                                   (network.generators.at[gen, "ramp_limit_down"],
                                    network.model.generator_p_nom[gen])],
                                  -p_prev)
            elif not network.generators.at[gen, "committable"]:
                lhs = LExpression([(1, network.model.generator_p[gen,sn])],
                                  network.generators.loc[gen, "ramp_limit_down"]*network.generators.at[gen, "p_nom"]-p_prev)
            else:
                status_prev = network.generators_t.status.at[network.snapshots[start_i-1],gen]
                lhs = LExpression([(1, network.model.generator_p[gen,sn]),
                                   ((network.generators.at[gen, "ramp_limit_down"] - network.generators.at[gen, "ramp_limit_shut_down"])*network.generators.at[gen, "p_nom"],
                                    network.model.generator_status[gen,sn])],
                                  -p_prev + status_prev*network.generators.at[gen, "ramp_limit_shut_down"]*network.generators.at[gen, "p_nom"])

            rd_start[gen] = LConstraint(lhs,">=")

        l_constraint(network.model, "ramp_down_start", rd_start, list(rd_gens))




def define_storage_variables_constraints(network,snapshots):

    sus = network.storage_units
    ext_sus_i = sus.index[sus.p_nom_extendable]
    fix_sus_i = sus.index[~ sus.p_nom_extendable]

    model = network.model

    ## Define storage dispatch variables ##

    p_max_pu = get_switchable_as_dense(network, 'StorageUnit', 'p_max_pu', snapshots)
    p_min_pu = get_switchable_as_dense(network, 'StorageUnit', 'p_min_pu', snapshots)

    bounds = {(su,sn) : (0,None) for su in ext_sus_i for sn in snapshots}
    bounds.update({(su,sn) :
                   (0,sus.at[su,"p_nom"]*p_max_pu.at[sn, su])
                   for su in fix_sus_i for sn in snapshots})

    def su_p_dispatch_bounds(model,su_name,snapshot):
        return bounds[su_name,snapshot]

    network.model.storage_p_dispatch = Var(list(network.storage_units.index), snapshots,
                                           domain=NonNegativeReals, bounds=su_p_dispatch_bounds)
    free_pyomo_initializers(network.model.storage_p_dispatch)



    bounds = {(su,sn) : (0,None) for su in ext_sus_i for sn in snapshots}
    bounds.update({(su,sn) :
                   (0,-sus.at[su,"p_nom"]*p_min_pu.at[sn, su])
                   for su in fix_sus_i
                   for sn in snapshots})

    def su_p_store_bounds(model,su_name,snapshot):
        return bounds[su_name,snapshot]

    network.model.storage_p_store = Var(list(network.storage_units.index), snapshots,
                                        domain=NonNegativeReals, bounds=su_p_store_bounds)
    free_pyomo_initializers(network.model.storage_p_store)

    ## Define spillage variables only for hours with inflow>0. ##
    inflow = get_switchable_as_dense(network, 'StorageUnit', 'inflow', snapshots)
    spill_sus_i = sus.index[inflow.max()>0] #skip storage units without any inflow
    inflow_gt0_b = inflow>0
    spill_bounds = {(su,sn) : (0,inflow.at[sn,su])
                    for su in spill_sus_i
                    for sn in snapshots
                    if inflow_gt0_b.at[sn,su]}
    spill_index = spill_bounds.keys()

    def su_p_spill_bounds(model,su_name,snapshot):
        return spill_bounds[su_name,snapshot]

    network.model.storage_p_spill = Var(list(spill_index),
                                        domain=NonNegativeReals, bounds=su_p_spill_bounds)
    free_pyomo_initializers(network.model.storage_p_spill)


    ## Define generator capacity variables if generator is extendable ##

    def su_p_nom_bounds(model, su_name):
        return (sus.at[su_name,"p_nom_min"],
                sus.at[su_name,"p_nom_max"])

    network.model.storage_p_nom = Var(list(ext_sus_i), domain=NonNegativeReals,
                                      bounds=su_p_nom_bounds)
    free_pyomo_initializers(network.model.storage_p_nom)


    ## Define generator dispatch constraints for extendable generators ##

    def su_p_upper(model,su_name,snapshot):
        return (model.storage_p_dispatch[su_name,snapshot] <=
                model.storage_p_nom[su_name]*p_max_pu.at[snapshot, su_name])

    network.model.storage_p_upper = Constraint(list(ext_sus_i),snapshots,rule=su_p_upper)
    free_pyomo_initializers(network.model.storage_p_upper)

    def su_p_lower(model,su_name,snapshot):
        return (model.storage_p_store[su_name,snapshot] <=
                -model.storage_p_nom[su_name]*p_min_pu.at[snapshot, su_name])

    network.model.storage_p_lower = Constraint(list(ext_sus_i),snapshots,rule=su_p_lower)
    free_pyomo_initializers(network.model.storage_p_lower)


    ## Now define state of charge constraints ##

    network.model.state_of_charge = Var(list(network.storage_units.index), snapshots,
                                        domain=NonNegativeReals, bounds=(0,None))

    upper = {(su,sn) : [[(1,model.state_of_charge[su,sn]),
                         (-sus.at[su,"max_hours"],model.storage_p_nom[su])],"<=",0.]
             for su in ext_sus_i for sn in snapshots}
    upper.update({(su,sn) : [[(1,model.state_of_charge[su,sn])],"<=",
                             sus.at[su,"max_hours"]*sus.at[su,"p_nom"]]
                  for su in fix_sus_i for sn in snapshots})

    l_constraint(model, "state_of_charge_upper", upper,
                 list(network.storage_units.index), snapshots)


    #this builds the constraint previous_soc + p_store - p_dispatch + inflow - spill == soc
    #it is complicated by the fact that sometimes previous_soc and soc are floats, not variables
    soc = {}

    #store the combinations with a fixed soc
    fixed_soc = {}

    state_of_charge_set = get_switchable_as_dense(network, 'StorageUnit', 'state_of_charge_set', snapshots)

    for su in sus.index:
        for i,sn in enumerate(snapshots):

            soc[su,sn] =  [[],"==",0.]

            elapsed_hours = network.snapshot_weightings.stores[sn]

            if i == 0 and not sus.at[su,"cyclic_state_of_charge"]:
                previous_state_of_charge = sus.at[su,"state_of_charge_initial"]
                soc[su,sn][2] -= ((1-sus.at[su,"standing_loss"])**elapsed_hours
                                  * previous_state_of_charge)
            else:
                previous_state_of_charge = model.state_of_charge[su,snapshots[i-1]]
                soc[su,sn][0].append(((1-sus.at[su,"standing_loss"])**elapsed_hours,
                                      previous_state_of_charge))


            state_of_charge = state_of_charge_set.at[sn,su]
            if pd.isnull(state_of_charge):
                state_of_charge = model.state_of_charge[su,sn]
                soc[su,sn][0].append((-1,state_of_charge))
            else:
                soc[su,sn][2] += state_of_charge
                #make sure the variable is also set to the fixed state of charge
                fixed_soc[su,sn] = [[(1,model.state_of_charge[su,sn])],"==",state_of_charge]

            soc[su,sn][0].append((sus.at[su,"efficiency_store"]
                                  * elapsed_hours,model.storage_p_store[su,sn]))
            soc[su,sn][0].append((-(1/sus.at[su,"efficiency_dispatch"]) * elapsed_hours,
                                  model.storage_p_dispatch[su,sn]))
            soc[su,sn][2] -= inflow.at[sn,su] * elapsed_hours

    for su,sn in spill_index:
        elapsed_hours = network.snapshot_weightings.stores[sn]
        storage_p_spill = model.storage_p_spill[su,sn]
        soc[su,sn][0].append((-1.*elapsed_hours,storage_p_spill))

    l_constraint(model,"state_of_charge_constraint",
                 soc,list(network.storage_units.index), snapshots)

    l_constraint(model, "state_of_charge_constraint_fixed",
                 fixed_soc, list(fixed_soc.keys()))



def define_store_variables_constraints(network,snapshots):

    stores = network.stores
    ext_stores = stores.index[stores.e_nom_extendable]
    fix_stores = stores.index[~ stores.e_nom_extendable]

    e_max_pu = get_switchable_as_dense(network, 'Store', 'e_max_pu', snapshots)
    e_min_pu = get_switchable_as_dense(network, 'Store', 'e_min_pu', snapshots)

    model = network.model

    ## Define store dispatch variables ##

    network.model.store_p = Var(list(stores.index), snapshots, domain=Reals)


    ## Define store energy variables ##

    bounds = {(store,sn) : (None,None) for store in ext_stores for sn in snapshots}

    bounds.update({(store,sn) :
                   (stores.at[store,"e_nom"]*e_min_pu.at[sn,store],stores.at[store,"e_nom"]*e_max_pu.at[sn,store])
                   for store in fix_stores for sn in snapshots})

    def store_e_bounds(model,store,snapshot):
        return bounds[store,snapshot]


    network.model.store_e = Var(list(stores.index), snapshots, domain=Reals,
                                bounds=store_e_bounds)
    free_pyomo_initializers(network.model.store_e)

    ## Define energy capacity variables if store is extendable ##

    def store_e_nom_bounds(model, store):
        return (stores.at[store,"e_nom_min"],
                stores.at[store,"e_nom_max"])

    network.model.store_e_nom = Var(list(ext_stores), domain=Reals,
                                    bounds=store_e_nom_bounds)
    free_pyomo_initializers(network.model.store_e_nom)

    ## Define energy capacity constraints for extendable generators ##

    def store_e_upper(model,store,snapshot):
        return (model.store_e[store,snapshot] <=
                model.store_e_nom[store]*e_max_pu.at[snapshot,store])

    network.model.store_e_upper = Constraint(list(ext_stores), snapshots, rule=store_e_upper)
    free_pyomo_initializers(network.model.store_e_upper)

    def store_e_lower(model,store,snapshot):
        return (model.store_e[store,snapshot] >=
                model.store_e_nom[store]*e_min_pu.at[snapshot,store])

    network.model.store_e_lower = Constraint(list(ext_stores), snapshots, rule=store_e_lower)
    free_pyomo_initializers(network.model.store_e_lower)

    ## Builds the constraint previous_e - p == e ##

    e = {}

    for store in stores.index:
        for i,sn in enumerate(snapshots):

            e[store,sn] =  LConstraint(sense="==")

            e[store,sn].lhs.variables.append((-1,model.store_e[store,sn]))

            elapsed_hours = network.snapshot_weightings.stores[sn]

            if i == 0 and not stores.at[store,"e_cyclic"]:
                previous_e = stores.at[store,"e_initial"]
                e[store,sn].lhs.constant += ((1-stores.at[store,"standing_loss"])**elapsed_hours
                                         * previous_e)
            else:
                previous_e = model.store_e[store,snapshots[i-1]]
                e[store,sn].lhs.variables.append(((1-stores.at[store,"standing_loss"])**elapsed_hours,
                                              previous_e))

            e[store,sn].lhs.variables.append((-elapsed_hours, model.store_p[store,sn]))

    l_constraint(model,"store_constraint", e, list(stores.index), snapshots)



def define_branch_extension_variables(network,snapshots):

    passive_branches = network.passive_branches()

    extendable_passive_branches = passive_branches[passive_branches.s_nom_extendable]

    bounds = {b : (extendable_passive_branches.at[b,"s_nom_min"],
                   extendable_passive_branches.at[b,"s_nom_max"])
              for b in extendable_passive_branches.index}

    def branch_s_nom_bounds(model, branch_type, branch_name):
        return bounds[branch_type,branch_name]

    network.model.passive_branch_s_nom = Var(list(extendable_passive_branches.index),
                                             domain=NonNegativeReals, bounds=branch_s_nom_bounds)
    free_pyomo_initializers(network.model.passive_branch_s_nom)

    extendable_links = network.links[network.links.p_nom_extendable]

    bounds = {b : (extendable_links.at[b,"p_nom_min"],
                   extendable_links.at[b,"p_nom_max"])
              for b in extendable_links.index}

    def branch_p_nom_bounds(model, branch_name):
        return bounds[branch_name]

    network.model.link_p_nom = Var(list(extendable_links.index),
                                   domain=NonNegativeReals, bounds=branch_p_nom_bounds)
    free_pyomo_initializers(network.model.link_p_nom)


def define_link_flows(network,snapshots):

    extendable_links_i = network.links.index[network.links.p_nom_extendable]

    fixed_links_i = network.links.index[~ network.links.p_nom_extendable]

    p_max_pu = get_switchable_as_dense(network, 'Link', 'p_max_pu', snapshots)
    p_min_pu = get_switchable_as_dense(network, 'Link', 'p_min_pu', snapshots)

    fixed_lower = p_min_pu.loc[:,fixed_links_i].multiply(network.links.loc[fixed_links_i, 'p_nom'])
    fixed_upper = p_max_pu.loc[:,fixed_links_i].multiply(network.links.loc[fixed_links_i, 'p_nom'])

    network.model.link_p = Var(list(network.links.index), snapshots)

    p_upper = {(cb, sn) : LConstraint(LExpression([(1, network.model.link_p[cb, sn])],
                                                 -fixed_upper.at[sn, cb]),"<=")
               for cb in fixed_links_i for sn in snapshots}

    p_upper.update({(cb,sn) : LConstraint(LExpression([(1, network.model.link_p[cb, sn]),
                                                       (-p_max_pu.at[sn, cb], network.model.link_p_nom[cb])]),
                                          "<=")
                    for cb in extendable_links_i for sn in snapshots})

    l_constraint(network.model, "link_p_upper", p_upper,
                 list(network.links.index), snapshots)


    p_lower = {(cb, sn) : LConstraint(LExpression([(1, network.model.link_p[cb, sn])],
                                                  -fixed_lower.at[sn, cb]),">=")
               for cb in fixed_links_i for sn in snapshots}

    p_lower.update({(cb,sn) : LConstraint(LExpression([(1, network.model.link_p[cb, sn]),
                                                       (-p_min_pu.at[sn, cb], network.model.link_p_nom[cb])]),
                                          ">=")
                    for cb in extendable_links_i for sn in snapshots})

    l_constraint(network.model, "link_p_lower", p_lower,
                 list(network.links.index), snapshots)



def define_passive_branch_flows(network,snapshots,formulation="angles",ptdf_tolerance=0.):

    if formulation == "angles":
        define_passive_branch_flows_with_angles(network,snapshots)
    elif formulation == "ptdf":
        define_passive_branch_flows_with_PTDF(network,snapshots,ptdf_tolerance)
    elif formulation == "cycles":
        define_passive_branch_flows_with_cycles(network,snapshots)
    elif formulation == "kirchhoff":
        define_passive_branch_flows_with_kirchhoff(network,snapshots)



def define_passive_branch_flows_with_angles(network,snapshots):

    network.model.voltage_angles = Var(list(network.buses.index), snapshots)

    slack = {(sub,sn) :
             [[(1,network.model.voltage_angles[network.sub_networks.slack_bus[sub],sn])], "==", 0.]
             for sub in network.sub_networks.index for sn in snapshots}

    l_constraint(network.model,"slack_angle",slack,list(network.sub_networks.index),snapshots)


    passive_branches = network.passive_branches()

    network.model.passive_branch_p = Var(list(passive_branches.index), snapshots)

    flows = {}
    for branch in passive_branches.index:
        bus0 = passive_branches.at[branch,"bus0"]
        bus1 = passive_branches.at[branch,"bus1"]
        bt = branch[0]
        bn = branch[1]
        sub = passive_branches.at[branch,"sub_network"]
        attribute = "r_pu_eff" if network.sub_networks.at[sub,"carrier"] == "DC" else "x_pu_eff"
        y = 1/ passive_branches.at[ branch, attribute]
        for sn in snapshots:
            lhs = LExpression([(y,network.model.voltage_angles[bus0,sn]),
                               (-y,network.model.voltage_angles[bus1,sn]),
                               (-1,network.model.passive_branch_p[bt,bn,sn])],
                              -y*(passive_branches.at[branch,"phase_shift"]*np.pi/180. if bt == "Transformer" else 0.))
            flows[bt,bn,sn] = LConstraint(lhs,"==",LExpression())

    l_constraint(network.model, "passive_branch_p_def", flows,
                 list(passive_branches.index), snapshots)


def define_passive_branch_flows_with_PTDF(network,snapshots,ptdf_tolerance=0.):

    passive_branches = network.passive_branches()

    network.model.passive_branch_p = Var(list(passive_branches.index), snapshots)

    flows = {}

    for sub_network in network.sub_networks.obj:
        find_bus_controls(sub_network)

        branches_i = sub_network.branches_i()
        if len(branches_i) > 0:
            calculate_PTDF(sub_network)

            #kill small PTDF values
            sub_network.PTDF[abs(sub_network.PTDF) < ptdf_tolerance] = 0

        for i,branch in enumerate(branches_i):
            bt = branch[0]
            bn = branch[1]

            for sn in snapshots:
                lhs = sum(sub_network.PTDF[i,j]*network._p_balance[bus,sn]
                          for j,bus in enumerate(sub_network.buses_o)
                          if sub_network.PTDF[i,j] != 0)
                rhs = LExpression([(1,network.model.passive_branch_p[bt,bn,sn])])
                flows[bt,bn,sn] = LConstraint(lhs,"==",rhs)


    l_constraint(network.model, "passive_branch_p_def", flows,
                 list(passive_branches.index), snapshots)


def define_sub_network_cycle_constraints( subnetwork, snapshots, passive_branch_p, attribute):
    """ Constructs cycle_constraints for a particular subnetwork
    """

    sub_network_cycle_constraints = {}
    sub_network_cycle_index = []

    matrix = subnetwork.C.tocsc()
    branches = subnetwork.branches()

    for col_j in range( matrix.shape[1] ):
        cycle_is = matrix.getcol(col_j).nonzero()[0]

        if len(cycle_is) == 0: continue

        sub_network_cycle_index.append((subnetwork.name, col_j))


        branch_idx_attributes = []

        for cycle_i in cycle_is:
            branch_idx = branches.index[cycle_i]
            attribute_value = 1e5 * branches.at[ branch_idx, attribute] * subnetwork.C[ cycle_i, col_j]
            branch_idx_attributes.append( (branch_idx, attribute_value))

        for snapshot in snapshots:
            expression_list = [ (attribute_value,
                                 passive_branch_p[branch_idx[0], branch_idx[1], snapshot]) for (branch_idx, attribute_value) in branch_idx_attributes]

            lhs = LExpression(expression_list)
            sub_network_cycle_constraints[subnetwork.name,col_j,snapshot] = LConstraint(lhs,"==",LExpression())

    return( sub_network_cycle_index, sub_network_cycle_constraints)

def define_passive_branch_flows_with_cycles(network,snapshots):

    for sub_network in network.sub_networks.obj:
        find_tree(sub_network)
        find_cycles(sub_network)

        #following is necessary to calculate angles post-facto
        find_bus_controls(sub_network)
        if len(sub_network.branches_i()) > 0:
            calculate_B_H(sub_network)


    passive_branches = network.passive_branches()


    network.model.passive_branch_p = Var(list(passive_branches.index), snapshots)

    cycle_index = []
    cycle_constraints = {}

    for subnetwork in network.sub_networks.obj:
        branches = subnetwork.branches()
        attribute = "r_pu_eff" if network.sub_networks.at[subnetwork.name,"carrier"] == "DC" else "x_pu_eff"

        sub_network_cycle_index, sub_network_cycle_constraints = define_sub_network_cycle_constraints( subnetwork,
                                                                                                       snapshots,
                                                                                                       network.model.passive_branch_p, attribute)

        cycle_index.extend( sub_network_cycle_index)
        cycle_constraints.update( sub_network_cycle_constraints)

    l_constraint(network.model, "cycle_constraints", cycle_constraints,
                 cycle_index, snapshots)


    network.model.cycles = Var(cycle_index, snapshots, domain=Reals, bounds=(None,None))

    flows = {}

    for subnetwork in network.sub_networks.obj:
        branches = subnetwork.branches()
        buses = subnetwork.buses()
        for i,branch in enumerate(branches.index):
            bt = branch[0]
            bn = branch[1]

            cycle_is = subnetwork.C[i,:].nonzero()[1]
            tree_is = subnetwork.T[i,:].nonzero()[1]

            if len(cycle_is) + len(tree_is) == 0: logger.error("The cycle formulation does not support infinite impedances, yet.")

            for snapshot in snapshots:
                expr = LExpression([(subnetwork.C[i,j], network.model.cycles[subnetwork.name,j,snapshot])
                                    for j in cycle_is])
                lhs = expr + sum(subnetwork.T[i,j]*network._p_balance[buses.index[j],snapshot]
                                 for j in tree_is)

                rhs = LExpression([(1,network.model.passive_branch_p[bt,bn,snapshot])])

                flows[bt,bn,snapshot] = LConstraint(lhs,"==",rhs)

    l_constraint(network.model, "passive_branch_p_def", flows,
                 list(passive_branches.index), snapshots)



def define_passive_branch_flows_with_kirchhoff(network,snapshots,skip_vars=False):
    """ define passive branch flows with the kirchoff method """

    for sub_network in network.sub_networks.obj:
        find_tree(sub_network)
        find_cycles(sub_network)

        #following is necessary to calculate angles post-facto
        find_bus_controls(sub_network)
        if len(sub_network.branches_i()) > 0:
            calculate_B_H(sub_network)

    passive_branches = network.passive_branches()

    if not skip_vars:
        network.model.passive_branch_p = Var(list(passive_branches.index), snapshots)

    cycle_index = []
    cycle_constraints = {}

    for subnetwork in network.sub_networks.obj:

        attribute = "r_pu_eff" if network.sub_networks.at[subnetwork.name,"carrier"] == "DC" else "x_pu_eff"

        sub_network_cycle_index, sub_network_cycle_constraints = define_sub_network_cycle_constraints( subnetwork,
                                                                                                       snapshots,
                                                                                                       network.model.passive_branch_p, attribute)

        cycle_index.extend( sub_network_cycle_index)
        cycle_constraints.update( sub_network_cycle_constraints)

    l_constraint(network.model, "cycle_constraints", cycle_constraints,
                 cycle_index, snapshots)

def define_passive_branch_constraints(network,snapshots):

    passive_branches = network.passive_branches()
    extendable_branches = passive_branches[passive_branches.s_nom_extendable]
    fixed_branches = passive_branches[~ passive_branches.s_nom_extendable]


    s_max_pu = pd.concat({c : get_switchable_as_dense(network, c, 's_max_pu', snapshots)
                          for c in network.passive_branch_components}, axis=1, sort=False)

    flow_upper = {(b[0],b[1],sn) : [[(1,network.model.passive_branch_p[b[0],b[1],sn])],
                                    "<=", s_max_pu.at[sn,b]*fixed_branches.at[b,"s_nom"]]
                  for b in fixed_branches.index
                  for sn in snapshots}

    flow_upper.update({(b[0],b[1],sn) : [[(1,network.model.passive_branch_p[b[0],b[1],sn]),
                                          (-s_max_pu.at[sn,b],network.model.passive_branch_s_nom[b[0],b[1]])],"<=",0]
                       for b in extendable_branches.index
                       for sn in snapshots})

    l_constraint(network.model, "flow_upper", flow_upper,
                 list(passive_branches.index), snapshots)

    flow_lower = {(b[0],b[1],sn) : [[(1,network.model.passive_branch_p[b[0],b[1],sn])],
                                    ">=", -s_max_pu.at[sn,b]*fixed_branches.at[b,"s_nom"]]
                  for b in fixed_branches.index
                  for sn in snapshots}

    flow_lower.update({(b[0],b[1],sn): [[(1,network.model.passive_branch_p[b[0],b[1],sn]),
                                         (s_max_pu.at[sn,b],network.model.passive_branch_s_nom[b[0],b[1]])],">=",0]
                       for b in extendable_branches.index
                       for sn in snapshots})

    l_constraint(network.model, "flow_lower", flow_lower,
                 list(passive_branches.index), snapshots)

def define_nodal_balances(network,snapshots):
    """Construct the nodal balance for all elements except the passive
    branches.

    Store the nodal balance expression in network._p_balance.
    """

    #dictionary for constraints
    network._p_balance = {(bus,sn) : LExpression()
                          for bus in network.buses.index
                          for sn in snapshots}

    efficiency = get_switchable_as_dense(network, 'Link', 'efficiency', snapshots)

    for cb in network.links.index:
        bus0 = network.links.at[cb,"bus0"]
        bus1 = network.links.at[cb,"bus1"]

        for sn in snapshots:
            network._p_balance[bus0,sn].variables.append((-1,network.model.link_p[cb,sn]))
            network._p_balance[bus1,sn].variables.append((efficiency.at[sn,cb],network.model.link_p[cb,sn]))

    #Add any other buses to which the links are attached
    for i in [int(col[3:]) for col in network.links.columns if col[:3] == "bus" and col not in ["bus0","bus1"]]:
        efficiency = get_switchable_as_dense(network, 'Link', 'efficiency{}'.format(i), snapshots)
        for cb in network.links.index[network.links["bus{}".format(i)] != ""]:
            bus = network.links.at[cb, "bus{}".format(i)]
            for sn in snapshots:
                network._p_balance[bus,sn].variables.append((efficiency.at[sn,cb],network.model.link_p[cb,sn]))


    for gen in network.generators.index:
        bus = network.generators.at[gen,"bus"]
        sign = network.generators.at[gen,"sign"]
        for sn in snapshots:
            network._p_balance[bus,sn].variables.append((sign,network.model.generator_p[gen,sn]))

    load_p_set = get_switchable_as_dense(network, 'Load', 'p_set', snapshots)
    for load in network.loads.index:
        bus = network.loads.at[load,"bus"]
        sign = network.loads.at[load,"sign"]
        for sn in snapshots:
            network._p_balance[bus,sn].constant += sign*load_p_set.at[sn,load]

    for su in network.storage_units.index:
        bus = network.storage_units.at[su,"bus"]
        sign = network.storage_units.at[su,"sign"]
        for sn in snapshots:
            network._p_balance[bus,sn].variables.append((sign,network.model.storage_p_dispatch[su,sn]))
            network._p_balance[bus,sn].variables.append((-sign,network.model.storage_p_store[su,sn]))

    for store in network.stores.index:
        bus = network.stores.at[store,"bus"]
        sign = network.stores.at[store,"sign"]
        for sn in snapshots:
            network._p_balance[bus,sn].variables.append((sign,network.model.store_p[store,sn]))


def define_nodal_balance_constraints(network,snapshots):

    passive_branches = network.passive_branches()


    for branch in passive_branches.index:
        bus0 = passive_branches.at[branch,"bus0"]
        bus1 = passive_branches.at[branch,"bus1"]
        bt = branch[0]
        bn = branch[1]
        for sn in snapshots:
            network._p_balance[bus0,sn].variables.append((-1,network.model.passive_branch_p[bt,bn,sn]))
            network._p_balance[bus1,sn].variables.append((1,network.model.passive_branch_p[bt,bn,sn]))

    power_balance = {k: LConstraint(v,"==",LExpression()) for k,v in network._p_balance.items()}

    l_constraint(network.model, "power_balance", power_balance,
                 list(network.buses.index), snapshots)


def define_sub_network_balance_constraints(network,snapshots):

    sn_balance = {}

    for sub_network in network.sub_networks.obj:
        for sn in snapshots:
            sn_balance[sub_network.name,sn] = LConstraint(LExpression(),"==",LExpression())
            for bus in sub_network.buses().index:
                sn_balance[sub_network.name,sn].lhs.variables.extend(network._p_balance[bus,sn].variables)
                sn_balance[sub_network.name,sn].lhs.constant += network._p_balance[bus,sn].constant

    l_constraint(network.model,"sub_network_balance_constraint", sn_balance,
                 list(network.sub_networks.index), snapshots)


def define_global_constraints(network,snapshots):


    global_constraints = {}

    for gc in network.global_constraints.index:
        if network.global_constraints.loc[gc,"type"] == "primary_energy":

            c = LConstraint(sense=network.global_constraints.loc[gc,"sense"])

            c.rhs.constant = network.global_constraints.loc[gc,"constant"]

            carrier_attribute = network.global_constraints.loc[gc,"carrier_attribute"]

            for carrier in network.carriers.index:
                attribute = network.carriers.at[carrier,carrier_attribute]
                if attribute == 0.:
                    continue
                #for generators, use the prime mover carrier
                gens = network.generators.index[network.generators.carrier == carrier]
                c.lhs.variables.extend([(attribute
                                         * (1/network.generators.at[gen,"efficiency"])
                                         * network.snapshot_weightings.generators[sn],
                                         network.model.generator_p[gen,sn])
                                        for gen in gens
                                        for sn in snapshots])

                #for storage units, use the prime mover carrier
                #take difference of energy at end and start of period
                sus = network.storage_units.index[(network.storage_units.carrier == carrier) & (~network.storage_units.cyclic_state_of_charge)]
                c.lhs.variables.extend([(-attribute, network.model.state_of_charge[su,snapshots[-1]])
                                        for su in sus])
                c.lhs.constant += sum(attribute*network.storage_units.at[su,"state_of_charge_initial"]
                                      for su in sus)

                #for stores, inherit the carrier from the bus
                #take difference of energy at end and start of period
                stores = network.stores.index[(network.stores.bus.map(network.buses.carrier) == carrier) & (~network.stores.e_cyclic)]
                c.lhs.variables.extend([(-attribute, network.model.store_e[store,snapshots[-1]])
                                        for store in stores])
                c.lhs.constant += sum(attribute*network.stores.at[store,"e_initial"]
                                      for store in stores)



            global_constraints[gc] = c

    l_constraint(network.model, "global_constraints",
                 global_constraints, list(network.global_constraints.index))




def define_linear_objective(network,snapshots):

    model = network.model

    extendable_generators = network.generators[network.generators.p_nom_extendable]

    ext_sus = network.storage_units[network.storage_units.p_nom_extendable]

    ext_stores = network.stores[network.stores.e_nom_extendable]

    passive_branches = network.passive_branches()

    extendable_passive_branches = passive_branches[passive_branches.s_nom_extendable]

    extendable_links = network.links[network.links.p_nom_extendable]

    suc_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable & (network.generators.start_up_cost > 0)]

    sdc_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable & (network.generators.shut_down_cost > 0)]


    marginal_cost_it = zip(get_switchable_as_iter(network, 'Generator', 'marginal_cost', snapshots),
                           get_switchable_as_iter(network, 'StorageUnit', 'marginal_cost', snapshots),
                           get_switchable_as_iter(network, 'Store', 'marginal_cost', snapshots),
                           get_switchable_as_iter(network, 'Link', 'marginal_cost', snapshots))

    objective = LExpression()


    for sn, marginal_cost in zip(snapshots, marginal_cost_it):
        gen_mc, su_mc, st_mc, link_mc = marginal_cost

        weight = network.snapshot_weightings.objective[sn]
        for gen in network.generators.index:
            coefficient = gen_mc.at[gen] * weight
            objective.variables.extend([(coefficient, model.generator_p[gen, sn])])

        for su in network.storage_units.index:
            coefficient = su_mc.at[su] * weight
            objective.variables.extend([(coefficient, model.storage_p_dispatch[su,sn])])

        for store in network.stores.index:
            coefficient = st_mc.at[store] * weight
            objective.variables.extend([(coefficient, model.store_p[store,sn])])

        for link in network.links.index:
            coefficient = link_mc.at[link] * weight
            objective.variables.extend([(coefficient, model.link_p[link,sn])])


    #NB: for capital costs we subtract the costs of existing infrastructure p_nom/s_nom

    objective.variables.extend([(extendable_generators.at[gen,"capital_cost"], model.generator_p_nom[gen])
                                for gen in extendable_generators.index])
    objective.constant -= (extendable_generators.capital_cost * extendable_generators.p_nom).zsum()

    objective.variables.extend([(ext_sus.at[su,"capital_cost"], model.storage_p_nom[su])
                                for su in ext_sus.index])
    objective.constant -= (ext_sus.capital_cost*ext_sus.p_nom).zsum()

    objective.variables.extend([(ext_stores.at[store,"capital_cost"], model.store_e_nom[store])
                                for store in ext_stores.index])
    objective.constant -= (ext_stores.capital_cost*ext_stores.e_nom).zsum()

    objective.variables.extend([(extendable_passive_branches.at[b,"capital_cost"], model.passive_branch_s_nom[b])
                                for b in extendable_passive_branches.index])
    objective.constant -= (extendable_passive_branches.capital_cost * extendable_passive_branches.s_nom).zsum()

    objective.variables.extend([(extendable_links.at[b,"capital_cost"], model.link_p_nom[b])
                                for b in extendable_links.index])
    objective.constant -= (extendable_links.capital_cost * extendable_links.p_nom).zsum()

    network.objective_constant = - objective.constant

    ## Unit commitment costs

    objective.variables.extend([(1, model.generator_start_up_cost[gen,sn]) for gen in suc_gens_i for sn in snapshots])

    objective.variables.extend([(1, model.generator_shut_down_cost[gen,sn]) for gen in sdc_gens_i for sn in snapshots])


    l_objective(model,objective)

def extract_optimisation_results(network, snapshots, formulation="angles", free_pyomo=True,
                                 extra_postprocessing=None):

    allocate_series_dataframes(network, {'Generator': ['p'],
                                         'Load': ['p'],
                                         'StorageUnit': ['p', 'state_of_charge', 'spill'],
                                         'Store': ['p', 'e'],
                                         'Bus': ['p', 'v_ang', 'v_mag_pu', 'marginal_price'],
                                         'Line': ['p0', 'p1', 'mu_lower', 'mu_upper'],
                                         'Transformer': ['p0', 'p1', 'mu_lower', 'mu_upper'],
                                         'Link': ["p"+col[3:] for col in network.links.columns if col[:3] == "bus"]
                                                  +['mu_lower', 'mu_upper']})

    #get value of objective function
    network.objective = network.results["Problem"][0]["Upper bound"]

    model = network.model

    duals = pd.Series(list(model.dual.values()), index=pd.Index(list(model.dual.keys())),
                      dtype=float)

    if free_pyomo:
        model.dual.clear()

    def clear_indexedvar(indexedvar):
        for v in indexedvar._data.values():
            v.clear()

    def get_values(indexedvar, free=free_pyomo):
        s = pd.Series(indexedvar.get_values(), dtype=float)
        if free:
            clear_indexedvar(indexedvar)
        return s

    def set_from_series(df, series):
        df.loc[snapshots] = series.unstack(0).reindex(columns=df.columns)

    def get_shadows(constraint, multiind=True):
        if len(constraint) == 0: return pd.Series(dtype=float)

        index = list(constraint.keys())
        if multiind:
            index = pd.MultiIndex.from_tuples(index)
        cdata = pd.Series(list(constraint.values()), index=index)
        return cdata.map(duals)

    if len(network.generators):
        set_from_series(network.generators_t.p, get_values(model.generator_p))

    if len(network.storage_units):
        set_from_series(network.storage_units_t.p,
                        get_values(model.storage_p_dispatch)
                        - get_values(model.storage_p_store))

        set_from_series(network.storage_units_t.state_of_charge,
                        get_values(model.state_of_charge))

        if (network.storage_units_t.inflow.max() > 0).any():
            set_from_series(network.storage_units_t.spill,
                            get_values(model.storage_p_spill))
        network.storage_units_t.spill.fillna(0, inplace=True) #p_spill doesn't exist if inflow=0

    if len(network.stores):
        set_from_series(network.stores_t.p, get_values(model.store_p))
        set_from_series(network.stores_t.e, get_values(model.store_e))

    if len(network.loads):
        load_p_set = get_switchable_as_dense(network, 'Load', 'p_set', snapshots)
        network.loads_t["p"].loc[snapshots] = load_p_set.loc[snapshots]

    if len(network.buses):
        network.buses_t.p.loc[snapshots] = \
            pd.concat({c.name:
                       c.pnl.p.loc[snapshots].multiply(c.df.sign, axis=1)
                       .groupby(c.df.bus, axis=1).sum()
                       for c in network.iterate_components(network.controllable_one_port_components)},
                      sort=False) \
              .sum(level=1) \
              .reindex(columns=network.buses_t.p.columns, fill_value=0.)


    # passive branches
    passive_branches = get_values(model.passive_branch_p)
    flow_lower = get_shadows(model.flow_lower)
    flow_upper = get_shadows(model.flow_upper)
    for c in network.iterate_components(network.passive_branch_components):
        set_from_series(c.pnl.p0, passive_branches.loc[c.name])
        c.pnl.p1.loc[snapshots] = - c.pnl.p0.loc[snapshots]

        set_from_series(c.pnl.mu_lower, flow_lower[c.name])
        set_from_series(c.pnl.mu_upper, -flow_upper[c.name])
    del flow_lower, flow_upper

    # active branches
    if len(network.links):
        set_from_series(network.links_t.p0, get_values(model.link_p))

        efficiency = get_switchable_as_dense(network, 'Link', 'efficiency', snapshots)

        network.links_t.p1.loc[snapshots] = - network.links_t.p0.loc[snapshots]*efficiency.loc[snapshots,:]

        network.buses_t.p.loc[snapshots] -= (network.links_t.p0.loc[snapshots]
                                             .groupby(network.links.bus0, axis=1).sum()
                                             .reindex(columns=network.buses_t.p.columns, fill_value=0.))

        network.buses_t.p.loc[snapshots] -= (network.links_t.p1.loc[snapshots]
                                             .groupby(network.links.bus1, axis=1).sum()
                                             .reindex(columns=network.buses_t.p.columns, fill_value=0.))

        #Add any other buses to which the links are attached
        for i in [int(col[3:]) for col in network.links.columns if col[:3] == "bus" and col not in ["bus0","bus1"]]:
            efficiency = get_switchable_as_dense(network, 'Link', 'efficiency{}'.format(i), snapshots)
            p_name = "p{}".format(i)
            links = network.links.index[network.links["bus{}".format(i)] != ""]
            network.links_t[p_name].loc[snapshots, links] = - network.links_t.p0.loc[snapshots, links]*efficiency.loc[snapshots, links]
            network.buses_t.p.loc[snapshots] -= (network.links_t[p_name].loc[snapshots, links]
                                                 .groupby(network.links["bus{}".format(i)], axis=1).sum()
                                                 .reindex(columns=network.buses_t.p.columns, fill_value=0.))


        set_from_series(network.links_t.mu_lower, get_shadows(model.link_p_lower))
        set_from_series(network.links_t.mu_upper, - get_shadows(model.link_p_upper))

    if len(network.buses):
        if formulation in {'angles', 'kirchhoff'}:
            set_from_series(network.buses_t.marginal_price,
                            pd.Series(list(model.power_balance.values()),
                                      index=pd.MultiIndex.from_tuples(list(model.power_balance.keys())))
                            .map(duals))

            #correct for snapshot weightings
            network.buses_t.marginal_price.loc[snapshots] = (
                network.buses_t.marginal_price.loc[snapshots].divide(
                    network.snapshot_weightings.objective.loc[snapshots],axis=0))

        if formulation == "angles":
            set_from_series(network.buses_t.v_ang,
                            get_values(model.voltage_angles))
        elif formulation in ["ptdf","cycles","kirchhoff"]:
            for sn in network.sub_networks.obj:
                network.buses_t.v_ang.loc[snapshots,sn.slack_bus] = 0.
                if len(sn.pvpqs) > 0:
                    network.buses_t.v_ang.loc[snapshots,sn.pvpqs] = spsolve(sn.B[1:, 1:], network.buses_t.p.loc[snapshots,sn.pvpqs].T).T

        network.buses_t.v_mag_pu.loc[snapshots,network.buses.carrier=="AC"] = 1.
        network.buses_t.v_mag_pu.loc[snapshots,network.buses.carrier=="DC"] = 1 + network.buses_t.v_ang.loc[snapshots,network.buses.carrier=="DC"]


    #now that we've used the angles to calculate the flow, set the DC ones to zero
    network.buses_t.v_ang.loc[snapshots,network.buses.carrier=="DC"] = 0.

    network.generators.p_nom_opt = network.generators.p_nom

    network.generators.loc[network.generators.p_nom_extendable, 'p_nom_opt'] = \
        get_values(network.model.generator_p_nom)

    network.storage_units.p_nom_opt = network.storage_units.p_nom

    network.storage_units.loc[network.storage_units.p_nom_extendable, 'p_nom_opt'] = \
        get_values(network.model.storage_p_nom)

    network.stores.e_nom_opt = network.stores.e_nom

    network.stores.loc[network.stores.e_nom_extendable, 'e_nom_opt'] = \
        get_values(network.model.store_e_nom)


    s_nom_extendable_passive_branches = get_values(model.passive_branch_s_nom)
    for c in network.iterate_components(network.passive_branch_components):
        c.df['s_nom_opt'] = c.df.s_nom
        if c.df.s_nom_extendable.any():
            c.df.loc[c.df.s_nom_extendable, 's_nom_opt'] = s_nom_extendable_passive_branches.loc[c.name]

    network.links.p_nom_opt = network.links.p_nom

    network.links.loc[network.links.p_nom_extendable, "p_nom_opt"] = \
        get_values(network.model.link_p_nom)

    try:
        network.global_constraints.loc[:,"mu"] = - get_shadows(model.global_constraints, multiind=False)
    except (AttributeError, KeyError):
        logger.warning("Could not read out global constraint shadow prices")

    #extract unit commitment statuses
    if network.generators.committable.any():
        allocate_series_dataframes(network, {'Generator': ['status']})

        fixed_committable_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable]

        if len(fixed_committable_gens_i) > 0:
            network.generators_t.status.loc[snapshots,fixed_committable_gens_i] = \
                get_values(model.generator_status).unstack(0)

    if extra_postprocessing is not None:
        extra_postprocessing(network, snapshots, duals)


def network_lopf_build_model(network, snapshots=None, skip_pre=False,
                             formulation="angles", ptdf_tolerance=0.):
    """
    Build pyomo model for linear optimal power flow for a group of snapshots.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    skip_pre : bool, default False
        Skip the preliminary steps of computing topology, calculating
        dependent values and finding bus controls.
    formulation : string
        Formulation of the linear power flow equations to use; must be
        one of ["angles","cycles","kirchhoff","ptdf"]
    ptdf_tolerance : float
        Value below which PTDF entries are ignored

    Returns
    -------
    network.model
    """
    if isinstance(network.snapshots, pd.MultiIndex):
        raise NotImplementedError("Optimization with multiindexed snapshots "
                                  "using pyomo is not supported.")

    if not skip_pre:
        network.determine_network_topology()
        calculate_dependent_values(network)
        for sub_network in network.sub_networks.obj:
            find_slack_bus(sub_network)
        logger.info("Performed preliminary steps")


    snapshots = _as_snapshots(network, snapshots)

    logger.info("Building pyomo model using `%s` formulation", formulation)
    network.model = ConcreteModel("Linear Optimal Power Flow")


    define_generator_variables_constraints(network,snapshots)

    define_storage_variables_constraints(network,snapshots)

    define_store_variables_constraints(network,snapshots)

    define_branch_extension_variables(network,snapshots)

    define_link_flows(network,snapshots)

    define_nodal_balances(network,snapshots)

    define_passive_branch_flows(network,snapshots,formulation,ptdf_tolerance)

    define_passive_branch_constraints(network,snapshots)

    if formulation in ["angles", "kirchhoff"]:
        define_nodal_balance_constraints(network,snapshots)
    elif formulation in ["ptdf", "cycles"]:
        define_sub_network_balance_constraints(network,snapshots)

    define_global_constraints(network,snapshots)

    define_linear_objective(network, snapshots)

    #tidy up auxilliary expressions
    del network._p_balance

    #force solver to also give us the dual prices
    network.model.dual = Suffix(direction=Suffix.IMPORT)

    return network.model

def network_lopf_prepare_solver(network, solver_name="glpk", solver_io=None):
    """
    Prepare solver for linear optimal power flow.

    Parameters
    ----------
    solver_name : string
        Must be a solver name that pyomo recognises and that is
        installed, e.g. "glpk", "gurobi"
    solver_io : string, default None
        Solver Input-Output option, e.g. "python" to use "gurobipy" for
        solver_name="gurobi"

    Returns
    -------
    None
    """

    network.opt = SolverFactory(solver_name, solver_io=solver_io)

    patch_optsolver_record_memusage_before_solving(network.opt, network)

    if isinstance(network.opt, PersistentSolver):
        network.opt.set_instance(network.model)

    return network.opt


def network_lopf_solve(network, snapshots=None, formulation="angles", solver_options={},solver_logfile=None,  keep_files=False,
                       free_memory={'pyomo'},extra_postprocessing=None):
    """
    Solve linear optimal power flow for a group of snapshots and extract results.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    formulation : string
        Formulation of the linear power flow equations to use; must be one of
        ["angles","cycles","kirchhoff","ptdf"]; must match formulation used for
        building the model.
    solver_options : dictionary
        A dictionary with additional options that get passed to the solver.
        (e.g. {'threads':2} tells gurobi to use only 2 cpus)
    solver_logfile : None|string
        If not None, sets the logfile option of the solver.
    keep_files : bool, default False
        Keep the files that pyomo constructs from OPF problem
        construction, e.g. .lp file - useful for debugging
    free_memory : set, default {'pyomo'}
        Any subset of {'pypsa', 'pyomo'}. Allows to stash `pypsa` time-series
        data away while the solver runs (as a pickle to disk) and/or free
        `pyomo` data after the solution has been extracted.
    extra_postprocessing : callable function
        This function must take three arguments
        `extra_postprocessing(network,snapshots,duals)` and is called after
        the model has solved and the results are extracted. It allows the user to
        extract further information about the solution, such as additional shadow prices.

    Returns
    -------
    None
    """

    snapshots = _as_snapshots(network, snapshots)

    logger.info("Solving model using %s", network.opt.name)

    if isinstance(network.opt, PersistentSolver):
        args = []
    else:
        args = [network.model]

    if isinstance(free_memory, str):
        free_memory = {free_memory}

    if 'pypsa' in free_memory:
        with empty_network(network):
            network.results = network.opt.solve(*args, suffixes=["dual"], keepfiles=keep_files, logfile=solver_logfile, options=solver_options)
    else:
        network.results = network.opt.solve(*args, suffixes=["dual"], keepfiles=keep_files, logfile=solver_logfile, options=solver_options)

    if logger.isEnabledFor(logging.INFO):
        network.results.write()

    status = network.results["Solver"][0]["Status"]
    termination_condition = network.results["Solver"][0]["Termination condition"]

    if status == "ok" and termination_condition == "optimal":
        logger.info("Optimization successful")
        extract_optimisation_results(network, snapshots, formulation,
                                     free_pyomo='pyomo' in free_memory,
                                     extra_postprocessing=extra_postprocessing)
    elif status == "warning" and termination_condition == "other":
        logger.warning("WARNING! Optimization might be sub-optimal. Writing output anyway")
        extract_optimisation_results(network, snapshots, formulation,
                                     free_pyomo='pyomo' in free_memory,
                                     extra_postprocessing=extra_postprocessing)
    else:
        logger.error("Optimisation failed with status %s and terminal condition %s"
              % (status, termination_condition))

    return status, termination_condition

def network_lopf(network, snapshots=None, solver_name="glpk", solver_io=None,
                 skip_pre=False, extra_functionality=None,
                 multi_investment_periods=False, solver_logfile=None,
                 solver_options={}, keep_files=False, formulation="angles",
                 ptdf_tolerance=0., free_memory={},
                 extra_postprocessing=None):
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
    solver_io : string, default None
        Solver Input-Output option, e.g. "python" to use "gurobipy" for
        solver_name="gurobi"
    skip_pre : bool, default False
        Skip the preliminary steps of computing topology, calculating
        dependent values and finding bus controls.
    extra_functionality : callable function
        This function must take two arguments
        `extra_functionality(network,snapshots)` and is called after
        the model building is complete, but before it is sent to the
        solver. It allows the user to
        add/change constraints and add/change the objective function.
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
    ptdf_tolerance : float
        Value below which PTDF entries are ignored
    free_memory : set, default {'pyomo'}
        Any subset of {'pypsa', 'pyomo'}. Allows to stash `pypsa` time-series
        data away while the solver runs (as a pickle to disk) and/or free
        `pyomo` data after the solution has been extracted.
    extra_postprocessing : callable function
        This function must take three arguments
        `extra_postprocessing(network,snapshots,duals)` and is called after
        the model has solved and the results are extracted. It allows the user to
        extract further information about the solution, such as additional shadow prices.

    Returns
    -------
    None
    """
    if multi_investment_periods:
        raise NotImplementedError("Multi period invesmtent is only supported for pyomo=False")
    if (type(network.snapshots)==pd.MultiIndex):
        raise NotImplementedError("Multi indexed snapshots is only supported for pyomo=False")


    snapshots = _as_snapshots(network, snapshots)

    network_lopf_build_model(network, snapshots, skip_pre=skip_pre,
                             formulation=formulation, ptdf_tolerance=ptdf_tolerance)

    if extra_functionality is not None:
        extra_functionality(network,snapshots)

    network_lopf_prepare_solver(network, solver_name=solver_name,
                                solver_io=solver_io)

    return network_lopf_solve(network, snapshots, formulation=formulation,
                              solver_logfile=solver_logfile, solver_options=solver_options,
                              keep_files=keep_files, free_memory=free_memory,
                              extra_postprocessing=extra_postprocessing)
