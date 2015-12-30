

## Copyright 2015 Tom Brown (FIAS), Jonas Hoersch (FIAS)

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

"""Python for Power Systems Analysis (PyPSA)

Grid calculation library.
"""


# make the code as Python 3 compatible as possible
from __future__ import print_function, division


__version__ = "0.1"
__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"



from pyomo.environ import ConcreteModel, Var, Objective, NonNegativeReals, Constraint, Reals, Suffix, Expression

from pyomo.opt import SolverFactory

from .pf import calculate_z_pu, find_slack_bus

from itertools import chain

import pandas as pd



#this function is necessary because pyomo doesn't deal with NaNs gracefully
def replace_nan_with_none(val):
    if pd.isnull(val):
        return None
    else:
        return val


def network_opf(network,snapshots=None):
    """Optimal power flow for snapshots."""

    raise NotImplementedError("Non-linear optimal power flow not supported yet.")




def define_generator_variables_constraints(network,snapshots):


    ## Define generator dispatch variables ##

    def gen_p_bounds(model,gen_name,snapshot):

        gen = network.generators_df.obj[gen_name]

        if gen.p_nom_extendable:
            return (None,None)
        else:
            if gen.dispatch == "flexible":
                return (gen.p_nom*gen.p_min_pu_fixed,gen.p_nom*gen.p_max_pu_fixed)
            elif gen.dispatch == "variable":
                return (gen.p_nom*gen.p_min_pu[snapshot],gen.p_nom*gen.p_max_pu[snapshot])
            else:
                raise NotImplementedError("Dispatch type %s is not supported yet." % (gen.dispatch))


    network.model.generator_p = Var(network.generators_df.index, snapshots, domain=Reals, bounds=gen_p_bounds)



    ## Define generator capacity variables if generator is extendble ##

    extendable_generators = network.generators_df[network.generators_df.p_nom_extendable]

    def gen_p_nom_bounds(model, gen_name):
        gen = network.generators_df.obj[gen_name]
        return (replace_nan_with_none(gen.p_nom_min), replace_nan_with_none(gen.p_nom_max))

    network.model.generator_p_nom = Var(extendable_generators.index, domain=NonNegativeReals, bounds=gen_p_nom_bounds)



    ## Define generator dispatch constraints for extendable generators ##

    def gen_p_lower(model,gen_name,snapshot):
        gen = network.generators_df.obj[gen_name]

        if gen.dispatch == "flexible":
            return model.generator_p[gen_name,snapshot] >= model.generator_p_nom[gen_name]*gen.p_min_pu_fixed
        elif gen.dispatch == "variable":
            return model.generator_p[gen_name,snapshot] >= model.generator_p_nom[gen_name]*gen.p_min_pu[snapshot]
        else:
            raise NotImplementedError("Dispatch type %s is not supported yet for extendability." % (gen.dispatch))

    network.model.generator_p_lower = Constraint(extendable_generators.index,snapshots,rule=gen_p_lower)


    def gen_p_upper(model,gen_name,snapshot):
        gen = network.generators_df.obj[gen_name]

        if gen.dispatch == "flexible":
            return model.generator_p[gen_name,snapshot] <= model.generator_p_nom[gen_name]*gen.p_max_pu_fixed
        elif gen.dispatch == "variable":
            return model.generator_p[gen_name,snapshot] <= model.generator_p_nom[gen_name]*gen.p_max_pu[snapshot]
        else:
            raise NotImplementedError("Dispatch type %s is not supported yet for extendability." % (gen.dispatch))

    network.model.generator_p_upper = Constraint(extendable_generators.index,snapshots,rule=gen_p_upper)





def define_storage_variables_constraints(network,snapshots):


    ## Define storage dispatch variables ##

    def su_p_dispatch_bounds(model,su_name,snapshot):
        su = network.storage_units_df.obj[su_name]

        if su.p_nom_extendable:
            return (0,None)
        else:
            return (0,su.p_nom*su.p_max_pu_fixed)

    network.model.storage_p_dispatch = Var(network.storage_units_df.index, snapshots, domain=NonNegativeReals, bounds=su_p_dispatch_bounds)



    def su_p_store_bounds(model,su_name,snapshot):
        su = network.storage_units_df.obj[su_name]

        if su.p_nom_extendable:
            return (0,None)
        else:
            return (0,-su.p_nom*su.p_min_pu_fixed)

    network.model.storage_p_store = Var(network.storage_units_df.index, snapshots, domain=NonNegativeReals, bounds=su_p_store_bounds)



    ## Define generator capacity variables if generator is extendble ##

    extendable_storage_units = network.storage_units_df[network.storage_units_df.p_nom_extendable]

    def su_p_nom_bounds(model, su_name):
        su = network.storage_units_df.obj[su_name]
        return (replace_nan_with_none(su.p_nom_min), replace_nan_with_none(su.p_nom_max))

    network.model.storage_p_nom = Var(extendable_storage_units.index, domain=NonNegativeReals, bounds=su_p_nom_bounds)



    ## Define generator dispatch constraints for extendable generators ##

    def su_p_upper(model,su_name,snapshot):
        su = network.storage_units_df.obj[su_name]
        return model.storage_p_dispatch[su_name,snapshot] <= model.storage_p_nom[su_name]*su.p_max_pu_fixed

    network.model.storage_p_upper = Constraint(extendable_storage_units.index,snapshots,rule=su_p_upper)


    def su_p_lower(model,su_name,snapshot):
        su = network.storage_units_df.obj[su_name]
        return model.storage_p_store[su_name,snapshot] <= -model.storage_p_nom[su_name]*su.p_min_pu_fixed

    network.model.storage_p_lower = Constraint(extendable_storage_units.index,snapshots,rule=su_p_lower)



    ## Now define state of charge constraints ##

    network.model.state_of_charge = Var(network.storage_units_df.index, snapshots, domain=NonNegativeReals, bounds=(0,None))

    def soc_upper(model,su_name,snapshot):
        su = network.storage_units_df.obj[su_name]
        if su.p_nom_extendable:
            return model.state_of_charge[su.name,snapshot] - su.max_hours*model.storage_p_nom[su_name] <= 0
        else:
            return model.state_of_charge[su.name,snapshot] - su.max_hours*su.p_nom <= 0

    network.model.state_of_charge_upper = Constraint(network.storage_units_df.index, snapshots, rule=soc_upper)


    def soc_constraint(model,su_name,snapshot):

        su = network.storage_units_df.obj[su_name]

        i = snapshots.get_loc(snapshot)

        if i == 0:
            previous_state_of_charge = su.state_of_charge_initial
        else:
            previous = snapshots[i-1]
            previous_state_of_charge = model.state_of_charge[su_name,previous]

        elapsed_hours = network.snapshot_weightings[snapshot]

        if pd.isnull(su.state_of_charge[snapshot]):
            state_of_charge = model.state_of_charge[su_name,snapshot]
        else:
            state_of_charge = su.state_of_charge[snapshot]

        return (1-su.standing_loss)**elapsed_hours*previous_state_of_charge\
            + su.efficiency_store*model.storage_p_store[su_name,snapshot]*elapsed_hours\
            - (1/su.efficiency_dispatch)*model.storage_p_dispatch[su_name,snapshot]*elapsed_hours\
            + su.inflow[snapshot]*elapsed_hours - state_of_charge == 0

    network.model.state_of_charge_constraint = Constraint(network.storage_units_df.index, snapshots, rule=soc_constraint)



    def soc_constraint_fixed(model,su_name,snapshot):

        su = network.storage_units_df.obj[su_name]

        if pd.isnull(su.state_of_charge[snapshot]):
            return Constraint.Feasible
        else:
            return model.state_of_charge[su_name,snapshot] == su.state_of_charge[snapshot]

    network.model.state_of_charge_constraint_fixed = Constraint(network.storage_units_df.index, snapshots, rule=soc_constraint_fixed)




def define_branch_extension_variables(network,snapshots):

    branches_df = network.branches_df

    extendable_branches = branches_df[branches_df.s_nom_extendable]


    def branch_s_nom_bounds(model, branch_type, branch_name):
        branch = extendable_branches.obj[(branch_type, branch_name)]
        return (replace_nan_with_none(branch.s_nom_min), replace_nan_with_none(branch.s_nom_max))

    network.model.branch_s_nom = Var(list(extendable_branches.index), domain=NonNegativeReals, bounds=branch_s_nom_bounds)



def define_controllable_branch_flows(network,snapshots):


    def tl_p_bounds(model,tl_name,snapshot):
        tl = network.transport_links_df.obj[tl_name]
        if tl.s_nom_extendable:
            return (None,None)
        else:
            return (tl.p_min,tl.p_max)

    network.model.transport_link_p = Var(network.transport_links_df.index, snapshots, domain=Reals, bounds=tl_p_bounds)


    extendable_transport_links = network.transport_links_df[network.transport_links_df.s_nom_extendable]

    def tl_p_upper(model,tl_name,snapshot):
        return model.transport_link_p[tl_name,snapshot] <= model.branch_s_nom["TransportLink",tl_name]

    network.model.transport_link_p_upper = Constraint(extendable_transport_links.index,snapshots,rule=tl_p_upper)


    def tl_p_lower(model,tl_name,snapshot):
        return model.transport_link_p[tl_name,snapshot] >= -model.branch_s_nom["TransportLink",tl_name]

    network.model.transport_link_p_lower = Constraint(extendable_transport_links.index,snapshots,rule=tl_p_lower)






def define_passive_branch_flows(network,snapshots):

    network.model.voltage_angles = Var(network.buses_df.index, snapshots, domain=Reals, bounds=(None,None))

    def slack(model,sn_name,snapshot):
        return model.voltage_angles[network.sub_networks_df.slack_bus[sn_name], snapshot] == 0

    network.model.slack_angle = Constraint(network.sub_networks_df.index, snapshots, rule=slack)

    passive_branches = network.passive_branches_df

    def flow(model,branch_type,branch_name,snapshot):
        branch = passive_branches.obj[branch_type,branch_name]
        return 1/branch.x_pu*(model.voltage_angles[branch.bus0.name,snapshot]- model.voltage_angles[branch.bus1.name,snapshot])

    network.model.flow = Expression(list(passive_branches.index),snapshots,rule=flow)


def define_passive_branch_constraints(network,snapshots):


    passive_branches = network.passive_branches_df

    extendable_branches = passive_branches[passive_branches.s_nom_extendable]

    def flow_upper(model,branch_type,branch_name,snapshot):
        branch = passive_branches.obj[branch_type,branch_name]
        if branch.s_nom_extendable:
            return model.flow[branch_type,branch_name,snapshot] <= model.branch_s_nom[branch_type,branch_name]
        else:
            return model.flow[branch_type,branch_name,snapshot] <= branch.s_nom

    network.model.flow_upper = Constraint(list(passive_branches.index),snapshots,rule=flow_upper)

    def flow_lower(model,branch_type,branch_name,snapshot):
        branch = passive_branches.obj[branch_type,branch_name]
        if branch.s_nom_extendable:
            return model.flow[branch_type,branch_name,snapshot] >= -model.branch_s_nom[branch_type,branch_name]
        else:
            return model.flow[branch_type,branch_name,snapshot] >= -branch.s_nom

    network.model.flow_lower = Constraint(list(passive_branches.index),snapshots,rule=flow_lower)


def define_nodal_balances(network,snapshots):

    passive_branches = network.passive_branches_df

    #create dictionary of inflow branches at each bus

    inflows = {bus_name : {"transport_links" : [], "branches" : []} for bus_name in network.buses_df.index}

    for tl in network.transport_links_df.obj:
        inflows[tl.bus0_name]["transport_links"].append((tl.name,-1))
        inflows[tl.bus1_name]["transport_links"].append((tl.name,1))

    for branch in passive_branches.obj:
        inflows[branch.bus0_name]["branches"].append(((branch.__class__.__name__,branch.name),-1))
        inflows[branch.bus1_name]["branches"].append(((branch.__class__.__name__,branch.name),1))



    def p_balance(model,bus_name,snapshot):

        bus = network.buses_df.obj[bus_name]

        p = sum(gen.sign*model.generator_p[gen.name,snapshot] for gen in bus.generators_df.obj)

        p += sum(su.sign*model.storage_p_dispatch[su.name,snapshot] for su in bus.storage_units_df.obj)

        p -= sum(su.sign*model.storage_p_store[su.name,snapshot] for su in bus.storage_units_df.obj)

        p += sum(load.sign*load.p_set[snapshot] for load in bus.loads_df.obj)

        p += sum(coeff*model.transport_link_p[tl_name,snapshot] for tl_name,coeff in inflows[bus_name]["transport_links"])

        p += sum(coeff*model.flow[bt,bn,snapshot] for (bt,bn),coeff in inflows[bus_name]["branches"])

        #beware if the p above sums to an integer, the below will return True or False, inducing a bug

        return p == 0

    network.model.power_balance = Constraint(network.buses_df.index, snapshots, rule=p_balance)



def define_co2_constraint(network,snapshots):

    def co2_constraint(model):
        return sum(gen.source.co2_emissions*(1/gen.efficiency)*model.generator_p[gen.name,snapshot]*network.snapshot_weightings[snapshot] for gen in network.generators_df.obj for snapshot in snapshots) <= network.co2_limit

    network.model.co2_constraint = Constraint(rule=co2_constraint)





def define_linear_objective(network,snapshots):

    extendable_generators = network.generators_df[network.generators_df.p_nom_extendable].obj

    extendable_storage_units = network.storage_units_df[network.storage_units_df.p_nom_extendable].obj

    branches = network.branches_df

    extendable_branches = branches[branches.s_nom_extendable].obj

    network.model.objective = Objective(expr=sum(gen.marginal_cost*network.model.generator_p[gen.name,snapshot]*network.snapshot_weightings[snapshot] for gen in network.generators_df.obj for snapshot in snapshots)\
                                        + sum(su.marginal_cost*network.model.storage_p_dispatch[su.name,snapshot]*network.snapshot_weightings[snapshot] for su in network.storage_units_df.obj for snapshot in snapshots)\
                                        + sum(gen.capital_cost*(network.model.generator_p_nom[gen.name] - gen.p_nom) for gen in extendable_generators)\
                                        + sum(su.capital_cost*(network.model.storage_p_nom[su.name] - su.p_nom) for su in extendable_storage_units)\
                                        + sum(branch.capital_cost*(network.model.branch_s_nom[branch.__class__.__name__,branch.name] - branch.s_nom) for branch in extendable_branches))




def extract_optimisation_results(network,snapshots):

    for snapshot in snapshots:

        for generator in network.generators_df.obj:
            generator.p[snapshot] = network.model.generator_p[generator.name,snapshot].value

        for su in network.storage_units_df.obj:
            su.p[snapshot] = network.model.storage_p_dispatch[su.name,snapshot].value - network.model.storage_p_store[su.name,snapshot].value
            su.state_of_charge[snapshot] = network.model.state_of_charge[su.name,snapshot].value



        for load in network.loads_df.obj:
            load.p[snapshot] = load.p_set[snapshot]

        for bus in network.buses_df.obj:
            bus.v_ang[snapshot] = network.model.voltage_angles[bus.name,snapshot].value

            bus.p[snapshot] = sum(asset.sign*asset.p[snapshot] for asset in chain(bus.generators_df.obj,bus.loads_df.obj,bus.storage_units_df.obj))


        for tl in network.transport_links_df.obj:
            tl.p1[snapshot] = network.model.transport_link_p[tl.name,snapshot].value
            tl.p0[snapshot] = -tl.p1[snapshot]
            tl.bus0.p[snapshot] += tl.p0[snapshot]
            tl.bus1.p[snapshot] += tl.p1[snapshot]


        for branch in network.passive_branches_df.obj:
            branch.p1[snapshot] = 1/branch.x_pu*(branch.bus0.v_ang[snapshot] - branch.bus1.v_ang[snapshot])
            branch.p0[snapshot] = -branch.p1[snapshot]



    for generator in network.generators_df[network.generators_df.p_nom_extendable].obj:
        generator.p_nom = network.model.generator_p_nom[generator.name].value


    for su in network.storage_units_df[network.storage_units_df.p_nom_extendable].obj:
        su.p_nom = network.model.storage_p_nom[su.name].value


    for branch in network.branches_df[network.branches_df.s_nom_extendable].obj:
        branch.s_nom = network.model.branch_s_nom[branch.__class__.__name__,branch.name].value




def network_lopf(network,snapshots=None,solver_name="glpk"):
    """Linear optimal power flow for snapshots."""

    if not network.topology_determined:
        network.build_graph()
        network.determine_network_topology()



    if snapshots is None:
        snapshots = [network.now]



    #calculate B,H or PTDF for each subnetwork.
    for sub_network in network.sub_networks_df.obj:
        calculate_z_pu(sub_network)
        find_slack_bus(sub_network)


    network.model = ConcreteModel("Linear Optimal Power Flow")


    define_generator_variables_constraints(network,snapshots)

    define_storage_variables_constraints(network,snapshots)

    define_branch_extension_variables(network,snapshots)

    define_controllable_branch_flows(network,snapshots)

    define_passive_branch_flows(network,snapshots)

    define_passive_branch_constraints(network,snapshots)

    define_nodal_balances(network,snapshots)

    if network.co2_limit is not None:
        define_co2_constraint(network,snapshots)

    define_linear_objective(network,snapshots)

    #force solver to also give us the dual prices
    network.model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    opt = SolverFactory(solver_name)

    instance = network.model.create()

    results = opt.solve(instance,suffixes=["dual"],keepfiles=True)

    results.write()

    network.model.load(results)

    extract_optimisation_results(network,snapshots)
