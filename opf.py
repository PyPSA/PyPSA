

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

from .dicthelpers import attrdata, attrfilter

from .pf import calculate_x_pu, find_slack_bus

from itertools import chain

import pandas as pd



def network_opf(network,subindex=None):
    """Optimal power flow for snapshots in subindex."""

    raise NotImplementedError("Non-linear optimal power flow not supported yet.")




def define_generator_variables_constraints(network,subindex):


    ## Define generator dispatch variables ##

    def gen_p_bounds(model,gen_name,snapshot):
        gen = network.generators[gen_name]

        if gen.p_nom_extendable:
            return (None,None)
        else:
            if gen.dispatch == "flexible":
                return (gen.p_nom*gen.p_min_pu_fixed,gen.p_nom*gen.p_max_pu_fixed)
            elif gen.dispatch == "variable":
                return (gen.p_nom*gen.p_min_pu[snapshot],gen.p_nom*gen.p_max_pu[snapshot])
            else:
                raise NotImplementedError("Dispatch type %s is not supported yet." % (gen.dispatch))


    network.model.generator_p = Var(network.generators.iterkeys(), subindex, domain=Reals, bounds=gen_p_bounds)



    ## Define generator capacity variables if generator is extendble ##

    extendable_generators = attrfilter(network.generators, p_nom_extendable=True)

    def gen_p_nom_bounds(model, gen_name):
        gen = network.generators[gen_name]
        return (gen.p_nom_min, gen.p_nom_max)

    network.model.generator_p_nom = Var([gen.name for gen in extendable_generators], domain=NonNegativeReals, bounds=gen_p_nom_bounds)



    ## Define generator dispatch constraints for extendable generators ##

    def gen_p_lower(model,gen_name,snapshot):
        gen = network.generators[gen_name]

        if gen.dispatch == "flexible":
            return network.model.generator_p[gen_name,snapshot] >= network.model.generator_p_nom[gen_name]*gen.p_min_pu_fixed
        elif gen.dispatch == "variable":
            return network.model.generator_p[gen_name,snapshot] >= network.model.generator_p_nom[gen_name]*gen.p_min_pu[snapshot]
        else:
            raise NotImplementedError("Dispatch type %s is not supported yet for extendability." % (gen.dispatch))

    network.model.generator_p_lower = Constraint([gen.name for gen in extendable_generators],subindex,rule=gen_p_lower)


    def gen_p_upper(model,gen_name,snapshot):
        gen = network.generators[gen_name]

        if gen.dispatch == "flexible":
            return network.model.generator_p[gen_name,snapshot] <= network.model.generator_p_nom[gen_name]*gen.p_max_pu_fixed
        elif gen.dispatch == "variable":
            return network.model.generator_p[gen_name,snapshot] <= network.model.generator_p_nom[gen_name]*gen.p_max_pu[snapshot]
        else:
            raise NotImplementedError("Dispatch type %s is not supported yet for extendability." % (gen.dispatch))

    network.model.generator_p_upper = Constraint([gen.name for gen in extendable_generators],subindex,rule=gen_p_upper)





def define_storage_variables_constraints(network,subindex):


    ## Define storage dispatch variables ##

    def su_p_dispatch_bounds(model,su_name,snapshot):
        su = network.storage_units[su_name]

        if su.p_nom_extendable:
            return (0,None)
        else:
            return (0,su.p_nom*su.p_max_pu_fixed)

    network.model.storage_p_dispatch = Var(network.storage_units.iterkeys(), subindex, domain=NonNegativeReals, bounds=su_p_dispatch_bounds)



    def su_p_store_bounds(model,su_name,snapshot):
        su = network.storage_units[su_name]

        if su.p_nom_extendable:
            return (0,None)
        else:
            return (0,-su.p_nom*su.p_min_pu_fixed)

    network.model.storage_p_store = Var(network.storage_units.iterkeys(), subindex, domain=NonNegativeReals, bounds=su_p_store_bounds)



    ## Define generator capacity variables if generator is extendble ##

    extendable_storage_units = attrfilter(network.storage_units, p_nom_extendable=True)

    def su_p_nom_bounds(model, su_name):
        su = network.storage_units[su_name]
        return (su.p_nom_min, su.p_nom_max)

    network.model.storage_p_nom = Var([su.name for su in extendable_storage_units], domain=NonNegativeReals, bounds=su_p_nom_bounds)



    ## Define generator dispatch constraints for extendable generators ##

    def su_p_upper(model,su_name,snapshot):
        su = network.storage_units[su_name]
        return network.model.storage_p_dispatch[su_name,snapshot] <= network.model.storage_p_nom[su_name]*su.p_max_pu_fixed

    network.model.storage_p_upper = Constraint([su.name for su in extendable_storage_units],subindex,rule=su_p_upper)


    def su_p_lower(model,su_name,snapshot):
        su = network.storage_units[su_name]
        return network.model.storage_p_store[su_name,snapshot] <= -network.model.storage_p_nom[su_name]*su.p_min_pu_fixed

    network.model.storage_p_lower = Constraint([su.name for su in extendable_storage_units],subindex,rule=su_p_lower)



    ## Now define state of charge constraints ##

    network.model.state_of_charge = Var(network.storage_units.iterkeys(), subindex, domain=NonNegativeReals, bounds=(0,None))

    def soc_upper(model,su_name,snapshot):
        su = network.storage_units[su_name]
        if su.p_nom_extendable:
            return network.model.state_of_charge[su.name,snapshot] - su.max_hours*network.model.storage_p_nom[su_name] <= 0
        else:
            return network.model.state_of_charge[su.name,snapshot] - su.max_hours*su.p_nom <= 0

    network.model.state_of_charge_upper = Constraint(network.storage_units.iterkeys(), subindex, rule=soc_upper)


    def soc_constraint(model,su_name,snapshot):

        su = network.storage_units[su_name]

        i = subindex.get_loc(snapshot)

        if i == 0:
            previous_state_of_charge = su.state_of_charge_start
            elapsed_hours = su.hours_from_state_of_charge_start
        else:
            previous = subindex[i-1]
            previous_state_of_charge = network.model.state_of_charge[su_name,previous]
            elapsed_hours = (snapshot - previous).total_seconds()/3600.

        if pd.isnull(su.state_of_charge[snapshot]):
            state_of_charge = network.model.state_of_charge[su_name,snapshot]
        else:
            state_of_charge = su.state_of_charge[snapshot]

        return (1-su.standing_loss)**elapsed_hours*previous_state_of_charge\
            + su.efficiency_store*network.model.storage_p_store[su_name,snapshot]*elapsed_hours\
            - (1/su.efficiency_dispatch)*network.model.storage_p_dispatch[su_name,snapshot]*elapsed_hours\
            + su.inflow[snapshot]*elapsed_hours - state_of_charge == 0

    network.model.state_of_charge_constraint = Constraint(network.storage_units.iterkeys(), subindex, rule=soc_constraint)



    def soc_constraint_fixed(model,su_name,snapshot):

        if pd.isnull(su.state_of_charge[snapshot]):
            return Constraint.Feasible
        else:
            return network.model.state_of_charge[su_name,snapshot] == su.state_of_charge[snapshot]

    network.model.state_of_charge_constraint_fixed = Constraint(network.storage_units.iterkeys(), subindex, rule=soc_constraint_fixed)




def define_branch_extension_variables(network,subindex):

    extendable_branches = attrfilter(network.branches, s_nom_extendable=True)


    def branch_s_nom_bounds(model, branch_name):
        branch = network.branches[branch_name]
        return (branch.s_nom_min, branch.s_nom_max)

    network.model.branch_s_nom = Var([branch.name for branch in extendable_branches], domain=NonNegativeReals, bounds=branch_s_nom_bounds)



def define_controllable_branch_flows(network,subindex):


    def tl_p_bounds(model,tl_name,snapshot):
        tl = network.transport_links[tl_name]
        if tl.s_nom_extendable:
            return (None,None)
        else:
            return (tl.p_min,tl.p_max)

    network.model.transport_link_p = Var(network.transport_links.iterkeys(), subindex, domain=Reals, bounds=tl_p_bounds)


    extendable_transport_links = attrfilter(network.transport_links, s_nom_extendable=True)

    def tl_p_upper(model,tl_name,snapshot):
        return network.model.transport_link_p[tl_name,snapshot] <= network.model.branch_s_nom[tl_name]

    network.model.transport_link_p_upper = Constraint([tl.name for tl in extendable_transport_links],subindex,rule=tl_p_upper)


    def tl_p_lower(model,tl_name,snapshot):
        return network.model.transport_link_p[tl_name,snapshot] >= -network.model.branch_s_nom[tl_name]

    network.model.transport_link_p_lower = Constraint([tl.name for tl in extendable_transport_links],subindex,rule=tl_p_lower)






def define_passive_branch_flows(network,subindex):


    network.model.voltage_angles = Var(network.buses.iterkeys(), subindex, domain=Reals, bounds=(-1,1))

    def slack(model,sn_name,snapshot):
        slack_bus = network.sub_networks[sn_name].slack_bus
        return network.model.voltage_angles[slack_bus.name,snapshot] == 0

    network.model.slack_angle = Constraint(network.sub_networks.iterkeys(), subindex, rule=slack)


    def flow(model,branch_name,snapshot):
        branch = network.branches[branch_name]
        return 1/branch.x_pu*(network.model.voltage_angles[branch.bus0.name,snapshot]- network.model.voltage_angles[branch.bus1.name,snapshot])

    network.model.flow = Expression([branch.name for sn in network.sub_networks.itervalues() for branch in sn.branches.itervalues()],subindex,rule=flow)


def define_passive_branch_constraints(network,subindex):


    extendable_branches = attrfilter((branch for sn in network.sub_networks.itervalues() for branch in sn.branches.itervalues()), s_nom_extendable=True)

    def flow_upper(model,branch_name,snapshot):
        branch = network.branches[branch_name]
        if branch.s_nom_extendable:
            return network.model.flow[branch_name,snapshot] <= network.model.branch_s_nom[branch_name]
        else:
            return network.model.flow[branch_name,snapshot] <= branch.s_nom

    network.model.flow_upper = Constraint([branch.name for sn in network.sub_networks.itervalues() for branch in sn.branches.itervalues()],subindex,rule=flow_upper)

    def flow_lower(model,branch_name,snapshot):
        branch = network.branches[branch_name]
        if branch.s_nom_extendable:
            return network.model.flow[branch_name,snapshot] >= -network.model.branch_s_nom[branch_name]
        else:
            return network.model.flow[branch_name,snapshot] >= -branch.s_nom

    network.model.flow_lower = Constraint([branch.name for sn in network.sub_networks.itervalues() for branch in sn.branches.itervalues()],subindex,rule=flow_lower)


def define_nodal_balances(network,subindex):


    def p_balance(model,bus_name,snapshot):
        bus = network.buses[bus_name]

        p = sum(gen.sign*network.model.generator_p[gen.name,snapshot] for gen in bus.generators.itervalues())

        p += sum(su.sign*network.model.storage_p_dispatch[su.name,snapshot] for su in bus.storage_units.itervalues())

        p -= sum(su.sign*network.model.storage_p_store[su.name,snapshot] for su in bus.storage_units.itervalues())

        p += sum(load.sign*load.p_set[snapshot] for load in bus.loads.itervalues())

        return p == 0

    network.model.power_balance = Constraint(network.buses.iterkeys(), subindex, rule=p_balance)

    #add branches to nodal power balance equation

    for tl in network.transport_links.itervalues():
        for snapshot in subindex:
            network.model.power_balance[tl.bus0.name,snapshot].body -= network.model.transport_link_p[tl.name,snapshot]
            network.model.power_balance[tl.bus1.name,snapshot].body += network.model.transport_link_p[tl.name,snapshot]

    for sub_network in network.sub_networks.itervalues():
        for branch in sub_network.branches.itervalues():
            for snapshot in subindex:
                network.model.power_balance[branch.bus0.name,snapshot].body -= network.model.flow[branch.name,snapshot]
                network.model.power_balance[branch.bus1.name,snapshot].body += network.model.flow[branch.name,snapshot]




def define_linear_objective(network,subindex):

    extendable_generators = attrfilter(network.generators, p_nom_extendable=True)

    extendable_storage_units = attrfilter(network.storage_units, p_nom_extendable=True)

    extendable_branches = attrfilter(network.branches, s_nom_extendable=True)

    network.model.objective = Objective(expr=sum(gen.marginal_cost*network.model.generator_p[gen.name,snapshot] for gen in network.generators.itervalues() for snapshot in subindex)\
                                        + sum(su.marginal_cost*network.model.storage_p_dispatch[su.name,snapshot] for su in network.storage_units.itervalues() for snapshot in subindex)\
                                        + sum(gen.capital_cost*(network.model.generator_p_nom[gen.name] - gen.p_nom) for gen in extendable_generators)\
                                        + sum(su.capital_cost*(network.model.storage_p_nom[su.name] - su.p_nom) for su in extendable_storage_units)\
                                        + sum(branch.capital_cost*(network.model.branch_s_nom[branch.name] - branch.s_nom) for branch in extendable_branches))




def extract_optimisation_results(network,subindex):

    for snapshot in subindex:

        for generator in network.generators.itervalues():
            generator.p[snapshot] = network.model.generator_p[generator.name,snapshot].value

        for su in network.storage_units.itervalues():
            su.p[snapshot] = network.model.storage_p_dispatch[su.name,snapshot].value - network.model.storage_p_store[su.name,snapshot].value
            su.state_of_charge[snapshot] = network.model.state_of_charge[su.name,snapshot].value



        for load in network.loads.itervalues():
            load.p[snapshot] = load.p_set[snapshot]

        for bus in network.buses.itervalues():
            bus.v_ang[snapshot] = network.model.voltage_angles[bus.name,snapshot].value

            bus.p[snapshot] = sum(asset.sign*asset.p[snapshot] for asset in chain(bus.generators.itervalues(),bus.loads.itervalues(),bus.storage_units.itervalues()))


        for tl in network.transport_links.itervalues():
            tl.p1[snapshot] = network.model.transport_link_p[tl.name,snapshot].value
            tl.p0[snapshot] = -tl.p1[snapshot]
            tl.bus0.p[snapshot] += tl.p0[snapshot]
            tl.bus1.p[snapshot] += tl.p1[snapshot]


        for sn in network.sub_networks.itervalues():
            for branch in sn.branches.itervalues():
                branch.p1[snapshot] = 1/branch.x_pu*(branch.bus0.v_ang[snapshot] - branch.bus1.v_ang[snapshot])
                branch.p0[snapshot] = -branch.p1[snapshot]


    for generator in attrfilter(network.generators, p_nom_extendable=True):
        generator.p_nom = network.model.generator_p_nom[generator.name].value


    for su in attrfilter(network.storage_units, p_nom_extendable=True):
        su.p_nom = network.model.storage_p_nom[su.name].value


    for branch in attrfilter(network.branches, s_nom_extendable=True):
        branch.s_nom = network.model.branch_s_nom[branch.name].value




def network_lopf(network,subindex=None,solver_name="glpk"):
    """Linear optimal power flow for snapshots in subindex."""


    if subindex is None:
        subindex = [network.now]



    #calculate B,H or PTDF for each subnetwork.
    for sub_network in network.sub_networks.itervalues():
        calculate_x_pu(sub_network)
        find_slack_bus(sub_network)


    network.model = ConcreteModel("Linear Optimal Power Flow")


    define_generator_variables_constraints(network,subindex)

    define_storage_variables_constraints(network,subindex)

    define_branch_extension_variables(network,subindex)

    define_controllable_branch_flows(network,subindex)

    define_passive_branch_flows(network,subindex)

    define_passive_branch_constraints(network,subindex)

    define_nodal_balances(network,subindex)

    define_linear_objective(network,subindex)

    #force solver to also give us the dual prices
    network.model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    opt = SolverFactory(solver_name)

    instance = network.model.create()

    results = opt.solve(instance,suffixes=["dual"],keepfiles=True)

    results.write()

    network.model.load(results)

    extract_optimisation_results(network,subindex)
