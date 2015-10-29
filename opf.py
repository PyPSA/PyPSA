

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



from pyomo.environ import ConcreteModel, Var, Objective, NonNegativeReals, Constraint, Reals, Suffix

from pyomo.opt import SolverFactory

from .dicthelpers import attrdata, attrfilter

from .pf import calculate_x_pu, find_slack_bus

from itertools import chain



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
            elif gen.dispatch == "inflexible":
                return (gen.p_set[snapshot],gen.p_set[snapshot])
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





def define_controllable_network_elements(network,subindex):

    def tl_p_bounds(model,tl_name,snapshot):
        tl = network.transport_links[tl_name]
        return (tl.p_min,tl.p_max)

    network.model.transport_link_p = Var(network.transport_links.iterkeys(), subindex, domain=Reals, bounds=tl_p_bounds)



def extract_optimisation_results(network,subindex):
    
    for snapshot in subindex:
    
        for generator in network.generators.itervalues():
            generator.p[snapshot] = network.model.generator_p[generator.name,snapshot].value
        
        for generator in attrfilter(network.generators, p_nom_extendable=True):
            generator.p_nom = network.model.generator_p_nom[generator.name].value


        for load in network.loads.itervalues():
            load.p[snapshot] = load.p_set[snapshot]
            
        for bus in network.buses.itervalues():
            bus.v_ang[snapshot] = network.model.voltage_angles[bus.name,snapshot].value
            
            bus.p[snapshot] = sum(asset.sign*asset.p[snapshot] for asset in chain(bus.generators.itervalues(),bus.loads.itervalues()))


        for tl in network.transport_links.itervalues():
            tl.p1[snapshot] = network.model.transport_link_p[tl.name,snapshot].value
            tl.p0[snapshot] = -tl.p1[snapshot]
            tl.bus0.p[snapshot] += tl.p0[snapshot]
            tl.bus1.p[snapshot] += tl.p1[snapshot]

        
        for sn in network.sub_networks.itervalues():
            for branch in sn.branches.itervalues():
                branch.p1[snapshot] = 1/branch.x_pu*(branch.bus0.v_ang[snapshot] - branch.bus1.v_ang[snapshot])
                branch.p0[snapshot] = -branch.p1[snapshot]



def network_lopf(network,subindex=None,solver_name="glpk"):
    """Linear optimal power flow for snapshots in subindex."""
    
    if subindex is None:
        subindex = [network.now]

    #calculate B,H or PTDF for each subnetwork.                                                                                            
    for sub_network in network.sub_networks.itervalues():
        calculate_x_pu(sub_network)
        find_slack_bus(sub_network)


    network.model = ConcreteModel("Linear Optimal Power Flow")
    
    
    ### Establish all optimisation variables ###

    define_generator_variables_constraints(network,subindex)

    define_controllable_network_elements(network,subindex)

    
    network.model.voltage_angles = Var(network.buses.iterkeys(), subindex, domain=Reals, bounds=(-1,1))

    
    #to include: s_nom for lines
    
    
    
    ### Establish all optimisation constraints ###



    def slack(model,sn_name,snapshot):
        slack_bus = network.sub_networks[sn_name].slack_bus
        return network.model.voltage_angles[slack_bus.name,snapshot] == 0

    network.model.slack_angle = Constraint(network.sub_networks.iterkeys(), subindex, rule=slack)

    
    def flow_upper(model,branch_name,snapshot):
        branch = network.branches[branch_name]
        return 1/branch.x_pu*(network.model.voltage_angles[branch.bus0.name,snapshot]- network.model.voltage_angles[branch.bus1.name,snapshot]) <= branch.s_nom
    
    network.model.flow_upper = Constraint([branch.name for sn in network.sub_networks.itervalues() for branch in sn.branches.itervalues()],subindex,rule=flow_upper)
    
    def flow_lower(model,branch_name,snapshot):
        branch = network.branches[branch_name]
        return 1/branch.x_pu*(network.model.voltage_angles[branch.bus0.name,snapshot]- network.model.voltage_angles[branch.bus1.name,snapshot]) >= -branch.s_nom
    
    network.model.flow_lower = Constraint([branch.name for sn in network.sub_networks.itervalues() for branch in sn.branches.itervalues()],subindex,rule=flow_lower)
        
    
    
    
    def p_balance(model,bus_name,snapshot):
        bus = network.buses[bus_name]
        
        p = sum(gen.sign*network.model.generator_p[gen.name,snapshot] for gen in bus.generators.itervalues())
        
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
                network.model.power_balance[branch.bus0.name,snapshot].body -= network.model.flow_upper[branch.name,snapshot].body
                network.model.power_balance[branch.bus1.name,snapshot].body += network.model.flow_upper[branch.name,snapshot].body


    extendable_generators = attrfilter(network.generators, p_nom_extendable=True)
                
    network.model.objective = Objective(expr=sum(gen.marginal_cost*network.model.generator_p[gen.name,snapshot] for gen in network.generators.itervalues() for snapshot in subindex) + sum(gen.capital_cost*(network.model.generator_p_nom[gen.name] - gen.p_nom) for gen in extendable_generators))
    
    #force solver to also give us the dual prices                                                                                              
    network.model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    opt = SolverFactory(solver_name)

    instance = network.model.create()

    results = opt.solve(instance,suffixes=["dual"],keepfiles=True)

    results.write()

    network.model.load(results)


    extract_optimisation_results(network,subindex)
