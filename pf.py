

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



from scipy.sparse import csr_matrix

from numpy import r_, ones, zeros, newaxis
from scipy.sparse.linalg import spsolve

import numpy as np

from .components import Line, Transformer
from .dicthelpers import attrfilter, attrdata

def network_pf(network):
    """Non-linear power flow for generic network."""
    #calls pf on each sub_network separately

    raise NotImplementedError("Non-linear power flow not supported yet.")

def sub_network_pf(sub_network):
    """Non-linear power flow for connected sub-network."""

    #calculate bus admittance matrix and then v_mag, v_ang at each bus

    #set p,q,v_mag,v_ang on each bus and on each load, generator and
    #branch

    raise NotImplementedError("Non-linear power flow not supported yet.")



def network_lpf(network,now=None,verbose=True):
    """Linear power flow for generic network."""

    if now is None:
        now=network.now

    #deal with DC networks (assumed to have no loads or generators)
    dc_networks = attrfilter(network.sub_networks, current_type="DC")

    for dc_network in dc_networks:
        if verbose:
            print("Performing linear load-flow on DC sub-network",ac_network)
        dc_network.lpf(now,verbose)

    #deal with transport links
    for transport_link in network.transport_links.itervalues():
        transport_link.p0[now] = -transport_link.p_set[now]
        transport_link.p1[now] = transport_link.p_set[now]


    #now deal with AC networks
    ac_networks = attrfilter(network.sub_networks, current_type="AC")

    for ac_network in ac_networks:
        if verbose:
            print("Performing linear load-flow on AC sub-network",ac_network)
        ac_network.lpf(now,verbose)





def find_slack_bus(sub_network,verbose=True):
    """Find the slack bus in a connected sub-network."""

    slack_buses, slack_bus_indices = attrfilter(sub_network.buses, control="Slack", indexed=True)

    if len(slack_buses) == 0:
        sub_network.slack_bus = next(sub_network.buses.itervalues())
        sub_network.slack_bus._i = 0
        if verbose:
            print("no slack bus found, taking %s to be the slack bus" % sub_network.slack_bus)
    else:
        sub_network.slack_bus = slack_buses[0]
        sub_network.slack_bus._i = slack_bus_indices[0]
        if len(slack_buses) >= 2 and verbose:
            print("more than one slack bus found, taking %s to be the slack bus" % sub_network.slack_bus)


def sub_network_lpf(sub_network,now=None,verbose=True):
    """Linear power flow for connected sub-network."""

    if now is None:
        now=sub_network.network.now

    if verbose:
        print("performing load-flow for snapshot %s",now)

    if len(sub_network.buses) == 1:
        return

    find_slack_bus(sub_network,verbose=verbose)


    if sub_network.current_type == "AC":

        calculate_x_pu(sub_network)

        calculate_B_H(sub_network,verbose=verbose)

        #set the power injection at each node and the bus's index in the OrderedDict
        for i,bus in enumerate(sub_network.buses.itervalues()):
            bus.p[now] = sum(g.sign*g.p_set[now] for g in bus.generators.itervalues()) \
                        + sum(l.sign*l.p_set[now] for l in bus.loads.itervalues())
            bus._i = i

        #power injection should include transport links and converters
        for t in sub_network.network.transport_links.itervalues():
            if t.bus0.name in sub_network.buses:
                t.bus0.p[now] += t.p0[now]
            if t.bus1.name in sub_network.buses:
                t.bus1.p[now] += t.p1[now]


        p = attrdata(sub_network.buses, "p", now)

        num_buses = len(sub_network.buses)

        v_ang = zeros(num_buses)

        non_slack_index = r_[0:sub_network.slack_bus._i, sub_network.slack_bus._i+1:num_buses][:,newaxis]
        v_ang[non_slack_index] = spsolve(sub_network.B[non_slack_index.T, non_slack_index], p[non_slack_index])

        #set slack bus power

        sub_network.slack_bus.p[now] = sub_network.B[sub_network.slack_bus._i,:].dot(v_ang)[0]

        flows = sub_network.H.dot(v_ang)

        for i,bus in enumerate(sub_network.buses.itervalues()):
            bus.v_ang[now] = v_ang[i]

            #allow all loads to dispatch as set
            for load in bus.loads.itervalues():
                load.p[now] = load.p_set[now]

            #allow all non-slack generators to dispatch as set
            if bus != sub_network.slack_bus:
                for generator in bus.generators.itervalues():
                    generator.p[now] = generator.p_set[now]
            else:
                num_generators = len(bus.generators)

                if num_generators == 0:
                    print("slack bus has no generators to take up the slack")
                    continue

                total_generator_p_set = sum([g.sign*g.p_set[now] for g in bus.generators.itervalues()])

                total_generator_p = bus.p[now] - sum([l.sign*l.p_set[now] for l in bus.loads.itervalues()])

                for t in sub_network.network.transport_links.itervalues():
                    if t.bus0 == bus:
                        total_generator_p -= t.p0[now]
                    if t.bus1 == bus:
                        total_generator_p -= t.p1[now]

                #now distribute slack power among generators
                if total_generator_p_set == 0:
                    for generator in bus.generators.itervalues():
                        generator.p[now] = total_generator_p/num_generators
                else:
                    for generator in bus.generators.itervalues():
                        generator.p[now] = total_generator_p*generator.p_set[now]/total_generator_p_set

        for i,branch in enumerate(sub_network.branches.itervalues()):
            branch.p1[now] = flows[i]
            branch.p0[now] = -flows[i]


    elif sub_network.current_type == "DC":
        print("DC networks not supported yet")


def calculate_x_pu(sub_network):

    #convert all branch reactances to per unit
    for branch in sub_network.branches.itervalues():
        if isinstance(branch, Line):
            branch.x_pu = branch.x*sub_network.base_power/(branch.bus0.v_nom**2)
        elif isinstance(branch, Transformer):
            branch.x_pu = branch.x*sub_network.base_power/branch.s_nom


def calculate_B_H(sub_network,verbose=True):
    """Calculate B and H matrices for AC or DC sub-networks."""


    if sub_network.current_type == "DC":
        attribute="r_pu"
    elif sub_network.current_type == "AC":
        attribute="x_pu"

    #following leans heavily on pypower.makeBdc

    for i,bus in enumerate(sub_network.buses.itervalues()):
        bus._i = i

    num_branches = len(sub_network.branches)
    num_buses = len(sub_network.buses)

    index = r_[:num_branches,:num_branches]

    #susceptances
    b = np.array([1/getattr(branch,attribute) for branch in sub_network.branches.itervalues()])

    from_bus = np.array([branch.bus0._i for branch in sub_network.branches.itervalues()])
    to_bus = np.array([branch.bus1._i for branch in sub_network.branches.itervalues()])


    #build weighted Laplacian
    sub_network.H = csr_matrix((r_[b,-b],(index,r_[from_bus,to_bus])))

    incidence = csr_matrix((r_[ones(num_branches),-ones(num_branches)],(index,r_[from_bus,to_bus])),(num_branches,num_buses))

    sub_network.B = incidence.T * sub_network.H





def network_batch_lpf(network,subindex=None):
    """Batched linear power flow with numpy.dot for several snapshots."""
