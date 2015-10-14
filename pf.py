

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



def network_pf(network):
    """Non-linear power flow for generic network."""
    #calls pf on each sub_network separately


def sub_network_pf(sub_network):
    """Non-linear power flow for connected sub-network."""

    #calculate bus admittance matrix and then v_mag, v_ang at each bus

    #set p,q,v_mag,v_ang on each bus and on each load, generator and
    #branch




def network_lpf(network,i=None,verbose=True):
    """Linear power flow for generic network."""

    if i is None:
        i=network.i

    #deal with DC networks (assumed to have no loads or generators)                                                
    dc_networks = filter(lambda sn: sn.current_type == "DC",network.sub_networks.itervalues())
    
    for dc_network in dc_networks:
        if verbose:
            print("Performing linear load-flow on DC sub-network",ac_network)
        dc_network.lpf(i,verbose)
        
    #deal with transport links
    for transport_link in network.transport_links.itervalues():
        transport_link.p0[i] = -transport_link.p_set[i]
        transport_link.p1[i] = transport_link.p_set[i]


    #now deal with AC networks
    ac_networks = filter(lambda sn: sn.current_type == "AC",network.sub_networks.itervalues())

    for ac_network in ac_networks:
        if verbose:
            print("Performing linear load-flow on AC sub-network",ac_network)
        ac_network.lpf(i,verbose)



def sub_network_lpf(sub_network,i=None,verbose=True):
    """Linear power flow for connected sub-network."""
    

    if i is None:
        i=sub_network.network.i

    if verbose:
        print("performing load-flow for snapshot %s",i)


    if len(sub_network.buses) == 1:
        return
    
    
    #first set the slack bus
    slack_buses = filter(lambda bus: bus.control=="Slack",sub_network.buses.itervalues())

    if len(slack_buses) == 0:
        
        slack_bus = next(sub_network.buses.itervalues())
        
        if verbose:
            print("no slack bus found, taking %s to be the slack bus" % slack_bus)
    elif len(slack_buses) == 1:
        
        slack_bus = slack_buses[0]
    
    elif len(slack_buses) == 2:
        
        slack_bus = slack_buses[0]
        
        if verbose:
            print("more than one slack bus found, taking %s to be the slack bus" % slack_bus)
    
    if sub_network.current_type == "AC":

        #convert all branch reactances to per unit
        for branch in sub_network.branches.itervalues():
            if branch.__class__.__name__ == "Line":
                branch.x_pu = branch.x*sub_network.base_power/(branch.bus0.v_nom**2)
            elif branch.__class__.__name__ == "Transformer":
                branch.x_pu = branch.x*sub_network.base_power/branch.s_nom

        #set the power injection at each node
        for bus in sub_network.buses.itervalues():
            bus.p[i] = sum([g.sign*g.p_set[i] for g in bus.generators.itervalues()]) \
                        + sum([l.sign*l.p_set[i] for l in bus.loads.itervalues()])
        
        for t in sub_network.network.transport_links.itervalues():
            if t.bus0.name in sub_network.buses:
                t.bus0.p[i] += t.p0[i]
            if t.bus1.name in sub_network.buses:
                t.bus1.p[i] += t.p1[i]

        
    
                                                                                                  
        #calculate B,H or PTDF using branch.x_pu
        
        sub_network.B = calculate_B(sub_network,verbose=verbose)
    
        slack_bus.v_ang.set_value(i,0)

        #delete row and column of slack bus and invert B to get v_ang
        #for other buses

        #then set slack_bus.p

        sub_network.H = calculate_H(sub_network,verbose=verbose)
    
        #now calculate branch.p0 and branch.p1

        #set p for all generators loads based on p_set; slack_bus is
        #different

    
    
    
def calculate_B(sub_network,attribute="x_pu",verbose=True):
    pass
    
    
def calculate_H(sub_network,attribute="x_pu",verbose=True):
    pass
    




def network_batch_lpf(network,subindex=None):
    """Batched linear power flow with numpy.dot for several snapshots."""

