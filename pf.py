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
import pandas as pd

from .components import Line, Transformer

from itertools import chain



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


    if not network.topology_determined:
        network.build_graph()
        network.determine_network_topology()

    if not network.dependent_values_calculated:
        calculate_dependent_values(network)

    if now is None:
        now=network.now


    #deal with transport links and converters
    network.converters.p0.loc[now] = network.converters.p_set.loc[now]
    network.converters.p1.loc[now] = -network.converters.p_set.loc[now]
    network.transport_links.p0.loc[now] = network.transport_links.p_set.loc[now]
    network.transport_links.p1.loc[now] = -network.transport_links.p_set.loc[now]


    for sub_network in network.sub_networks.obj:
        if verbose:
            print("Performing linear load-flow on %s sub-network %s" % (sub_network.current_type,sub_network))
        sub_network.lpf(now,verbose)





def find_slack_bus(sub_network,verbose=True):
    """Find the slack bus in a connected sub-network."""

    gens = sub_network.generators

    if len(gens) == 0:
        if verbose:
            print("No generators in %s, better hope power is already balanced" % sub_network)
        sub_network.slack_generator = None
        sub_network.slack_bus = sub_network.buses.index[0]

    else:

        slacks = gens[gens.control == "Slack"]

        if len(slacks) == 0:
            sub_network.slack_generator = gens.index[0]
            sub_network.network.generators.loc[sub_network.slack_generator,"control"] = "Slack"
            if verbose:
                print("No slack generator found, using %s as the slack generator" % sub_network.slack_generator)

        elif len(slacks) == 1:
            sub_network.slack_generator = slacks.index[0]
        else:
            sub_network.slack_generator = slacks.index[0]
            sub_network.network.generators.loc[slacks.index[1:],"control"] = "PV"
            if verbose:
                print("More than one slack generator found, taking %s to be the slack generator" % sub_network.slack_generator)

        sub_network.slack_bus = gens.bus[sub_network.slack_generator]

    if verbose:
        print("Slack bus is %s" % sub_network.slack_bus)


def find_bus_controls(sub_network,verbose=True):
    """Find slack and all PV and PQ buses for a sub_network.
    This function also fixes sub_network.buses_o, a DataFrame
    ordered by control type."""

    network = sub_network.network

    find_slack_bus(sub_network,verbose)

    gens = sub_network.generators
    buses = sub_network.buses

    network.buses.loc[buses.index,"control"] = "PQ"

    pvs = gens[gens.control == "PV"]

    network.buses.loc[pvs.bus,"control"] = "PV"

    network.buses.loc[sub_network.slack_bus,"control"] = "Slack"

    buses = sub_network.buses

    sub_network.pvs = buses[buses.control == "PV"]
    sub_network.pqs = buses[buses.control == "PQ"]

    sub_network.pvpqs = pd.concat((sub_network.pvs,sub_network.pqs))

    #order buses
    sub_network.buses_o = pd.concat((buses.loc[[sub_network.slack_bus]],sub_network.pvpqs))
    sub_network.buses_o["i"] = range(len(sub_network.buses_o))


def calculate_dependent_values(network):
    """Calculate per unit impedances and append voltages to lines and shunt impedances."""

    #add voltages to components from bus voltage
    for list_name in ["lines","shunt_impedances"]:
        df = getattr(network,list_name)

        if "v_nom" in df.columns:
            df.drop(["v_nom"],axis=1,inplace=True)

        bus_attr = "bus0" if list_name == "lines" else "bus"

        join = pd.merge(df,network.buses,
                        how="left",
                        left_on=bus_attr,
                        right_index=True)

        df.loc[:,"v_nom"] = join["v_nom"]


    network.lines["x_pu"] = network.lines.x/(network.lines.v_nom**2)
    network.lines["r_pu"] = network.lines.r/(network.lines.v_nom**2)
    network.lines["b_pu"] = network.lines.b*network.lines.v_nom**2
    network.lines["g_pu"] = network.lines.g*network.lines.v_nom**2
    #convert transformer impedances from base power s_nom to base = 1 MVA
    network.transformers["x_pu"] = network.transformers.x/network.transformers.s_nom
    network.transformers["r_pu"] = network.transformers.r/network.transformers.s_nom
    network.transformers["b_pu"] = network.transformers.b*network.transformers.s_nom
    network.transformers["g_pu"] = network.transformers.g*network.transformers.s_nom
    network.shunt_impedances["b_pu"] = network.shunt_impedances.b*network.shunt_impedances.v_nom**2
    network.shunt_impedances["g_pu"] = network.shunt_impedances.g*network.shunt_impedances.v_nom**2

    network.dependent_values_calculated = True


def calculate_B_H(sub_network,verbose=True):
    """Calculate B and H matrices for AC or DC sub-networks."""


    if sub_network.current_type == "DC":
        attribute="r_pu"
    elif sub_network.current_type == "AC":
        attribute="x_pu"

    branches = sub_network.branches
    buses = sub_network.buses_o

    #following leans heavily on pypower.makeBdc

    num_branches = len(branches)
    num_buses = len(buses)

    index = r_[:num_branches,:num_branches]

    #susceptances
    b = 1/branches[attribute]

    from_bus = np.array([buses["i"][bus] for bus in branches.bus0])
    to_bus = np.array([buses["i"][bus] for bus in branches.bus1])


    #build weighted Laplacian
    sub_network.H = csr_matrix((r_[b,-b],(index,r_[from_bus,to_bus])))

    incidence = csr_matrix((r_[ones(num_branches),-ones(num_branches)],(index,r_[from_bus,to_bus])),(num_branches,num_buses))

    sub_network.B = incidence.T * sub_network.H



def sub_network_lpf(sub_network,now=None,verbose=True):
    """Linear power flow for connected sub-network."""

    network = sub_network.network

    if now is None:
        now = network.now

    if verbose:
        print("Performing load-flow for snapshot %s" % (now))

    if not network.dependent_values_calculated:
        calculate_dependent_values(network)

    find_bus_controls(sub_network,verbose=verbose)


    if len(sub_network.branches) > 0:
        calculate_B_H(sub_network,verbose=verbose)

    branches = sub_network.branches
    buses = sub_network.buses_o

    #set the power injection at each node
    for bus in buses.obj:
        bus.p[now] = sum(g.sign*g.p_set[now] for g in bus.generators.obj) \
                     + sum(l.sign*l.p_set[now] for l in bus.loads.obj) \
                     + sum(sh.sign*sh.g_pu for sh in bus.shunt_impedances.obj)

    #power injection should include transport links and converters
    for t in chain(network.transport_links.obj,network.converters.obj):
        if t.bus0 in buses.index:
            buses.obj[t.bus0].p[now] -= t.p0[now]
        if t.bus1 in buses.index:
            buses.obj[t.bus1].p[now] -= t.p1[now]


    p = network.buses.p.loc[now,buses.index]

    num_buses = len(buses)

    v_diff = zeros(num_buses)

    if len(sub_network.branches) > 0:
        v_diff[1:] = spsolve(sub_network.B[1:, 1:], p[1:])

        branches["flows"] = sub_network.H.dot(v_diff)

        lines = branches.loc["Line"]
        trafos = branches.loc["Transformer"]

        network.lines.p1.loc[now,lines.index] = -lines["flows"]
        network.lines.p0.loc[now,lines.index] = lines["flows"]

        network.transformers.p1.loc[now,trafos.index] = -trafos["flows"]
        network.transformers.p0.loc[now,trafos.index] = trafos["flows"]



    #set slack bus power to pick up remained
    network.buses.p.loc[now,sub_network.slack_bus] = -sum(p[1:])

    if sub_network.current_type == "AC":
        network.buses.v_ang.loc[now,buses.index] = v_diff
    elif sub_network.current_type == "DC":
        network.buses.v_mag.loc[now,buses.index] = buses.v_nom + v_diff*buses.v_nom

    #allow all loads to dispatch as set
    loads = sub_network.loads
    network.loads.p.loc[now,loads.index] = network.loads.p_set.loc[now,loads.index]

    #allow all loads to dispatch as set
    shunt_impedances = sub_network.shunt_impedances
    network.shunt_impedances.p.loc[now,shunt_impedances.index] = network.shunt_impedances.g_pu.loc[shunt_impedances.index].values

    #allow all generators to dispatch as set
    generators = sub_network.generators
    network.generators.p.loc[now,generators.index] = network.generators.p_set.loc[now,generators.index]

    #let slack generator take up the slack
    if sub_network.slack_generator is not None:
        network.generators.p.loc[now,sub_network.slack_generator] += network.buses.p.loc[now,sub_network.slack_bus] - p[0]



def network_batch_lpf(network,snapshots=None):
    """Batched linear power flow with numpy.dot for several snapshots."""
