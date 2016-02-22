## Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS)

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

"""Power flow functionality.
"""


# make the code as Python 3 compatible as possible
from __future__ import print_function, division
from __future__ import absolute_import
from six.moves import range

__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"



from scipy.sparse import issparse, csr_matrix, csc_matrix, hstack as shstack, vstack as svstack

from numpy import r_, ones, zeros, newaxis
from scipy.sparse.linalg import spsolve

import numpy as np
import pandas as pd

from itertools import chain


from numpy.linalg import norm

import time

def _as_snapshots(network, snapshots, now):
    if snapshots is None:
        if now is None:
            now = network.now
        snapshots = [now]
    try:
        return pd.Index(snapshots)
    except TypeError:
        return pd.Index([snapshots])

def network_pf(network,now=None,verbose=True,skip_pre=False,x_tol=1e-6):
    """
    Full non-linear power flow for generic network.

    Parameters
    ----------
    now : object
        A member of network.snapshots on which to run the power flow, defaults to network.now
    verbose: bool, default True
    skip_pre: bool, default False
        Skip the preliminary steps of computing topology, calculating dependent values and finding bus controls.
    x_tol: float
        Tolerance for Newton-Raphson power flow.

    Returns
    -------
    None
    """




    if not skip_pre:
        network.build_graph()
        network.determine_network_topology()
        calculate_dependent_values(network)

    if now is None:
        now=network.now


    #deal with transport links and converters
    network.converters_t.p0.loc[now] = network.converters_t.p_set.loc[now]
    network.converters_t.p1.loc[now] = -network.converters_t.p_set.loc[now]
    network.transport_links_t.p0.loc[now] = network.transport_links_t.p_set.loc[now]
    network.transport_links_t.p1.loc[now] = -network.transport_links_t.p_set.loc[now]


    for sub_network in network.sub_networks.obj:

        if sub_network.current_type == "DC":
            raise NotImplementedError("Non-linear power flow for DC networks not supported yet.")
            continue

        if verbose:
            print("Performing full non-linear load-flow on %s sub-network %s" % (sub_network.current_type,sub_network))


        if not skip_pre:
            find_bus_controls(sub_network,verbose=verbose)

            branches = sub_network.branches()

            if len(branches) > 0:
                calculate_Y(sub_network,verbose=verbose,skip_pre=True)

        sub_network.pf(now,verbose=verbose,skip_pre=True)




def newton_raphson_sparse(f,guess,dfdx,x_tol=1e-10,lim_iter=100,verbose=True):
    """Solve f(x) = 0 with initial guess for x and dfdx(x). dfdx(x) should
    return a sparse Jacobian.  Terminate if error on norm of f(x) is <
    x_tol or there were more than lim_iter iterations.

    """

    n_iter = 0
    F = f(guess)
    diff = norm(F,np.Inf)

    if verbose:
        print("Error at iteration %d: %f" % (n_iter,diff))

    while diff > x_tol and n_iter < lim_iter:

        n_iter +=1

        guess = guess - spsolve(dfdx(guess),F)

        F = f(guess)
        diff = norm(F,np.Inf)

        if verbose:
            print("Error at iteration %d: %f" % (n_iter,diff))

    if verbose and diff > x_tol:
        print("Warning, looks like we didn't reach the required tolerance within %d iterations" % (n_iter,))

    return guess,n_iter,diff



def sub_network_pf(sub_network,now=None,verbose=True,skip_pre=False,x_tol=1e-6):
    """
    Non-linear power flow for connected sub-network.

    Parameters
    ----------
    now : object
        A member of network.snapshots on which to run the power flow, defaults to network.now
    verbose: bool, default True
    skip_pre: bool, default False
        Skip the preliminary steps of computing topology, calculating dependent values and finding bus controls.
    x_tol: float
        Tolerance for Newton-Raphson power flow.

    Returns
    -------
    None
    """

    from .components import passive_branch_types

    network = sub_network.network

    if now is None:
        now = network.now

    if verbose:
        print("Performing full non-linear load-flow for snapshot %s" % (now))

    if not skip_pre:
        calculate_dependent_values(network)
        find_bus_controls(sub_network,verbose=verbose)

    branches = sub_network.branches()
    buses = sub_network.buses_o

    if not skip_pre and len(branches) > 0:
        calculate_Y(sub_network,verbose=verbose,skip_pre=True)




    #set the power injection at each node
    network.buses_t.p.loc[now,buses.index] =  pd.DataFrame({list_name :
                          (getattr(network,list_name+"_t").p_set.loc[now]*getattr(network,list_name).sign)
                          .groupby(getattr(network,list_name).bus).sum()
    for list_name in ["generators","loads","storage_units"]})\
              .sum(axis=1).reindex(buses.index,fill_value = 0.)


    network.buses_t.q.loc[now,buses.index] =  pd.DataFrame({list_name:
                          (getattr(network,list_name+"_t").q_set.loc[now]*getattr(network,list_name).sign)
                          .groupby(getattr(network,list_name).bus).sum()
    for list_name in ["generators","loads","storage_units"]})\
              .sum(axis=1).reindex(buses.index,fill_value = 0.)

    network.buses_t.p.loc[now,buses.index] -=  pd.DataFrame({list_name+str(i) :
                          getattr(getattr(network,list_name+"_t"),"p"+str(i)).loc[now]
                          .groupby(getattr(getattr(network,list_name),"bus"+str(i))).sum()
                                    for list_name in ["transport_links","converters"] for i in [0,1]})\
              .sum(axis=1).reindex(buses.index,fill_value = 0.)


    p = network.buses_t.p.loc[now,buses.index]
    q = network.buses_t.q.loc[now,buses.index]

    s = p + 1j*q



    def f(guess):
        network.buses_t.v_ang.loc[now,sub_network.pvpqs.index] = guess[:len(sub_network.pvpqs)]

        network.buses_t.v_mag_pu.loc[now,sub_network.pqs.index] = guess[len(sub_network.pvpqs):]

        v_mag_pu = network.buses_t.v_mag_pu.loc[now,buses.index]
        v_ang = network.buses_t.v_ang.loc[now,buses.index]
        V = v_mag_pu*np.exp(1j*v_ang)

        mismatch = V*np.conj(sub_network.Y*V) - s

        F = r_[mismatch.real[1:],mismatch.imag[1+len(sub_network.pvs):]]

        return F


    def dfdx(guess):

        network.buses_t.v_ang.loc[now,sub_network.pvpqs.index] = guess[:len(sub_network.pvpqs)]

        network.buses_t.v_mag_pu.loc[now,sub_network.pqs.index] = guess[len(sub_network.pvpqs):]

        v_mag_pu = network.buses_t.v_mag_pu.loc[now,buses.index]
        v_ang = network.buses_t.v_ang.loc[now,buses.index]

        V = v_mag_pu*np.exp(1j*v_ang)

        index = r_[:len(buses)]

        #make sparse diagonal matrices
        V_diag = csr_matrix((V,(index,index)))
        V_norm_diag = csr_matrix((V/abs(V),(index,index)))
        I_diag = csr_matrix((sub_network.Y*V,(index,index)))

        dS_dVa = 1j*V_diag*np.conj(I_diag - sub_network.Y*V_diag)

        dS_dVm = V_norm_diag*np.conj(I_diag) + V_diag * np.conj(sub_network.Y*V_norm_diag)

        J00 = dS_dVa[1:,1:].real
        J01 = dS_dVm[1:,1+len(sub_network.pvs):].real
        J10 = dS_dVa[1+len(sub_network.pvs):,1:].imag
        J11 = dS_dVm[1+len(sub_network.pvs):,1+len(sub_network.pvs):].imag

        J = svstack([
            shstack([J00, J01]),
            shstack([J10, J11])
        ], format="csr")

        return J


    #Set what we know: slack V and v_mag_pu for PV buses
    network.buses_t.v_mag_pu.loc[now,sub_network.pvs.index] = network.buses_t.v_mag_pu_set.loc[now,sub_network.pvs.index]

    network.buses_t.v_mag_pu.loc[now,sub_network.slack_bus] = network.buses_t.v_mag_pu_set.loc[now,sub_network.slack_bus]

    network.buses_t.v_ang.loc[now,sub_network.slack_bus] = 0.

    #Make a guess for what we don't know: V_ang for PV and PQs and v_mag_pu for PQ buses
    guess = r_[zeros(len(sub_network.pvpqs)),ones(len(sub_network.pqs))]

    #Now try and solve
    start = time.time()
    roots,n_iter,diff = newton_raphson_sparse(f,guess,dfdx,x_tol=x_tol,verbose=verbose)
    if verbose:
        print("Newton-Raphson solved in %d iterations with error of %f in %f seconds" % (n_iter,diff,time.time()-start))


    #now set everything

    network.buses_t.v_ang.loc[now,sub_network.pvpqs.index] = roots[:len(sub_network.pvpqs)]
    network.buses_t.v_mag_pu.loc[now,sub_network.pqs.index] = roots[len(sub_network.pvpqs):]

    v_mag_pu = network.buses_t.v_mag_pu.loc[now,buses.index]
    v_ang = network.buses_t.v_ang.loc[now,buses.index]

    V = v_mag_pu*np.exp(1j*v_ang)

    #add voltages to branches
    branches = pd.merge(branches,pd.DataFrame({"v0" :V}),how="left",left_on="bus0",right_index=True)
    branches = pd.merge(branches,pd.DataFrame({"v1" :V}),how="left",left_on="bus1",right_index=True)

    i0 = sub_network.Y0*V
    i1 = sub_network.Y1*V

    branches["s0"] = branches["v0"]*np.conj(i0)
    branches["s1"] = branches["v1"]*np.conj(i1)

    for t in network.iterate_components(passive_branch_types):
        df = branches.loc[t.name]
        t.pnl.loc["p0",now,df.index] = df["s0"].real
        t.pnl.loc["q0",now,df.index] = df["s0"].imag
        t.pnl.loc["p1",now,df.index] = df["s1"].real
        t.pnl.loc["q1",now,df.index] = df["s1"].imag


    s_calc = V*np.conj(sub_network.Y*V)

    network.buses_t.p.loc[now,sub_network.slack_bus] = s_calc[sub_network.slack_bus].real
    network.buses_t.q.loc[now,sub_network.slack_bus] = s_calc[sub_network.slack_bus].imag
    network.buses_t.q.loc[now,sub_network.pvs.index] = s_calc[sub_network.pvs.index].imag

    #allow all loads to dispatch as set
    loads = sub_network.loads()
    network.loads_t.p.loc[now,loads.index] = network.loads_t.p_set.loc[now,loads.index]
    network.loads_t.q.loc[now,loads.index] = network.loads_t.q_set.loc[now,loads.index]

    #set shunt impedance powers
    shunt_impedances = sub_network.shunt_impedances()
    #add voltages
    shunt_impedances = pd.merge(shunt_impedances,pd.DataFrame({"v_mag_pu" :v_mag_pu}),how="left",left_on="bus",right_index=True)
    network.shunt_impedances_t.p.loc[now,shunt_impedances.index] = (shunt_impedances.v_mag_pu**2)*shunt_impedances.g_pu
    network.shunt_impedances_t.q.loc[now,shunt_impedances.index] = (shunt_impedances.v_mag_pu**2)*shunt_impedances.b_pu

    #allow all generators to dispatch as set
    generators = sub_network.generators()
    network.generators_t.p.loc[now,generators.index] = network.generators_t.p_set.loc[now,generators.index]
    network.generators_t.q.loc[now,generators.index] = network.generators_t.q_set.loc[now,generators.index]

    #let slack generator take up the slack
    network.generators_t.p.loc[now,sub_network.slack_generator] += network.buses_t.p.loc[now,sub_network.slack_bus] - s[sub_network.slack_bus].real
    network.generators_t.q.loc[now,sub_network.slack_generator] += network.buses_t.q.loc[now,sub_network.slack_bus] - s[sub_network.slack_bus].imag

    #set the Q of the PV generators
    network.generators_t.q.loc[now,sub_network.pvs.generator] += network.buses_t.q.loc[now,sub_network.pvs.index] - s[sub_network.pvs.index].imag


def network_lpf(network, snapshots=None, verbose=True, skip_pre=False, now=None):
    """
    Linear power flow for generic network.

    Parameters
    ----------
    snapshots : list-like|single snapshot
        A subset or an elements of network.snapshots on which to run
        the power flow, defaults to [now]
    verbose: bool, default True
    skip_pre: bool, default False
        Skip the preliminary steps of computing topology, calculating
        dependent values and finding bus controls.
    now : object
        Deprecated: A member of network.snapshots on which to run the
        power flow, defaults to network.now

    Returns
    -------
    None
    """

    if not skip_pre:
        network.build_graph()
        network.determine_network_topology()
        calculate_dependent_values(network)

    snapshots = _as_snapshots(network, snapshots, now=now)

    #deal with transport links and converters
    network.converters_t.p0.loc[snapshots] = network.converters_t.p_set.loc[snapshots]
    network.converters_t.p1.loc[snapshots] = -network.converters_t.p_set.loc[snapshots]
    network.transport_links_t.p0.loc[snapshots] = network.transport_links_t.p_set.loc[snapshots]
    network.transport_links_t.p1.loc[snapshots] = -network.transport_links_t.p_set.loc[snapshots]

    for sub_network in network.sub_networks.obj:
        if not skip_pre:
            find_bus_controls(sub_network, verbose=verbose)

            branches = sub_network.branches()
            if len(branches) > 0:
                calculate_B_H(sub_network, verbose=verbose, skip_pre=True)
        sub_network.lpf(snapshots=snapshots, verbose=verbose, skip_pre=True)


def find_slack_bus(sub_network,verbose=True):
    """Find the slack bus in a connected sub-network."""

    gens = sub_network.generators()

    if len(gens) == 0:
        if verbose:
            print("No generators in %s, better hope power is already balanced" % sub_network)
        sub_network.slack_generator = None
        sub_network.slack_bus = sub_network.buses().index[0]

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

    gens = sub_network.generators()
    buses = sub_network.buses()

    #default bus control is PQ
    network.buses.loc[buses.index,"control"] = "PQ"

    #find all buses with one or more gens with PV
    gen_pvs = pd.DataFrame(data=0.,index=gens.index,columns=["pvs"])
    gen_pvs.loc[gens.control=="PV","pvs"] = 1.
    gen_pvs["name"] = gens.index

    bus_pvs = gen_pvs["pvs"].groupby(gens.bus).sum().reindex(buses.index)
    bus_pv_names = gen_pvs["name"].groupby(gens.bus).first().reindex(buses.index)

    pvs = bus_pv_names[bus_pvs>0]

    network.buses.loc[pvs.index,"control"] = "PV"
    network.buses.loc[pvs.index,"generator"] = pvs

    network.buses.loc[sub_network.slack_bus,"control"] = "Slack"
    network.buses.loc[sub_network.slack_bus,"generator"] = sub_network.slack_generator

    buses = sub_network.buses()

    sub_network.pvs = buses[buses.control == "PV"]
    sub_network.pqs = buses[buses.control == "PQ"]

    sub_network.pvpqs = pd.concat((sub_network.pvs,sub_network.pqs))

    #order buses
    sub_network.buses_o = pd.concat((buses.loc[[sub_network.slack_bus]],sub_network.pvpqs))
    sub_network.buses_o["i"] = list(range(len(sub_network.buses_o)))



def calculate_dependent_values(network):
    """Calculate per unit impedances and append voltages to lines and shunt impedances."""

    network.lines["v_nom"] = network.lines.bus0.map(network.buses.v_nom)

    network.lines["x_pu"] = network.lines.x/(network.lines.v_nom**2)
    network.lines["r_pu"] = network.lines.r/(network.lines.v_nom**2)
    network.lines["b_pu"] = network.lines.b*network.lines.v_nom**2
    network.lines["g_pu"] = network.lines.g*network.lines.v_nom**2

    #convert transformer impedances from base power s_nom to base = 1 MVA
    network.transformers["x_pu"] = network.transformers.x/network.transformers.s_nom
    network.transformers["r_pu"] = network.transformers.r/network.transformers.s_nom
    network.transformers["b_pu"] = network.transformers.b*network.transformers.s_nom
    network.transformers["g_pu"] = network.transformers.g*network.transformers.s_nom

    network.shunt_impedances["v_nom"] = network.shunt_impedances["bus"].map(network.buses.v_nom)
    network.shunt_impedances["b_pu"] = network.shunt_impedances.b*network.shunt_impedances.v_nom**2
    network.shunt_impedances["g_pu"] = network.shunt_impedances.g*network.shunt_impedances.v_nom**2


def calculate_B_H(sub_network,verbose=True,skip_pre=False):
    """Calculate B and H matrices for AC or DC sub-networks."""

    if not skip_pre:
        calculate_dependent_values(sub_network.network)

    if sub_network.current_type == "DC":
        attribute="r_pu"
    elif sub_network.current_type == "AC":
        attribute="x_pu"

    branches = sub_network.branches()
    buses = sub_network.buses_o

    #following leans heavily on pypower.makeBdc

    num_branches = len(branches)
    num_buses = len(buses)

    index = r_[:num_branches,:num_branches]

    #susceptances
    b = 1/branches[attribute]

    from_bus = branches.bus0.map(buses["i"]).values
    to_bus = branches.bus1.map(buses["i"]).values

    #build weighted Laplacian
    sub_network.H = csr_matrix((r_[b,-b],(index,r_[from_bus,to_bus])))

    incidence = csr_matrix((r_[ones(num_branches),-ones(num_branches)],(index,r_[from_bus,to_bus])),(num_branches,num_buses))

    sub_network.B = incidence.T * sub_network.H



def calculate_PTDF(sub_network,verbose=True,skip_pre=False):
    """Calculate the PTDF for sub_network based on the already calculated sub_network.B and sub_network.H."""

    if not skip_pre:
        calculate_dependent_values(sub_network.network)


    #calculate inverse of B with slack removed

    n_pvpq = sub_network.pvpqs.shape[0]
    index = np.r_[:n_pvpq]

    I = csc_matrix((np.ones((n_pvpq)),(index,index)))

    B_inverse = spsolve(sub_network.B[1:, 1:],I)

    #exception for two-node networks, where B_inverse is a 1d array
    if issparse(B_inverse):
        B_inverse = B_inverse.toarray()
    elif B_inverse.shape == (1,):
        B_inverse = B_inverse.reshape((1,1))

    #add back in zeroes for slack
    B_inverse = np.hstack((np.zeros((n_pvpq,1)),B_inverse))
    B_inverse = np.vstack((np.zeros(n_pvpq+1),B_inverse))

    sub_network.PTDF = sub_network.H*B_inverse


def calculate_Y(sub_network,verbose=True,skip_pre=False):
    """Calculate bus admittance matrices for AC sub-networks."""

    if not skip_pre:
        calculate_dependent_values(sub_network.network)

    if sub_network.current_type == "DC":
        print("DC networks not supported for Y!")
        return

    branches = sub_network.branches()
    buses = sub_network.buses_o

    network = sub_network.network

    #following leans heavily on pypower.makeYbus
    #Copyright Richard Lincoln, Ray Zimmerman, BSD-style licence

    num_branches = len(branches)
    num_buses = len(buses)

    y_se = 1/(branches["r_pu"] + 1.j*branches["x_pu"])

    y_sh = branches["g_pu"]+ 1.j*branches["b_pu"]

    tau = branches["tap_ratio"].fillna(1.)

    #catch some transformers falsely set with tau = 0 by pypower
    tau[tau==0] = 1.

    phase_shift = np.exp(1.j*branches["phase_shift"].fillna(0.)*np.pi/180.)

    #build the admittance matrix elements for each branch
    Y11 = y_se + 0.5*y_sh
    Y01 = -y_se/tau/phase_shift
    Y10 = -y_se/tau/np.conj(phase_shift)
    Y00 = Y11/tau**2

    #bus shunt impedances
    b_sh = network.shunt_impedances.b_pu.groupby(network.shunt_impedances.bus).sum().reindex(buses.index,fill_value = 0.)
    g_sh = network.shunt_impedances.g_pu.groupby(network.shunt_impedances.bus).sum().reindex(buses.index,fill_value = 0.)
    Y_sh = g_sh + 1.j*b_sh

    #get bus indices
    join = pd.merge(branches,buses,how="left",left_on="bus0",right_index=True,suffixes=("","_0"))
    join = pd.merge(join,buses,how="left",left_on="bus1",right_index=True,suffixes=("","_1"))
    bus0 = join.i
    bus1 = join.i_1

    #connection matrices
    C0 = csr_matrix((ones(num_branches), (np.arange(num_branches), bus0)), (num_branches, num_buses))
    C1 = csr_matrix((ones(num_branches), (np.arange(num_branches), bus1)), (num_branches, num_buses))

    #build Y{0,1} such that Y{0,1} * V is the vector complex branch currents

    i = r_[np.arange(num_branches), np.arange(num_branches)]
    sub_network.Y0 = csr_matrix((r_[Y00,Y01],(i,r_[bus0,bus1])), (num_branches,num_buses))
    sub_network.Y1 = csr_matrix((r_[Y10,Y11],(i,r_[bus0,bus1])), (num_branches,num_buses))

    #now build bus admittance matrix
    sub_network.Y = C0.T * sub_network.Y0 + C1.T * sub_network.Y1 + \
       csr_matrix((Y_sh, (np.arange(num_buses), np.arange(num_buses))))



def aggregate_multi_graph(sub_network,verbose=True):
    """Aggregate branches between same buses and replace with a single
branch with aggregated properties (e.g. s_nom is summed, length is
averaged).

    """

    network = sub_network.network

    count = 0
    seen = []
    for u,v in sub_network.graph.edges():
        if (u,v) in seen:
            continue
        line_objs = sub_network.graph.edge[u][v].keys()
        if len(line_objs) > 1:
            lines = network.lines.loc[[l.name for l in line_objs]]
            aggregated = {}

            attr_inv = ["x","r"]
            attr_sum = ["s_nom","b","g","s_nom_max","s_nom_min"]
            attr_mean = ["capital_cost","length","terrain_factor"]

            for attr in attr_inv:
                aggregated[attr] = 1/(1/lines[attr]).sum()

            for attr in attr_sum:
                aggregated[attr] = lines[attr].sum()

            for attr in attr_mean:
                aggregated[attr] = lines[attr].mean()

            count += len(line_objs) - 1

            #remove all but first line
            for line in line_objs[1:]:
                network.remove("Line",line.name)

            rep = line_objs[0]

            for key,value in aggregated.items():
                setattr(rep,key,value)

            seen.append((u,v))

    if verbose:
        print("Removed %d excess lines from sub-network %s and replaced with aggregated lines" % (count,sub_network.name))



def sub_network_lpf(sub_network, snapshots=None, verbose=True, skip_pre=False, now=None):
    """
    Linear power flow for connected sub-network.

    Parameters
    ----------
    snapshots : list-like|single snapshot
        A subset or an elements of network.snapshots on which to run
        the power flow, defaults to [now]
    verbose: bool, default True
    skip_pre: bool, default False
        Skip the preliminary steps of computing topology, calculating
        dependent values and finding bus controls.
    now : object
        Deprecated: A member of network.snapshots on which to run the
        power flow, defaults to network.now

    Returns
    -------
    None
    """

    from .components import \
        one_port_types, controllable_one_port_types, \
        passive_branch_types, controllable_branch_types

    network = sub_network.network

    snapshots = _as_snapshots(network, snapshots, now=now)

    if verbose:
        print("Performing linear load-flow on {} sub-network {} for snapshot(s) {}"
              .format(sub_network.current_type,sub_network,snapshots))

    if not skip_pre:
        calculate_dependent_values(network)
        find_bus_controls(sub_network,verbose=verbose)

    # get indices for the components on this subnetwork
    buses_i = sub_network.buses_o.index
    branches_i = sub_network.branches().index

    # allow all shunt impedances to dispatch as set
    shunt_impedances_i = sub_network.shunt_impedances_i()
    network.shunt_impedances_t.p.loc[snapshots, shunt_impedances_i] = \
        network.shunt_impedances.g_pu.loc[shunt_impedances_i].values

    # allow all one ports to dispatch as set
    for t in sub_network.iterate_components(controllable_one_port_types):
        t.pnl.p.loc[snapshots, t.ind] = t.pnl.p_set.loc[snapshots, t.ind]

    # set the power injection at each node
    network.buses_t.p.loc[snapshots, buses_i] = \
        sum([((t.pnl.p.loc[snapshots, t.ind] * t.df.loc[t.ind, 'sign'])
              .groupby(t.df.loc[t.ind, 'bus'], axis=1).sum()
              .reindex(columns=buses_i, fill_value=0.))
             for t in sub_network.iterate_components(one_port_types)]
            +
            [(- t.pnl.loc["p"+str(i), snapshots].groupby(t.df["bus"+str(i)], axis=1).sum()
              .reindex(columns=buses_i, fill_value=0))
             for t in network.iterate_components(controllable_branch_types)
             for i in [0,1]])

    if not skip_pre and len(branches_i) > 0:
        calculate_B_H(sub_network, verbose=verbose, skip_pre=True)

    v_diff = np.zeros((len(snapshots), len(buses_i)))
    if len(branches_i) > 0:
        p = network.buses_t.loc['p', snapshots, buses_i].values
        v_diff[:,1:] = spsolve(sub_network.B[1:, 1:], p[:,1:].T).T
        flows = pd.DataFrame(v_diff * sub_network.H.T,
                             columns=branches_i, index=snapshots)

        for t in network.iterate_components(passive_branch_types):
            f = flows.loc[:, t.name]
            t.pnl.p0.loc[snapshots, f.columns] = f
            t.pnl.p1.loc[snapshots, f.columns] = -f

    if sub_network.current_type == "AC":
        network.buses_t.v_ang.loc[snapshots, buses_i] = v_diff
        network.buses_t.v_mag_pu.loc[snapshots, buses_i] = 1.
    elif sub_network.current_type == "DC":
        network.buses_t.v_mag_pu.loc[snapshots, buses_i] = 1 + v_diff
        network.buses_t.v_ang.loc[snapshots, buses_i] = 0.

    # set slack bus power to pick up remained
    slack_adjustment = (- network.buses_t.loc['p', snapshots, buses_i[1:]].sum(axis=1)
                        - network.buses_t.loc['p', snapshots, buses_i[0]])
    network.buses_t.loc["p", snapshots, buses_i[0]] += slack_adjustment

    # let slack generator take up the slack
    if sub_network.slack_generator is not None:
        network.generators_t.p.loc[snapshots, sub_network.slack_generator] += slack_adjustment



def network_batch_lpf(network,snapshots=None):
    """Batched linear power flow with numpy.dot for several snapshots."""

    raise NotImplementedError("Batch linear power flow not supported yet.")
