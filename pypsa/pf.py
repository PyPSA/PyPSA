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



from scipy.sparse import issparse, csr_matrix, csc_matrix, hstack as shstack, vstack as svstack, dok_matrix

from numpy import r_, ones, zeros, newaxis
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm

import numpy as np
import pandas as pd
import scipy as sp, scipy.sparse
import networkx as nx

import collections, six
from itertools import chain
import time

def _as_snapshots(network, snapshots):
    if snapshots is None:
        snapshots = [network.now]
    if (isinstance(snapshots, six.string_types) or
        not isinstance(snapshots, (collections.Sequence, pd.Index))):
        return pd.Index([snapshots])
    else:
        return pd.Index(snapshots)

def _incidence_matrix(sub_network, busorder=None):
    from .components import passive_branch_types

    if busorder is None:
        busorder = sub_network.buses_i()

    num_buses = len(busorder)
    num_branches = 0
    bus0_indices = []
    bus1_indices = []
    for t in sub_network.iterate_components(passive_branch_types):
        num_branches += len(t.ind)
        bus0_indices.append(busorder.get_indexer(t.df.loc[t.ind, 'bus0']))
        bus1_indices.append(busorder.get_indexer(t.df.loc[t.ind, 'bus1']))
    bus0_indices = np.concatenate(bus0_indices)
    bus1_indices = np.concatenate(bus1_indices)
    K = sp.sparse.csr_matrix((np.r_[np.ones(num_branches), -np.ones(num_branches)],
                              (np.r_[bus0_indices, bus1_indices], np.r_[:num_branches, :num_branches])),
                             (num_buses, num_branches))
    return K

def _network_prepare_and_run_pf(network, snapshots, verbose, skip_pre, sub_network_pf_fun, sub_network_prepare_fun, **kwargs):

    if not skip_pre:
        network.determine_network_topology()
        calculate_dependent_values(network)

    snapshots = _as_snapshots(network, snapshots)

    #deal with transport links and converters
    if not network.converters.empty:
        network.converters_t.p0.loc[snapshots] = network.converters_t.p_set.loc[snapshots]
        network.converters_t.p1.loc[snapshots] = -network.converters_t.p_set.loc[snapshots]
    if not network.transport_links.empty:
        network.transport_links_t.p0.loc[snapshots] = network.transport_links_t.p_set.loc[snapshots]
        network.transport_links_t.p1.loc[snapshots] = -network.transport_links_t.p_set.loc[snapshots]

    for sub_network in network.sub_networks.obj:
        if not skip_pre:
            find_bus_controls(sub_network, verbose=verbose)

            branches_i = sub_network.branches_i()
            if len(branches_i) > 0:
                sub_network_prepare_fun(sub_network, verbose=verbose, skip_pre=True)
        sub_network_pf_fun(sub_network, snapshots=snapshots, verbose=verbose, skip_pre=True, **kwargs)

def network_pf(network, snapshots=None, verbose=True, skip_pre=False, x_tol=1e-6):
    """
    Full non-linear power flow for generic network.

    Parameters
    ----------
    snapshots : list-like|single snapshot
        A subset or an elements of network.snapshots on which to run
        the power flow, defaults to [now]
    verbose: bool, default True
    skip_pre: bool, default False
        Skip the preliminary steps of computing topology, calculating dependent values and finding bus controls.
    x_tol: float
        Tolerance for Newton-Raphson power flow.

    Returns
    -------
    None
    """

    _network_prepare_and_run_pf(network, snapshots, verbose, skip_pre, sub_network_pf, calculate_Y, x_tol=x_tol)


def newton_raphson_sparse(f,guess,dfdx,x_tol=1e-10,lim_iter=100,verbose=True):
    """Solve f(x) = 0 with initial guess for x and dfdx(x). dfdx(x) should
    return a sparse Jacobian.  Terminate if error on norm of f(x) is <
    x_tol or there were more than lim_iter iterations.

    """

    n_iter = 0
    F = f(guess)
    diff = norm(F,np.Inf)

    if verbose:
        print("Error at iteration %d: %f" % (n_iter, diff))

    while diff > x_tol and n_iter < lim_iter:

        n_iter +=1

        guess = guess - spsolve(dfdx(guess),F)

        F = f(guess)
        diff = norm(F,np.Inf)

        if verbose:
            print("Error at iteration %d: %f" % (n_iter, diff))

    if diff > x_tol:
        print("Warning, we didn't reach the required tolerance within %d iterations, error is at %f" % (n_iter, diff))

    return guess,n_iter,diff



def sub_network_pf(sub_network, snapshots=None, verbose=True, skip_pre=False, x_tol=1e-6):
    """
    Non-linear power flow for connected sub-network.

    Parameters
    ----------
    snapshots : list-like|single snapshot
        A subset or an elements of network.snapshots on which to run
        the power flow, defaults to [now]
    verbose: bool, default True
    skip_pre: bool, default False
        Skip the preliminary steps of computing topology, calculating dependent values and finding bus controls.
    x_tol: float
        Tolerance for Newton-Raphson power flow.

    Returns
    -------
    None
    """

    snapshots = _as_snapshots(sub_network.network, snapshots)
    if verbose:
        print("Performing non-linear load-flow on {} sub-network {} for snapshots {}"
              .format(sub_network.current_type, sub_network, snapshots))

    # _sub_network_prepare_pf(sub_network, snapshots, verbose, skip_pre, calculate_Y)
    network = sub_network.network

    from .components import passive_branch_types, controllable_branch_types, controllable_one_port_types

    if not skip_pre:
        calculate_dependent_values(network)
        find_bus_controls(sub_network,verbose=verbose)

    # get indices for the components on this subnetwork
    branches_i = sub_network.branches_i()
    buses_o = sub_network.buses_o

    if not skip_pre and len(branches_i) > 0:
        calculate_Y(sub_network,verbose=verbose,skip_pre=True)

    for n in ("q", "p"):
        # allow all one ports to dispatch as set
        for t in sub_network.iterate_components(controllable_one_port_types):
            t.pnl[n].loc[snapshots, t.ind] = t.pnl[n+"_set"].loc[snapshots, t.ind]

        # set the power injection at each node from controllable components
        network.buses_t[n].loc[snapshots, buses_o] = \
            sum([((t.pnl[n].loc[snapshots, t.ind] * t.df.loc[t.ind, 'sign'])
                  .groupby(t.df.loc[t.ind, 'bus'], axis=1).sum()
                  .reindex(columns=buses_o, fill_value=0.))
                 for t in sub_network.iterate_components(controllable_one_port_types)]
                +
                [(- t.pnl.loc[n+str(i), snapshots].groupby(t.df["bus"+str(i)], axis=1).sum()
                  .reindex(columns=buses_o, fill_value=0))
                 for t in network.iterate_components(controllable_branch_types)
                 for i in [0,1]])

    def f(guess):
        network.buses_t.v_ang.loc[now,sub_network.pvpqs] = guess[:len(sub_network.pvpqs)]

        network.buses_t.v_mag_pu.loc[now,sub_network.pqs] = guess[len(sub_network.pvpqs):]

        v_mag_pu = network.buses_t.v_mag_pu.loc[now,buses_o]
        v_ang = network.buses_t.v_ang.loc[now,buses_o]
        V = v_mag_pu*np.exp(1j*v_ang)

        mismatch = V*np.conj(sub_network.Y*V) - s

        F = r_[mismatch.real[1:],mismatch.imag[1+len(sub_network.pvs):]]

        return F


    def dfdx(guess):

        network.buses_t.v_ang.loc[now,sub_network.pvpqs] = guess[:len(sub_network.pvpqs)]

        network.buses_t.v_mag_pu.loc[now,sub_network.pqs] = guess[len(sub_network.pvpqs):]

        v_mag_pu = network.buses_t.v_mag_pu.loc[now,buses_o]
        v_ang = network.buses_t.v_ang.loc[now,buses_o]

        V = v_mag_pu*np.exp(1j*v_ang)

        index = r_[:len(buses_o)]

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
    network.buses_t.v_mag_pu.loc[snapshots,sub_network.pvs] = network.buses_t.v_mag_pu_set.loc[snapshots,sub_network.pvs]
    network.buses_t.v_mag_pu.loc[snapshots,sub_network.slack_bus] = network.buses_t.v_mag_pu_set.loc[snapshots,sub_network.slack_bus]
    network.buses_t.v_ang.loc[snapshots,sub_network.slack_bus] = 0.

    ss = np.empty((len(snapshots), len(buses_o)), dtype=np.complex)
    roots = np.empty((len(snapshots), len(sub_network.pvpqs) + len(sub_network.pqs)))
    for i, now in enumerate(snapshots):
        p = network.buses_t.p.loc[now,buses_o]
        q = network.buses_t.q.loc[now,buses_o]
        ss[i] = s = p + 1j*q

        #Make a guess for what we don't know: V_ang for PV and PQs and v_mag_pu for PQ buses
        guess = r_[zeros(len(sub_network.pvpqs)),ones(len(sub_network.pqs))]

        #Now try and solve
        start = time.time()
        roots[i], n_iter, diff = newton_raphson_sparse(f,guess,dfdx,x_tol=x_tol,verbose=verbose)
        if verbose:
            print("Newton-Raphson solved in %d iterations with error of %f in %f seconds" % (n_iter,diff,time.time()-start))

    #now set everything
    network.buses_t.v_ang.loc[snapshots,sub_network.pvpqs] = roots[:,:len(sub_network.pvpqs)]
    network.buses_t.v_mag_pu.loc[snapshots,sub_network.pqs] = roots[:,len(sub_network.pvpqs):]

    v_mag_pu = network.buses_t.v_mag_pu.loc[snapshots,buses_o].values
    v_ang = network.buses_t.v_ang.loc[snapshots,buses_o].values

    V = v_mag_pu*np.exp(1j*v_ang)

    #add voltages to branches
    buses_indexer = buses_o.get_indexer
    branch_bus0 = []; branch_bus1 = []
    for t in sub_network.iterate_components(passive_branch_types):
        branch_bus0 += list(t.df.loc[t.ind, 'bus0'])
        branch_bus1 += list(t.df.loc[t.ind, 'bus1'])
    v0 = V[:,buses_indexer(branch_bus0)]
    v1 = V[:,buses_indexer(branch_bus1)]

    i0 = np.empty((len(snapshots), sub_network.Y0.shape[0]), dtype=np.complex)
    i1 = np.empty((len(snapshots), sub_network.Y1.shape[0]), dtype=np.complex)
    for i, now in enumerate(snapshots):
        i0[i] = sub_network.Y0*V[i]
        i1[i] = sub_network.Y1*V[i]

    s0 = pd.DataFrame(v0*np.conj(i0), columns=branches_i, index=snapshots)
    s1 = pd.DataFrame(v1*np.conj(i1), columns=branches_i, index=snapshots)
    for t in network.iterate_components(passive_branch_types):
        s0t = s0.loc[:,t.name]
        s1t = s1.loc[:,t.name]
        t.pnl.p0.loc[snapshots,s0t.columns] = s0t.values.real
        t.pnl.q0.loc[snapshots,s0t.columns] = s0t.values.imag
        t.pnl.p1.loc[snapshots,s1t.columns] = s1t.values.real
        t.pnl.q1.loc[snapshots,s1t.columns] = s1t.values.imag

    s_calc = np.empty((len(snapshots), len(buses_o)), dtype=np.complex)
    for i in np.arange(len(snapshots)):
        s_calc[i] = V[i]*np.conj(sub_network.Y*V[i])
    slack_index = buses_o.get_loc(sub_network.slack_bus)
    network.buses_t.p.loc[snapshots,sub_network.slack_bus] = s_calc[:,slack_index].real
    network.buses_t.q.loc[snapshots,sub_network.slack_bus] = s_calc[:,slack_index].imag
    network.buses_t.q.loc[snapshots,sub_network.pvs] = s_calc[:,buses_indexer(sub_network.pvs)].imag

    #set shunt impedance powers
    shunt_impedances_i = sub_network.shunt_impedances_i()
    if len(shunt_impedances_i):
        #add voltages
        shunt_impedances_v_mag_pu = v_mag_pu[:,buses_indexer(network.shunt_impedances.loc[shunt_impedances_i, 'bus'])]
        network.shunt_impedances_t.p.loc[snapshots,shunt_impedances_i] = (shunt_impedances_v_mag_pu**2)*network.shunt_impedances.loc[shunt_impedances_i, 'g_pu'].values
        network.shunt_impedances_t.q.loc[snapshots,shunt_impedances_i] = (shunt_impedances_v_mag_pu**2)*network.shunt_impedances.loc[shunt_impedances_i, 'b_pu'].values

    #let slack generator take up the slack
    network.generators_t.p.loc[snapshots,sub_network.slack_generator] += network.buses_t.p.loc[snapshots,sub_network.slack_bus] - ss[:,slack_index].real
    network.generators_t.q.loc[snapshots,sub_network.slack_generator] += network.buses_t.q.loc[snapshots,sub_network.slack_bus] - ss[:,slack_index].imag

    #set the Q of the PV generators
    network.generators_t.q.loc[snapshots,network.buses.loc[sub_network.pvs, "generator"]] += np.asarray(network.buses_t.q.loc[snapshots,sub_network.pvs] - ss[:,buses_indexer(sub_network.pvs)].imag)

def network_lpf(network, snapshots=None, verbose=True, skip_pre=False):
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

    Returns
    -------
    None
    """

    _network_prepare_and_run_pf(network, snapshots, verbose, skip_pre, sub_network_lpf, calculate_B_H)


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


def find_slack_bus(sub_network,verbose=True):
    """Find the slack bus in a connected sub-network."""

    gens = sub_network.generators()

    if len(gens) == 0:
        if verbose:
            print("No generators in %s, better hope power is already balanced" % sub_network)
        sub_network.slack_generator = None
        sub_network.slack_bus = sub_network.buses_i()[0]

    else:

        slacks = gens[gens.control == "Slack"].index

        if len(slacks) == 0:
            sub_network.slack_generator = gens.index[0]
            sub_network.network.generators.loc[sub_network.slack_generator,"control"] = "Slack"
            if verbose:
                print("No slack generator found, using %s as the slack generator" % sub_network.slack_generator)

        elif len(slacks) == 1:
            sub_network.slack_generator = slacks[0]
        else:
            sub_network.slack_generator = slacks[0]
            sub_network.network.generators.loc[slacks[1:],"control"] = "PV"
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
    buses_i = sub_network.buses_i()

    #default bus control is PQ
    network.buses.loc[buses_i, "control"] = "PQ"

    #find all buses with one or more gens with PV
    pvs = gens[gens.control == 'PV'].reset_index().groupby('bus').first()['name']
    network.buses.loc[pvs.index, "control"] = "PV"
    network.buses.loc[pvs.index, "generator"] = pvs

    network.buses.loc[sub_network.slack_bus, "control"] = "Slack"
    network.buses.loc[sub_network.slack_bus, "generator"] = sub_network.slack_generator

    buses_control = network.buses.loc[buses_i, "control"]
    sub_network.pvs = buses_control.index[buses_control == "PV"]
    sub_network.pqs = buses_control.index[buses_control == "PQ"]

    sub_network.pvpqs = sub_network.pvs.append(sub_network.pqs)

    # order buses
    sub_network.buses_o = sub_network.pvpqs.insert(0, sub_network.slack_bus)


def calculate_B_H(sub_network,verbose=True,skip_pre=False):
    """Calculate B and H matrices for AC or DC sub-networks."""

    from .components import passive_branch_types

    if not skip_pre:
        calculate_dependent_values(sub_network.network)
        find_bus_controls(sub_network,verbose)

    if sub_network.current_type == "DC":
        attribute="r_pu"
    elif sub_network.current_type == "AC":
        attribute="x_pu"

    #following leans heavily on pypower.makeBdc

    #susceptances
    b = 1./np.concatenate([t.df.loc[t.ind, attribute].values
                           for t in sub_network.iterate_components(passive_branch_types)])
    if verbose and np.isnan(b).any():
        print("Warning! Some series impedances are zero - this will cause a singularity in LPF!")
    b_diag = csr_matrix((b, (r_[:len(b)], r_[:len(b)])))

    #incidence matrix
    sub_network.K = _incidence_matrix(sub_network, sub_network.buses_o)

    sub_network.H = b_diag*sub_network.K.T

    #weighted Laplacian
    sub_network.B = sub_network.K * sub_network.H



def calculate_PTDF(sub_network,verbose=True,skip_pre=False):
    """
    Calculate the Power Transfer Distribution Factor (PTDF) for
    sub_network.

    Sets sub_network.PTDF as a (dense) numpy array.

    Parameters
    ----------
    sub_network : pypsa.SubNetwork
    verbose: bool, default True
    skip_pre: bool, default False
        Skip the preliminary steps of computing topology, calculating dependent values,
        finding bus controls and computing B and H.

    """

    if not skip_pre:
        calculate_B_H(sub_network,verbose)

    #calculate inverse of B with slack removed

    n_pvpq = len(sub_network.pvpqs)
    index = np.r_[:n_pvpq]

    I = csc_matrix((np.ones(n_pvpq), (index, index)))

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
    buses_o = sub_network.buses_o

    network = sub_network.network

    #following leans heavily on pypower.makeYbus
    #Copyright Richard Lincoln, Ray Zimmerman, BSD-style licence

    num_branches = len(branches)
    num_buses = len(buses_o)

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
    b_sh = network.shunt_impedances.b_pu.groupby(network.shunt_impedances.bus).sum().reindex(buses_o, fill_value = 0.)
    g_sh = network.shunt_impedances.g_pu.groupby(network.shunt_impedances.bus).sum().reindex(buses_o, fill_value = 0.)
    Y_sh = g_sh + 1.j*b_sh

    #get bus indices
    bus0 = buses_o.get_indexer(branches.bus0)
    bus1 = buses_o.get_indexer(branches.bus1)

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
        line_objs = list(sub_network.graph.adj[u][v].keys())
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




def find_tree(sub_network,verbose=True):
    """Get the spanning tree of the graph, choose the node with the
    highest degree as a central "tree slack" and then see for each
    branch which paths from the slack to each node go through the
    branch.

    """

    branches_i = sub_network.branches_i()
    buses_i = sub_network.buses_i()

    sub_network.tree = nx.minimum_spanning_tree(sub_network.graph)

    #find bus with highest degree to use as slack

    tree_slack_bus = None
    slack_degree = -1

    for bus,degree in sub_network.tree.degree_iter():
        if degree > slack_degree:
            tree_slack_bus = bus
            slack_degree = degree

    if verbose:
        print("Tree slack bus is %s with degree %d." % (tree_slack_bus,slack_degree))


    #determine which buses are supplied in tree through branch from slack

    #matrix to store tree structure
    sub_network.T = dok_matrix((len(branches_i),len(buses_i)))


    for j,bus in enumerate(buses_i):
        path = nx.shortest_path(sub_network.tree,bus,tree_slack_bus)
        for i in range(len(path)-1):
            branch = list(sub_network.graph[path[i]][path[i+1]].keys())[0]
            if branch.bus0 == path[i]:
                sign = +1
            else:
                sign = -1

            branch_i = branches_i.get_loc((branch.__class__.__name__, branch.name))

            sub_network.T[branch_i,j] = sign


def find_cycles(sub_network,verbose=True):
    """
    Find all cycles in the sub_network and record them in sub_network.C.

    networkx collects the cycles with more than 2 edges; then the 2-edge cycles
    from the MultiGraph must be collected separately (for cases where there
    are multiple lines between the same pairs of buses).

    """

    branches_i = sub_network.branches_i()

    #reduce to a non-multi-graph for cycles with > 2 edges
    graph = nx.OrderedGraph(sub_network.graph)

    cycles = nx.cycle_basis(graph)

    #number of 2-edge cycles
    num_multi = len(sub_network.graph.edges()) - len(graph.edges())

    sub_network.C = dok_matrix((len(branches_i),len(cycles)+num_multi))


    for j,cycle in enumerate(cycles):

        for i in range(len(cycle)):
            branch = list(sub_network.graph[cycle[i]][cycle[(i+1)%len(cycle)]].keys())[0]
            if branch.bus0 == cycle[i]:
                sign = +1
            else:
                sign = -1

            branch_i = branches_i.get_loc((branch.__class__.__name__,branch.name))
            sub_network.C[branch_i,j] = sign

    #counter for multis
    c = len(cycles)

    #add multi-graph 2-edge cycles for multiple branches between same pairs of buses
    for u,v in graph.edges():
        bs = list(sub_network.graph[u][v].keys())
        if len(bs) > 1:
            first = bs[0]
            first_i = branches_i.get_loc((first.__class__.__name__,first.name))
            for b in bs[1:]:
                b_i = branches_i.index.get_loc((b.__class__.__name__,b.name))
                if b.bus0 == first.bus0:
                    sign = -1
                else:
                    sign = 1

                sub_network.C[first_i,c] = 1
                sub_network.C[b_i,c] = sign
                c+=1

def sub_network_lpf(sub_network, snapshots=None, verbose=True, skip_pre=False):
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

    Returns
    -------
    None
    """

    snapshots = _as_snapshots(sub_network.network, snapshots)
    if verbose:
        print("Performing linear load-flow on {} sub-network {} for snapshot(s) {}"
              .format(sub_network.current_type,sub_network,snapshots))

    from .components import \
        one_port_types, controllable_one_port_types, \
        passive_branch_types, controllable_branch_types

    network = sub_network.network


    if not skip_pre:
        calculate_dependent_values(network)
        find_bus_controls(sub_network,verbose=verbose)

    # get indices for the components on this subnetwork
    buses_o = sub_network.buses_o
    branches_i = sub_network.branches_i()

    # allow all shunt impedances to dispatch as set
    shunt_impedances_i = sub_network.shunt_impedances_i()
    network.shunt_impedances_t.p.loc[snapshots, shunt_impedances_i] = \
        network.shunt_impedances.g_pu.loc[shunt_impedances_i].values

    # allow all one ports to dispatch as set
    for t in sub_network.iterate_components(controllable_one_port_types):
        t.pnl.p.loc[snapshots, t.ind] = t.pnl.p_set.loc[snapshots, t.ind]

    # set the power injection at each node
    network.buses_t.p.loc[snapshots, buses_o] = \
        sum([((t.pnl.p.loc[snapshots, t.ind] * t.df.loc[t.ind, 'sign'])
              .groupby(t.df.loc[t.ind, 'bus'], axis=1).sum()
              .reindex(columns=buses_o, fill_value=0.))
             for t in sub_network.iterate_components(one_port_types)]
            +
            [(- t.pnl.loc["p"+str(i), snapshots].groupby(t.df["bus"+str(i)], axis=1).sum()
              .reindex(columns=buses_o, fill_value=0))
             for t in network.iterate_components(controllable_branch_types)
             for i in [0,1]])

    if not skip_pre and len(branches_i) > 0:
        calculate_B_H(sub_network, verbose=verbose, skip_pre=True)

    v_diff = np.zeros((len(snapshots), len(buses_o)))
    if len(branches_i) > 0:
        p = network.buses_t.loc['p', snapshots, buses_o].values
        v_diff[:,1:] = spsolve(sub_network.B[1:, 1:], p[:,1:].T).T
        flows = pd.DataFrame(v_diff * sub_network.H.T,
                             columns=branches_i, index=snapshots)

        for t in network.iterate_components(passive_branch_types):
            f = flows.loc[:, t.name]
            t.pnl.p0.loc[snapshots, f.columns] = f
            t.pnl.p1.loc[snapshots, f.columns] = -f

    if sub_network.current_type == "AC":
        network.buses_t.v_ang.loc[snapshots, buses_o] = v_diff
        network.buses_t.v_mag_pu.loc[snapshots, buses_o] = 1.
    elif sub_network.current_type == "DC":
        network.buses_t.v_mag_pu.loc[snapshots, buses_o] = 1 + v_diff
        network.buses_t.v_ang.loc[snapshots, buses_o] = 0.

    # set slack bus power to pick up remained
    slack_adjustment = (- network.buses_t.loc['p', snapshots, buses_o[1:]].sum(axis=1)
                        - network.buses_t.loc['p', snapshots, buses_o[0]])
    network.buses_t.loc["p", snapshots, buses_o[0]] += slack_adjustment

    # let slack generator take up the slack
    if sub_network.slack_generator is not None:
        network.generators_t.p.loc[snapshots, sub_network.slack_generator] += slack_adjustment



def network_batch_lpf(network,snapshots=None):
    """Batched linear power flow with numpy.dot for several snapshots."""

    raise NotImplementedError("Batch linear power flow not supported yet.")
