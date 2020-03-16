## Copyright 2016-2017 Tom Brown (FIAS)

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

"""Functionality for contingency analysis, such as branch outages.
"""


__author__ = "Tom Brown (FIAS)"
__copyright__ = "Copyright 2016-2017 Tom Brown (FIAS), GNU GPL 3"


from scipy.sparse import issparse, csr_matrix, csc_matrix, hstack as shstack

from numpy import r_, ones, zeros

import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

import collections

from .pf import calculate_PTDF, _as_snapshots

from .opt import l_constraint


def calculate_BODF(sub_network, skip_pre=False):
    """
    Calculate the Branch Outage Distribution Factor (BODF) for
    sub_network.

    Sets sub_network.BODF as a (dense) numpy array.

    The BODF is a num_branch x num_branch 2d array.

    For the outage of branch l, the new flow on branch k is
    given in terms of the flow before the outage

    f_k^after = f_k^before + BODF_{kl} f_l^before

    Note that BODF_{ll} = -1.

    Parameters
    ----------
    sub_network : pypsa.SubNetwork
    skip_pre : bool, default False
        Skip the preliminary step of computing the PTDF.

    Examples
    --------
    >>> sub_network.caculate_BODF()
    """

    if not skip_pre:
        calculate_PTDF(sub_network)

    num_branches = sub_network.PTDF.shape[0]

    #build LxL version of PTDF
    branch_PTDF = sub_network.PTDF*sub_network.K

    denominator = csr_matrix((1/(1-np.diag(branch_PTDF)),(r_[:num_branches],r_[:num_branches])))

    sub_network.BODF = branch_PTDF*denominator

    #make sure the flow on the branch itself is zero
    np.fill_diagonal(sub_network.BODF,-1)


def network_lpf_contingency(network, snapshots=None, branch_outages=None):
    """
    Computes linear power flow for a selection of branch outages.

    Parameters
    ----------
    snapshots : list-like|single snapshot
        A subset or an elements of network.snapshots on which to run
        the power flow, defaults to network.snapshots
        NB: currently this only works for a single snapshot
    branch_outages : list-like
        A list of passive branches which are to be tested for outages.
        If None, it's take as all network.passive_branches_i()

    Returns
    -------
    p0 : pandas.DataFrame
        num_passive_branch x num_branch_outages DataFrame of new power flows

    Examples
    --------
    >>> network.lpf_contingency(snapshot, branch_outages)
    """

    if snapshots is None:
        snapshots = network.snapshots

    if isinstance(snapshots, collections.Iterable):
        logger.warning("Apologies LPF contingency, this only works for single snapshots at the moment, taking the first snapshot.")
        snapshot = snapshots[0]
    else:
        snapshot = snapshots

    network.lpf(snapshot)

    # Store the flows from the base case

    passive_branches = network.passive_branches()

    if branch_outages is None:
        branch_outages = passive_branches.index


    p0_base = pd.Series(index=passive_branches.index)

    for c in network.passive_branch_components:
        pnl = network.pnl(c)
        p0_base[c] = pnl.p0.loc[snapshot]

    for sn in network.sub_networks.obj:
        sn._branches = sn.branches()
        sn.calculate_BODF()

    p0 = pd.DataFrame(index=passive_branches.index)

    p0["base"] = p0_base

    for branch in branch_outages:
        if not isinstance(branch, tuple):
            logger.warning("No type given for {}, assuming it is a line".format(branch))
            branch = ("Line",branch)

        sn = network.sub_networks.obj[passive_branches.sub_network[branch]]

        branch_i = sn._branches.index.get_loc(branch)

        p0_new = p0_base + pd.Series(sn.BODF[:,branch_i]*p0_base[branch],sn._branches.index)

        p0[branch] = p0_new

    return p0




def network_sclopf(network, snapshots=None, branch_outages=None, solver_name="glpk",
                   skip_pre=False, extra_functionality=None, solver_options={},
                   keep_files=False, formulation="angles", ptdf_tolerance=0.):
    """
    Computes Security-Constrained Linear Optimal Power Flow (SCLOPF).

    This ensures that no branch is overloaded even given the branch outages.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    branch_outages : list-like
        A list of passive branches which are to be tested for outages.
        If None, it's take as all network.passive_branches_i()
    solver_name : string
        Must be a solver name that pyomo recognises and that is
        installed, e.g. "glpk", "gurobi"
    skip_pre : bool, default False
        Skip the preliminary steps of computing topology, calculating
        dependent values and finding bus controls.
    extra_functionality : callable function
        This function must take two arguments
        `extra_functionality(network,snapshots)` and is called after
        the model building is complete, but before it is sent to the
        solver. It allows the user to add/change constraints and
        add/change the objective function.
    solver_options : dictionary
        A dictionary with additional options that get passed to the solver.
        (e.g. {'threads':2} tells gurobi to use only 2 cpus)
    keep_files : bool, default False
        Keep the files that pyomo constructs from OPF problem
        construction, e.g. .lp file - useful for debugging
    formulation : string
        Formulation of the linear power flow equations to use; must be
        one of ["angles","cycles","kirchoff","ptdf"]
    ptdf_tolerance : float

    Returns
    -------
    None

    Examples
    --------
    >>> network.sclopf(network, branch_outages)
    """

    if not skip_pre:
        network.determine_network_topology()

    snapshots = _as_snapshots(network, snapshots)

    passive_branches = network.passive_branches()

    if branch_outages is None:
        branch_outages = passive_branches.index

    #prepare the sub networks by calculating BODF and preparing helper DataFrames

    for sn in network.sub_networks.obj:

        sn.calculate_BODF()

        sn._branches = sn.branches()
        sn._branches["_i"] = range(sn._branches.shape[0])

        sn._extendable_branches = sn._branches[sn._branches.s_nom_extendable]
        sn._fixed_branches = sn._branches[~ sn._branches.s_nom_extendable]


    def add_contingency_constraints(network,snapshots):

        #a list of tuples with branch_outage and passive branches in same sub_network
        branch_outage_keys = []
        flow_upper = {}
        flow_lower = {}

        for branch in branch_outages:
            if type(branch) is not tuple:
                logger.warning("No type given for {}, assuming it is a line".format(branch))
                branch = ("Line",branch)

            sub = network.sub_networks.at[passive_branches.at[branch,"sub_network"],"obj"]

            branch_i = sub._branches.at[branch,"_i"]

            branch_outage_keys.extend([(branch[0],branch[1],b[0],b[1]) for b in sub._branches.index])

            flow_upper.update({(branch[0],branch[1],b[0],b[1],sn) : [[(1,network.model.passive_branch_p[b[0],b[1],sn]),(sub.BODF[sub._branches.at[b,"_i"],branch_i],network.model.passive_branch_p[branch[0],branch[1],sn])],"<=",sub._fixed_branches.at[b,"s_nom"]] for b in sub._fixed_branches.index for sn in snapshots})

            flow_upper.update({(branch[0],branch[1],b[0],b[1],sn) : [[(1,network.model.passive_branch_p[b[0],b[1],sn]),(sub.BODF[sub._branches.at[b,"_i"],branch_i],network.model.passive_branch_p[branch[0],branch[1],sn]),(-1,network.model.passive_branch_s_nom[b[0],b[1]])],"<=",0] for b in sub._extendable_branches.index for sn in snapshots})


            flow_lower.update({(branch[0],branch[1],b[0],b[1],sn) : [[(1,network.model.passive_branch_p[b[0],b[1],sn]),(sub.BODF[sub._branches.at[b,"_i"],branch_i],network.model.passive_branch_p[branch[0],branch[1],sn])],">=",-sub._fixed_branches.at[b,"s_nom"]] for b in sub._fixed_branches.index for sn in snapshots})

            flow_lower.update({(branch[0],branch[1],b[0],b[1],sn) : [[(1,network.model.passive_branch_p[b[0],b[1],sn]),(sub.BODF[sub._branches.at[b,"_i"],branch_i],network.model.passive_branch_p[branch[0],branch[1],sn]),(1,network.model.passive_branch_s_nom[b[0],b[1]])],">=",0] for b in sub._extendable_branches.index for sn in snapshots})


        l_constraint(network.model,"contingency_flow_upper",flow_upper,branch_outage_keys,snapshots)


        l_constraint(network.model,"contingency_flow_lower",flow_lower,branch_outage_keys,snapshots)

        if extra_functionality is not None:
            extra_functionality(network, snapshots)

    #need to skip preparation otherwise it recalculates the sub-networks

    network.lopf(snapshots=snapshots, solver_name=solver_name, skip_pre=True,
                 extra_functionality=add_contingency_constraints,
                 solver_options=solver_options, keep_files=keep_files,
                 formulation=formulation, ptdf_tolerance=ptdf_tolerance)
