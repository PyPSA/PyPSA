## Copyright 2015-2016 Tom Brown (FIAS)

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


# make the code as Python 3 compatible as possible
from __future__ import print_function, division, absolute_import


__author__ = "Tom Brown (FIAS)"
__copyright__ = "Copyright 2016 Tom Brown (FIAS), GNU GPL 3"



from scipy.sparse import issparse, csr_matrix, csc_matrix, hstack as shstack, vstack as svstack

from numpy import r_, ones, zeros, newaxis

import numpy as np
import pandas as pd

import collections

from .pf import calculate_PTDF

def calculate_BODF(sub_network,verbose=True,skip_pre=False):
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
    verbose: bool, default True
    skip_pre: bool, default False
        Skip the preliminary step of computing the PTDF.

    """

    if not skip_pre:
        calculate_PTDF(sub_network,verbose)

    num_branches = sub_network.PTDF.shape[0]

    #build LxL version of PTDF
    branch_PTDF = sub_network.PTDF*sub_network.K

    denominator = csr_matrix((1/(1-np.diag(branch_PTDF)),(r_[:num_branches],r_[:num_branches])))

    sub_network.BODF = branch_PTDF*denominator

    #make sure the flow on the branch itself is zero
    np.fill_diagonal(sub_network.BODF,-1)


def network_lpf_contingency(network,snapshots=None,branch_outages=None,verbose=True,skip_pre=True):
    """
    Computes linear power flow for a selection of branch outages.

    Parameters
    ----------
    snapshots : list-like|single snapshot
        A subset or an elements of network.snapshots on which to run
        the power flow, defaults to [now]
        NB: currently this only works for a single snapshot
    branch_outages : list-like
        A list of passive branches which are to be tested for outages.
        If None, it's take as all network.passive_branches_i()
    verbose: bool, default True
    skip_pre: bool, default False
        Skip the preliminary steps of computing topology, calculating
        dependent values and finding bus controls.
    now : object
        Deprecated: A member of network.snapshots on which to run the
        power flow, defaults to network.now

    Returns
    -------
    p0 : pandas.DataFrame
        num_passive_branch x num_branch_outages DataFrame of new power flows

    """

    from .components import passive_branch_types

    if snapshots is None:
        snapshot = network.now
    elif isinstance(snapshots, collections.Iterable):
        print("Apologies LPF contingency, this only works for single snapshots at the moment, taking the first snapshot.")
        snapshot = snapshots[0]
    else:
        snapshot = snapshots


    network.lpf(snapshot)

    # Store the flows from the base case

    passive_branches = network.passive_branches()

    p0_base = pd.Series(index=passive_branches.index)

    for typ in passive_branch_types:
        pnl = getattr(network,typ.list_name + "_t")
        p0_base[typ.__name__] = pnl.p0.loc[network.now]

    for sn in network.sub_networks.obj:
        sn._branches = sn.branches()
        sn.calculate_BODF(verbose)

    p0 = pd.DataFrame(index=passive_branches.index)

    p0["base"] = p0_base

    for branch in branch_outages:
        if type(branch) is not tuple and verbose:
            print("No type given for {}, assuming it is a line".format(branch))
            branch = ("Line",branch)

        sn = network.sub_networks.obj[passive_branches.sub_network[branch]]

        branch_i = sn._branches.index.get_loc(branch)

        p0_new = p0_base + pd.Series(sn.BODF[:,branch_i]*p0_base[branch],sn._branches.index)

        p0[branch] = p0_new

    return p0
