"""
Functionality for contingency analysis, such as branch outages.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy import r_
from scipy.sparse import csr_matrix

from pypsa.pf import calculate_PTDF
from pypsa.utils import deprecated_common_kwargs

if TYPE_CHECKING:
    from pypsa import Network, SubNetwork

logger = logging.getLogger(__name__)


def calculate_BODF(sub_network: SubNetwork, skip_pre: bool = False) -> None:
    """
    Calculate the Branch Outage Distribution Factor (BODF) for sub_network.

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

    # build LxL version of PTDF
    branch_PTDF = sub_network.PTDF * sub_network.K

    with np.errstate(divide="ignore"):
        denominator = csr_matrix(
            (1 / (1 - np.diag(branch_PTDF)), (r_[:num_branches], r_[:num_branches]))
        )

    sub_network.BODF = branch_PTDF * denominator

    # make sure the flow on the branch itself is zero
    np.fill_diagonal(sub_network.BODF, -1)


@deprecated_common_kwargs
def network_lpf_contingency(
    n: Network,
    snapshots: Sequence | str | int | pd.Timestamp | None = None,
    branch_outages: Sequence | None = None,
) -> pd.DataFrame:
    """
    Computes linear power flow for a selection of branch outages.

    Parameters
    ----------
    snapshots : list-like|single snapshot
        A subset or an elements of n.snapshots on which to run
        the power flow, defaults to n.snapshots
        NB: currently this only works for a single snapshot
    branch_outages : list-like
        A list of passive branches which are to be tested for outages.
        If None, it's take as all n.passive_branches_i()

    Returns
    -------
    p0 : pandas.DataFrame
        num_passive_branch x num_branch_outages DataFrame of new power flows

    Examples
    --------
    >>> n.lpf_contingency(snapshot, branch_outages)
    """
    if snapshots is None:
        snapshots = n.snapshots

    if isinstance(snapshots, Sequence):
        logger.warning(
            "Apologies LPF contingency, this only works for single snapshots at the moment, taking the first snapshot."
        )
        snapshot = snapshots[0]
    else:
        snapshot = snapshots

    n.lpf(snapshot)

    # Store the flows from the base case

    passive_branches = n.passive_branches()

    if branch_outages is None:
        branch_outages = passive_branches.index

    p0_base = pd.concat(
        {c: n.dynamic(c).p0.loc[snapshot] for c in n.passive_branch_components}
    )
    p0 = p0_base.to_frame("base")

    for sub_network in n.sub_networks.obj:
        sub_network._branches = sub_network.branches()
        sub_network.calculate_BODF()

    for branch in branch_outages:
        if not isinstance(branch, tuple):
            logger.warning(f"No type given for {branch}, assuming it is a line")
            branch = ("Line", branch)

        sub_network = n.sub_networks.obj[passive_branches.sub_network[branch]]

        branch_i = sub_network._branches.index.get_loc(branch)
        p0_new = p0_base + pd.Series(
            sub_network.BODF[:, branch_i] * p0_base[branch], sub_network._branches.index
        )
        p0_new.name = branch

        p0 = pd.concat([p0, p0_new], axis=1)

    return p0
