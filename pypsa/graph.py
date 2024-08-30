"""
Graph helper functions, which are attached to network and sub_network.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy as sp

from pypsa.descriptors import OrderedGraph, get_active_assets

if TYPE_CHECKING:
    from pypsa import Network, SubNetwork


def graph(
    network: Network | SubNetwork,
    branch_components: Collection[str] | None = None,
    weight: str | None = None,
    inf_weight: bool | float = False,
) -> OrderedGraph:
    """
    Build NetworkX graph.

    Parameters
    ----------
    network : Network|SubNetwork

    branch_components : [str]
        Components to use as branches. The default are
        passive_branch_components in the case of a SubNetwork and
        branch_components in the case of a Network.

    weight : str
        Branch attribute to use as weight

    inf_weight : bool|float
        How to treat infinite weights (default: False). True keeps the infinite
        weight. False skips edges with infinite weight. If a float is given it
        is used instead.

    Returns
    -------
    graph : OrderedGraph
        NetworkX graph
    """
    from pypsa import components

    if isinstance(network, components.Network):
        if branch_components is None:
            branch_components = network.branch_components
        else:
            branch_components = set(branch_components)
        buses_i = network.buses.index
    elif isinstance(network, components.SubNetwork):
        if branch_components is None:
            branch_components = network.network.passive_branch_components
        buses_i = network.buses_i()
    else:
        raise TypeError("graph must be called with a Network or a SubNetwork")

    graph = OrderedGraph()

    # add nodes first, in case there are isolated buses not connected with branches
    graph.add_nodes_from(buses_i)

    # Multigraph uses the branch type and name as key
    def gen_edges() -> Iterable[tuple[str, str, tuple[str, int], dict]]:
        for c in network.iterate_components(branch_components):
            for branch in c.df.loc[
                slice(None) if c.ind is None else c.ind
            ].itertuples():
                if weight is None:
                    data = {}
                else:
                    data = dict(weight=getattr(branch, weight, 0))
                    if np.isinf(data["weight"]) and inf_weight is not True:
                        if inf_weight is False:
                            continue
                        data["weight"] = inf_weight

                yield (branch.bus0, branch.bus1, (c.name, branch.Index), data)

    graph.add_edges_from(gen_edges())

    return graph


def adjacency_matrix(
    network: Network,
    branch_components: Collection[str] | None = None,
    investment_period: int | str | None = None,
    busorder: pd.Index | None = None,
    weights: pd.Series | None = None,
) -> sp.sparse.coo_matrix:
    """
    Construct a sparse adjacency matrix (directed)

    Parameters
    ----------
    branch_components : iterable sublist of `branch_components`
       Buses connected by any of the selected branches are adjacent
       (default: branch_components (network) or passive_branch_components (sub_network))
    busorder : pd.Index subset of network.buses.index
       Basis to use for the matrix representation of the adjacency matrix
       (default: buses.index (network) or buses_i() (sub_network))
    weights : pd.Series or None (default)
       If given must provide a weight for each branch, multi-indexed
       on branch_component name and branch name.

    Returns
    -------
    adjacency_matrix : sp.sparse.coo_matrix
       Directed adjacency matrix
    """

    from pypsa import components

    if isinstance(network, components.Network):
        if branch_components is None:
            branch_components = network.branch_components
        if busorder is None:
            busorder = network.buses.index
    elif isinstance(network, components.SubNetwork):
        if branch_components is None:
            branch_components = network.network.passive_branch_components
        if busorder is None:
            busorder = network.buses_i()
    else:
        raise TypeError(" must be called with a Network or a SubNetwork")

    no_buses = len(busorder)
    no_branches = 0
    bus0_inds = []
    bus1_inds = []
    weight_vals = []
    for c in network.iterate_components(branch_components):
        if c.ind is None:
            if investment_period is None:
                sel = slice(None)
            else:
                active = get_active_assets(network, c.name, investment_period)
                sel = c.df.loc[active].index
        elif investment_period is None:
            sel = c.ind
        else:
            active = get_active_assets(network, c.name, investment_period)
            sel = c.ind & c.df.loc[active].index

        no_branches = len(c.df.loc[sel])
        bus0_inds.append(busorder.get_indexer(c.df.loc[sel, "bus0"]))
        bus1_inds.append(busorder.get_indexer(c.df.loc[sel, "bus1"]))
        weight_vals.append(
            np.ones(no_branches) if weights is None else weights[c.name][sel].values
        )

    if no_branches == 0:
        return sp.sparse.coo_matrix((no_buses, no_buses))

    bus0_inds = np.concatenate(bus0_inds)
    bus1_inds = np.concatenate(bus1_inds)
    weight_vals = np.concatenate(weight_vals)

    return sp.sparse.coo_matrix(
        (weight_vals, (bus0_inds, bus1_inds)), shape=(no_buses, no_buses)
    )


def incidence_matrix(
    network: Network | SubNetwork,
    branch_components: Collection[str] | None = None,
    busorder: pd.Index | None = None,
) -> sp.sparse.csr_matrix:
    """
    Construct a sparse incidence matrix (directed)

    Parameters
    ----------
    branch_components : iterable sublist of `branch_components`
       Buses connected by any of the selected branches are adjacent
       (default: branch_components (network) or passive_branch_components (sub_network))
    busorder : pd.Index subset of network.buses.index
       Basis to use for the matrix representation of the adjacency matrix
       (default: buses.index (network) or buses_i() (sub_network))

    Returns
    -------
    incidence_matrix : sp.sparse.csr_matrix
       Directed incidence matrix
    """
    from pypsa import components

    if isinstance(network, components.Network):
        if branch_components is None:
            branch_components = network.branch_components
        if busorder is None:
            busorder = network.buses.index
    elif isinstance(network, components.SubNetwork):
        if branch_components is None:
            branch_components = network.network.passive_branch_components
        if busorder is None:
            busorder = network.buses_i()
    else:
        raise TypeError(" must be called with a Network or a SubNetwork")

    no_buses = len(busorder)
    no_branches = 0
    bus0_inds = []
    bus1_inds = []
    for c in network.iterate_components(branch_components):
        if c.ind is None:
            sel = slice(None)
            no_branches += len(c.df)
        else:
            sel = c.ind
            no_branches += len(c.ind)
        bus0_inds.append(busorder.get_indexer(c.df.loc[sel, "bus0"]))
        bus1_inds.append(busorder.get_indexer(c.df.loc[sel, "bus1"]))
    bus0_inds = np.concatenate(bus0_inds)
    bus1_inds = np.concatenate(bus1_inds)

    return sp.sparse.csr_matrix(
        (
            np.r_[np.ones(no_branches), -np.ones(no_branches)],
            (np.r_[bus0_inds, bus1_inds], np.r_[:no_branches, :no_branches]),
        ),
        (no_buses, no_branches),
    )
