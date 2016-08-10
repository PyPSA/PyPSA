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

"""Graph helper functions, which are attached to network and sub_network
"""

# Functions which will be attached to network and sub_network

import scipy as sp, scipy.sparse
import numpy as np

from .descriptors import OrderedGraph

def graph(network, branch_types=None):
    """Build networkx graph."""
    from . import components

    if isinstance(network, components.Network):
        if branch_types is None:
            branch_types = components.branch_types
        buses_i = network.buses.index
    elif isinstance(network, components.SubNetwork):
        if branch_types is None:
            branch_types = components.passive_branch_types
        buses_i = network.buses_i()
    else:
        raise TypeError("build_graph must be called with a Network or a SubNetwork")

    graph = OrderedGraph()

    # add nodes first, in case there are isolated buses not connected with branches
    graph.add_nodes_from(buses_i)

    # Multigraph uses object itself as key
    graph.add_edges_from((branch.bus0, branch.bus1, branch.obj, {})
                         for t in network.iterate_components(branch_types)
                         for branch in t.df.loc[slice(None)
                                                if t.ind is None
                                                else t.ind].itertuples())

    return graph

def adjacency_matrix(network, branch_types=None, busorder=None):
    """
    Construct a sparse adjacency matrix (directed)

    Parameters
    ----------
    branch_types : iterable sublist of `branch_types`
       Buses connected by any of the selected branches are adjacent
       (default: branch_types (network) or passive_branch_types (sub_network))
    busorder : pd.Index subset of network.buses.index
       Basis to use for the matrix representation of the adjacency matrix
       (default: buses.index (network) or buses_i() (sub_network))

    Returns
    -------
    adjacency_matrix : sp.sparse.coo_matrix
       Directed adjacency matrix
    """
    from . import components

    if isinstance(network, components.Network):
        if branch_types is None:
            branch_types = components.branch_types
        if busorder is None:
            busorder = network.buses.index
    elif isinstance(network, components.SubNetwork):
        if branch_types is None:
            branch_types = components.passive_branch_types
        if busorder is None:
            busorder = network.buses_i()
    else:
        raise TypeError(" must be called with a Network or a SubNetwork")

    no_buses = len(busorder)
    no_branches = 0
    bus0_inds = []
    bus1_inds = []
    for t in network.iterate_components(branch_types):
        if t.ind is None:
            sel = slice(None)
            no_branches += len(t.df)
        else:
            sel = t.ind
            no_branches += len(t.ind)
        bus0_inds.append(busorder.get_indexer(t.df.loc[sel, "bus0"]))
        bus1_inds.append(busorder.get_indexer(t.df.loc[sel, "bus1"]))

    if no_branches == 0:
        return sp.sparse.coo_matrix((no_buses, no_buses))

    bus0_inds = np.concatenate(bus0_inds)
    bus1_inds = np.concatenate(bus1_inds)

    return sp.sparse.coo_matrix((np.ones(no_branches), (bus0_inds, bus1_inds)),
                                shape=(no_buses, no_buses))

def incidence_matrix(network, branch_types=None, busorder=None):
    """
    Construct a sparse incidence matrix (directed)

    Parameters
    ----------
    branch_types : iterable sublist of `branch_types`
       Buses connected by any of the selected branches are adjacent
       (default: branch_types (network) or passive_branch_types (sub_network))
    busorder : pd.Index subset of network.buses.index
       Basis to use for the matrix representation of the adjacency matrix
       (default: buses.index (network) or buses_i() (sub_network))

    Returns
    -------
    incidence_matrix : sp.sparse.csr_matrix
       Directed incidence matrix
    """
    from . import components

    if isinstance(network, components.Network):
        if branch_types is None:
            branch_types = components.branch_types
        if busorder is None:
            busorder = network.buses.index
    elif isinstance(network, components.SubNetwork):
        if branch_types is None:
            branch_types = components.passive_branch_types
        if busorder is None:
            busorder = network.buses_i()
    else:
        raise TypeError(" must be called with a Network or a SubNetwork")

    no_buses = len(busorder)
    no_branches = 0
    bus0_inds = []
    bus1_inds = []
    for t in network.iterate_components(branch_types):
        if t.ind is None:
            sel = slice(None)
            no_branches += len(t.df)
        else:
            sel = t.ind
            no_branches += len(t.ind)
        bus0_inds.append(busorder.get_indexer(t.df.loc[sel, "bus0"]))
        bus1_inds.append(busorder.get_indexer(t.df.loc[sel, "bus1"]))
    bus0_inds = np.concatenate(bus0_inds)
    bus1_inds = np.concatenate(bus1_inds)

    return sp.sparse.csr_matrix((np.r_[np.ones(no_branches), -np.ones(no_branches)],
                                 (np.r_[bus0_inds, bus1_inds], np.r_[:no_branches, :no_branches])),
                                (no_buses, no_branches))
