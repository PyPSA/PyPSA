## Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS)

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

import scipy as sp
import numpy as np

from .descriptors import OrderedGraph
from .utils import branch_select_i

def graph(network, branch_components=None, weight=None, inf_weight=False,
          sel='operative'):
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
    sel : string|None
        Specifies subset of lines.
        If `None` (default) it includes both operative and inoperative lines.
        If `"operative"` it includes only operative lines.
        If `"inoperative"` it includes only inoperative lines.
        If `"potential"` it includes operative or candidate lines (i.e. not operative but extendable).
        If `"candidate"` it includes candidate lines; i.e. not operative but extendable lines.
        If `"used"` it includes operative and built candidate lines. Can only be called after successful optimisation.

    Returns
    -------
    graph : OrderedGraph
        NetworkX graph
    """

    from . import components

    if isinstance(network, components.Network):
        if branch_components is None:
            branch_components = network.branch_components
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
    def gen_edges():
        for c in network.iterate_components(branch_components):
            for branch in c.df.loc[branch_select_i(c, sel=sel)].itertuples():
                if weight is None:
                    data = {}
                else:
                    data = dict(weight=getattr(branch, weight, 0))
                    if np.isinf(data['weight']) and inf_weight is not True:
                        if inf_weight is False:
                            continue
                        else:
                            data['weight'] = inf_weight

                yield (branch.bus0, branch.bus1, (c.name, branch.Index), data)

    graph.add_edges_from(gen_edges())

    return graph

def adjacency_matrix(network, branch_components=None, busorder=None, weights=None,
                     sel='operative'):
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
    sel : string|None
        Specifies subset of lines.
        If `None` (default) it includes both operative and inoperative lines.
        If `"operative"` it includes only operative lines.
        If `"inoperative"` it includes only inoperative lines.
        If `"potential"` it includes operative or candidate lines (i.e. not operative but extendable).
        If `"candidate"` it includes candidate lines; i.e. not operative but extendable lines.
        If `"used"` it includes operative and built candidate lines. Can only be called after successful optimisation.

    Returns
    -------
    adjacency_matrix : sp.sparse.coo_matrix
       Directed adjacency matrix
    """

    from . import components

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
        selection = branch_select_i(c, sel=sel)
        no_branches = len(selection) if type(selection) != slice else len(c.df)
        bus0_inds.append(busorder.get_indexer(c.df.loc[selection, "bus0"]))
        bus1_inds.append(busorder.get_indexer(c.df.loc[selection, "bus1"]))
        weight_vals.append(np.ones(no_branches)
                           if weights is None
                           else weights[c.name][selection].values)

    if no_branches == 0:
        return sp.sparse.coo_matrix((no_buses, no_buses))

    bus0_inds = np.concatenate(bus0_inds)
    bus1_inds = np.concatenate(bus1_inds)
    weight_vals = np.concatenate(weight_vals)

    return sp.sparse.coo_matrix((weight_vals, (bus0_inds, bus1_inds)),
                                shape=(no_buses, no_buses))

def incidence_matrix(network, branch_components=None, busorder=None,
                     sel='operative'):
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
    sel : string|None
        Specifies subset of lines.
        If `None` (default) it includes both operative and inoperative lines.
        If `"operative"` it includes only operative lines.
        If `"inoperative"` it includes only inoperative lines.
        If `"potential"` it includes operative or candidate lines (i.e. not operative but extendable).
        If `"candidate"` it includes candidate lines; i.e. not operative but extendable lines.
        If `"used"` it includes operative and built candidate lines. Can only be called after successful optimisation.

    Returns
    -------
    incidence_matrix : sp.sparse.csr_matrix
       Directed incidence matrix
    """
    from . import components

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
        selection = branch_select_i(c, sel=sel)
        no_branches += len(selection) if type(selection) != slice else len(c.df)
        bus0_inds.append(busorder.get_indexer(c.df.loc[selection, "bus0"]))
        bus1_inds.append(busorder.get_indexer(c.df.loc[selection, "bus1"]))
    bus0_inds = np.concatenate(bus0_inds)
    bus1_inds = np.concatenate(bus1_inds)

    return sp.sparse.csr_matrix((np.r_[np.ones(no_branches), -np.ones(no_branches)],
                                 (np.r_[bus0_inds, bus1_inds], np.r_[:no_branches, :no_branches])),
                                (no_buses, no_branches))
