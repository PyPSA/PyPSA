# Copyright 2019-2020 Fabian Neumann (KIT)

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Optimal Power Flow functions with Integer Transmission Expansion Planning.
"""

# make the code as Python 3 compatible as possible
from __future__ import division, absolute_import
from six import iteritems, itervalues, string_types

__author__ = "Fabian Neumann (KIT)"
__copyright__ = "Copyright 2019-2020 Fabian Neumann (KIT), GNU GPL 3"

import pandas as pd
import numpy as np
import networkx as nx
import itertools

from collections import deque

from scipy.sparse import issparse, csr_matrix, csc_matrix, hstack as shstack, vstack as svstack, dok_matrix

from pyomo.environ import (ConcreteModel, Var, Objective,
                           NonNegativeReals, Constraint, Reals,
                           Suffix, Expression, Binary, SolverFactory)

import logging
logger = logging.getLogger(__name__)

from .opf import (define_generator_variables_constraints,
                  define_branch_extension_variables,
                  define_storage_variables_constraints,
                  define_store_variables_constraints,
                  define_link_flows,
                  define_passive_branch_flows,
                  define_passive_branch_flows_with_angles,
                  define_passive_branch_flows_with_kirchhoff,
                  define_sub_network_cycle_constraints,
                  define_passive_branch_constraints,
                  define_nodal_balances,
                  define_sub_network_balance_constraints,
                  define_global_constraints,
                  define_linear_objective,
                  extract_optimisation_results,
                  network_lopf_build_model,
                  network_lopf_prepare_solver,
                  network_lopf_solve)

from .pf import (_as_snapshots, calculate_dependent_values,
                 find_slack_bus, find_cycles)

from .opt import (free_pyomo_initializers, l_constraint,
                  LExpression, LConstraint)

from .descriptors import (get_switchable_as_dense, get_switchable_as_iter)

from .utils import (_make_consense, _haversine, _normed)

RESCALING = 1e5


def sub_networks_graph(network):
    """
    Creates a networkx.MultiGraph() from the pypsa.Network
    with sub_networks represented as vertices and
    candidate lines that connect sub_networks as edges.

    Parameters
    ----------
    network : pypsa.Network

    Returns
    -------
    graph : networkx.MultiGraph
    """

    graph = nx.MultiGraph()

    graph.add_nodes_from(network.sub_networks.index)

    def gen_sub_network_edges():
        candidate_branches = network.passive_branches(sel='candidate')
        data = {}
        for cnd_i, cnd, in candidate_branches.iterrows():
            sn0 = network.buses.loc[cnd.bus0].sub_network
            sn1 = network.buses.loc[cnd.bus1].sub_network
            if sn0 != sn1:
                yield (sn0, sn1, cnd_i, data)

    graph.add_edges_from(gen_sub_network_edges())

    return graph


def _within_sub_network_b(sub_network, lines):
    return [True if bus1 in sub_network.buses().index else False for bus1 in lines.bus1]


def equivalent_cycle(c, d):
    """
    Checks whether two cycles are equivalent
    when disregarding orientation and first vertex.

    Parameters
    ----------
    c : list
        List of vertices representing the first cycle.
    d : list
        List of vertices representing the second cycle.

    Returns
    -------
    bool
        Cycles are equivalent.
    """

    dc, dd = (deque(c), deque(d))
    dd_rev = dd.copy()
    dd_rev.reverse()
    for _ in range(len(dd)):
        dd.rotate(1)
        dd_rev.rotate(1)
        if dd == dc or dd_rev == dc:
            return True
    return False


def add_cycle_b(cycle, cycles):
    """
    Checks whether an equivalent cycle of `cycle`
    is already in `cycles`.

    Parameters
    ----------
    cycle : list
        List of vertices representing a cycle.
    cycles : list
        List of cycles, where elements are
        lists of vertices representing a cycle.

    Returns
    -------
    bool
        `cycle` is already in `cycles`.
    """

    for c in cycles:
        if equivalent_cycle(c, cycle):
            return False
    return True


def get_line_sub_networks(line_i, network):
    """
    Determines which sub_networks a line connects to.
    Notes sub_network index, orientation and buses.

    Parameters
    ----------
    line_i : tuple
        Line index from network.passive_branches(),
        e.g. ('Line', 'my_line')
    network : pypsa.Network

    Returns
    -------
    dict
        Dictionary with sub_networks as keys and a tuple
        with orientation and connecting bus as values, e.g.
        {'0': ('bus0': 'my_bus4'), '1': ('bus1', 'my_bus6')}
    """

    line = network.lines.loc[line_i[1]]
    sn0 = network.buses.loc[line.bus0].sub_network
    sn1 = network.buses.loc[line.bus1].sub_network

    return {sn0: ('bus0', line.bus0), sn1: ('bus1', line.bus1)}


def common_sub_network_vertices(line0, line1, network):
    """
    Determines a list with an entry for each sub_network
    `line0` and `line1` commonly connect to. Containing
    information regarding their connection point and orientation.

    Parameters
    ----------
    line0 : tuple
        Line index from network.passive_branches(),
        e.g. ('Line', 'my_line')
    line1 : tuple
        Line index from network.passive_branches(),
        e.g. ('Line', 'my_line')
    network : pypsa.Network

    Returns
    -------
    list
        For instance [(('bus0', 'bus1'), ('b1', 'b3'))] menas
        `line0` connects at 'bus0' to 'b1' and `line1` connects
        at 'bus1' to 'b3'.
    """

    sub_networks_line0 = get_line_sub_networks(line0, network)
    sub_networks_line1 = get_line_sub_networks(line1, network)

    commons = list(set(sub_networks_line0.keys()).intersection(
        set(sub_networks_line1.keys())))

    return [tuple(zip(*(sub_networks_line0[common], sub_networks_line1[common]))) for common in commons]


def get_cycles_as_branches(graph, cycles_deduplicated):
    """
    Converts cycles based on vertices to cycles based
    on candidate line indices.

    Parameters
    ----------
    graph : networkx.MultiGraph
    cycles_deduplicated : list
        For example [['0','1','2'], ['1','3','4']]

    Returns
    -------
    cycles_as_branches : list
        For example [(('Line', 'l4'), ('Line', 'l6'), ('Line', 'l22')),
        (('Line', 'l4'), ('Line', 'l21'), ('Line', 'l22'))]
    """

    ordered_graph = nx.OrderedGraph(graph)

    branches_in_corridors = []
    for cycle in cycles_deduplicated:
        l = len(cycle)
        for i in range(l):
            corridor_branches = list(graph[cycle[i]][cycle[(i+1) % l]])
            branches_in_corridors.append(corridor_branches)

    cycles_as_branches = list(itertools.product(*branches_in_corridors))

    # add 2-edge cycles
    for u, v in ordered_graph.edges():
        corridor_branches = tuple(graph[u][v])
        if len(corridor_branches) > 1:
            cycles_as_branches.append(corridor_branches)

    return cycles_as_branches


def find_candidate_cycles(network):
    """
    Constructs a cycle matrix based on cycles added by candidate lines.
    Differentiates between cycles within sub_network and
    cycles across sub_networks.

    Parameters
    ----------
    network : pypsa.SubNetwork | pypsa.Network

    """

    if '_network' in network.__dict__:
        find_candidate_cycles_sub_network(network)
    else:
        find_candidate_cycles_network(network)


def find_candidate_cycles_network(network):
    """
    Constructs an additional cycle matrix based on cycles added by
    candidate lines across sub_networks and records them in network.CC.
    """

    potential_branches = network.passive_branches(sel='potential')

    # skip if network is just a single bus
    if len(potential_branches) == 0:
        network.CC = dok_matrix((0, 0))
        return

    ngraph = network.graph()
    g = sub_networks_graph(network)

    g_ordered = nx.OrderedGraph(g)
    g_di = g_ordered.to_directed()

    cycles = list(nx.simple_cycles(g_di))
    cycles_long = [c for c in cycles if len(c) > 2]

    cycles_deduplicated = []
    for cycle in cycles_long:
        if add_cycle_b(cycle, cycles_deduplicated):
            cycles_deduplicated.append(cycle)

    cycles_branch = get_cycles_as_branches(g, cycles_deduplicated)

    branches_i = potential_branches.index
    branches_bus0 = potential_branches.bus0

    network.CC = dok_matrix((len(branches_i), len(cycles_branch)))

    for j, cycle in enumerate(cycles_branch):
        l = len(cycle)
        for i in range(l):

            line0 = cycle[i]
            line1 = cycle[(i+1) % l]

            csn_vertices = common_sub_network_vertices(line0, line1, network)
            # switch to other orientation
            csn_i = 1 if (l <= 2) & ((i+1) % l == 0) else 0
            orientation, from_to = csn_vertices[csn_i]

            # add sync
            branch_i = branches_i.get_loc(cycle[i])
            sign = +1 if orientation[0] == 'bus1' else -1
            network.CC[branch_i, j] = sign

            # add route with sub_network
            path = nx.dijkstra_path(ngraph, *from_to)
            for k in range(len(path)-1):
                corridor_branches = dict(ngraph[path[k]][path[k+1]])
                # if multiple existing lines pick one
                branch_name = list(corridor_branches.keys())[0]
                branch_i = branches_i.get_loc(branch_name)
                sign = +1 if branches_bus0.iat[branch_i] == path[k] else -1
                network.CC[branch_i, j] = sign


def find_candidate_cycles_sub_network(sub_network):
    """
    Constructs an additional cycle matrix based on cycles added by
    candidate lines within the sub_network and records them in sub_network.CC.
    """

    potential_lines = sub_network.branches(sel='potential')
    candidate_lines = sub_network.branches(sel='candidate')

    candidate_lines_sub = candidate_lines.loc[_within_sub_network_b(
        sub_network, candidate_lines)]
    potential_lines_sub = potential_lines.loc[_within_sub_network_b(
        sub_network, potential_lines)]

    # skip if sub_network is just a single bus
    if len(potential_lines_sub) == 0:
        sub_network.CC = dok_matrix((0, 0))
        return

    cnd_edges = candidate_lines_sub.apply(
        lambda x: (x.bus0, x.bus1, x.name), axis=1)

    mgraph = sub_network.graph(sel='operative')

    cycles = {}
    for cnd_edge in cnd_edges:
        cycle = nx.dijkstra_path(mgraph, cnd_edge[0], cnd_edge[1])
        cycles[cnd_edge] = cycle

    branches_bus0 = potential_lines_sub.bus0
    branches_i = potential_lines_sub.index

    sub_network.CC = dok_matrix((len(branches_i), len(cycles)))

    for j, (candidate, cycle) in enumerate(iteritems(cycles)):
        for i in range(len(cycle)-1):
            corridor_branches = dict(mgraph[cycle[i]][cycle[i+1]])
            # if multiple existing lines pick one
            branch_name = list(corridor_branches.keys())[0]
            branch_i = branches_i.get_loc(branch_name)
            sign = +1 if branches_bus0.iat[branch_i] == cycle[i] else -1
            sub_network.CC[branch_i, j] += sign

        # add candidate line
        branch_i = branches_i.get_loc(candidate[2])
        sub_network.CC[branch_i, j] += -1


def infer_candidates_from_existing(network):
    """
    Infer candidate lines from existing lines.

    Parameters
    ----------
    network : pypsa.Network

    Returns
    -------
    network.lines : pandas.DataFrame
    """

    network.calculate_dependent_values()

    network.lines = add_candidate_lines(network)

    # extendability is transferred to candidate lines
    network.lines.loc[network.lines.operative, 's_nom_extendable'] = False

    return network.lines


def potential_num_parallels(network):
    """
    Determine the number of allowable additional parallel circuits
    per line based on `s_nom_extendable`, the difference between
    `s_nom_max` and `s_nom`, and the `type` if specified.
    Otherwise, the line type and its electrical parameters 
    will be inferred from `num_parallel`.

    Parameters
    ----------
    network : pypsa.Network

    Returns
    -------
    pandas.Series
    """

    ext_lines = network.lines[network.lines.s_nom_extendable &
                              network.lines.operative]

    assert (ext_lines.s_nom_max != np.inf).all(
    ), "Calculating potential for additional circuits at `pypsa.tepopf.potential_num_parallels()` requires `s_nom_max` to be a finite number."

    ext_lines.s_nom_max = ext_lines.s_nom_max.apply(
        np.ceil)  # to avoid rounding errors
    investment_potential = ext_lines.s_nom_max - ext_lines.s_nom
    unit_s_nom = np.sqrt(
        3) * ext_lines.type.map(network.line_types.i_nom) * ext_lines.v_nom
    # fallback if no line type given
    unit_s_nom = unit_s_nom.fillna(ext_lines.s_nom/ext_lines.num_parallel)
    num_parallels = investment_potential.divide(
        unit_s_nom).map(np.floor).map(int)

    return num_parallels


def add_candidate_lines(network):
    """
    Create a DataFrame of individual candidate lines that can be built.

    Parameters
    ----------
    network : pypsa.Network

    Returns
    -------
    pandas.DataFrame
    """

    num_parallels = potential_num_parallels(network)
    c_sets = num_parallels.apply(lambda num: np.arange(1, num+1))

    candidates = pd.DataFrame(columns=network.lines.columns)

    for ind, c_set in c_sets.iteritems():
        for c in c_set:
            candidate = network.lines.loc[ind].copy()
            try:
                type_params = network.line_types.loc[candidate.type]
            except KeyError:
                type_params = pd.Series({'x_per_length': c.x/c.length,
                                         'r_per_length': c.r/c.length,
                                         'i_nom': c.s_nom/c.num_parallel/np.sqrt(3)/c.v_nom})
            candidate.num_parallel = 1.
            candidate.x = type_params.x_per_length * candidate.length
            candidate.r = type_params.r_per_length * candidate.length
            candidate.s_nom = np.sqrt(3) * type_params.i_nom * candidate.v_nom
            candidate.s_nom_max = candidate.s_nom
            candidate.s_nom_min = 0.
            candidate.operative = False
            candidate.name = "{}_c{}".format(candidate.name, c)
            candidates.loc[candidate.name] = candidate

    lines = pd.concat([network.lines, candidates])
    lines.operative = lines .operative.astype('bool')

    return lines.loc[~lines.index.duplicated(keep='first')]


def _corridors(lines):
    """
    From a set of lines, determine unique corridors between
    pairs of buses which describe all connections.

    Parameters
    ----------
    lines : pandas.DataFrame
        For example a set of inoperative lines
        `n.lines.loc[n.lines.operative==False]`.

    Returns
    -------
    corridors : list
        For example `[('Line', 'bus_name_0', 'bus_name_1')]
    """

    if len(lines) > 0:
        return list(lines.apply(lambda ln: ('Line', ln.bus0, ln.bus1), axis=1).unique())
    else:
        return []


def calculate_big_m(network, formulation):
    """
    Determines minimal Big-M parameters.

    Parameters
    ----------
    network : pypsa.Network|pypsa.sub_network
    formulation : string
        Power flow formulation used. E.g. `"angles"` or `"kirchhoff"`.

    Returns
    -------
    big_m : dict
    """

    if formulation == "angles":
        big_m = calculate_big_m_for_angles(network)
    elif formulation == "kirchhoff":
        big_m = calculate_big_m_for_kirchhoff(network)
    else:
        raise NotImplementedError(f"Calculating Big-M for formulation `{formulation}` not implemented.\
                                   Try `angles` or `kirchhoff`.")

    return big_m


def calculate_big_m_for_angles(network, keep_weights=False):
    """
    Determines the minimal Big-M parameters for the `angles` formulation following [1]_.

    Parameters
    ----------
    network : pypsa.Network
    keep_weights : bool
        Keep the weights used for calculating the Big-M parameters.

    Returns
    -------
    big_m : dict
        Keys are lines.

    References
    ----------
    .. [1] S. Binato, M. V. F. Pereira and S. Granville,
           "A new Benders decomposition approach to solve power
           transmission network design problems,"
           in IEEE Transactions on Power Systems,
           vol. 16, no. 2, pp. 235-240, May 2001.
           doi: https://doi.org/10.1109/59.918292
    """

    network.calculate_dependent_values()

    network.lines['big_m_weight'] = network.lines.apply(
        lambda l: l.s_nom * l.x_pu_eff, axis=1)

    candidates = network.lines[(network.lines.operative == False) & (
        network.lines.s_nom_extendable == True)]

    n_graph = network.graph(sel='operative',
                            branch_components=['Line'],
                            weight='big_m_weight')

    big_m = {}
    for name, candidate in candidates.iterrows():
        bus0 = candidate.bus0
        bus1 = candidate.bus1
        if nx.has_path(n_graph, bus0, bus1):
            path_length = nx.dijkstra_path_length(n_graph, bus0, bus1)
            big_m[name] = path_length / candidate.x_pu_eff
        else:
            # fallback if no path exists
            big_m[name] = network.lines.s_nom.sum()

    if not keep_weights:
        network.lines.drop("big_m_weight", axis=1)

    return big_m


def calculate_big_m_for_kirchhoff(network):
    """
    Determines the minimal Big-M parameters for the `kirchhoff` formulation.

    Parameters
    ----------
    network : pypsa.SubNetwork | pypsa.Network

    Returns
    -------
    big_m : dict
        Keys are candidate cycles starting from 0.
    """

    # make sure network has a candidate cycle matrix
    if not hasattr(network, 'CC'):
        find_candidate_cycles(network)

    if '_network' in network.__dict__:
        branches = network.branches(sel='potential')
        branches = branches.loc[_within_sub_network_b(network, branches)]
    else:
        branches = network.passive_branches(sel='potential')

    matrix = network.CC.tocsc()

    big_m = {}
    for col_j in range(matrix.shape[1]):
        cycle_is = matrix.getcol(col_j).nonzero()[0]
        big_m_cycle_i = 0
        for cycle_i in cycle_is:
            b = branches.iloc[cycle_i]
            if b.operative:
                big_m_cycle_i += branches.loc[b.name,
                                              ['x_pu_eff', 's_nom']].product()
            else:
                # take maximum x_pu_eff * s_nom of any one parallel candidate line
                branch_idx = (branches.operative == False) & \
                             (
                                 ((branches.bus0 == b.bus0) & (branches.bus1 == b.bus1)) |
                                 ((branches.bus0 == b.bus1) &
                                  (branches.bus1 == b.bus0))
                )
                big_m_cycle_i += branches.loc[branch_idx,
                                              ['x_pu_eff', 's_nom']].product(axis=1).max()
        big_m[col_j] = RESCALING * big_m_cycle_i

    return big_m


def sub_networks_graph_is_forest(network):
    graph = sub_networks_graph(network)
    return nx.is_forest(nx.OrderedGraph(graph))


def determine_sub_networks_hierarchy(network):
    """
    Allocates a distance to a central sub_network to each sub_network
    representing the level of the sub_network tree it belongs to.
    
    The central sub_network is the sub_network for which the slack constraint
    is kept if all candidate lines are chosen for investment.
    
    The graph of sub_networks must be a forest.
    Otherwise allocating slack dependencies in the 'angles' formulation
    requires considering interdependencies of investments.
    
    Parameters
    ----------
    network : pypsa.Network
    
    Returns
    -------
    hierarchy : dict
        Keys are sub_network names, values are their
        distance to the most central sub_network.
    route : dict
        Keys are sub_network names, values are their
        route to the most central sub_network.
    """

    assert sub_networks_graph_is_forest(network), ("To allocate slack dependencies in the 'angles' formulation without interdependencies "
                                                   "of investments the sub_networks graph must classify as forest (excluding parallel edges).")

    graph = sub_networks_graph(network)
    
    hierarchy = {}
    route = {}
    for nodes in nx.connected_components(graph):
        subgraph = graph.subgraph(nodes)
        
        centralities = nx.closeness_centrality(subgraph)
        central_source = max(centralities, key=centralities.get)
        
        hierarchy.update(nx.single_source_shortest_path_length(subgraph, central_source))
        route.update(nx.single_source_shortest_path(subgraph, central_source))

    return hierarchy, route


def find_slack_dependencies(network):
    """
    Allocates candidate lines connecting two sub_networks to the downstream
    sub_network according to `pypsa.tepopf.determine_sub_networks_hierarchy`
    marking the slack of which sub_network should be
    disregarded if that candidate line is built.

    Parameters
    ----------
    network : pypsa.Network

    Returns
    -------
    slack_dependencies : dict
        A dictionary where keys are sub_network names and
        values are a list of tuples identifying the candidate lines
        associated with this sub_network's slack constraint; e.g. 
        {'0': [('Line', 'c1'),('Line', 'c2')], '1': [('Line','c3')], '2': []}
    """

    if not len(network.sub_networks) > 0:
        network.determine_network_topology()

    candidate_branches = network.passive_branches(sel='candidate')

    hierarchy, _ = determine_sub_networks_hierarchy(network)

    slack_dependencies = {sn_i: [] for sn_i in network.sub_networks.index}

    for cnd_i, cnd in candidate_branches.iterrows():
        sn0 = network.buses.loc[cnd.bus0].sub_network
        sn1 = network.buses.loc[cnd.bus1].sub_network
        if sn0 != sn1:
            allocated_sn = sn0 if hierarchy[sn0] >= hierarchy[sn1] else sn1
            slack_dependencies[allocated_sn].append(cnd_i)

    return slack_dependencies


def define_candidate_cycle_constraints(network, snapshots,
                                       passive_branch_p, passive_branch_inv_p,
                                       passive_branch_inv,
                                       attribute):
    """
    Constructs cycle constraints for candidate cycles
    both of a particular sub_network and across multiple sub_networks.
    """

    big_m = calculate_big_m(network, "kirchhoff")

    cycle_index = []
    cycle_constraints_upper = {}
    cycle_constraints_lower = {}

    if '_network' in network.__dict__:
        branches = network.branches(sel='potential')
        branches = branches.loc[_within_sub_network_b(network, branches)]
        network_name = network.name
    else:
        branches = network.passive_branches(sel='potential')
        network_name = 'main'

    matrix = network.CC.tocsc()

    for col_j in range(matrix.shape[1]):
        cycle_is = matrix.getcol(col_j).nonzero()[0]

        if len(cycle_is) == 0:
            continue

        cycle_index.append((network_name, col_j))

        branch_idx_attributes = []
        branch_inv_idx_attributes = []

        for cycle_i in cycle_is:
            branch_idx = branches.index[cycle_i]
            attribute_value = RESCALING * \
                branches.at[branch_idx, attribute] * network.CC[cycle_i, col_j]
            if branches.at[branch_idx, 'operative']:
                branch_idx_attributes.append((branch_idx, attribute_value))
            else:
                branch_inv_idx_attributes.append((branch_idx, attribute_value))

        for snapshot in snapshots:
            expression_list = [(attribute_value,
                                passive_branch_p[branch_idx[0], branch_idx[1], snapshot])
                               for (branch_idx, attribute_value) in branch_idx_attributes]

            expression_list += [(attribute_value,
                                 passive_branch_inv_p[branch_idx[0], branch_idx[1], snapshot])
                                for (branch_idx, attribute_value) in branch_inv_idx_attributes]

            lhs = LExpression(expression_list)

            rhs = LExpression(variables=[(-big_m[col_j], passive_branch_inv[b]) for b, _ in branch_inv_idx_attributes],
                              constant=len(branch_inv_idx_attributes) * big_m[col_j])

            cycle_constraints_upper[network_name, col_j,
                                    snapshot] = LConstraint(lhs, "<=", rhs)
            cycle_constraints_lower[network_name, col_j,
                                    snapshot] = LConstraint(lhs, ">=", -rhs)

    return (cycle_index, cycle_constraints_upper, cycle_constraints_lower)


def define_integer_branch_extension_variables(network, snapshots):
    """
    Defines binary candidate line investment variables based on 'inoperative' and 'extendable' lines.
    """

    candidate_branches = network.passive_branches(sel='candidate')

    network.model.passive_branch_inv = Var(
        list(candidate_branches.index), domain=Binary)

    free_pyomo_initializers(network.model.passive_branch_inv)


def define_integer_passive_branch_constraints(network, snapshots):
    """
    Capacity constraints of flow variables linked to binary candidate line investment variables.
    """

    candidate_branches = network.passive_branches(sel='candidate')

    s_max_pu = pd.concat({c: get_switchable_as_dense(network, c, 's_max_pu', snapshots)
                          for c in network.passive_branch_components}, axis=1, sort=False)

    flow_upper = {(b[0], b[1], sn): [[(1, network.model.passive_branch_inv_p[b[0], b[1], sn]),
                                      (-s_max_pu.at[sn, b] * candidate_branches.at[b, "s_nom"],
                                       network.model.passive_branch_inv[b[0], b[1]])], "<=", 0]
                  for b in candidate_branches.index
                  for sn in snapshots}

    l_constraint(network.model, "inv_flow_upper", flow_upper,
                 list(candidate_branches.index), snapshots)

    flow_lower = {(b[0], b[1], sn): [[(1, network.model.passive_branch_inv_p[b[0], b[1], sn]),
                                      (s_max_pu.at[sn, b] * candidate_branches.at[b, "s_nom"],
                                       network.model.passive_branch_inv[b[0], b[1]])], ">=", 0]
                  for b in candidate_branches.index
                  for sn in snapshots}

    l_constraint(network.model, "inv_flow_lower", flow_lower,
                 list(candidate_branches.index), snapshots)


def define_rank_constraints(network, snapshots):
    """
    Iterate through candidate line investment option duplicates of the same investment corridor
    and require a distinct order of investment to avoid problem degeneracy.

    Notes
    -----
    A duplicate is identified by the parameters `s_nom`, `x` and `capital_cost`.
    """

    ranks = {}

    candidate_branches = cb = network.passive_branches(sel='candidate')

    corridors = _corridors(candidate_branches)
    for c in corridors:
        corridor_candidates = cb.loc[(cb.bus0 == c[1]) & (cb.bus1 == c[2])]
        for gn, group in corridor_candidates.groupby(['s_nom', 'x', 'capital_cost']):
            if len(group) > 1:
                for i in range(len(group)-1):
                    lhs = LExpression(
                        [(1, network.model.passive_branch_inv[group.iloc[i].name])])
                    rhs = LExpression(
                        [(1, network.model.passive_branch_inv[group.iloc[i+1].name])])
                    ranks[c[0], c[1], c[2], gn[0], gn[1],
                          gn[2], i] = LConstraint(lhs, ">=", rhs)

    l_constraint(network.model, "corridor_rank_constraints",
                 ranks, list(ranks.keys()))


def define_integer_passive_branch_flows(network, snapshots, formulation='angles'):
    """
    Enforce Kirchhoff's Voltage Law only if candidate line is
    invested in by using the disjunctive Big-M reformulation.

    Parameters
    ----------
    network : pypsa.Network
    snapshots : network.snapshots
    formulation : string
        Power flow formulation used; e.g. `"angles"` or `"kirchhoff"`.        
    """

    if formulation == "angles":
        define_integer_passive_branch_flows_with_angles(network, snapshots)
    elif formulation == "kirchhoff":
        define_integer_passive_branch_flows_with_kirchhoff(network, snapshots)


def big_m_slack(network, slack_dependencies=None, keep_weights=False):
    """
    Calculates a big-M parameter for each candidate line that relaxes 
    the a slack constraint in the `angle` formulation.

    The parameter is determined based on the maximum angle difference
    regardless of the investment of other candidate lines.

    A recursive strategy if multiple sub_networks are synchronized; adds
    maximum big-M parameter of upstream sub_network to all big-M parameters
    of the downstream sub_network (where the slack is relaxed).

    These are not minimal values (detour via slack bus of intermediate
    sub_networks even if shorter path available), but low computational
    effort to calculate and guarantee non-binding slack constraint.

    Parameters
    ----------
    network : pypsa.Network
    slack_dependencies : dict
        Output of function `pypsa.tepopf.find_slack_dependencies(network)`.
    keep_weights : bool
        Keep the weights used for calculating the Big-M parameters in `network.lines`.

    Returns
    -------
    big_m : dict
        Keys are candidate lines that synchronize sub_networks.
    """

    network.calculate_dependent_values()

    if slack_dependencies is None:
        slack_dependencies = find_slack_dependencies(network)

    if not len(network.sub_networks) > 0:
        network.determine_network_topology()

    if any(network.sub_networks.slack_bus == ""):
        for sn in network.sub_networks.obj:
            find_slack_bus(sn)

    hierarchy, route = determine_sub_networks_hierarchy(network)
    hierarchy = pd.Series(hierarchy).sort_values()

    network.lines['big_m_weight'] = network.lines.apply(
        lambda l: l.s_nom * l.x_pu_eff, axis=1)

    n_graph = network.graph(sel='operative',
                            branch_components=['Line'],
                            weight='big_m_weight')

    big_m = {}
    for idx, rank in hierarchy.iteritems():
        for cnd_i in slack_dependencies[idx]:
            cnd = network.lines.loc[cnd_i[1]]
            sn0 = network.buses.loc[cnd.bus0].sub_network
            sn1 = network.buses.loc[cnd.bus1].sub_network
            slack0 = network.sub_networks.slack_bus[sn0]
            slack1 = network.sub_networks.slack_bus[sn1]
            
            n_graph.add_edge(cnd.bus0, cnd.bus1,
                            weight=cnd.x_pu_eff * cnd.s_nom)
            path_length = nx.dijkstra_path_length(n_graph, slack0, slack1)
            big_m[cnd_i] = path_length
            
            inbetween_sub_networks = route[idx][1:-1]
            if len(inbetween_sub_networks) > 0:
                predecessor = inbetween_sub_networks[-1]
                big_m[cnd_i] += max([big_m[line]
                                    for line in slack_dependencies[predecessor]])
            
            n_graph.remove_edge(cnd.bus0, cnd.bus1)

    if not keep_weights:
        network.lines.drop("big_m_weight", axis=1)

    return big_m


def define_integer_slack_angle(network, snapshots):

    slack_dependencies = find_slack_dependencies(network)
    big_m = big_m_slack(network, slack_dependencies=slack_dependencies)

    slack_upper = {}
    slack_lower = {}
    for sub, lines in slack_dependencies.items():
        for sn in snapshots:
            lhs = LExpression(
                [(1, network.model.voltage_angles[network.sub_networks.slack_bus[sub], sn])])
            rhs = LExpression(
                [(big_m[l], network.model.passive_branch_inv[l]) for l in lines])
            slack_upper[sub, sn] = LConstraint(lhs, "<=", rhs)
            slack_lower[sub, sn] = LConstraint(lhs, ">=", -rhs)

    l_constraint(network.model, "slack_angle_upper", slack_upper,
                 list(network.sub_networks.index), snapshots)
    l_constraint(network.model, "slack_angle_lower", slack_lower,
                 list(network.sub_networks.index), snapshots)


def define_integer_passive_branch_flows_with_angles(network, snapshots):
    """
    Enforce Kirchhoff's Second Law with angles formulation only if invested with Big-M reformulation.
    """

    candidate_branches = network.passive_branches(sel='candidate')

    network.model.passive_branch_inv_p = Var(
        list(candidate_branches.index), snapshots)

    big_m = calculate_big_m(network, "angles")

    flows_upper = {}
    flows_lower = {}
    for branch in candidate_branches.index:
        bus0 = candidate_branches.at[branch, "bus0"]
        bus1 = candidate_branches.at[branch, "bus1"]
        bt = branch[0]
        bn = branch[1]
        sub = candidate_branches.at[branch, "sub_network"]
        attribute = "r_pu_eff" if network.sub_networks.at[sub,
                                                          "carrier"] == "DC" else "x_pu_eff"
        y = 1 / candidate_branches.at[branch, attribute]
        for sn in snapshots:
            lhs = LExpression([(y, network.model.voltage_angles[bus0, sn]),
                               (-y, network.model.voltage_angles[bus1, sn]),
                               (-1, network.model.passive_branch_inv_p[bt, bn, sn])],
                              -y*(candidate_branches.at[branch, "phase_shift"]*np.pi/180. if bt == "Transformer" else 0.))
            rhs = LExpression(variables=[(-big_m[bn], network.model.passive_branch_inv[bt, bn])],
                              constant=big_m[bn])
            flows_upper[bt, bn, sn] = LConstraint(lhs, "<=", rhs)
            flows_lower[bt, bn, sn] = LConstraint(lhs, ">=", -rhs)

    l_constraint(network.model, "passive_branch_inv_p_upper_def", flows_upper,
                 list(candidate_branches.index), snapshots)

    l_constraint(network.model, "passive_branch_inv_p_lower_def", flows_lower,
                 list(candidate_branches.index), snapshots)


def define_integer_passive_branch_flows_with_kirchhoff(network, snapshots):
    """
    Enforce Kirchhoff's Second Law with angles formulation only if invested with Big-M reformulation.
    """

    find_candidate_cycles(network)

    for sub_network in network.sub_networks.obj:
        # bus_controls and B H calculation to be done ex-post given optimised candidate investment decisions!
        find_cycles(sub_network)
        find_candidate_cycles(sub_network)

    candidate_branches = network.passive_branches(sel='candidate')

    network.model.passive_branch_inv_p = Var(
        list(candidate_branches.index), snapshots)

    pb_p = network.model.passive_branch_p
    pb_inv_p = network.model.passive_branch_inv_p
    pb_inv = network.model.passive_branch_inv

    cycle_index = []
    cycle_constraints_upper = {}
    cycle_constraints_lower = {}

    n_cycle_index, n_cycle_constraints_upper, n_cycle_constraints_lower = \
        define_candidate_cycle_constraints(
            network, snapshots, pb_p, pb_inv_p, pb_inv, 'x_pu_eff')

    cycle_index.extend(n_cycle_index)
    cycle_constraints_upper.update(n_cycle_constraints_upper)
    cycle_constraints_lower.update(n_cycle_constraints_lower)

    for sub_network in network.sub_networks.obj:

        attribute = "r_pu_eff" if network.sub_networks.at[sub_network.name,
                                                          "carrier"] == "DC" else "x_pu_eff"

        subn_cycle_index, subn_cycle_constraints_upper, subn_cycle_constraints_lower = \
            define_candidate_cycle_constraints(
                sub_network, snapshots, pb_p, pb_inv_p, pb_inv, attribute)

        cycle_index.extend(subn_cycle_index)
        cycle_constraints_upper.update(subn_cycle_constraints_upper)
        cycle_constraints_lower.update(subn_cycle_constraints_lower)

    l_constraint(network.model, "cycle_constraints_upper", cycle_constraints_upper,
                 cycle_index, snapshots)

    l_constraint(network.model, "cycle_constraints_lower", cycle_constraints_lower,
                 cycle_index, snapshots)


def define_integer_nodal_balance_constraints(network, snapshots):
    """
    Identical to `pypsa.opf.define_nodal_balance_constraints` but including candidate line flow variables.
    """

    passive_branches = network.passive_branches(sel='operative')

    for branch in passive_branches.index:
        bus0 = passive_branches.at[branch, "bus0"]
        bus1 = passive_branches.at[branch, "bus1"]
        bt = branch[0]
        bn = branch[1]
        for sn in snapshots:
            network._p_balance[bus0, sn].variables.append(
                (-1, network.model.passive_branch_p[bt, bn, sn]))
            network._p_balance[bus1, sn].variables.append(
                (1, network.model.passive_branch_p[bt, bn, sn]))

    candidate_branches = network.passive_branches(sel='candidate')

    for branch in candidate_branches.index:
        bus0 = candidate_branches.at[branch, "bus0"]
        bus1 = candidate_branches.at[branch, "bus1"]
        bt = branch[0]
        bn = branch[1]
        for sn in snapshots:
            network._p_balance[bus0, sn].variables.append(
                (-1, network.model.passive_branch_inv_p[bt, bn, sn]))
            network._p_balance[bus1, sn].variables.append(
                (1, network.model.passive_branch_inv_p[bt, bn, sn]))

    power_balance = {k: LConstraint(v, "==", LExpression())
                     for k, v in iteritems(network._p_balance)}

    l_constraint(network.model, "power_balance", power_balance,
                 list(network.buses.index), snapshots)


def network_teplopf_build_model(network, snapshots=None, skip_pre=False,
                                formulation="angles"):
    """
    Build pyomo model for transmission expansion planning version of 
    linear optimal power flow for a group of snapshots.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    skip_pre: bool, default False
        Skip the preliminary steps of computing topology, calculating
        dependent values and finding bus controls.
    formulation : string
        Formulation of the linear power flow equations to use; must be
        one of ["angles","kirchhoff"]

    Returns
    -------
    network.model : pyomo.core.base.PyomoModel.ConcreteModel
    """

    if len(network.passive_branches(sel='candidate')) == 0:
        logger.info("No candidate lines given. Building model with `pypsa.opf.network_lopf_build_model` instead!")
        return network_lopf_build_model(network, snapshots, skip_pre, formulation)

    if not skip_pre:
        network.determine_network_topology()
        calculate_dependent_values(network)
        for sub_network in network.sub_networks.obj:
            find_slack_bus(sub_network)
        logger.info("Performed preliminary steps")

    snapshots = _as_snapshots(network, snapshots)

    logger.info("Building pyomo model using `%s` formulation", formulation)
    network.model = ConcreteModel(
        "Linear Optimal Power Flow for Transmission Expansion Planning")

    define_generator_variables_constraints(network, snapshots)

    define_storage_variables_constraints(network, snapshots)

    define_store_variables_constraints(network, snapshots)

    define_branch_extension_variables(network, snapshots)
    define_integer_branch_extension_variables(network, snapshots)

    define_rank_constraints(network, snapshots)

    define_link_flows(network, snapshots)

    define_nodal_balances(network, snapshots)

    define_passive_branch_flows(network, snapshots, formulation)
    define_integer_passive_branch_flows(network, snapshots, formulation)

    if formulation == "angles" and sub_networks_graph_is_forest(network):
        # skip if complex synchronisation hierarchy present
        # which introduces potentially thwarting rotational degeneracy!
        define_integer_slack_angle(network, snapshots)

    define_passive_branch_constraints(network, snapshots)
    define_integer_passive_branch_constraints(network, snapshots)

    if formulation in ["angles", "kirchhoff"]:
        define_integer_nodal_balance_constraints(network, snapshots)

    define_global_constraints(network, snapshots)

    define_linear_objective(network, snapshots, candidate_lines=True)

    # tidy up auxilliary expressions
    del network._p_balance

    # force solver to also give us the dual prices
    network.model.dual = Suffix(direction=Suffix.IMPORT)

    return network.model


def network_teplopf(network, snapshots=None, solver_name="glpk", solver_io=None,
                    skip_pre=False, extra_functionality=None, solver_logfile=None, solver_options={},
                    keep_files=False, formulation="angles",
                    free_memory={}, extra_postprocessing=None,
                    infer_candidates=False):
    """
    Transmission Expansion Planning with linear optimal power flow for a group of snapshots.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    solver_name : string
        Must be a solver name that pyomo recognises and that is
        installed, e.g. "glpk", "gurobi"
    solver_io : string, default None
        Solver Input-Output option, e.g. "python" to use "gurobipy" for
        solver_name="gurobi"
    skip_pre: bool, default False
        Skip the preliminary steps of computing topology, calculating
        dependent values and finding bus controls.
    extra_functionality : callable function
        This function must take two arguments
        `extra_functionality(network,snapshots)` and is called after
        the model building is complete, but before it is sent to the
        solver. It allows the user to
        add/change constraints and add/change the objective function.
    solver_logfile : None|string
        If not None, sets the logfile option of the solver.
    solver_options : dictionary
        A dictionary with additional options that get passed to the solver.
        (e.g. {'threads':2} tells gurobi to use only 2 cpus)
    keep_files : bool, default False
        Keep the files that pyomo constructs from OPF problem
        construction, e.g. .lp file - useful for debugging
    formulation : string
        Formulation of the linear power flow equations to use; must be
        one of ["angles","cycles","kirchhoff","ptdf"]
    ptdf_tolerance : float
        Value below which PTDF entries are ignored
    free_memory : set, default {'pyomo'}
        Any subset of {'pypsa', 'pyomo'}. Allows to stash `pypsa` time-series
        data away while the solver runs (as a pickle to disk) and/or free
        `pyomo` data after the solution has been extracted.
    extra_postprocessing : callable function
        This function must take three arguments
        `extra_postprocessing(network,snapshots,duals)` and is called after
        the model has solved and the results are extracted. It allows the user to
        extract further information about the solution, such as additional shadow prices.
    infer_candidates : bool
        Indicator whether candidate lines should be inferred
        based on potential and line type using
        `pypsa.tepopf.infer_candidates_from_existing()`.

    Returns
    -------
    status : string
    termination_condition : string
    """

    if infer_candidates:
        network.lines = infer_candidates_from_existing(network)

    snapshots = _as_snapshots(network, snapshots)

    network_teplopf_build_model(
        network, snapshots, skip_pre=False, formulation=formulation)

    if extra_functionality is not None:
        extra_functionality(network, snapshots)

    network_lopf_prepare_solver(network, solver_name=solver_name,
                                solver_io=solver_io)

    status, termination_condition = network_lopf_solve(network, snapshots, formulation=formulation,
                                                       solver_logfile=solver_logfile, solver_options=solver_options,
                                                       keep_files=keep_files, free_memory=free_memory,
                                                       extra_postprocessing=extra_postprocessing,
                                                       candidate_lines=True)

    return status, termination_condition
