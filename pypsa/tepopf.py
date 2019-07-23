## Copyright 2019 Fabian Neumann (KIT)

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

"""Optimal Power Flow functions with Integer Transmission Expansion Planning.
"""

# TODO: make insensitive to order of bus0 bus1
# at the moment it is assumed corridors have unique order from bus0 to bus1

# TODO: duals in mixed-integer programming?

# TODO: unify candidates arguments in functions {exclusive_candidates, candidates}
# possibly allow string an boolean, where boolean has some default behaviour (e.g. exclusive)
# e.g. extracting optimisation results varies whether investment is exclusive or not.

# TODO: discuss whether moving networkclustering utilities to utils is acceptable (used in .tepopf)
# should these maybe be in .descriptors?
# haversine function seems to be duplicated in .geo and (now) .utils?

# TODO: discuss whether it is better to duplicate some code from opf.py in tepopf.py
# or whether it is also recommendable to avoid code duplication through if / else in opf.py functions

# TODO: write tests for tepopf()

# TODO: add line_selector = 'operable' : either candidate or existing
# adapt functionality to presence of lines that are not operative and not extendable

# TODO: what happens if separate subnetworks are connected through candidate lines?
# **angles:**
# say a candidate line connects 2 sub_networks, then
# (1) treat them as one sub_network (for calculating K, B, H, etc.)
# (2) enforce slack theta_0 = 0 in one sub_network only if bridging line is not invested in
#     use 2*pi~=6.3 as big-M as it is a voltage angle in radians
# (3) gets more complicated if there are more sub_networks joining; need an order of slacks in an
#     investment group (e.g. individual subnetworks that could join through candidates),
#     lower ranking slack constraint is coupled to investment of connecting candidate line
# **kirchhoff:**
# no need to worry;
# just treat possibly connected subnetworks as one sub_network;
# just calculate B and H matrices based on candidate investment decisions ex-post!

# TODO: handling inoperative but not-extendable lines

# TODO: infer candidates should ignore operative but extendable lines
# (two consecutive runs give different results if inferred in both cases)

# make the code as Python 3 compatible as possible
from __future__ import division, absolute_import
from six import iteritems, itervalues, string_types

__author__ = "Fabian Neumann (KIT)"
__copyright__ = "Copyright 2019 Fabian Neumann (KIT), GNU GPL 3"

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
                  define_nodal_balance_constraints,
                  define_sub_network_balance_constraints,
                  define_global_constraints,
                  define_linear_objective,
                  extract_optimisation_results,
                  network_lopf_prepare_solver,
                  network_lopf_solve)

from .pf import (_as_snapshots, calculate_dependent_values, find_slack_bus, find_cycles)

from .opt import (free_pyomo_initializers, l_constraint, LExpression, LConstraint)

from .descriptors import (get_switchable_as_dense, get_switchable_as_iter)

from .utils import (_make_consense, _haversine, _normed)


def _corridors(lines):
    """
    From a set of lines, determine unique corridors between
    pairs of buses which describe all connections.

    Parameters
    ----------
    lines : pandas.DataFrame
        For example a set of candidate lines
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


def infer_candidates_from_existing(network, exclusive_candidates=True):
    """
    Infer candidate lines from existing lines.

    Parameters
    ----------
    network : pypsa.Network
    exclusive_candidates : bool
        Indicator whether individual candidate lines should be 
        transformed into combinations of investments

    Returns
    -------
    network.lines : pandas.DataFrame
    """

    network.calculate_dependent_values()

    network.lines = add_candidate_lines(network)

    if exclusive_candidates:
        network.lines = candidate_lines_to_investment(network)

    # extendability is transferred to candidate lines
    network.lines.loc[network.lines.operative, 's_nom_extendable'] = False

    return network.lines


def potential_num_parallels(network):
    """
    Determine the number of allowable additional parallel circuits
    per line based on `s_nom_extendable`, the difference between
    `s_nom_max` and `s_nom`, and the `type` if specified.
    Otherwise, the line type will be inferred from `num_parallel`
    and its electrical parameters.

    Parameters
    ----------
    network : pypsa.Network

    Returns
    -------
    pandas.Series
    """

    ext_lines = network.lines[network.lines.s_nom_extendable & network.lines.operative]

    # TODO: derive line type from num_parallel, s_nom, r, x if no valid line type given
    assert pd.Series([lt in network.line_types.index for lt in ext_lines.type]).all(), "Currently all extendable lines must have a `type` in TEP-LOPF"
    
    assert (ext_lines.s_nom_max!=np.inf).all(), "TEP-LOPF requires `s_nom_max` to be a finite number"

    ext_lines.s_nom_max = ext_lines.s_nom_max.apply(np.ceil) # to avoid rounding errors
    investment_potential = ext_lines.s_nom_max - ext_lines.s_nom
    unit_s_nom = np.sqrt(3) * ext_lines.type.map(network.line_types.i_nom) * ext_lines.v_nom
    num_parallels = investment_potential.divide(unit_s_nom).map(np.floor).map(int)
    
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
    c_sets = num_parallels.apply(lambda num: np.arange(1,num+1))

    candidates = pd.DataFrame(columns=network.lines.columns)
    
    for ind, c_set in c_sets.iteritems():
        for c in c_set:
            candidate = network.lines.loc[ind].copy()
            type_params = network.line_types.loc[candidate.type]
            candidate.num_parallel = 1.
            candidate.x = type_params.x_per_length * candidate.length
            candidate.r = type_params.r_per_length * candidate.length
            candidate.s_nom = np.sqrt(3) * type_params.i_nom * candidate.v_nom
            candidate.s_nom_max = candidate.s_nom
            candidate.s_nom_min = 0.
            candidate.operative = False
            candidate.name = "{}_c{}".format(candidate.name,c)
            candidates.loc[candidate.name] = candidate

    lines = pd.concat([network.lines,candidates])
    lines.operative = lines .operative.astype('bool')

    return  lines.loc[~lines.index.duplicated(keep='first')]


# this is an adapted version from pypsa.networkclustering
def aggregate_candidates(network, l):
    """
    Aggregates multiple lines into a single line.

    Parameters
    ----------
    l : pandas.DataFrame
        Dataframe of lines which are to be aggregated.

    Returns
    -------
    pandas.Series
    """
    
    attrs = network.components["Line"]["attrs"]
    columns = set(attrs.index[attrs.static & attrs.status.str.startswith('Input')]).difference(('name',))
    
    consense = {
        attr: _make_consense('Bus', attr)
        for attr in (columns | {'sub_network'}
                        - {'r', 'x', 'g', 'b', 'terrain_factor', 's_nom',
                        's_nom_min', 's_nom_max', 's_nom_extendable',
                        'length', 'v_ang_min', 'v_ang_max'})
    }

    line_length_factor = 1.0
    buses = l.iloc[0][['bus0', 'bus1']].values
    length_s = _haversine(network.buses.loc[buses,['x', 'y']])*line_length_factor
    v_nom_s = network.buses.loc[buses,'v_nom'].max()

    voltage_factor = (np.asarray(network.buses.loc[l.bus0,'v_nom'])/v_nom_s)**2
    length_factor = (length_s/l['length'])

    data = dict(
        r=1./(voltage_factor/(length_factor * l['r'])).sum(),
        x=1./(voltage_factor/(length_factor * l['x'])).sum(),
        g=(voltage_factor * length_factor * l['g']).sum(),
        b=(voltage_factor * length_factor * l['b']).sum(),
        terrain_factor=l['terrain_factor'].mean(),
        s_nom=l['s_nom'].sum(),
        s_nom_min=l['s_nom_min'].sum(),
        s_nom_max=l['s_nom_max'].sum(),
        s_nom_extendable=l['s_nom_extendable'].any(),
        num_parallel=l['num_parallel'].sum(),
        capital_cost=(length_factor * _normed(l['s_nom']) * l['capital_cost']).sum(),
        length=length_s,
        sub_network=consense['sub_network'](l['sub_network']),
        v_ang_min=l['v_ang_min'].max(),
        v_ang_max=l['v_ang_max'].min()
    )

    data.update((f, consense[f](l[f])) for f in columns.difference(data))

    return pd.Series(data, index=[f for f in l.columns if f in columns])


def get_investment_combinations(candidate_group):
    """
    Find all possible investment combinations from a set of candidate
    lines that connects the same pair of buses.

    Parameters
    ----------
    candidate_group : pandas.DataFrame
        Group of candidate lines connecting the same pair of buses.

    Returns
    -------
    combinations : list
        For example `[["cand1", "cand2"], ["cand1"], ["cand2"]]`
        where list contents are values from `candidate_group.index`.
    """

    for bus in ['bus0', 'bus1']:
        assert len(candidate_group[bus].unique()) == 1

    combinations = []

    for r in range(1, len(candidate_group.index)+1):
        for i in itertools.combinations(candidate_group.index, r):
            combinations.append(list(i))

    return combinations


# TODO: need to ensure unique order, possibly by sorting, cf. networkclustering
def candidate_lines_to_investment(network):
    """
    Merge combinations of candidate lines to
    candididate investment combinations.

    Parameters
    ----------
    network : pypsa.Network

    Returns
    -------
    lines : pandas.DataFrame
    """
    
    lines = network.lines
    candidate_lines = lines[lines.operative==False]
    candidate_inv = pd.DataFrame(columns=lines.columns)
    candidate_inv.astype(lines.dtypes)
    
    for name, group in candidate_lines.groupby(['bus0', 'bus1']):
        combinations = get_investment_combinations(group)
        for c in combinations:
            candidate_block = group.loc[c]
            cinv = aggregate_candidates(network, candidate_block)
            names = pd.Series(c).apply(lambda x: x.split('_'))
            cinv.name = ("{}"+"_{}"*len(c)).format(names.iloc[0][0], *names.apply(lambda x: x[1]))
            candidate_inv.loc[cinv.name] = cinv

    updated_lines = pd.concat([lines[lines.operative], candidate_inv.drop_duplicates()])
    updated_lines.operative = updated_lines.operative.astype('bool')

    return updated_lines


def bigm(n, formulation):
    """
    Determines the minimal Big-M parameters.

    Parameters
    ----------
    n : pypsa.Network|pypsa.SubNetwork
    formulation : string
        Power flow formulation used. E.g. `"angles"` or `"kirchhoff"`.

    Returns
    -------
    m : dict
    """

    if formulation == "angles":
        m = bigm_for_angles(n)
    elif formulation == "kirchhoff":
        m = bigm_for_kirchhoff(n)
    else:
        raise NotImplementedError("Calculating Big-M for formulation `{}` not implemented.\
                                   Try `angles` or `kirchhoff`.")

    return m

# TODO: connecting sub_networks
def bigm_for_angles(n, keep_weights=False):
    """
    Determines the minimal Big-M parameters for the `angles` formulation following [1]_.

    Parameters
    ----------
    n : pypsa.Network
    keep_weights : bool
        Keep the weights used for calculating the Big-M parameters.

    Returns
    -------
    m : dict
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

    n.calculate_dependent_values()

    n.lines['bigm_weight'] = n.lines.apply(lambda l: l.s_nom * l.x_pu_eff, axis=1)

    candidates = n.lines[n.lines.operative==False]

    ngraph = n.graph(line_selector='operative', branch_components=['Line'], weight='bigm_weight')

    m = {}
    for name, candidate in candidates.iterrows():
        if nx.has_path(ngraph, candidate.bus0, candidate.bus1):
            path_length = nx.dijkstra_path_length(ngraph, candidate.bus0, candidate.bus1)
            m[name] = path_length / candidate.x_pu_eff 
        else:
            # no path through existing network
            # Binato proposes solving non-polynomial longest path problem
            m[name] = 4 * np.pi / candidate.x_pu_eff + candidate.s_nom

    if not keep_weights:
        n.lines.drop("bigm_weight", axis=1)

    return m


def bigm_for_kirchhoff(sub_network):
    """
    Determines the minimal Big-M parameters for the `kirchhoff` formulation.

    Parameters
    ----------
    sub_network : pypsa.SubNetwork

    Returns
    -------
    m : dict
        Keys are candidate cycles starting from 0.
    """

    # make sure sub_network has a candidate cycle matrix
    if not hasattr(sub_network, 'CC'):
        find_candidate_cycles(sub_network)


    branches = sub_network.branches(line_selector='potential')
    matrix = sub_network.CC.tocsc()

    m = {}
    for col_j in range(matrix.shape[1]):
        cycle_is = matrix.getcol(col_j).nonzero()[0]
        
        bigm_cycle_i = 0
        for cycle_i in cycle_is:
            b = branches.iloc[cycle_i]
            if b.operative:
                branch_idx = b.name
                bigm_cycle_i += branches.loc[branch_idx, ['x_pu_eff', 's_nom']].product()
            else:
                branch_idx = ( branches.operative==False ) & \
                             ( 
                                ( (branches.bus0==b.bus0) & (branches.bus1==b.bus1) ) | \
                                ( (branches.bus0==b.bus1) & (branches.bus1==b.bus0) )
                             )
                bigm_cycle_i += branches.loc[branch_idx, ['x_pu_eff', 's_nom']].product(axis=1).max()
            
        m[col_j] = 1e5 * bigm_cycle_i

    return m

def find_candidate_cycles(sub_network):
    """
    Constructs an additional cycle matrix based cycles added by
    candidate lines and records them in sub_network.CC.
    """

    pot_sub_lines = sub_network.branches(line_selector='potential')
    cand_sub_lines = sub_network.branches(line_selector='candidate')

    cand_edges = cand_sub_lines.apply(lambda x: (x.bus0, x.bus1, x.name), axis=1)
    cand_weight = len(pot_sub_lines)
    
    weights = pot_sub_lines.apply(lambda x: [(x.bus0, x.bus1, x.name), 1 if x.operative else cand_weight], result_type='expand', axis=1)
    weights.columns = ['index', 'weight']
    weights.set_index('index', inplace=True)
    weights = weights.to_dict()['weight']
    
    mgraph = sub_network.graph(inf_weight=False, line_selector='potential')
    nx.set_edge_attributes(mgraph, weights, name='weight')
    
    def equivalent_cycle(c,d):
        dc, dd = (deque(c), deque(d))
        dd_rev = dd.copy()
        dd_rev.reverse()
        for i in range(len(dd)):
            dd.rotate(1)
            dd_rev.rotate(1)
            if dd==dc or dd_rev==dc:
                return True         
        return False
    
    cycles = {}
    for candidate in cand_edges:
        mgraph.remove_edge(*candidate)
        if nx.has_path(mgraph, candidate[0], candidate[1]):
            cycle = nx.dijkstra_path(mgraph, candidate[0], candidate[1])

            def add_cycle_b():
                for k, v in cycles.items():
                    if k[:2] != candidate[:2] and equivalent_cycle(v,cycle):
                        return False
                return True

            if add_cycle_b():
                cycles[candidate] = cycle
        mgraph.add_edge(*candidate, weight=weights[candidate])
    
    branches_bus0 = pot_sub_lines['bus0']
    branches_i = branches_bus0.index
    
    sub_network.CC = dok_matrix((len(branches_bus0),len(cycles)))

    for j, (candidate, cycle)  in enumerate(iteritems(cycles)):
        for i in range(len(cycle)-1):
            branch, weight = (None, np.inf)
            corridor_branches = mgraph[cycle[i]][cycle[i+1]]
            for k,v in iteritems(corridor_branches):
                if v['weight'] < weight or branch is None:
                    branch = k 
                    weight = v['weight']
            branch_i = branches_i.get_loc(branch)
            sign = +1 if branches_bus0.iat[branch_i] == cycle[i] else -1
            sub_network.CC[branch_i,j] += sign

        branch_i = branches_i.get_loc(candidate[2])
        sub_network.CC[branch_i,j] += -1


def find_slack_dependencies(network):
    """
    Allocates candidate lines connecting two subnetworks to the one with
    the lower order that defines the slack of which subnetwork should be
    disregarded if that candidate line is built.
    
    Parameters
    ----------
    network : pypsa.Network
    
    Returns
    -------
    slack_dependencies : dict
        A dictionary where keys are sub_network names and
        values are a list of tuples identifying the candidate lines
        associated with this subnetwork's slack constraint; e.g. 
        {'0': [('Line', 'c1'),('Line', 'c1')], '1': [('Line','c3')], '2': []}
    """

    if not len(network.sub_networks) > 0:
        network.determine_network_topology()

    candidate_branches = network.passive_branches(sel='candidate')

    slack_dependencies = {sn_i: [] for sn_i in network.sub_networks.index}
    
    for cnd_i, cnd in candidate_branches.iterrows():
        sn0 = network.buses.loc[cnd.bus0].sub_network
        sn1 = network.buses.loc[cnd.bus1].sub_network
        if sn0 != sn1:
            order = lambda sn: len(network.sub_networks.loc[sn].obj.buses())
            allocated_sn = sn0 if order(sn0) <= order(sn1) else sn1
            slack_dependencies[allocated_sn].append(cnd_i)

    return slack_dependencies


def define_sub_network_candidate_cycle_constraints(subnetwork, snapshots,
                                                   passive_branch_p, passive_branch_inv_p,
                                                   passive_branch_inv,
                                                   attribute):
    """
    Constructs cycle constraints for candidate cycles
    of a particular subnetwork.
    """

    big_m = bigm(subnetwork, "kirchhoff")

    subn_cycle_index = []
    subn_cycle_constraints_upper = {}
    subn_cycle_constraints_lower = {}

    matrix = subnetwork.CC.tocsc()
    branches = subnetwork.branches(line_selector='potential')

    for col_j in range( matrix.shape[1] ):
        cycle_is = matrix.getcol(col_j).nonzero()[0]

        if len(cycle_is) == 0: continue

        subn_cycle_index.append((subnetwork.name, col_j))

        branch_idx_attributes = []
        branch_inv_idx_attributes = []
        candidates_idx = []

        for cycle_i in cycle_is:
            branch_idx = branches.index[cycle_i]
            attribute_value = 1e5 * branches.at[branch_idx, attribute] * subnetwork.CC[ cycle_i, col_j]
            if branches.at[branch_idx,'operative']:
                branch_idx_attributes.append((branch_idx, attribute_value))
            else:
                candidates_idx.append(branch_idx)
                corridor_idx = ('Line', branches.at[branch_idx,'bus0'], branches.at[branch_idx,'bus1'])
                branch_inv_idx_attributes.append((corridor_idx, attribute_value))

        for snapshot in snapshots:
            expression_list = [ (attribute_value,
                                 passive_branch_p[branch_idx[0], branch_idx[1], snapshot])
                                 for (branch_idx, attribute_value) in branch_idx_attributes]

            expression_list += [ (attribute_value,
                                 passive_branch_inv_p[corr_idx[0], corr_idx[1], corridor_idx[2], snapshot])
                                 for (corr_idx, attribute_value) in branch_inv_idx_attributes]

            lhs = LExpression(expression_list)

            rhs = LExpression(variables= [ (-big_m[col_j], passive_branch_inv[c]) for c in candidates_idx],
                              constant= len(candidates_idx) * big_m[col_j] )

            subn_cycle_constraints_upper[subnetwork.name, col_j, snapshot] = LConstraint(lhs,"<=",rhs)
            subn_cycle_constraints_lower[subnetwork.name, col_j, snapshot] = LConstraint(lhs,">=",-rhs)

    return (subn_cycle_index, subn_cycle_constraints_upper, subn_cycle_constraints_lower)

def assert_candidate_kvl_duals(network):
    """
    Assert that all KVL constraints of candidate lines are
    non-binding if they are not invested in. If this check fails,
    Big-M parameters must be larger. 

    Parameters
    ----------
    network : pypsa.Network

    Returns
    -------
    None
    """

    pass


def define_integer_branch_extension_variables(network, snapshots):
    """
    Defines candidate line investment variables based on 'inoperative' and 'extendable' lines.
    """

    candidate_passive_branches = network.passive_branches(sel='candidate')

    network.model.passive_branch_inv = Var(list(candidate_passive_branches.index),
                                           domain=Binary)

    free_pyomo_initializers(network.model.passive_branch_inv)


# TODO: multiple flow variables (one per candidate line)
def define_integer_passive_branch_constraints(network, snapshots): 
    """
    Capacity constraints of investment corridor flows.
    There is currently only one flow variable for all candidate lines per corridor.
    """

    candidate_branches = cb = network.passive_branches(sel='candidate')

    s_max_pu = pd.concat({c : get_switchable_as_dense(network, c, 's_max_pu', snapshots)
                          for c in network.passive_branch_components}, axis=1, sort=False)

    investment_corridors = _corridors(candidate_branches)

    flow_upper = {(*c,sn) : [[(1,network.model.passive_branch_inv_p[c,sn]),
                            *[(
                                -s_max_pu.at[sn,b] * candidate_branches.at[b,"s_nom"],
                                network.model.passive_branch_inv[b[0],b[1]])
                            for b, bd in cb.loc[(cb.bus0==c[1]) & (cb.bus1==c[2])].iterrows()]
                            ],"<=",0]
                  for c in investment_corridors
                  for sn in snapshots}

    l_constraint(network.model, "inv_flow_upper", flow_upper,
                 investment_corridors, snapshots)

    flow_lower = {(*c,sn): [[(1,network.model.passive_branch_inv_p[c,sn]),
                            *[(
                                s_max_pu.at[sn,b] * candidate_branches.at[b,"s_nom"],
                                network.model.passive_branch_inv[b[0],b[1]])
                            for b, bd in cb.loc[(cb.bus0==c[1]) & (cb.bus1==c[2])].iterrows()]
                            ],">=",0]
                   for c in investment_corridors
                   for sn in snapshots}

    l_constraint(network.model, "inv_flow_lower", flow_lower,
                 investment_corridors, snapshots)


def define_rank_constraints(network, snapshots):
    """
    Iterate through candidate line duplicates of the same investment corridor
    and require a distinct order of investment to avoid problem degeneracy.

    Notes
    -----
    A duplicate is identified by the parameters `s_nom`, `x` and `capital_cost`.
    """

    ranks = {}

    candidate_branches = cb = network.passive_branches(sel='candidate')

    corridors = _corridors(candidate_branches)
    for c in corridors:
        corridor_candidates = cb.loc[(cb.bus0==c[1]) & (cb.bus1==c[2])]
        for gn, group in corridor_candidates.groupby(['s_nom','x', 'capital_cost']):
            if len(group) > 1:
                for i in range(len(group)-1):
                    lhs = LExpression([(1,network.model.passive_branch_inv[group.iloc[i].name])])
                    rhs = LExpression([(1,network.model.passive_branch_inv[group.iloc[i+1].name])])
                    ranks[c[0],c[1],c[2],gn[0],gn[1],gn[2],i] = LConstraint(lhs,">=",rhs)
                    
    l_constraint(network.model, "corridor_rank_constraints", ranks, list(ranks.keys()))


def define_exclusive_candidates_constraints(network, snapshots):
    """
    Only one candidate line investment can be selected per corridor.
    """

    extendable_branches = eb = network.passive_branches(sel='candidate')

    investment_corridors = _corridors(extendable_branches)

    investment_groups = {c : [[
            *[(1,network.model.passive_branch_inv[b[0],b[1]])
            for b, bd in eb.loc[(eb.bus0==c[1]) & (eb.bus1==c[2])].iterrows()]
            ], "<=", 1] for c in investment_corridors}

    l_constraint(network.model, "investment_groups",
                 investment_groups, investment_corridors)


def define_integer_passive_branch_flows(network, snapshots, formulation='angles'):
    """
    Enforce Kirchhoff's Second Law only if candidate line is
    invested in using the disjunctive Big-M reformulation.

    Parameters
    ----------
    formulation : string
        Power flow formulation used; e.g. `"angles"` or `"kirchhoff"`.        
    """
    
    if formulation == "angles":
        define_integer_passive_branch_flows_with_angles(network, snapshots)
    elif formulation == "kirchhoff":
        define_integer_passive_branch_flows_with_kirchhoff(network, snapshots)


def define_integer_slack_angle(network, snapshots):

    slack_dependencies = find_slack_dependencies(network)

    slack_upper = {}
    slack_lower = {}
    for sub, lines in slack_dependencies.items():
        for sn in snapshots:
            lhs = LExpression([(1,network.model.voltage_angles[network.sub_networks.slack_bus[sub],sn])])
            rhs = LExpression([(20*np.pi, network.model.passive_branch_inv[l]) for l in lines])
            slack_upper[sub,sn] = LConstraint(lhs,"<=",rhs)
            slack_lower[sub,sn] = LConstraint(lhs,">=",-rhs)

    l_constraint(network.model,"slack_angle_upper",slack_upper,list(network.sub_networks.index), snapshots)
    l_constraint(network.model,"slack_angle_lower",slack_lower,list(network.sub_networks.index), snapshots)
    

def define_integer_passive_branch_flows_with_angles(network, snapshots):
    """
    Enforce Kirchhoff's Second Law with angles formulation only if invested with Big-M reformulation.
    """

    extendable_branches = network.passive_branches(sel='candidate')

    investment_corridors = _corridors(extendable_branches)

    network.model.passive_branch_inv_p = Var(investment_corridors, snapshots)

    big_m = bigm(network, "angles")

    flows_upper = {}
    flows_lower = {}
    for branch in extendable_branches.index:
        bus0 = extendable_branches.at[branch, "bus0"]
        bus1 = extendable_branches.at[branch, "bus1"]
        bt = branch[0]
        bn = branch[1]
        sub = extendable_branches.at[branch,"sub_network"]
        attribute = "r_pu_eff" if network.sub_networks.at[sub,"carrier"] == "DC" else "x_pu_eff"
        y = 1/ extendable_branches.at[ branch, attribute]
        for sn in snapshots:
            lhs = LExpression([(y,network.model.voltage_angles[bus0,sn]),
                               (-y,network.model.voltage_angles[bus1,sn]),
                               (-1,network.model.passive_branch_inv_p[bt,bus0,bus1,sn])],
                              -y*(extendable_branches.at[branch,"phase_shift"]*np.pi/180. if bt == "Transformer" else 0.))
            rhs = LExpression(variables=[(-big_m[bn],network.model.passive_branch_inv[bt,bn])],
                              constant=big_m[bn])
            flows_upper[bt,bn,sn] = LConstraint(lhs,"<=",rhs)
            flows_lower[bt,bn,sn] = LConstraint(lhs,">=",-rhs)
        

    l_constraint(network.model, "passive_branch_inv_p_upper_def", flows_upper,
                 list(extendable_branches.index), snapshots)

    l_constraint(network.model, "passive_branch_inv_p_lower_def", flows_lower,
                 list(extendable_branches.index), snapshots)

def define_integer_passive_branch_flows_with_kirchhoff(network, snapshots):
    """
    Enforce Kirchhoff's Second Law with angles formulation only if invested with Big-M reformulation.
    """

    for sub_network in network.sub_networks.obj:
        # find_tree(sub_network) # TODO: is this necessary?
        find_cycles(sub_network)
        find_candidate_cycles(sub_network)

        # omitted bus_controls and B H calculation should be done ex-post given candidate investment decisions!

    extendable_branches = network.passive_branches(sel='candidate')

    investment_corridors = _corridors(extendable_branches)

    network.model.passive_branch_inv_p = Var(investment_corridors, snapshots)

    cycle_index = []
    cycle_constraints_upper = {}
    cycle_constraints_lower = {}

    for subnetwork in network.sub_networks.obj:

        attribute = "r_pu_eff" if network.sub_networks.at[subnetwork.name,"carrier"] == "DC" else "x_pu_eff"

        subn_cycle_index, subn_cycle_constraints_upper, subn_cycle_constraints_lower = \
            define_sub_network_candidate_cycle_constraints(subnetwork, snapshots, 
                                                network.model.passive_branch_p,
                                                network.model.passive_branch_inv_p,
                                                network.model.passive_branch_inv,
                                                attribute)

        cycle_index.extend(subn_cycle_index)
        cycle_constraints_upper.update(subn_cycle_constraints_upper)
        cycle_constraints_lower.update(subn_cycle_constraints_lower)

    l_constraint(network.model, "cycle_constraints_upper", cycle_constraints_upper,
                 cycle_index, snapshots)

    l_constraint(network.model, "cycle_constraints_lower", cycle_constraints_lower,
                 cycle_index, snapshots)


# TODO: this very nicely separates integer from continuous flow variables
# therefore, possibly include in pypsa.opf.define_nodal_balance_constraints
def define_integer_nodal_balance_constraints(network,snapshots):
    """
    Identical to `pypsa.opf.define_nodal_balance_constraints` but including candidate corridor flows.
    """

    # copied from pypsa.opf.define_nodal_balance_constraints
    passive_branches = network.passive_branches(sel='operative')

    for branch in passive_branches.index:
        bus0 = passive_branches.at[branch,"bus0"]
        bus1 = passive_branches.at[branch,"bus1"]
        bt = branch[0]
        bn = branch[1]
        for sn in snapshots:
            network._p_balance[bus0,sn].variables.append((-1,network.model.passive_branch_p[bt,bn,sn]))
            network._p_balance[bus1,sn].variables.append((1,network.model.passive_branch_p[bt,bn,sn]))

    # similar to pypsa.opf.define_nodal_balance_constraints
    candidate_branches = network.passive_branches(sel='candidate')

    investment_corridors = _corridors(candidate_branches)

    for c in investment_corridors:
        bus0 = c[1]
        bus1 = c[2]
        for sn in snapshots:
            network._p_balance[bus0,sn].variables.append((-1,network.model.passive_branch_inv_p[c,sn])) 
            network._p_balance[bus1,sn].variables.append((1,network.model.passive_branch_inv_p[c,sn]))

    # copied from pypsa.opf.define_nodal_balance_constraints
    power_balance = {k: LConstraint(v,"==",LExpression()) for k,v in iteritems(network._p_balance)}

    l_constraint(network.model, "power_balance", power_balance,
                 list(network.buses.index), snapshots)


def network_teplopf_build_model(network, snapshots=None, skip_pre=False,
                                formulation="angles", exclusive_candidates=True):
    """
    Description

    Parameters
    ----------
    exclusive_candidates : bool
        Indicator whether only one candidate line investment
        can be chosen per corridor.

    Returns
    -------
    None
    """

    if not skip_pre:
        # considered network topology depends on formulation.
        ls = 'operative' if formulation=="angles" else 'potential'
        network.determine_network_topology(line_selector=ls)
        
        calculate_dependent_values(network)
        for sub_network in network.sub_networks.obj:
            find_slack_bus(sub_network)
        logger.info("Performed preliminary steps")


    snapshots = _as_snapshots(network, snapshots)

    logger.info("Building pyomo model using `%s` formulation", formulation)
    network.model = ConcreteModel("Linear Optimal Power Flow for Transmission Expansion Planning")

    define_generator_variables_constraints(network,snapshots)

    define_storage_variables_constraints(network,snapshots)

    define_store_variables_constraints(network,snapshots)

    define_branch_extension_variables(network,snapshots)
    define_integer_branch_extension_variables(network,snapshots)

    if exclusive_candidates:
        define_exclusive_candidates_constraints(network,snapshots)
        
    define_rank_constraints(network, snapshots)

    define_link_flows(network,snapshots)

    define_nodal_balances(network,snapshots)

    define_passive_branch_flows(network,snapshots,formulation)
    define_integer_passive_branch_flows(network,snapshots,formulation)
    
    if formulation == "angles":
        define_integer_slack_angle(network,snapshots)

    define_passive_branch_constraints(network,snapshots)
    define_integer_passive_branch_constraints(network,snapshots)

    if formulation in ["angles", "kirchhoff"]:
        define_integer_nodal_balance_constraints(network,snapshots)

    define_global_constraints(network,snapshots)

    define_linear_objective(network, snapshots, candidates=True)

    #tidy up auxilliary expressions
    del network._p_balance

    #force solver to also give us the dual prices
    network.model.dual = Suffix(direction=Suffix.IMPORT)

    return network.model


def network_teplopf(network, snapshots=None, solver_name="glpk", solver_io=None,
                    skip_pre=False, extra_functionality=None, solver_logfile=None, solver_options={},
                    keep_files=False, formulation="angles",
                    free_memory={}, extra_postprocessing=None,
                    infer_candidates=False, exclusive_candidates=True):
    """
    Description

    Parameters
    ----------
    infer_candidates : bool
        Indicator whether candidate lines should be inferred
        based on potential and line type using
        `pypsa.tepopf.infer_candidates_from_existing()`.
    exclusive_candidates : bool
        Indicator whether only one candidate line investment
        can be chosen per corridor.

    Returns
    -------
    status
    termination_condition
    """

    if infer_candidates:
        network.lines = infer_candidates_from_existing(network, exclusive_candidates=exclusive_candidates)

    snapshots = _as_snapshots(network, snapshots)

    network_teplopf_build_model(network, snapshots, skip_pre=False, formulation=formulation,
                                exclusive_candidates=exclusive_candidates)

    if extra_functionality is not None:
        extra_functionality(network, snapshots)

    network_lopf_prepare_solver(network, solver_name=solver_name,
                                solver_io=solver_io)
    
    status, termination_condition = network_lopf_solve(network, snapshots, formulation=formulation,
                              solver_logfile=solver_logfile, solver_options=solver_options,
                              keep_files=keep_files, free_memory=free_memory,
                              extra_postprocessing=extra_postprocessing,
                              candidates=True)

    return status, termination_condition
