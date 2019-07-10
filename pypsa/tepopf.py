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

# make the code as Python 3 compatible as possible
from __future__ import division, absolute_import
from six import iteritems, itervalues, string_types

__author__ = "Fabian Neumann (KIT)"
__copyright__ = "Copyright 2019 Fabian Neumann (KIT), GNU GPL 3"

import pandas as pd
import numpy as np
import networkx as nx
import itertools

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

from .pf import (_as_snapshots, calculate_dependent_values, find_slack_bus)

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

    # TODO: assert that all extendable lines have line type that is in network.line_types
    # otherwise derive line type from num_parallel, s_nom, r, x
    
    ext_lines = network.lines[network.lines.s_nom_extendable]
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
            candidate.num_parallel = 1
            candidate.x = type_params.x_per_length * candidate.length
            candidate.r = type_params.r_per_length * candidate.length
            candidate.s_nom = np.sqrt(3) * type_params.i_nom * candidate.v_nom
            candidate.s_nom_max = candidate.s_nom
            candidate.s_nom_min = 0.
            candidate.operative = False
            candidate.name = "{}_c{}".format(candidate.name,c)
            candidates.loc[candidate.name] = candidate

    lines = pd.concat([network.lines,candidates])

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
    
    for name, group in candidate_lines.groupby(['bus0', 'bus1']):
        combinations = get_investment_combinations(group)
        for c in combinations:
            candidate_block = group.loc[c]
            cinv = aggregate_candidates(network, candidate_block)
            names = pd.Series(c).apply(lambda x: x.split('_'))
            cinv.name = ("{}"+"_{}"*len(c)).format(names.iloc[0][0], *names.apply(lambda x: x[1]))
            candidate_inv.loc[cinv.name] = cinv

    return pd.concat([lines[lines.operative], candidate_inv.drop_duplicates()])


def bigm(n, formulation):
    """
    Determines the minimal Big-M parameters.

    Parameters
    ----------
    n : pypsa.Network
    formulation : string
        Power flow formulation used. E.g. `"angles"` or `"kirchhoff"`.

    Returns
    -------
    big_m : dict
    """

    if formulation == "angles":
        m = bigm_for_angles(n)
    elif formulation == "kirchhoff":
        m = bigm_for_kirchhoff(n)
    else:
        raise NotImplementedError("Calculating Big-M for formulation `{}` not implemented.\
                                   Try `angles` or `kirchhoff`.")

    return m


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
    big_m : dict

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

    n.lines['bigm_weight'] = n.lines.apply(lambda l: l.s_nom * l.x_pu_eff 
                                      if l.operative
                                      else np.nan, axis=1)

    candidates = n.lines[n.lines.operative==False]

    ngraph = n.graph(line_selector='operative', branch_components=['Line'], weight='bigm_weight')

    bigm = {}
    for name, candidate in candidates.iterrows():
        path_length = nx.dijkstra_path_length(ngraph, candidate.bus0, candidate.bus1)
        bigm[name] = path_length / candidate.x_pu_eff

    if not keep_weights:
        n.lines.drop("bigm_weight")

    return bigm


def bigm_for_kirchhoff(n):
    """
    Determines the minimal Big-M parameters for the `kirchhoff` formulation.

    Parameters
    ----------
    n : pypsa.Network

    Returns
    -------
    big_m : dict
    """

    m = None

    return m


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

    passive_branches = network.passive_branches(sel='inoperative')

    extendable_passive_branches = passive_branches[passive_branches.s_nom_extendable]

    network.model.passive_branch_inv = Var(list(extendable_passive_branches.index),
                                           domain=Binary)

    free_pyomo_initializers(network.model.passive_branch_inv)


# TODO: multiple flow variables (one per candidate line)
def define_integer_passive_branch_constraints(network, snapshots): 
    """
    Capacity constraints of investment corridor flows.
    There is currently only one flow variable for all candidate lines per corridor.
    """

    passive_branches = network.passive_branches(sel='inoperative')
    extendable_branches = eb = passive_branches[passive_branches.s_nom_extendable]

    s_max_pu = pd.concat({c : get_switchable_as_dense(network, c, 's_max_pu', snapshots)
                          for c in network.passive_branch_components}, axis=1, sort=False)

    investment_corridors = _corridors(extendable_branches)

    flow_upper = {(*c,sn) : [[(1,network.model.passive_branch_inv_p[c,sn]),
                            *[(
                                -s_max_pu.at[sn,b] * extendable_branches.at[b,"s_nom"],
                                network.model.passive_branch_inv[b[0],b[1]])
                            for b, bd in eb.loc[(eb.bus0==c[1]) & (eb.bus1==c[2])].iterrows()]
                            ],"<=",0]
                  for c in investment_corridors
                  for sn in snapshots}

    l_constraint(network.model, "inv_flow_upper", flow_upper,
                 investment_corridors, snapshots)

    flow_lower = {(*c,sn): [[(1,network.model.passive_branch_inv_p[c,sn]),
                            *[(
                                s_max_pu.at[sn,b] * extendable_branches.at[b,"s_nom"],
                                network.model.passive_branch_inv[b[0],b[1]])
                            for b, bd in eb.loc[(eb.bus0==c[1]) & (eb.bus1==c[2])].iterrows()]
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

    passive_branches = network.passive_branches(sel='inoperative')
    candidate_branches = cb = passive_branches[passive_branches.s_nom_extendable]

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

    passive_branches = network.passive_branches(sel='inoperative')
    extendable_branches = eb = passive_branches[passive_branches.s_nom_extendable]

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

# TODO: needs to consider combinations of investment if investments are not exclusive! 
def define_integer_passive_branch_flows_with_angles(network, snapshots):
    """
    Enforce Kirchhoff's Second Law with angles formulation only if invested with Big-M reformulation.
    """

    passive_branches = network.passive_branches(sel='inoperative')
    extendable_branches = passive_branches[passive_branches.s_nom_extendable]

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

    pass


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
    inoperative_passive_branches = network.passive_branches(sel='inoperative')
    candidate_branches = inoperative_passive_branches[inoperative_passive_branches.s_nom_extendable]

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
        network.determine_network_topology(line_selector=None)
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
        `pypsa.tepopf.infer_candidates_from_exiting()`.
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
