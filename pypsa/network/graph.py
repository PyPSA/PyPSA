# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Graph helper functions, which are attached to network and sub_network, and cycle-basis algorithms."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp

if TYPE_CHECKING:
    from collections.abc import Collection, Hashable, Iterable


class OrderedGraph(nx.MultiGraph):
    """Ordered graph."""

    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict


class NetworkGraphMixin:
    """Mixin class for network graph methods.

    Class inherits to [pypsa.Network][]/[pypsa.SubNetwork][]. All attributes and
    methods can be used within any Network/SubNetwork instance.

    """

    c: Any
    components: Any
    iterate_components: Any
    passive_branches: pd.DataFrame
    has_scenarios: Any
    scenarios: pd.DataFrame

    def graph(
        self,
        branch_components: Collection[str] | None = None,
        weight: str | None = None,
        inf_weight: bool | float = False,
        include_inactive: bool = True,
    ) -> OrderedGraph:
        """Build NetworkX graph.

        Parameters
        ----------
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
        include_inactive : bool
            Whether to include inactive components in the graph.

        Returns
        -------
        graph : OrderedGraph
            NetworkX graph

        """
        n = self
        from pypsa import Network, SubNetwork  # noqa: PLC0415

        if branch_components is not None:
            branch_components = set(branch_components)
        elif isinstance(n, Network):
            branch_components = n.branch_components
        elif isinstance(n, SubNetwork):
            branch_components = n.n.passive_branch_components
        else:
            msg = "graph must be called with a Network or a SubNetwork"
            raise TypeError(msg)
        # sort for a hash-seed-independent edge order (and cycle basis); see GH #1356
        branch_components = sorted(branch_components)

        buses_i = n.c.buses.static.index

        if n.has_scenarios:
            buses_i = buses_i.unique("name")

        graph = OrderedGraph()

        # add nodes first, in case there are isolated buses not connected with branches
        graph.add_nodes_from(buses_i)

        # Multigraph uses the branch type and name as key
        def gen_edges() -> Iterable[tuple[str, str, tuple[str, int], dict]]:
            for c in n.iterate_components(branch_components):
                static = c.static
                if n.has_scenarios:
                    static = c.static.loc[n.scenarios[0]]

                for branch in static.loc[
                    slice(None) if include_inactive else static.query("active").index
                ].itertuples():
                    if weight is None:
                        data = {}
                    else:
                        data = {"weight": getattr(branch, weight, 0)}
                        if np.isinf(data["weight"]) and inf_weight is not True:
                            if inf_weight is False:
                                continue
                            data["weight"] = inf_weight
                    yield (branch.bus0, branch.bus1, (c.name, branch.Index), data)

        with warnings.catch_warnings():
            # TODO Resolve
            warnings.filterwarnings(
                "ignore",
                message=".*iterate_components is deprecated.*",
                category=DeprecationWarning,
            )
            graph.add_edges_from(gen_edges())

        return graph

    def adjacency_matrix(
        self,
        branch_components: Collection[str] | None = None,
        investment_period: int | str | None = None,
        busorder: pd.Index | None = None,
        weights: pd.Series | None = None,
        return_dataframe: bool | None = None,
    ) -> pd.DataFrame | sp.sparse.coo_matrix:
        """Construct an adjacency matrix (directed) as a pandas DataFrame or sparse matrix.

        Parameters
        ----------
        branch_components : iterable sublist of `branch_components`
            Buses connected by any of the selected branches are adjacent
            (default: branch_components (network) or passive_branch_components (sub_network))
        investment_period : int | str | None, default None
            If given, only assets active in the given investment period are considered
            in the network topology.
        busorder : pd.Index subset of n.buses.index
            Basis to use for the matrix representation of the adjacency matrix
            (default: buses.index (network) or buses_i() (sub_network))
        weights : pd.Series or None (default)
            If given must provide a weight for each branch, multi-indexed
            on branch_component name and branch name.
        return_dataframe : bool | None, default None
            If True, returns a pandas DataFrame. If False, returns a sparse coo_matrix
            for backwards compatibility. If None (default), returns a sparse coo_matrix
            with a deprecation warning.

        Returns
        -------
        adjacency_matrix : pd.DataFrame or sp.sparse.coo_matrix
            Directed adjacency matrix as DataFrame (if return_dataframe=True) or
            sparse matrix (if return_dataframe=False) with bus indices

        """
        from pypsa.networks import Network, SubNetwork  # noqa: PLC0415

        n = self
        if not isinstance(n, Network | SubNetwork):
            msg = "graph must be called with a Network or a SubNetwork"
            raise TypeError(msg)

        if branch_components is not None:
            branch_components = set(branch_components)
        elif isinstance(n, Network):
            branch_components = n.branch_components
        elif isinstance(n, SubNetwork):
            branch_components = n.n.passive_branch_components
        else:
            msg = " must be called with a Network or a SubNetwork"
            raise TypeError(msg)

        if busorder is None:
            busorder = n.c.buses.static.index

        # Initialize empty DataFrame with buses as both rows and columns
        if n.has_scenarios:
            busorder = busorder.unique("name")

        dtype = int if weights is None else float
        adjacency_df = pd.DataFrame(0, index=busorder, columns=busorder, dtype=dtype)

        # Build adjacency matrix component by component
        for c in n.components:
            if c.name not in branch_components:
                continue
            active = c.get_active_assets(investment_period)
            sel = c.static[active].index.unique("name")
            static = c.static.reindex(sel, level="name")

            # Skip if no branches in this component
            if len(static) == 0:
                continue

            # Get bus0 and bus1 from static data
            bus0 = static.bus0
            bus1 = static.bus1

            # Set weights for these connections
            if weights is None:
                # Set default weights of 1 for all branches
                for b0, b1 in zip(bus0, bus1, strict=False):
                    adjacency_df.at[b0, b1] = 1
            else:
                # Use provided weights
                for b0, b1, idx in zip(bus0, bus1, sel, strict=False):
                    adjacency_df.at[b0, b1] = weights[c.name][idx]

        # Handle deprecation warning for None case
        if return_dataframe is None:
            warnings.warn(
                "In future versions, adjacency_matrix will return a pandas DataFrame by default. "
                "To maintain the current behavior, explicitly set return_dataframe=False. "
                "To adopt the new behavior and silence this warning, set return_dataframe=True.",
                FutureWarning,
                stacklevel=2,
            )
            return_dataframe = False

        if return_dataframe:
            return adjacency_df
        else:
            # Convert to sparse matrix for backwards compatibility
            return sp.sparse.coo_matrix(adjacency_df.values)

    def incidence_matrix(
        self,
        branch_components: Collection[str] | None = None,
        busorder: pd.Index | None = None,
    ) -> sp.sparse.csr_matrix:
        """Construct a sparse incidence matrix (directed).

        Parameters
        ----------
        branch_components : iterable sublist of `branch_components`
            Buses connected by any of the selected branches are adjacent
            (default: branch_components (network) or passive_branch_components (sub_network))
        busorder : pd.Index subset of n.buses.index
            Basis to use for the matrix representation of the adjacency matrix
            (default: buses.index (network) or buses_i() (sub_network))

        Returns
        -------
        incidence_matrix : sp.sparse.csr_matrix
        Directed incidence matrix

        Examples
        --------
        >>> n.incidence_matrix()
        <Compressed Sparse Row sparse matrix of dtype 'float64'
                with 22 stored elements and shape (9, 11)>

        """
        from pypsa.networks import Network, SubNetwork  # noqa: PLC0415

        if branch_components is not None:
            branch_components = set(branch_components)
        elif isinstance(self, Network):
            branch_components = self.branch_components
        elif isinstance(self, SubNetwork):
            branch_components = self.n.passive_branch_components
        else:
            msg = " must be called with a Network or a SubNetwork"
            raise TypeError(msg)

        if busorder is None:
            busorder = self.c.buses.static.index

        no_buses = len(busorder)
        no_branches = 0
        bus0_inds_list = []
        bus1_inds_list = []
        for c in self.components:
            if c.name not in branch_components:
                continue
            sel = c.static.query("active").index
            no_branches += len(c.static.loc[sel])
            bus0_inds_list.append(busorder.get_indexer(c.static.loc[sel, "bus0"]))
            bus1_inds_list.append(busorder.get_indexer(c.static.loc[sel, "bus1"]))
        bus0_inds = np.concatenate(bus0_inds_list)
        bus1_inds = np.concatenate(bus1_inds_list)

        return sp.sparse.csr_matrix(
            (
                np.r_[np.ones(no_branches), -np.ones(no_branches)],
                (np.r_[bus0_inds, bus1_inds], np.r_[:no_branches, :no_branches]),
            ),
            (no_buses, no_branches),
        )


def _bfs_fundamental_cycles(graph: nx.Graph, root: Hashable) -> list[list[Hashable]]:
    """Return the fundamental cycles of a breadth-first spanning tree.

    A breadth-first spanning tree connects every node to ``root`` by a
    shortest-hop path; each remaining non-tree edge then closes exactly one
    fundamental cycle with the tree path between its endpoints [1].

    References
    ----------
    [1] N. Deo, G. M. Prabhu, M. S. Krishnamoorthy (1982), Algorithms for
    Generating Fundamental Cycles in a Graph, ACM Transactions on Mathematical
    Software 8 (1), 26-42, https://doi.org/10.1145/355984.355988

    """
    tree = nx.bfs_tree(graph, root)
    parent = {root: root} | {child: parent_node for parent_node, child in tree.edges()}
    tree_edges = {frozenset(edge) for edge in tree.edges()}
    cycles = []
    for u, v in graph.edges():
        if frozenset((u, v)) in tree_edges:
            continue
        ancestors = {u}
        current = u
        while current != root:
            current = parent[current]
            ancestors.add(current)
        left = [v]
        current = v
        while current not in ancestors:
            current = parent[current]
            left.append(current)
        lca = current
        right = [u]
        while right[-1] != lca:
            right.append(parent[right[-1]])
        cycles.append(right + left[-2::-1])
    return cycles


def _bfs_cycle_basis(graph: nx.Graph, num_roots: int = 5) -> list[list[Hashable]]:
    """Choose the best of several high-degree-root BFS fundamental bases.

    Each candidate is the fundamental cycle basis of a breadth-first spanning
    tree [1] grown from a different high-degree root; the basis with the shortest
    cycles is kept as a heuristic to reduce the total basis size.

    References
    ----------
    [1] N. Deo, G. M. Prabhu, M. S. Krishnamoorthy (1982), Algorithms for
    Generating Fundamental Cycles in a Graph, ACM Transactions on Mathematical
    Software 8 (1), 26-42, https://doi.org/10.1145/355984.355988

    """
    cycles: list[list[Hashable]] = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        roots = [
            node
            for node, _ in sorted(subgraph.degree(), key=itemgetter(1), reverse=True)[
                :num_roots
            ]
        ]
        candidates = [_bfs_fundamental_cycles(subgraph, root) for root in roots]
        cycles.extend(
            min(
                candidates,
                key=lambda candidate: (
                    max((len(cycle) for cycle in candidate), default=0),
                    sum(map(len, candidate)),
                ),
                default=[],
            )
        )
    return cycles


def _edge_set_to_cycle(edges: set[frozenset[Hashable]]) -> list[Hashable] | None:
    """Return a cyclic node ordering if ``edges`` form exactly one simple cycle."""
    adjacency: dict[Hashable, list[Hashable]] = {}
    for edge in edges:
        u, v = tuple(edge)
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return None
    start = next(iter(adjacency))
    order, previous, current = [start], None, start
    while True:
        left, right = adjacency[current]
        following = right if left == previous else left
        if following == start:
            return order if len(order) == len(edges) else None
        order.append(following)
        previous, current = current, following


def bfs_refined_cycle_basis(
    graph: nx.Graph, max_passes: int = 50
) -> list[list[Hashable]]:
    """Shorten a BFS cycle basis by greedy pairwise XOR exchanges.

    Replacing a basis cycle by its XOR (symmetric difference) with another is an
    elementary operation in the GF(2) cycle space, so the basis stays valid;
    each pass greedily applies exchanges that yield a shorter simple cycle. This
    is a heuristic without a minimality guarantee; see Kavitha et al. (2009) [1].

    References
    ----------
    [1] T. Kavitha, C. Liebchen, K. Mehlhorn, et al. (2009), Cycle bases in
    graphs: characterization, algorithms, complexity, and applications, Computer
    Science Review 3 (4), 199-243, https://doi.org/10.1016/j.cosrev.2009.08.001

    """
    cycles = _bfs_cycle_basis(graph)
    edge_sets = [
        {frozenset((cycle[i], cycle[(i + 1) % len(cycle)])) for i in range(len(cycle))}
        for cycle in cycles
    ]
    for _ in range(max_passes):
        edge_to_cycles: dict[frozenset[Hashable], list[int]] = {}
        for index, edges in enumerate(edge_sets):
            for edge in edges:
                edge_to_cycles.setdefault(edge, []).append(index)
        swaps = 0
        for left in sorted(range(len(cycles)), key=lambda index: -len(cycles[index])):
            candidates = {
                index for edge in edge_sets[left] for index in edge_to_cycles[edge]
            }
            candidates.discard(left)
            for right in candidates:
                if len(cycles[right]) >= len(cycles[left]):
                    continue
                replacement_edges = edge_sets[left] ^ edge_sets[right]
                if len(replacement_edges) >= len(cycles[left]):
                    continue
                replacement = _edge_set_to_cycle(replacement_edges)
                if replacement is None:
                    continue
                cycles[left], edge_sets[left] = replacement, replacement_edges
                swaps += 1
                break
        if not swaps:
            break
    return cycles
