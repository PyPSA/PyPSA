# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Graph helper functions, which are attached to network and sub_network."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable


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
        bus0_inds = []
        bus1_inds = []
        for c in self.components:
            if c.name not in branch_components:
                continue
            sel = c.static.query("active").index
            no_branches += len(c.static.loc[sel])
            bus0_inds.append(busorder.get_indexer(c.static.loc[sel, "bus0"]))
            bus1_inds.append(busorder.get_indexer(c.static.loc[sel, "bus1"]))
        bus0_inds = np.concatenate(bus0_inds)
        bus1_inds = np.concatenate(bus1_inds)

        return sp.sparse.csr_matrix(
            (
                np.r_[np.ones(no_branches), -np.ones(no_branches)],
                (np.r_[bus0_inds, bus1_inds], np.r_[:no_branches, :no_branches]),
            ),
            (no_buses, no_branches),
        )
