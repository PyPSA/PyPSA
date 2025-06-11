"""Graph helper functions, which are attached to network and sub_network."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import scipy as sp
from deprecation import deprecated

from pypsa.common import deprecated_common_kwargs

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable

    import pandas as pd

    from pypsa import Network, SubNetwork


class OrderedGraph(nx.MultiGraph):
    """Ordered graph."""

    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `n.graph` instead.",
)
@deprecated_common_kwargs
def graph(n: Network | SubNetwork, *args: Any, **kwargs: Any) -> Any:
    """Use `n.graph` instead."""
    return n.graph(*args, **kwargs)


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `n.adjacency_matrix` instead.",
)
@deprecated_common_kwargs
def adjacency_matrix(n: Network | SubNetwork, *args: Any, **kwargs: Any) -> Any:
    """Use `n.adjacency_matrix` instead."""
    return n.adjacency_matrix(*args, **kwargs)


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `n.incidence_matrix` instead.",
)
@deprecated_common_kwargs
def incidence_matrix(n: Network | SubNetwork, *args: Any, **kwargs: Any) -> Any:
    """Use `n.incidence_matrix` instead."""
    return n.incidence_matrix(*args, **kwargs)


class NetworkGraphMixin:
    """Mixin class for network graph methods.

    Class only inherits to [pypsa.Network][]/[pypsa.SubNetwork][] and should not be
    used directly.
    All attributes and methods can be used within any Network/SubNetwork instance.
    """

    # Type Hints
    iterate_components: Callable

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
        n : Network|SubNetwork
            Network or sub-network.
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
            Whether to include inactive branches in the graph.

        Returns
        -------
        graph : OrderedGraph
            NetworkX graph

        """
        from pypsa import Network, SubNetwork

        if isinstance(self, Network):
            if branch_components is None:
                branch_components = self.branch_components
            else:
                branch_components = set(branch_components)
            buses_i = self.buses.index
        elif isinstance(self, SubNetwork):
            if branch_components is None:
                branch_components = self.n.passive_branch_components
            buses_i = self.buses_i()
        else:
            msg = "graph must be called with a Network or a SubNetwork"
            raise TypeError(msg)

        graph = OrderedGraph()

        # add nodes first, in case there are isolated buses not connected with branches
        graph.add_nodes_from(buses_i)

        # Multigraph uses the branch type and name as key
        def gen_edges() -> Iterable[tuple[str, str, tuple[str, int], dict]]:
            for c in self.iterate_components(branch_components):
                for branch in c.static.loc[
                    slice(None) if include_inactive else c.static.query("active").index
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

        graph.add_edges_from(gen_edges())

        return graph

    def adjacency_matrix(
        self,
        branch_components: Collection[str] | None = None,
        investment_period: int | str | None = None,
        busorder: pd.Index | None = None,
        weights: pd.Series | None = None,
    ) -> sp.sparse.coo_matrix:
        """Construct a sparse adjacency matrix (directed).

        Parameters
        ----------
        n : Network | SubNetwork
            Network or sub-network.
        branch_components : iterable sublist of `branch_components`
            Buses connected by any of the selected branches are adjacent
            (default: branch_components (network) or passive_branch_components (sub_network))
        investment_period : int | str | None
            Investment period to use for the matrix representation of the adjacency matrix.
        busorder : pd.Index subset of n.buses.index
            Basis to use for the matrix representation of the adjacency matrix
            (default: buses.index (network) or buses_i() (sub_network))
        weights : pd.Series or None (default)
            If given must provide a weight for each branch, multi-indexed
            on branch_component name and branch name.

        Returns
        -------
        adjacency_matrix : sp.sparse.coo_matrix
        Directed adjacency matrix

        Examples
        --------
        >>> n.adjacency_matrix()  # doctest: +ELLIPSIS
        <COOrdinate sparse matrix of dtype 'float64'
            with ... stored elements and shape (..., ...)>

        """
        from pypsa.networks import Network, SubNetwork

        if isinstance(self, Network):
            if branch_components is None:
                branch_components = self.branch_components
            if busorder is None:
                busorder = self.buses.index
        elif isinstance(self, SubNetwork):
            if branch_components is None:
                branch_components = self.n.passive_branch_components
            if busorder is None:
                busorder = self.buses_i()
        else:
            msg = " must be called with a Network or a SubNetwork"
            raise TypeError(msg)

        no_buses = len(busorder)
        no_branches = 0
        bus0_inds = []
        bus1_inds = []
        weight_vals = []
        for c in self.iterate_components(branch_components):
            active = c.get_active_assets(investment_period)
            sel = c.static[active].index

            no_branches = len(c.static.loc[sel])
            bus0_inds.append(busorder.get_indexer(c.static.loc[sel, "bus0"]))
            bus1_inds.append(busorder.get_indexer(c.static.loc[sel, "bus1"]))
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
        from pypsa import Network, SubNetwork

        if isinstance(self, Network):
            if branch_components is None:
                branch_components = self.branch_components
            if busorder is None:
                busorder = self.buses.index
        elif isinstance(self, SubNetwork):
            if branch_components is None:
                branch_components = self.n.passive_branch_components
            if busorder is None:
                busorder = self.buses_i()
        else:
            msg = "The 'n' parameter must be an instance of 'Network' or 'SubNetwork'."
            raise TypeError(msg)

        no_buses = len(busorder)
        no_branches = 0
        bus0_inds = []
        bus1_inds = []
        for c in self.iterate_components(branch_components):
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
