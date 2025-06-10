"""Graph helper functions, which are attached to network and sub_network."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import networkx as nx
from deprecation import deprecated

from pypsa.common import deprecated_common_kwargs

if TYPE_CHECKING:
    from collections.abc import Collection

    import pandas as pd
    import scipy as sp

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
def graph(
    n: Network | SubNetwork,
    branch_components: Collection[str] | None = None,
    weight: str | None = None,
    inf_weight: bool | float = False,
    include_inactive: bool = True,
) -> OrderedGraph:
    """Use `n.graph` instead."""
    return n.graph(
        branch_components=branch_components,
        weight=weight,
        inf_weight=inf_weight,
        include_inactive=include_inactive,
    )


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `n.adjacency_matrix` instead.",
)
@deprecated_common_kwargs
def adjacency_matrix(
    n: Network | SubNetwork,
    branch_components: Collection[str] | None = None,
    investment_period: int | str | None = None,
    busorder: pd.Index | None = None,
    weights: pd.Series | None = None,
) -> sp.sparse.coo_matrix:
    """Use `n.adjacency_matrix` instead."""
    return n.adjacency_matrix(
        branch_components=branch_components,
        investment_period=investment_period,
        busorder=busorder,
        weights=weights,
    )


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `n.incidence_matrix` instead.",
)
@deprecated_common_kwargs
def incidence_matrix(
    n: Network | SubNetwork,
    branch_components: Collection[str] | None = None,
    busorder: pd.Index | None = None,
) -> sp.sparse.csr_matrix:
    """Use `n.incidence_matrix` instead."""
    return n.incidence_matrix(branch_components=branch_components, busorder=busorder)
