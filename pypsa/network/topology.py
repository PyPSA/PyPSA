# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The topology of the passive network, as functions over plain tables.

Each function reads plain arrays and tables of buses and passive branches,
so it needs no network. `pypsa.Network` and `pypsa.SubNetwork` call them
with their own data.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csgraph, csr_matrix, dok_matrix, issparse
from scipy.sparse.linalg import spsolve

from pypsa.network.graph import OrderedGraph, bfs_refined_cycle_basis

logger = logging.getLogger(__name__)

CYCLE_BASIS_METHODS = ("bfs-refined", "paton")


def sub_network_labels(
    buses: pd.Index, bus0: pd.Index, bus1: pd.Index
) -> tuple[int, np.ndarray]:
    """Return the number of connected sub-networks and the one each bus is in.

    Parameters
    ----------
    buses : pandas.Index
        The buses, in the order that numbers the sub-networks.
    bus0, bus1 : pandas.Index
        The two ends of each branch that connects.

    Raises
    ------
    ValueError
        A branch ends at a bus that `buses` lacks.

    """
    ends = buses.get_indexer(bus0), buses.get_indexer(bus1)
    if any((end == -1).any() for end in ends):
        msg = "A branch ends at a bus the network does not hold."
        raise ValueError(msg)
    size = len(buses)
    adjacency = csr_matrix((np.ones(len(ends[0])), ends), shape=(size, size))
    return csgraph.connected_components(adjacency, directed=False)


def cycle_basis(
    buses: pd.Index,
    bus0: pd.Series,
    bus1: pd.Series,
    edges: np.ndarray,
    method: str = "bfs-refined",
) -> dok_matrix:
    """Return the cycle matrix of one sub-network, branches by cycles.

    The basis of `method` holds the cycles of more than 2 edges; each
    further branch between two buses then closes a cycle of 2 edges with the
    first one between them. An entry is `+1` where the cycle runs along
    the branch from `bus0` to `bus1` and `-1` against it.

    Parameters
    ----------
    buses : pandas.Index
        The buses of the sub-network, in the order of its graph's nodes.
    bus0, bus1 : pandas.Series
        The two ends of each branch of the sub-network, by its label.
    edges : numpy.ndarray
        Whether each branch is an edge of the graph: active and of finite
        impedance.
    method : str, default "bfs-refined"
        Either `"bfs-refined"` or `"paton"`.

    """
    if method not in CYCLE_BASIS_METHODS:
        msg = 'cycle_basis_method must be one of "bfs-refined" or "paton"'
        raise ValueError(msg)
    branches_i = bus0.index
    mgraph = OrderedGraph()
    mgraph.add_nodes_from(buses)
    mgraph.add_edges_from(
        (u, v, key)
        for u, v, key, edge in zip(bus0, bus1, branches_i, edges, strict=True)
        if edge
    )
    graph = nx.Graph(mgraph)
    if method == "paton":
        cycles = nx.cycle_basis(graph)
    else:
        cycles = bfs_refined_cycle_basis(graph)

    num_multi = len(mgraph.edges()) - len(graph.edges())
    C = dok_matrix((len(bus0), len(cycles) + num_multi))

    for j, cycle in enumerate(cycles):
        for i in range(len(cycle)):
            branch = next(iter(mgraph[cycle[i]][cycle[(i + 1) % len(cycle)]].keys()))
            branch_i = branches_i.get_loc(branch)
            sign = +1 if bus0.iat[branch_i] == cycle[i] else -1
            C[branch_i, j] += sign

    c = len(cycles)
    for u, v in graph.edges():
        bs = list(mgraph[u][v].keys())
        if len(bs) > 1:
            first_i = branches_i.get_loc(bs[0])
            for b in bs[1:]:
                b_i = branches_i.get_loc(b)
                sign = -1 if bus0.iat[b_i] == bus0.iat[first_i] else +1
                C[first_i, c] = 1
                C[b_i, c] = sign
                c += 1
    return C


def combine_cycles(
    cycles: list[pd.DataFrame], weights: pd.Series, branches_i: pd.MultiIndex
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the cycle matrix of a network, plain and weighted, from those of its sub-networks.

    Parameters
    ----------
    cycles : list of pandas.DataFrame
        The cycle matrix of each sub-network that has a cycle, by branch.
    weights : pandas.Series
        The impedance of each branch in a cycle.
    branches_i : pandas.MultiIndex
        Every passive branch, `(type, name)`; the result holds those of each
        type that has a branch in a cycle.

    """
    if not cycles:
        return pd.DataFrame(), pd.DataFrame()
    cycles_df = pd.concat(cycles, axis=1, ignore_index=True).fillna(0)
    existing = cycles_df.index.unique("type")
    branches_i = pd.Series(0, branches_i).loc[existing].index
    weights = weights[cycles_df.index]
    weighted = cycles_df.where(cycles_df == 0, cycles_df.multiply(weights, axis=0))
    plain, weighted = (
        df.reindex(branches_i, fill_value=0).rename_axis(columns="cycle")
        for df in (cycles_df, weighted)
    )
    return plain, weighted


def incidence(bus0: np.ndarray, bus1: np.ndarray, buses: int) -> csr_matrix:
    """Return the incidence matrix, buses by branches, of branches from position `bus0` to `bus1`."""
    branches = len(bus0)
    return csr_matrix(
        (
            np.r_[np.ones(branches), -np.ones(branches)],
            (np.r_[bus0, bus1], np.r_[:branches, :branches]),
        ),
        (buses, branches),
    )


def susceptance(
    z: np.ndarray, K: csr_matrix
) -> tuple[np.ndarray, csr_matrix, csr_matrix]:
    """Return the susceptance of each branch, and the matrices `H` and `B` of the linear power flow.

    Parameters
    ----------
    z : numpy.ndarray
        The series impedance of each branch.
    K : scipy.sparse.csr_matrix
        The incidence matrix, buses by branches.

    """
    b = np.divide(1.0, z, out=np.full_like(z, np.inf), where=z != 0)
    if np.isnan(b).any():
        logger.warning(
            "Warning! Some series impedances are zero - this will cause a singularity in LPF!"
        )
    b_diag = csr_matrix((b, (np.r_[: len(b)], np.r_[: len(b)])))
    H = b_diag * K.T
    return b, H, K * H


def ptdf(B: csr_matrix, H: csr_matrix) -> np.ndarray:
    """Return the power transfer distribution factors, branches by buses, the first bus the slack."""
    n_pvpq = B.shape[0] - 1
    index = np.r_[:n_pvpq]
    identity = csc_matrix((np.ones(n_pvpq), (index, index)))
    B_inverse = spsolve(csc_matrix(B[1:, 1:]), identity)
    if issparse(B_inverse):
        B_inverse = B_inverse.toarray()
    elif B_inverse.shape == (1,):
        B_inverse = B_inverse.reshape((1, 1))
    B_inverse = np.hstack((np.zeros((n_pvpq, 1)), B_inverse))
    B_inverse = np.vstack((np.zeros(n_pvpq + 1), B_inverse))
    return H * B_inverse


def bodf(PTDF: np.ndarray, K: csr_matrix) -> np.ndarray:
    """Return the branch outage distribution factors, branches by outaged branches.

    After the outage of branch `l`, the flow on branch `k` is its flow
    before plus `BODF[k, l]` times the flow on `l` before.
    """
    num_branches = PTDF.shape[0]
    branch_PTDF = PTDF * K
    with np.errstate(divide="ignore"):
        denominator = csr_matrix(
            (
                1 / (1 - np.diag(branch_PTDF)),
                (np.r_[:num_branches], np.r_[:num_branches]),
            )
        )
    factors = branch_PTDF * denominator
    np.fill_diagonal(factors, -1)
    return factors
