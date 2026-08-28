# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Cycle-basis algorithms."""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, eye, hstack

if TYPE_CHECKING:
    from collections.abc import Hashable


def _bfs_fundamental_cycles(graph: nx.Graph, root: Hashable) -> list[list[Hashable]]:
    """Return the fundamental cycles of a breadth-first spanning tree."""
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


def bfs_cycle_basis(graph: nx.Graph, num_roots: int = 5) -> list[list[Hashable]]:
    """Choose the best of several high-degree-root BFS fundamental bases."""
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
    """Locally shorten a BFS basis by independent pairwise XOR exchanges."""
    cycles = bfs_cycle_basis(graph)
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


def minimum_cycle_basis_ip(
    initial_basis: np.ndarray,
    *,
    solver: str = "scipy",
    options: dict[str, object] | None = None,
) -> np.ndarray:
    """Return a minimum-cardinality cycle basis.

    Parameters
    ----------
    initial_basis : numpy.ndarray
        Binary, full-row-rank cycle-edge incidence matrix with cycles in rows.
        Each row must represent a cycle in the same undirected graph.
    options : dict, optional
        Options passed to the selected MILP solver.
    solver : {"scipy", "gurobi"}, default "scipy"
        MILP backend.

    Returns
    -------
    numpy.ndarray
        A binary minimum cycle basis with the same shape as ``initial_basis``.

    """
    basis = np.asarray(initial_basis, dtype=np.int8).copy()
    if basis.ndim != 2 or not basis.size:
        msg = "initial_basis must be a non-empty 2-D array"
        raise ValueError(msg)
    if not np.isin(basis, (0, 1)).all():
        msg = "initial_basis must be binary"
        raise ValueError(msg)

    n_cycles, n_edges = basis.shape
    if np.linalg.matrix_rank(basis.astype(float)) != n_cycles:
        msg = "initial_basis rows must be linearly independent"
        raise ValueError(msg)

    if solver == "gurobi":
        return _minimum_cycle_basis_ip_gurobi(basis, options=options)
    if solver != "scipy":
        msg = "solver must be 'scipy' or 'gurobi'"
        raise ValueError(msg)

    zeta_upper = n_cycles
    lower = np.zeros(n_cycles + n_edges + n_edges)
    upper = np.concatenate(
        [np.ones(n_cycles), np.full(n_edges, zeta_upper), np.ones(n_edges)]
    )
    integrality = np.ones_like(lower, dtype=np.uint8)
    objective = np.concatenate([np.zeros(n_cycles + n_edges), np.ones(n_edges)])

    for k in range(n_cycles):
        lhs = hstack(
            [
                csc_matrix(basis.T),
                -2 * eye(n_edges, format="csc"),
                -eye(n_edges, format="csc"),
            ],
            format="csc",
        )
        constraint = LinearConstraint(lhs, np.zeros(n_edges), np.zeros(n_edges))
        row_lower = lower.copy()
        row_upper = upper.copy()
        row_lower[k] = row_upper[k] = 1
        milp_options: dict[str, object] = {"presolve": False}
        if options is not None:
            milp_options.update(options)
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(row_lower, row_upper),
            constraints=constraint,
            options=milp_options,
        )
        if not result.success or result.x is None:
            msg = f"minimum-cycle-basis IP failed at row {k}: {result.message}"
            raise RuntimeError(msg)

        replacement = np.rint(result.x[n_cycles + n_edges :]).astype(np.int8)
        if not np.isin(replacement, (0, 1)).all():
            msg = "MILP returned a non-binary cycle incidence vector"
            raise RuntimeError(msg)
        basis[k] = replacement

    return basis


def _minimum_cycle_basis_ip_gurobi(
    basis: np.ndarray, *, options: dict[str, object] | None
) -> np.ndarray:
    """Run equation (28) with an optional Gurobi backend."""
    try:
        import gurobipy as gp  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on local license
        msg = "solver='gurobi' requires gurobipy"
        raise ImportError(msg) from exc

    n_cycles, n_edges = basis.shape
    gurobi_options = {"OutputFlag": 0}
    if options is not None:
        gurobi_options.update(options)

    for k in range(n_cycles):
        model = gp.Model("minimum-cycle-basis")
        for name, value in gurobi_options.items():
            setattr(model.Params, name, value)
        w = model.addMVar(n_cycles, vtype=gp.GRB.BINARY, name="w")
        zeta = model.addMVar(
            n_edges, lb=0, ub=n_cycles, vtype=gp.GRB.INTEGER, name="zeta"
        )
        v = model.addMVar(n_edges, vtype=gp.GRB.BINARY, name="v")
        model.addConstr(basis.T.astype(float) @ w == 2 * zeta + v)
        model.addConstr(w[k] == 1)
        model.setObjective(v.sum(), gp.GRB.MINIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            msg = f"minimum-cycle-basis IP failed at row {k}: Gurobi status {model.Status}"
            raise RuntimeError(msg)
        basis[k] = np.rint(v.X).astype(np.int8)

    return basis


def initial_cycle_basis_incidence(
    graph: nx.MultiGraph,
) -> tuple[np.ndarray, list[tuple[Hashable, Hashable, Hashable]]]:
    """Build a binary cycle-edge incidence matrix."""
    if graph.is_directed():
        msg = "minimum cycle basis requires an undirected graph"
        raise nx.NetworkXNotImplemented(msg)
    if any(u == v for u, v, _ in graph.edges(keys=True)):
        msg = "self-loops are not supported"
        raise ValueError(msg)

    edge_order = list(graph.edges(keys=True))
    edge_position = {edge: i for i, edge in enumerate(edge_order)}
    pair_edges: dict[
        frozenset[Hashable], list[tuple[Hashable, Hashable, Hashable]]
    ] = {}
    for edge in edge_order:
        pair_edges.setdefault(frozenset(edge[:2]), []).append(edge)

    simple = nx.Graph(graph)
    rows: list[np.ndarray] = []
    for cycle in nx.cycle_basis(simple):
        row = np.zeros(len(edge_order), dtype=np.int8)
        for i, node in enumerate(cycle):
            edge = pair_edges[frozenset((node, cycle[(i + 1) % len(cycle)]))][0]
            row[edge_position[edge]] = 1
        rows.append(row)

    for edges in pair_edges.values():
        for edge in edges[1:]:
            row = np.zeros(len(edge_order), dtype=np.int8)
            row[edge_position[edges[0]]] = 1
            row[edge_position[edge]] = 1
            rows.append(row)

    expected_rank = (
        graph.number_of_edges()
        - graph.number_of_nodes()
        + nx.number_connected_components(graph)
    )
    if len(rows) != expected_rank:
        msg = "initial cycle basis has unexpected rank"
        raise RuntimeError(msg)
    return np.asarray(rows, dtype=np.int8), edge_order
