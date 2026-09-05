# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Cycle-basis algorithms."""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING

import networkx as nx

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
