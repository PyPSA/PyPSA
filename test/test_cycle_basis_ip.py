# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the IP minimum-cycle-basis implementation."""

import networkx as nx
import numpy as np

from pypsa.network.cycle_basis import (
    initial_cycle_basis_incidence,
    minimum_cycle_basis_ip,
)


def test_ip_minimum_cycle_basis_finds_grid_faces() -> None:
    """A 3x3 grid has four length-four face cycles as an MCB."""
    graph = nx.grid_2d_graph(3, 3, create_using=nx.MultiGraph)
    initial, _ = initial_cycle_basis_incidence(graph)
    minimum = minimum_cycle_basis_ip(initial)

    assert minimum.shape == initial.shape == (4, 12)
    assert np.linalg.matrix_rank(minimum.astype(float)) == 4
    assert minimum.sum() == 16
    assert (minimum.sum(axis=1) == 4).all()


def test_ip_minimum_cycle_basis_keeps_parallel_edge_cycle() -> None:
    """The IP formulation retains the required two-edge parallel cycle."""
    graph = nx.MultiGraph()
    graph.add_edges_from([(0, 1, "a"), (0, 1, "b"), (1, 2, "c"), (2, 0, "d")])
    initial, _ = initial_cycle_basis_incidence(graph)
    minimum = minimum_cycle_basis_ip(initial)

    assert minimum.shape == (2, 4)
    assert np.linalg.matrix_rank(minimum.astype(float)) == 2
    assert minimum.sum() == 5
