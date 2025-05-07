#!/usr/bin/env python3
"""
Created on Wed May  6 12:00:00 2025

Tests for the graph module
"""

import pandas as pd

from pypsa import Network
from pypsa.graph import adjacency_matrix


def test_adjacency_matrix_returns_dataframe():
    """Test that adjacency_matrix returns a pandas DataFrame"""
    # Create a simple network with 3 buses and 2 lines
    n = Network()
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")
    n.add("Line", "line0", bus0="bus0", bus1="bus1")
    n.add("Line", "line1", bus0="bus1", bus1="bus2")

    # Get the adjacency matrix
    adj = adjacency_matrix(n)

    # Check that the result is a DataFrame
    assert isinstance(adj, pd.DataFrame)

    # Check dimensions
    assert adj.shape == (3, 3)

    # Check indices
    assert list(adj.index) == ["bus0", "bus1", "bus2"]
    assert list(adj.columns) == ["bus0", "bus1", "bus2"]

    # Check values
    assert adj.at["bus0", "bus1"] == 1
    assert adj.at["bus1", "bus2"] == 1
    assert adj.at["bus0", "bus0"] == 0
    assert adj.at["bus0", "bus2"] == 0
    assert adj.at["bus1", "bus0"] == 0
    assert adj.at["bus1", "bus1"] == 0
    assert adj.at["bus2", "bus0"] == 0
    assert adj.at["bus2", "bus1"] == 0
    assert adj.at["bus2", "bus2"] == 0


def test_adjacency_matrix_with_weights():
    """Test adjacency_matrix with custom weights"""
    # Create a simple network with 3 buses and 2 lines
    n = Network()
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")
    n.add("Line", "line0", bus0="bus0", bus1="bus1")
    n.add("Line", "line1", bus0="bus1", bus1="bus2")

    # Create weights
    weights = pd.Series(
        [2.5, 3.5],
        index=pd.MultiIndex.from_tuples(
            [("Line", "line0"), ("Line", "line1")], names=["component", "name"]
        ),
    )

    # Get the adjacency matrix with weights
    adj = adjacency_matrix(n, weights=weights)

    # Check values
    assert adj.at["bus0", "bus1"] == 2.5
    assert adj.at["bus1", "bus2"] == 3.5
