# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""
Created on Wed May  6 12:00:00 2025

Tests for the graph module
"""

import warnings

import pandas as pd
import scipy.sparse as sp

from pypsa import Network


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
    adj = n.adjacency_matrix(return_dataframe=True)

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
    adj = n.adjacency_matrix(weights=weights, return_dataframe=True)

    # Check values
    assert adj.at["bus0", "bus1"] == 2.5
    assert adj.at["bus1", "bus2"] == 3.5


def test_adjacency_matrix_returns_sparse():
    """Test that adjacency_matrix returns a sparse matrix when return_dataframe=False"""
    # Create a simple network with 3 buses and 2 lines
    n = Network()
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")
    n.add("Line", "line0", bus0="bus0", bus1="bus1")
    n.add("Line", "line1", bus0="bus1", bus1="bus2")

    # Get the adjacency matrix as sparse
    adj = n.adjacency_matrix(return_dataframe=False)

    # Check that the result is a sparse matrix
    assert isinstance(adj, sp.coo_matrix)

    # Check dimensions
    assert adj.shape == (3, 3)

    # Convert to dense for easier checking
    adj_dense = adj.toarray()

    # Check values (bus indices are in order: bus0=0, bus1=1, bus2=2)
    assert adj_dense[0, 1] == 1  # bus0 -> bus1
    assert adj_dense[1, 2] == 1  # bus1 -> bus2
    assert adj_dense[0, 0] == 0
    assert adj_dense[0, 2] == 0
    assert adj_dense[1, 0] == 0
    assert adj_dense[1, 1] == 0
    assert adj_dense[2, 0] == 0
    assert adj_dense[2, 1] == 0
    assert adj_dense[2, 2] == 0


def test_adjacency_matrix_sparse_with_weights():
    """Test adjacency_matrix with custom weights returns sparse matrix correctly"""
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

    # Get the adjacency matrix with weights as sparse
    adj = n.adjacency_matrix(weights=weights, return_dataframe=False)

    # Check that the result is a sparse matrix
    assert isinstance(adj, sp.coo_matrix)

    # Convert to dense for easier checking
    adj_dense = adj.toarray()

    # Check values
    assert adj_dense[0, 1] == 2.5  # bus0 -> bus1
    assert adj_dense[1, 2] == 3.5  # bus1 -> bus2


def test_adjacency_matrix_compatibility():
    """Test that both formats contain the same information"""
    # Create a simple network with 3 buses and 2 lines
    n = Network()
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")
    n.add("Line", "line0", bus0="bus0", bus1="bus1")
    n.add("Line", "line1", bus0="bus1", bus1="bus2")

    # Get both formats
    adj_df = n.adjacency_matrix(return_dataframe=True)
    adj_sparse = n.adjacency_matrix(return_dataframe=False)

    # Convert sparse to dense
    adj_sparse_dense = adj_sparse.toarray()

    # Check that they contain the same information
    for i, bus_i in enumerate(["bus0", "bus1", "bus2"]):
        for j, bus_j in enumerate(["bus0", "bus1", "bus2"]):
            assert adj_df.at[bus_i, bus_j] == adj_sparse_dense[i, j]


def test_adjacency_matrix_deprecation_warning():
    """Test that adjacency_matrix shows deprecation warning when return_dataframe is not specified"""
    # Create a simple network with 3 buses and 2 lines
    n = Network()
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")
    n.add("Line", "line0", bus0="bus0", bus1="bus1")
    n.add("Line", "line1", bus0="bus1", bus1="bus2")

    # Check that calling without return_dataframe parameter raises FutureWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        adj = n.adjacency_matrix()

        # Check that a warning was raised
        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert "adjacency_matrix will return a pandas DataFrame by default" in str(
            w[0].message
        )

        # Check that it still returns sparse matrix
        assert isinstance(adj, sp.coo_matrix)

    # Check that calling with explicit return_dataframe=False doesn't raise warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        adj = n.adjacency_matrix(return_dataframe=False)

        # Check that no warning was raised
        assert len(w) == 0
        assert isinstance(adj, sp.coo_matrix)

    # Check that calling with explicit return_dataframe=True doesn't raise warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        adj = n.adjacency_matrix(return_dataframe=True)

        # Check that no warning was raised
        assert len(w) == 0
        assert isinstance(adj, pd.DataFrame)
