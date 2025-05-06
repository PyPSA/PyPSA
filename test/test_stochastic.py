"""
Test stochastic functionality of PyPSA networks.
"""

import os

import pandas as pd
import pytest

import pypsa


def test_network_properties():
    """Test basic properties of a stochastic network."""
    snapshots = pd.date_range("2024-01-01", periods=5, freq="h")
    n = pypsa.Network(snapshots=snapshots)
    n.add("Bus", "bus1")
    n.add("Load", "load1", bus="bus1", p_set=100)

    # Set up scenarios
    n.set_scenarios({"low": 0.33, "medium": 0.34, "high": 0.33})

    # Check scenarios were properly set up
    assert len(n.scenarios) == 3
    assert "low" in n.scenarios
    assert "medium" in n.scenarios
    assert "high" in n.scenarios
    assert n.has_scenarios

    # Check probabilities sum to 1
    assert abs(n.scenarios.sum() - 1.0) < 1e-10

    # Check data shape for each scenario
    for scenario in n.scenarios.index:
        assert n.get_switchable_as_dense("Load", "p_set").loc[:, scenario].shape[0] == 5

    # Check string representation contains scenario information
    assert "Scenarios:" in repr(n)


def test_variable_dimensions():
    """Placeholder: verify that variables are cast to the correct dimensions."""
    pass


def test_scenario_constraints():
    """Placeholder: ensure constraints expressions are correctly defined."""
    pass


def test_optimization_simple(ac_dc_network):
    """
    Simple test case for the optimization of a stochastic network.
    """
    n = ac_dc_network
    n.set_scenarios({"low": 0.5, "high": 0.5})

    # Check that the optimization problem can be solved
    # status, _ = n.optimize(solver_name="highs")
    # assert status == "ok"


def test_solved_network_simple(stochastic_benchmark_network):
    """
    Solve the stochastic problem and compare results with benchmark data.
    Simple test case with a single bus and multiple generators.
    """
    # Load the benchmark results
    benchmark_path = os.path.join(os.path.dirname(__file__), "data", "benchmark-sp")

    if not os.path.exists(benchmark_path):
        pytest.skip("Benchmark data not available")

    # n_r = pypsa.Network(benchmark_path)

    # Create a new network for the stochastic model
    # n = stochastic_benchmark_network
    # TODO

    # # Check that it solves to optimality
    # status, _ = n.optimize(solver_name="highs")
    # assert status == "ok"

    # # Compare generator capacities (these are the main result of stochastic planning)
    # equal(
    #     n.generators.p_nom_opt,
    #     n_r.generators.p_nom_opt,
    #     decimal=2,
    # )

    # # Compare objective value
    # equal(n.objective, n_r.objective, decimal=2)


def test_solved_network_advanced():
    """
    Placeholder: solve the stochastic problem and compare a solution with a known result.
    Advanced test case with multiple components and constraints.
    """
    pass


def test_solved_network_multiperiod():
    """
    Placeholder: solve the stochastic problem and compare a solution with a known result.
    Advanced test case with multiple optimization periods.
    """
    pass


def test_objective_scaling_with_weights():
    """
    Placeholder: verify that the objective function is correctly scaled with scenario weights.
    Check that generation costs and investment costs are properly weighted by scenario probabilities.
    """
    pass


def test_weight_sensitivity():
    """
    Placeholder: test that changing scenario weights affects the solution accordingly.
    Compare solutions with different probability distributions across the same scenarios.
    """
    pass
