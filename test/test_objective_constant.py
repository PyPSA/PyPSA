# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for include_objective_constant parameter in optimization."""

import warnings

import pytest

import pypsa


@pytest.fixture
def network_with_extendable_assets() -> pypsa.Network:
    """Create a network with extendable generators that have existing capacity."""
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", "bus")
    # Generator with existing capacity (p_nom > 0) and extendable
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,  # Existing capacity
        p_nom_extendable=True,
        capital_cost=1000,
        marginal_cost=10,
    )
    n.add("Load", "load", bus="bus", p_set=50)
    return n


@pytest.fixture
def simple_network() -> pypsa.Network:
    """Create a simple network without extendable assets (no objective constant)."""
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", "bus")
    n.add("Generator", "gen", bus="bus", p_nom=100, marginal_cost=10)
    n.add("Load", "load", bus="bus", p_set=50)
    return n


def test_include_objective_constant_true(
    network_with_extendable_assets: pypsa.Network,
) -> None:
    """Test that include_objective_constant=True includes the constant variable."""
    n = network_with_extendable_assets
    n.optimize(include_objective_constant=True, log_to_console=False)

    assert n.objective_constant is not None
    assert n.objective_constant > 0
    # The objective constant should be included, affecting the model objective
    assert "objective_constant" in n.model.variables


def test_include_objective_constant_false(
    network_with_extendable_assets: pypsa.Network,
) -> None:
    """Test that include_objective_constant=False excludes the constant variable."""
    n = network_with_extendable_assets
    n.optimize(include_objective_constant=False, log_to_console=False)

    # The objective constant should still be calculated and stored
    assert n.objective_constant is not None
    assert n.objective_constant > 0
    # But it should NOT be included as a variable in the model
    assert "objective_constant" not in n.model.variables


def test_include_objective_constant_none_raises_future_warning(
    network_with_extendable_assets: pypsa.Network,
) -> None:
    """Test that include_objective_constant=None raises FutureWarning."""
    n = network_with_extendable_assets
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        n.optimize(log_to_console=False)
        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert "include_objective_constant" in str(w[0].message)


def test_include_objective_constant_true_suppresses_warning(
    network_with_extendable_assets: pypsa.Network,
) -> None:
    """Test that setting include_objective_constant explicitly suppresses warning."""
    n = network_with_extendable_assets
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        n.optimize(include_objective_constant=True, log_to_console=False)
        future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(future_warnings) == 0


def test_include_objective_constant_false_suppresses_warning(
    network_with_extendable_assets: pypsa.Network,
) -> None:
    """Test that setting include_objective_constant=False suppresses warning."""
    n = network_with_extendable_assets
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        n.optimize(include_objective_constant=False, log_to_console=False)
        future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(future_warnings) == 0


def test_objective_value_consistency(
    network_with_extendable_assets: pypsa.Network,
) -> None:
    """Test that total system cost is consistent regardless of include_objective_constant.

    When include_objective_constant=True, the objective function subtracts the constant,
    so total cost = n.objective + n.objective_constant.

    When include_objective_constant=False, the objective function does NOT include the
    constant (neither as an addition nor subtraction), so total cost = n.objective + n.objective_constant.

    Both cases should give the same total system cost.
    """
    n = network_with_extendable_assets

    # Run with constant included (subtracts constant from objective function)
    n1 = n.copy()
    n1.optimize(include_objective_constant=True, log_to_console=False)
    # When included, n.objective = model_obj_value = CAPEX_new + OPEX - constant
    # Total cost = n.objective + n.objective_constant = CAPEX_new + OPEX
    assert n1.objective is not None
    assert n1.objective_constant is not None
    total_cost_with = n1.objective + n1.objective_constant

    # Run with constant excluded
    n2 = n.copy()
    n2.optimize(include_objective_constant=False, log_to_console=False)
    # When excluded, n.objective = model_obj_value = CAPEX_new + OPEX
    # Total cost = n.objective (no need to add constant, it's simply not in the objective)
    assert n2.objective is not None
    assert n2.objective_constant is not None
    total_cost_without = n2.objective

    # Both should represent the same system costs (without the fixed infrastructure costs)
    assert abs(total_cost_with - total_cost_without) < 1e-6

    # The objective_constant should be the same regardless of whether it was included
    assert abs(n1.objective_constant - n2.objective_constant) < 1e-6


def test_create_model_with_include_objective_constant(
    network_with_extendable_assets: pypsa.Network,
) -> None:
    """Test create_model method with include_objective_constant parameter."""
    n = network_with_extendable_assets

    # Create model with constant included
    n.optimize.create_model(include_objective_constant=True)
    assert "objective_constant" in n.model.variables

    # Create model with constant excluded
    n2 = network_with_extendable_assets
    n2.optimize.create_model(include_objective_constant=False)
    assert "objective_constant" not in n2.model.variables


def test_no_objective_constant_when_no_extendables(
    simple_network: pypsa.Network,
) -> None:
    """Test behavior when there's no objective constant (no extendable assets)."""
    n = simple_network
    n.optimize(include_objective_constant=True, log_to_console=False)

    # No extendable assets means no objective constant
    assert n.objective_constant == 0.0
    # The variable should not be created when constant is 0
    assert "objective_constant" not in n.model.variables
