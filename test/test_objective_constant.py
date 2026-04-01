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
    """Test that include_objective_constant=False skips constant calculation."""
    n = network_with_extendable_assets
    n.optimize(include_objective_constant=False, log_to_console=False)

    # The objective constant should not be calculated
    assert n.objective_constant == 0.0
    # And should NOT be included as a variable in the model
    assert "objective_constant" not in n.model.variables


def test_include_objective_constant_none_raises_future_warning(
    network_with_extendable_assets: pypsa.Network,
) -> None:
    """Test that include_objective_constant=None raises FutureWarning."""
    n = network_with_extendable_assets
    pypsa.options.reset_option("params.optimize.include_objective_constant")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        n.optimize(log_to_console=False)
        future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(future_warnings) == 1
        assert "include_objective_constant" in str(future_warnings[0].message)


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
    """Test that optimization results are consistent regardless of include_objective_constant.

    When include_objective_constant=True, the objective function subtracts the constant,
    so: n.objective = CAPEX_new + OPEX - constant

    When include_objective_constant=False, no constant is calculated or subtracted,
    so: n.objective = CAPEX_new + OPEX

    Both should yield the same operational result (n.objective + n.objective_constant).
    """
    n = network_with_extendable_assets

    # Run with constant included (subtracts constant from objective function)
    n1 = n.copy()
    n1.optimize(include_objective_constant=True, log_to_console=False)
    assert n1.objective is not None
    assert n1.objective_constant is not None
    assert n1.objective_constant > 0  # Should be calculated

    # Run with constant excluded
    n2 = n.copy()
    n2.optimize(include_objective_constant=False, log_to_console=False)
    assert n2.objective is not None
    assert n2.objective_constant == 0.0  # Not calculated

    # Total cost (objective + constant) should be the same
    # With True: objective already has constant subtracted, so add it back
    # With False: objective is raw, constant is 0
    total_cost_with = n1.objective + n1.objective_constant
    total_cost_without = n2.objective + n2.objective_constant
    assert abs(total_cost_with - total_cost_without) < 1e-6


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


def test_create_model_raises_future_warning(
    network_with_extendable_assets: pypsa.Network,
) -> None:
    """Test that create_model raises FutureWarning when parameter not set."""
    n = network_with_extendable_assets
    pypsa.options.reset_option("params.optimize.include_objective_constant")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        n.optimize.create_model()
        future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(future_warnings) == 1
        assert "include_objective_constant" in str(future_warnings[0].message)


def test_options_include_objective_constant(
    network_with_extendable_assets: pypsa.Network,
) -> None:
    """Test that pypsa.options.params.optimize.include_objective_constant works."""
    n = network_with_extendable_assets

    # Set the option to False
    pypsa.options.params.optimize.include_objective_constant = False
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n.optimize(log_to_console=False)
            # No FutureWarning since option is set
            future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
            assert len(future_warnings) == 0
        # Constant should not be calculated
        assert n.objective_constant == 0.0
        assert "objective_constant" not in n.model.variables
    finally:
        # Reset to default
        pypsa.options.reset_option("params.optimize.include_objective_constant")


def test_options_include_objective_constant_create_model(
    network_with_extendable_assets: pypsa.Network,
) -> None:
    """Test that options work with create_model."""
    n = network_with_extendable_assets

    # Set the option to True
    pypsa.options.params.optimize.include_objective_constant = True
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n.optimize.create_model()
            future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
            assert len(future_warnings) == 0
        assert "objective_constant" in n.model.variables
    finally:
        pypsa.options.reset_option("params.optimize.include_objective_constant")
