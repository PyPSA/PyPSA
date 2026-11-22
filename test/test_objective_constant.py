# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for include_objective_constant parameter in optimization."""

import warnings
from typing import Any

import pytest

import pypsa
from pypsa import PyPSA2Warning


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


@pytest.mark.parametrize("build", ["optimize", "create_model"])
def test_none_warns_and_keeps_constant(
    network_with_extendable_assets: pypsa.Network, build: str
) -> None:
    """Test that the None flag warns and keeps the objective constant."""
    n = network_with_extendable_assets
    run = n.optimize if build == "optimize" else n.optimize.create_model
    kwargs: dict[str, Any] = {"log_to_console": False} if build == "optimize" else {}
    with pypsa.option_context("future.exclude_objective_constant", None):
        with pytest.warns(PyPSA2Warning, match="exclude_objective_constant") as record:
            run(**kwargs)
    assert record[0].filename == __file__
    assert "objective_constant" in n.model.variables


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


@pytest.mark.parametrize("flag", [True, False, None])
def test_flag_and_old_key_agree(
    network_with_extendable_assets: pypsa.Network, flag: bool | None
) -> None:
    """Test that the old key is the inverted alias of the new flag."""
    n_new = network_with_extendable_assets
    n_old = n_new.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyPSA2Warning)
        with pypsa.option_context("future.exclude_objective_constant", flag):
            n_new.optimize(log_to_console=False)
            expected = None if flag is None else not flag
            with pytest.warns(DeprecationWarning, match="deprecated"):
                pypsa.options.params.optimize.include_objective_constant = expected
            assert pypsa.options.future.exclude_objective_constant is flag
            n_old.optimize(log_to_console=False)
    assert n_old.objective == pytest.approx(n_new.objective)
    assert n_old.objective_constant == pytest.approx(n_new.objective_constant)
    if flag:
        assert n_new.objective_constant == 0.0
    else:
        assert n_new.objective_constant != 0.0


def test_keyword_overrides_flag(
    network_with_extendable_assets: pypsa.Network,
) -> None:
    """Test that an explicit keyword wins over the flag without warning."""
    n = network_with_extendable_assets
    with warnings.catch_warnings():
        warnings.simplefilter("error", PyPSA2Warning)
        with pypsa.option_context("future.exclude_objective_constant", True):
            n.optimize(include_objective_constant=True, log_to_console=False)
        assert n.objective_constant != 0.0
        with pypsa.option_context("future.exclude_objective_constant", None):
            n.optimize(include_objective_constant=False, log_to_console=False)
        assert n.objective_constant == 0.0
        with pypsa.option_context("future.exclude_objective_constant", False):
            n.optimize.create_model(include_objective_constant=False)
        assert n.objective_constant == 0.0


@pytest.mark.parametrize(
    ("multi_invest", "scenarios", "expected"),
    [
        (False, None, 101000.0),
        (True, None, 100500.0),
        (True, {"a": 0.3, "b": 0.7}, 100500.0),
    ],
)
def test_objective_constant_investment_periods(
    multi_invest: bool, scenarios: dict[str, float] | None, expected: float
) -> None:
    """Test objective constant with investment period weightings (#1957)."""

    def build() -> pypsa.Network:
        n = pypsa.Network(snapshots=range(3))
        if multi_invest:
            n.investment_periods = [2030, 2040]
            n.investment_period_weightings["objective"] = [1.0, 0.5]
        n.add("Bus", "bus")
        n.add(
            "Generator",
            ["gen", "late"],
            bus="bus",
            p_nom=[100, 10],
            p_nom_extendable=True,
            capital_cost=[1000, 100],
            marginal_cost=[10, 20],
            build_year=[2030, 2040],
            lifetime=[10, 50],
        )
        n.add("Load", "load", bus="bus", p_set=50)
        if scenarios:
            n.set_scenarios(scenarios)
        return n

    n1, n2 = build(), build()
    for n, include in [(n1, True), (n2, False)]:
        n.optimize(
            multi_investment_periods=multi_invest,
            include_objective_constant=include,
            log_to_console=False,
        )
    assert n1.objective is not None
    assert n2.objective is not None
    assert n1.objective_constant is not None
    assert n1.objective_constant == pytest.approx(expected)
    assert n1.objective + n1.objective_constant == pytest.approx(n2.objective)
