# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.optimization.mga import (
    generate_directions_evenly_spaced,
    generate_directions_halton,
    generate_directions_random,
)

SOLVER_KWARGS = {"solver_name": "highs", "log_to_console": False}


@pytest.fixture
def network_factory():
    """Two substitutable technologies and optional brownfield/fixed capacity."""

    def factory(
        existing_capacity=0, fixed_capacity=0, multi_investment=False, fom=False
    ):
        n = pypsa.Network()
        if multi_investment:
            n.set_snapshots(pd.MultiIndex.from_product([[2030, 2040], [0]]))
            n.investment_period_weightings["objective"] = [2, 3]
        n.add("Carrier", "electricity")
        n.add("Bus", "bus", carrier="electricity")
        n.add(
            "Generator",
            "cheap",
            bus="bus",
            carrier="electricity",
            capital_cost=10,
            marginal_cost=1,
            fom_cost=1 if fom else 0,
            p_nom=existing_capacity,
            p_nom_min=existing_capacity,
            p_nom_extendable=True,
        )
        n.add(
            "Generator",
            "expensive",
            bus="bus",
            carrier="electricity",
            capital_cost=20,
            marginal_cost=1,
            fom_cost=2 if fom else 0,
            p_nom_extendable=True,
        )
        if fixed_capacity:
            n.add(
                "Generator",
                "fixed",
                bus="bus",
                carrier="electricity",
                p_nom=fixed_capacity,
                capital_cost=100,
                fom_cost=5 if fom else 0,
            )
        n.add("Load", "load", bus="bus", carrier="electricity", p_set=100)
        return n

    return factory


@pytest.fixture
def dimensions():
    return {
        name: {"Generator": {"p_nom": {name: 1}}} for name in ["cheap", "expensive"]
    }


def _solve(n, include_constant=True, multi_investment=False):
    assert n.optimize(
        include_objective_constant=include_constant,
        multi_investment_periods=multi_investment,
        **SOLVER_KWARGS,
    ) == ("ok", "optimal")


@pytest.mark.parametrize("method", ["weights", "direction"])
@pytest.mark.parametrize("include_constant", [True, False])
@pytest.mark.parametrize(
    ("existing_capacity", "fixed_capacity"),
    [(0, 0), (20, 0), (0, 20), (20, 20)],
    ids=["greenfield", "brownfield", "fixed", "mixed"],
)
@pytest.mark.parametrize("multi_investment", [False, True], ids=["single", "multi"])
@pytest.mark.parametrize("slack", [0, 0.1])
def test_total_cost_budget(
    network_factory,
    dimensions,
    method,
    include_constant,
    existing_capacity,
    fixed_capacity,
    multi_investment,
    slack,
):
    """Both entry points obey the same analytical total-cost budget."""
    n = network_factory(existing_capacity, fixed_capacity, multi_investment)
    _solve(n, include_constant, multi_investment)
    optimal_cost = fixed_capacity * 100 + (100 - fixed_capacity) * 11
    kwargs = dict(
        slack=slack,
        multi_investment_periods=multi_investment,
        model_kwargs={"include_objective_constant": include_constant},
        **SOLVER_KWARGS,
    )
    if method == "weights":
        result = n.optimize.optimize_mga(weights=dimensions["cheap"], **kwargs)
        assert n.meta["weights"] == dimensions["cheap"]
        assert n.meta["sense"] == 1
    else:
        direction = {"cheap": -1, "expensive": 1}
        result = n.optimize.optimize_mga_in_direction(
            direction=direction,
            dimensions=dimensions,
            **kwargs,
        )
        assert n.meta["direction"] == direction
        assert n.meta["dimensions"] == dimensions
        assert result[:2] == ("ok", "optimal")
        pd.testing.assert_series_equal(
            result[2],
            n.c.generators.static.p_nom_opt.reindex(dimensions),
            check_names=False,
        )
    assert result[:2] == ("ok", "optimal")
    assert n.meta["slack"] == slack

    # Replacing one unit of cheap capacity costs ten extra currency units.
    # Calculate expectations independently of the statistics building the budget.
    replacement = slack * optimal_cost / 10
    capacities = n.c.generators.static.p_nom_opt
    assert capacities["cheap"] == pytest.approx(100 - fixed_capacity - replacement)
    assert capacities["expensive"] == pytest.approx(replacement)
    total_cost = (
        capacities["cheap"] * 10
        + capacities["expensive"] * 20
        + fixed_capacity * 100
        + 100
        - fixed_capacity
    )
    period_weight = 5 if multi_investment else 1
    assert total_cost * period_weight == pytest.approx(
        (1 + slack) * optimal_cost * period_weight
    )


@pytest.mark.parametrize("include_constant", [True, False])
@pytest.mark.parametrize("multi_investment", [False, True])
def test_budget_with_fixed_om(
    network_factory, dimensions, include_constant, multi_investment
):
    n = network_factory(20, 20, multi_investment, fom=True)
    _solve(n, include_constant, multi_investment)
    assert n.optimize.optimize_mga(
        weights=dimensions["cheap"],
        slack=0.1,
        multi_investment_periods=multi_investment,
        model_kwargs={"include_objective_constant": include_constant},
        **SOLVER_KWARGS,
    ) == ("ok", "optimal")
    # Full cost is 20 * (100 + 5) + 80 * (10 + 1 + 1) = 3060.
    # Replacing capacity costs (20 + 2) - (10 + 1) = 11 per unit.
    replacement = 0.1 * 3060 / 11
    assert n.c.generators.static.p_nom_opt["cheap"] == pytest.approx(80 - replacement)
    assert n.c.generators.static.p_nom_opt["expensive"] == pytest.approx(replacement)


def test_budget_with_retiring_fixed_capacity(network_factory, dimensions):
    """Fixed costs must match the periods in which the asset is active."""
    n = network_factory(20, 20, multi_investment=True)
    n.c.generators.static.loc["fixed", ["build_year", "lifetime"]] = [2030, 5]
    _solve(n, include_constant=False, multi_investment=True)
    assert n.optimize.optimize_mga(
        weights=dimensions["cheap"],
        slack=0.1,
        multi_investment_periods=True,
        model_kwargs={"include_objective_constant": False},
        **SOLVER_KWARGS,
    ) == ("ok", "optimal")
    # The fixed generator operates only in 2030; 2040 needs 100 units of capacity.
    optimal_cost = 2 * (100 * 10 + 20 * 100 + 80) + 3 * (100 * 10 + 100)
    replacement = 0.1 * optimal_cost / ((2 + 3) * (20 - 10))
    assert n.c.generators.static.p_nom_opt["cheap"] == pytest.approx(100 - replacement)


def test_budget_with_fixed_storage(network_factory, dimensions):
    """Include fixed assets with energy capacity as well as power capacity."""
    n = network_factory(20, 20)
    n.add(
        "Store",
        "store",
        bus="bus",
        carrier="electricity",
        e_nom=10,
        capital_cost=7,
        e_cyclic=True,
    )
    _solve(n, include_constant=False)
    assert n.optimize.optimize_mga(
        weights=dimensions["cheap"],
        slack=0.1,
        model_kwargs={"include_objective_constant": False},
        **SOLVER_KWARGS,
    ) == ("ok", "optimal")
    optimal_cost = 80 * 11 + 20 * 100 + 10 * 7
    replacement = 0.1 * optimal_cost / (20 - 10)
    assert n.c.generators.static.p_nom_opt["cheap"] == pytest.approx(80 - replacement)


def test_budget_ignores_statistics_rounding(network_factory, dimensions):
    n = network_factory()
    n.c.generators.static.loc["cheap", "capital_cost"] = 10.00347
    _solve(n)
    with pypsa.option_context("params.statistics.round", 1):
        assert n.optimize.optimize_mga(
            weights=dimensions["cheap"],
            slack=0.1,
            **SOLVER_KWARGS,
        ) == ("ok", "optimal")
    replacement = 0.1 * 1100.347 / (20 - 10.00347)
    assert n.c.generators.static.p_nom_opt["cheap"] == pytest.approx(100 - replacement)


@pytest.mark.parametrize(
    ("initial_constant", "mga_constant"), [(True, False), (False, True)]
)
def test_changed_objective_constant(
    network_factory, dimensions, initial_constant, mga_constant
):
    n = network_factory(20, 20)
    _solve(n, initial_constant)
    assert n.optimize.optimize_mga(
        weights=dimensions["cheap"],
        slack=0.1,
        model_kwargs={"include_objective_constant": mga_constant},
        **SOLVER_KWARGS,
    ) == ("ok", "optimal")
    assert n.c.generators.static.p_nom_opt["cheap"] == pytest.approx(51.2)


@pytest.mark.parametrize(
    ("sense", "capacity"), [("min", 89), (1, 89), ("max", 111), (-1, 111)]
)
def test_optimization_sense(network_factory, dimensions, sense, capacity):
    n = network_factory()
    _solve(n)
    weights = {"Generator": {"p_nom": pd.Series({"cheap": 1})}}
    assert n.optimize.optimize_mga(
        weights=weights,
        sense=sense,
        slack=0.1,
        **SOLVER_KWARGS,
    ) == ("ok", "optimal")
    assert n.c.generators.static.p_nom_opt["cheap"] == pytest.approx(capacity)
    assert n.meta["weights"] == dimensions["cheap"]


@pytest.mark.parametrize("as_dataframe", [False, True])
def test_multiple_directions(network_factory, dimensions, as_dataframe):
    n = network_factory(20, 20)
    _solve(n)
    directions = pd.DataFrame(
        [{"cheap": -1, "expensive": 1}, {"cheap": 1, "expensive": -1}]
    )
    successful, coordinates = n.optimize.optimize_mga_in_multiple_directions(
        directions=directions if as_dataframe else directions.to_dict("records"),
        dimensions=dimensions,
        slack=0.1,
        max_parallel=1,
        model_kwargs={"include_objective_constant": False},
        **SOLVER_KWARGS,
    )
    pd.testing.assert_frame_equal(successful, directions)
    np.testing.assert_allclose(
        coordinates[["cheap", "expensive"]], [[51.2, 28.8], [108.8, 0]]
    )


@pytest.mark.parametrize(
    "method",
    [
        "optimize_mga",
        "optimize_mga_in_direction",
        "project_solved",
        "_add_near_opt_constraint",
    ],
)
def test_requires_solved_network(network_factory, dimensions, method):
    n = network_factory()
    kwargs = {}
    if method in ["optimize_mga_in_direction", "project_solved"]:
        kwargs["dimensions"] = dimensions
    if method == "optimize_mga_in_direction":
        kwargs["direction"] = {"cheap": -1, "expensive": 1}
    with pytest.raises(ValueError, match="Network needs to be solved"):
        getattr(n.optimize, method)(**kwargs)


def test_mismatched_dimensions(network_factory, dimensions):
    n = network_factory()
    with pytest.raises(
        ValueError, match="Keys of `direction` and `dimensions` arguments must match"
    ):
        n.optimize.optimize_mga_in_direction(
            direction={"cheap": 1}, dimensions=dimensions
        )


@pytest.mark.parametrize("sense", ["invalid", 0])
def test_invalid_sense(network_factory, dimensions, sense):
    n = network_factory()
    _solve(n)
    with pytest.raises(ValueError, match="Could not parse optimization sense"):
        n.optimize.optimize_mga(weights=dimensions["cheap"], sense=sense)


@pytest.mark.parametrize(
    "generate", [generate_directions_random, generate_directions_halton]
)
@pytest.mark.parametrize("keys", [["x", "y"], ["x", "y", "z"]])
def test_sampled_directions(generate, keys):
    directions = generate(keys, 8, seed=42)
    assert directions.shape == (8, len(keys))
    assert list(directions.columns) == keys
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1)
    pd.testing.assert_frame_equal(directions, generate(keys, 8, seed=42))
    assert not directions.equals(generate(keys, 8, seed=0))


def test_evenly_spaced_directions():
    directions = generate_directions_evenly_spaced(["x", "y"], 4)
    assert list(directions.columns) == ["x", "y"]
    np.testing.assert_allclose(
        directions, [[1, 0], [0, 1], [-1, 0], [0, -1]], atol=1e-15
    )


@pytest.mark.parametrize("keys", [["x"], ["x", "y", "z"]])
def test_evenly_spaced_requires_two_dimensions(keys):
    with pytest.raises(ValueError, match="This function only supports two keys"):
        generate_directions_evenly_spaced(keys, 4)
