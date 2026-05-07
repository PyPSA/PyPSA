# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for piecewise linear cost curves in the optimisation objective."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import pypsa


def _formulation(n: pypsa.Network, name: str, **kwargs: Any) -> Any:
    model = n.optimize.create_model(include_objective_constant=False, **kwargs)
    return model._piecewise_formulations[name]


@pytest.fixture
def convex_marginal_cost_network() -> pypsa.Network:
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add(
        "Generator",
        "gen",
        bus="bus0",
        p_nom=100,
        marginal_cost=pd.DataFrame(
            {"p_pu": [1.0, 0.0, 0.5], "marginal_cost": [40.0, 10.0, 20.0]}
        ),
    )
    n.add("Load", "load", bus="bus0", p_set=50)
    return n


@pytest.fixture
def convex_capital_cost_network() -> pypsa.Network:
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add(
        "Generator",
        "gen",
        bus="bus0",
        p_nom_extendable=True,
        p_nom_max=100,
        capital_cost=pd.DataFrame(
            {"p_nom": [100.0, 0.0, 50.0], "capital_cost": [4.0, 1.0, 2.0]}
        ),
    )
    n.add("Load", "load", bus="bus0", p_set=50)
    return n


@pytest.fixture
def concave_marginal_cost_network() -> pypsa.Network:
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add(
        "Generator",
        "gen",
        bus="bus0",
        p_nom=100,
        marginal_cost=pd.DataFrame(
            {"p_pu": [0.0, 0.5, 1.0], "marginal_cost": [4.0, 2.0, 1.0]}
        ),
    )
    n.add("Load", "load", bus="bus0", p_set=50)
    return n


def test_piecewise_marginal_cost() -> None:
    """gen1 runs at 25 MW (where its marginal cost then becomes more expensive than gen0); gen0 covers 55 MW."""
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=20)
    n.add(
        "Generator",
        "gen1",
        bus="bus0",
        p_nom=100,
        marginal_cost={0.1: 60, 0.5: 35.0, 1.0: 100.0},
    )
    n.add("Load", "load", bus="bus0", p_set=80)
    n.optimize()
    assert n.generators_t.p["gen1"].iloc[0] == pytest.approx(10.0, rel=1e-3)
    assert n.generators_t.p["gen0"].iloc[0] == pytest.approx(70.0, rel=1e-3)
    assert n.objective < 4000.0


@pytest.mark.parametrize(
    ("fixture_name", "formulation_name", "piecewise_options", "expected_convexity"),
    [
        (
            "convex_marginal_cost_network",
            "Generator-marginal_cost_piecewise",
            None,
            "convex",
        ),
        (
            "convex_capital_cost_network",
            "Generator-capital_cost_piecewise",
            [
                {
                    "component": "Generator",
                    "attribute": "capital_cost",
                    "operator": ">=",
                    "method": "lp",
                }
            ],
            "convex",
        ),
        (
            "concave_marginal_cost_network",
            "Generator-marginal_cost_piecewise",
            [
                {
                    "component": "Generator",
                    "attribute": "marginal_cost",
                    "operator": "<=",
                    "method": "lp",
                }
            ],
            "concave",
        ),
    ],
)
def test_piecewise_lp_formulation_method_and_convexity(
    request: pytest.FixtureRequest,
    fixture_name: str,
    formulation_name: str,
    piecewise_options: list[dict[str, Any]] | None,
    expected_convexity: str,
) -> None:
    n = request.getfixturevalue(fixture_name)
    formulation = _formulation(
        n, formulation_name, piecewise_options=piecewise_options or []
    )
    assert formulation.method == "lp"
    assert formulation.convexity == expected_convexity


def test_piecewise_unsorted_rows_get_sorted() -> None:
    """Shuffled segment rows must produce the same dispatch as sorted rows."""
    sorted_costs = pd.DataFrame(
        {"p_pu": [0.0, 0.5, 1.0], "marginal_cost": [10.0, 20.0, 40.0]}
    )
    shuffled_costs = sorted_costs.iloc[[2, 0, 1]].reset_index(drop=True)

    results = []
    for costs in (sorted_costs, shuffled_costs):
        n = pypsa.Network()
        n.add("Bus", "bus0")
        n.add("Generator", "gen", bus="bus0", p_nom=100, marginal_cost=costs)
        n.add("Load", "load", bus="bus0", p_set=50)
        n.optimize()
        results.append((n.objective, n.generators_t.p["gen"].iloc[0]))

    assert results[0] == pytest.approx(results[1], rel=1e-6)


def test_piecewise_ragged_trailing_nan_across_components() -> None:
    """Two generators with different breakpoint counts share one segment frame."""
    nan = float("nan")
    segments = pd.DataFrame(
        [
            [0.0, 10.0, 0.0, 5.0],
            [0.5, 20.0, 1.0, 25.0],
            [1.0, 40.0, nan, nan],
        ],
        columns=pd.MultiIndex.from_tuples(
            [
                ("gen0", "p_pu"),
                ("gen0", "marginal_cost"),
                ("gen1", "p_pu"),
                ("gen1", "marginal_cost"),
            ],
            names=["name", "attribute"],
        ),
    )

    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add(
        "Generator",
        ["gen0", "gen1"],
        bus="bus0",
        p_nom=100,
        marginal_cost=segments,
    )
    n.add("Load", "load", bus="bus0", p_set=80)
    n.optimize()

    assert n.generators_t.p["gen0"].iloc[0] + n.generators_t.p["gen1"].iloc[0] == (
        pytest.approx(80.0, rel=1e-3)
    )


# TODO-1603: unmark once we have dealt with initial cost
@pytest.mark.xfail(reason="Need to fix how we handle initial cost (at zero p_nom)")
def test_piecewise_capital_cost() -> None:
    """Network with piecewise capital cost (diseconomies of scale = convex total cost).

    gen2: extendable, p_nom_extendable=True, p_nom_max=100 MW
          piecewise capital_cost — marginal capital cost (€/MW) at breakpoints:
          p_nom=0 -> 0.5 €/MW, p_nom=100 -> 1.5 €/MW  (linearly increasing)
    load: 80 MW

    Optimal: gen2 builds exactly 80 MW.
    Total cost at 80 MW = ∫₀^80 (0.5 + p/100) dp = [0.5p + p²/200]₀^80 = 40 + 32 = 72 €
    """
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add(
        "Generator",
        "gen2",
        bus="bus0",
        p_nom_extendable=True,
        p_nom_max=100,
        capital_cost=pd.DataFrame({"p_nom": [0.0, 100.0], "capital_cost": [0.5, 1.5]}),
    )
    n.add("Load", "load", bus="bus0", p_set=80)
    n.optimize()

    assert n.objective == pytest.approx(70.0, rel=1e-3)
    assert n.generators.p_nom_opt["gen2"] == pytest.approx(80.0, rel=1e-3)
