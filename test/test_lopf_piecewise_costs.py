# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for piecewise linear cost curves in the optimisation objective."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import pypsa


class TestPiecewiseCostsDefine:
    @pytest.fixture
    def convex_marginal_cost_network(self) -> pypsa.Network:
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
    def convex_capital_cost_network(self) -> pypsa.Network:
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
    def concave_marginal_cost_network(self) -> pypsa.Network:
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
                        "sign": ">=",
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
                        "sign": "<=",
                        "method": "lp",
                    }
                ],
                "concave",
            ),
        ],
    )
    def test_piecewise_lp_formulation_method_and_convexity(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
        formulation_name: str,
        piecewise_options: list[dict[str, Any]] | None,
        expected_convexity: str,
    ) -> None:
        n = request.getfixturevalue(fixture_name)

        model = n.optimize.create_model(
            include_objective_constant=False, piecewise_options=piecewise_options or []
        )
        formulation = model._piecewise_formulations[formulation_name]
        assert formulation.method == "lp"
        assert formulation.convexity == expected_convexity

    def test_piecewise_unsorted_rows_get_sorted(self) -> None:
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

    def test_piecewise_ragged_trailing_nan_across_components(self) -> None:
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


class TestPiecewiseCostsResults:
    def test_piecewise_marginal_cost(self) -> None:
        """gen1 runs at 50 MW (where its marginal cost then becomes more expensive than gen0); gen0 covers 30 MW."""
        n = pypsa.Network()
        n.add("Bus", "bus0")
        n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=50)
        n.add(
            "Generator",
            "gen1",
            bus="bus0",
            p_nom=100,
            marginal_cost={0.0: 0.0, 0.1: 60.0, 0.5: 35.0, 1.0: 100.0},
        )
        n.add("Load", "load", bus="bus0", p_set=80)
        n.optimize()
        assert n.c.generators.dynamic.p.squeeze().equals(
            pd.Series({"gen0": 30.0, "gen1": 50})
        )
        # expected: gen1 overall marginal cost is (0.1 * 60 + 0.4 * 35) / 0.5 = 40
        assert n.c.generators.dynamic.marginal_cost_piecewise_opt.squeeze().equals(
            pd.Series({"gen0": 0.0, "gen1": 40.0})
        )

    def test_piecewise_capital_cost(self) -> None:
        """Network with piecewise capital cost (diseconomies of scale = convex total cost).

        gen1 is cheaper than gen0 (1.8 EUR/MW) until 50MW capacity, where the marginal capital cost rate increases to 2.0 EUR/MW.
        """
        n = pypsa.Network()
        n.add("Bus", "bus0")
        n.add(
            "Generator",
            "gen0",
            bus="bus0",
            p_nom_extendable=True,
            p_nom_max=100,
            capital_cost=1.8,
        )
        n.add(
            "Generator",
            "gen1",
            bus="bus0",
            p_nom_extendable=True,
            p_nom_max=100,
            capital_cost=pd.DataFrame(
                {"p_nom": [0.0, 10, 50, 100.0], "capital_cost": [0.0, 1, 1.5, 2.0]}
            ),
        )
        n.add("Load", "load", bus="bus0", p_set=150)
        n.optimize()

        assert n.c.generators.static.p_nom_opt.equals(
            pd.Series({"gen0": 100.0, "gen1": 50})
        )
        # expected: gen1 overall capital cost is (0.1 * 1 + 0.4 * 1.5) / 0.5 = 1.4
        assert n.c.generators.static.capital_cost_piecewise_opt.equals(
            pd.Series({"gen0": 0.0, "gen1": 1.4})
        )
