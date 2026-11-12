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
                {"p_pu": [0.0, 0.5, 1.0], "marginal_cost": [0.0, 1.0, 4.0]}
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
                {"p_nom": [0.0, 50.0, 100.0], "capital_cost": [0.0, 2.0, 4.0]}
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
                {"p_pu": [0.0, 0.5, 1.0], "marginal_cost": [0.0, 4.0, 1.0]}
            ),
        )
        n.add("Load", "load", bus="bus0", p_set=50)
        return n

    @pytest.mark.parametrize(
        ("fixture_name", "formulation_name", "piecewise_options"),
        [
            pytest.param(
                "convex_marginal_cost_network",
                "Generator-marginal_cost_piecewise",
                None,
                id="convex-default",
            ),
            pytest.param(
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
                id="convex-capital-lp",
            ),
            pytest.param(
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
                id="concave-lp-upper",
            ),
        ],
    )
    def test_piecewise_lp_formulation(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
        formulation_name: str,
        piecewise_options: list[dict[str, Any]] | None,
    ) -> None:
        n = request.getfixturevalue(fixture_name)

        model = n.optimize.create_model(
            include_objective_constant=False, piecewise_options=piecewise_options or []
        )
        assert f"{formulation_name}_chord" in model.constraints, "lp states chords"
        assert not model.binaries, "lp needs no binaries"

    @pytest.mark.parametrize(
        ("sign", "method"),
        [
            pytest.param(">=", "incremental", id="incremental"),
            pytest.param("==", "auto", id="auto-equality"),
        ],
    )
    def test_piecewise_incremental_option_overrides_lp(
        self, convex_marginal_cost_network: pypsa.Network, sign: str, method: str
    ) -> None:
        option = {
            "component": "Generator",
            "attribute": "marginal_cost",
            "sign": sign,
            "method": method,
        }
        model = convex_marginal_cost_network.optimize.create_model(
            include_objective_constant=False, piecewise_options=[option]
        )
        assert "Generator-marginal_cost_piecewise_order_binary" in model.binaries
        assert "Generator-marginal_cost_piecewise_chord" not in model.constraints

    @pytest.mark.parametrize(
        ("options", "error", "match"),
        [
            pytest.param(
                [{"name": "gen", "method": "lp"}],
                NotImplementedError,
                "named units",
                id="name-filter",
            ),
            pytest.param(
                [{"method": "lp"}, {"method": "incremental"}],
                NotImplementedError,
                "Several piecewise options",
                id="two-options",
            ),
            pytest.param([{"method": "sos2"}], NotImplementedError, "sos2", id="sos2"),
            pytest.param(
                [{"sign": "<="}], NotImplementedError, "name the method", id="auto-sign"
            ),
            pytest.param(
                [{"method": "lp", "committable": True}],
                ValueError,
                "status gate",
                id="lp-committable",
            ),
        ],
    )
    def test_piecewise_options_refused(
        self,
        convex_marginal_cost_network: pypsa.Network,
        options: list[dict[str, Any]],
        error: type[Exception],
        match: str,
    ) -> None:
        n = convex_marginal_cost_network
        base = {"component": "Generator", "attribute": "marginal_cost", "sign": ">="}
        if any(option.pop("committable", False) for option in options):
            n.c.generators.static.loc["gen", "committable"] = True
        with pytest.raises(error, match=match):
            n.optimize.create_model(
                include_objective_constant=False,
                piecewise_options=[base | option for option in options],
            )

    @pytest.mark.parametrize(
        ("attr", "value"),
        [("p_nom_extendable", True), ("p_nom", 0.0), ("p_nom", float("inf"))],
    )
    def test_piecewise_per_unit_curve_needs_fixed_nominal(
        self, convex_marginal_cost_network: pypsa.Network, attr: str, value: Any
    ) -> None:
        n = convex_marginal_cost_network
        n.c.generators.static.loc["gen", attr] = value
        with pytest.raises(
            ValueError, match="Generator_marginal_cost_curve_on_a_fixed_build"
        ):
            n.optimize.create_model(include_objective_constant=False)

    def test_piecewise_unsorted_rows_get_sorted(self) -> None:
        """Shuffled segment rows must produce the same dispatch as sorted rows."""
        sorted_costs = pd.DataFrame(
            {"p_pu": [0.0, 0.5, 1.0], "marginal_cost": [0.0, 20.0, 40.0]}
        )
        shuffled_costs = sorted_costs.iloc[[2, 0, 1]].reset_index(drop=True)

        results = []
        for costs in (sorted_costs, shuffled_costs):
            n = pypsa.Network()
            n.add("Bus", "bus0")
            n.add("Generator", "gen", bus="bus0", p_nom=100, marginal_cost=costs)
            n.add("Load", "load", bus="bus0", p_set=50)
            n.optimize()
            results.append((n.objective, n.c.generators.dynamic.p["gen"].iloc[0]))

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

        assert n.c.generators.dynamic.p["gen0"].iloc[0] + n.c.generators.dynamic.p[
            "gen1"
        ].iloc[0] == (pytest.approx(80.0, rel=1e-3))


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

    def test_storage_unit_piecewise_marginal_cost(self) -> None:
        """StorageUnit piecewise marginal cost applies to the dispatch variable."""
        n = pypsa.Network()
        n.set_snapshots(range(2))
        n.add("Bus", "bus0")
        n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=50)
        n.add(
            "StorageUnit",
            "su0",
            bus="bus0",
            p_nom=100,
            max_hours=1,
            state_of_charge_initial=75,
            marginal_cost={0.0: 0.0, 0.5: 3.0, 1.0: 10.0},
        )
        n.add("Load", "load", bus="bus0", p_set=50)
        n.optimize()

        mc = n.c.storage_units.dynamic.marginal_cost_piecewise_opt["su0"]
        p = n.c.storage_units.dynamic.p["su0"]
        assert n.objective == pytest.approx((mc * p).sum() + (25 * 50), rel=1e-6)

    def test_store_piecewise_marginal_cost_applies_to_dispatch(self) -> None:
        """Store piecewise marginal cost prices dispatch ``p``, not the storage level."""
        n = pypsa.Network()
        n.set_snapshots(range(2))
        n.add("Bus", "bus0")
        n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=50)
        n.add(
            "Store",
            "store0",
            bus="bus0",
            e_nom=100,
            e_initial=75,
            marginal_cost={0.0: 0.0, 0.5: 3.0, 1.0: 10.0},
        )
        n.add("Load", "load", bus="bus0", p_set=50)
        n.optimize()
        mc = n.c.stores.dynamic.marginal_cost_piecewise_opt["store0"]
        p = n.c.stores.dynamic.p["store0"]
        assert n.objective == pytest.approx((mc * p).sum() + (25 * 50), rel=1e-6)

    def test_multi_invest_piecewise_capital_cost(self) -> None:
        """A constant-slope piecewise capital cost matches its linear equivalent."""

        def build(piecewise: bool) -> pypsa.Network:
            n = pypsa.Network(snapshots=range(2))
            n.investment_periods = [2020, 2030]
            n.add("Bus", "bus0")
            capital_cost: Any = {0.0: 2.0, 100.0: 2.0} if piecewise else 2.0
            n.add(
                "Generator",
                "gen0",
                bus="bus0",
                p_nom_extendable=True,
                p_nom_max=100,
                build_year=2020,
                lifetime=30,
                capital_cost=capital_cost,
                marginal_cost=1,
            )
            n.add("Load", "load", bus="bus0", p_set=50)
            n.optimize(multi_investment_periods=True)
            return n

        n_pw = build(piecewise=True)
        n_lin = build(piecewise=False)
        assert n_pw.objective == pytest.approx(n_lin.objective)
        assert n_pw.c.generators.static.p_nom_opt.item() == pytest.approx(
            n_lin.c.generators.static.p_nom_opt.item()
        )

    @pytest.mark.parametrize(
        ("attr", "value", "committable"),
        [
            ("marginal_cost", 10.0, False),
            ("marginal_cost", 10.0, True),
            ("efficiency", 0.5, False),
        ],
    )
    def test_multi_invest_piecewise_per_snapshot(
        self, attr: str, value: float, committable: bool
    ) -> None:
        """A constant-slope per-snapshot curve matches its linear equivalent (#2116)."""

        def build(piecewise: bool) -> pypsa.Network:
            n = pypsa.Network()
            n.set_snapshots(pd.MultiIndex.from_product([[2030, 2040], [0, 1]]))
            n.investment_periods = [2030, 2040]
            n.add("Bus", "bus0")
            n.add("Carrier", "gas", co2_emissions=1.0)
            n.add("Generator", "backup", bus="bus0", p_nom=200, marginal_cost=100)
            curve: Any = {0.0: 0.0, 1.0: value} if piecewise else value
            n.add(
                "Generator",
                "gen0",
                bus="bus0",
                carrier="gas",
                p_nom=100,
                p_min_pu=0.2 if committable else 0.0,
                committable=committable,
                **{"marginal_cost": 1.0, attr: curve},
            )
            n.add(
                "GlobalConstraint",
                "co2_limit",
                sense="<=",
                carrier_attribute="co2_emissions",
                constant=200,
            )
            n.add("Load", "load", bus="bus0", p_set=[50, 60, 70, 80])
            n.optimize(multi_investment_periods=True)
            return n

        n_pw = build(piecewise=True)
        n_lin = build(piecewise=False)
        assert n_pw.objective == pytest.approx(n_lin.objective)


# TODO: update to a non-zero starting x point on the piecewise curve once we handle non-zero intercepts on cumulative curves
class TestPiecewiseCostsStatus:
    @pytest.fixture
    def base_network(self):
        n = pypsa.Network()
        n.add("Bus", "bus0")
        n.add("Load", "load", bus="bus0", p_set=80)
        return n

    def test_piecewise_marginal_cost_considers_status(self, base_network):
        """Piecewise marginal cost considers the binary unit commitment status of the component."""
        n = base_network
        n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=50)
        n.add(
            "Generator",
            "gen-committable",
            bus="bus0",
            p_nom=100,
            # effectively setting a p_min_pu of 0.1 which shouldn't be binding because the generator is committable and can be turned off
            marginal_cost={0.0: 60, 0.5: 75, 1.0: 100.0},
            committable=True,
        )
        n.optimize()
        assert (n.c.generators.dynamic.p["gen-committable"] == 0).all()

    def test_piecewise_marginal_cost_considers_status_two_piecewise_costs(
        self, base_network
    ):
        """Piecewise marginal cost considers the binary unit commitment status of the component when there is a mix of committable and non-committable piecewise generators."""
        n = base_network
        n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=50)
        n.add(
            "Generator",
            "gen1",
            bus="bus0",
            p_nom=100,
            marginal_cost={0.0: 60.0, 0.1: 60.0, 0.5: 35.0, 1.0: 100.0},
        )
        n.add(
            "Generator",
            "gen-committable",
            bus="bus0",
            p_nom=100,
            marginal_cost={0.0: 60, 0.5: 75, 1.0: 100.0},
            committable=True,
        )
        n.optimize()
        assert (n.c.generators.dynamic.p["gen-committable"] == 0).all()

    def test_piecewise_marginal_cost_considers_status_two_committable_generators(
        self, base_network
    ):
        """Piecewise marginal cost considers the binary unit commitment status of the component when there is a mix of committable and non-committable piecewise generators."""
        n = base_network
        n.add(
            "Generator",
            "gen0",
            bus="bus0",
            p_nom=100,
            marginal_cost=50,
            committable=True,
        )
        n.add(
            "Generator",
            "gen-committable",
            bus="bus0",
            p_nom=100,
            marginal_cost={0.0: 60, 0.5: 75, 1.0: 100.0},
            committable=True,
        )
        n.optimize()
        assert (n.c.generators.dynamic.p["gen-committable"] == 0).all()


class TestPiecewiseCostsErrors:
    def test_piecewise_capital_cost_with_overnight_cost_raises(self) -> None:
        """Defining overnight_cost next to a piecewise capital_cost curve is ambiguous."""
        n = pypsa.Network(snapshots=range(2))
        n.add("Bus", "bus0")
        n.add(
            "Generator",
            "gen0",
            bus="bus0",
            p_nom_extendable=True,
            p_nom_max=100,
            capital_cost={0.0: 2.0, 100.0: 2.0},
            overnight_cost=1000,
            lifetime=20,
        )
        n.add("Load", "load", bus="bus0", p_set=50)
        with pytest.raises(ValueError, match="overnight_cost"):
            n.optimize.create_model(include_objective_constant=False)


SUPPLIERS = {
    "Generator": {"bus": "bus0"},
    "Link": {"bus0": "bus1", "bus1": "bus0"},
    "Process": {"bus0": "bus1", "bus1": "bus0", "rate0": -1, "rate1": 1},
    "StorageUnit": {"bus": "bus0", "max_hours": 1, "state_of_charge_initial": 100},
    "Store": {"bus": "bus0", "e_initial": 100},
    "Line": {"bus0": "bus1", "bus1": "bus0", "x": 0.1},
    "Transformer": {"bus0": "bus1", "bus1": "bus0", "x": 0.1},
}
CURVES = [
    *((c, "marginal_cost") for c in list(SUPPLIERS)[:5]),
    *((c, "capital_cost") for c in SUPPLIERS),
]


def _supplied_load(component: str, attribute: str, cost: Any) -> pypsa.Network:
    """Return a load that `component` supplies cheaper than a backup, at `cost`."""
    n = pypsa.Network()
    n.add("Bus", ["bus0", "bus1"])
    n.add("Generator", "source", bus="bus1", p_nom=1000, marginal_cost=1)
    n.add("Generator", "backup", bus="bus0", p_nom=1000, marginal_cost=100)
    n.add("Load", "load", bus="bus0", p_set=50)
    nom = n.components[component]._operational_attrs["nom"]
    build = {nom: 100}
    if attribute == "capital_cost":
        build = {f"{nom}_extendable": True, f"{nom}_max": 100}
    n.add(component, "unit", **SUPPLIERS[component], **build, **{attribute: cost})
    return n


@pytest.mark.parametrize("method", ["lp", "incremental"])
@pytest.mark.parametrize(("component", "attribute"), CURVES)
def test_constant_slope_curve_costs_its_slope(
    component: str, attribute: str, method: str
) -> None:
    linear = _supplied_load(component, attribute, 2.0)
    linear.optimize()
    end = 100.0 if attribute == "capital_cost" else 1.0
    n = _supplied_load(component, attribute, {0.0: 0.0, end / 2: 2.0, end: 2.0})
    option = {
        "component": component,
        "attribute": attribute,
        "method": method,
        "sign": ">=",
    }
    n.optimize(piecewise_options=[option])
    assert n.objective == pytest.approx(linear.objective, rel=1e-6)
