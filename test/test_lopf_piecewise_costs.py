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


def snapshot_curve() -> pd.DataFrame:
    return pd.concat(
        {
            0: pd.DataFrame(
                {"p_pu": [0, 0.9, 0.95, 1], "marginal_cost": [0, 50, 56, 60]}
            ),
            1: pd.DataFrame(
                {
                    "p_pu": [0, 0.45, 0.475, 0.5, 1],
                    "marginal_cost": [0, 55, 61, 65, 65],
                }
            ),
            2: pd.DataFrame({"p_pu": [0, 1], "marginal_cost": [0, 47]}),
        },
        names=["snapshot", "breakpoint"],
    )


def snapshot_curve_network(curve: pd.DataFrame | None = None) -> pypsa.Network:
    n = pypsa.Network(snapshots=range(3))
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        p_max_pu=[1, 0.5, 0],
        marginal_cost=snapshot_curve() if curve is None else curve,
    )
    n.add("Generator", "backup", bus="bus", p_nom=100, marginal_cost=1000)
    n.add("Load", "load", bus="bus", p_set=[97, 48.5, 10])
    return n


def assert_snapshot_curve_dispatch(n: pypsa.Network, snapshots: list[int]) -> None:
    status, _ = n.optimize(snapshots=snapshots, include_objective_constant=False)
    assert status == "ok"
    objective = pd.Series([4900, 2692.5, 10000])
    assert n.objective == pytest.approx(objective[snapshots].sum())
    p = n.generators_t.p.loc[snapshots, "gen"]
    price = n.buses_t.marginal_price.loc[snapshots, "bus"]
    assert p.tolist() == pytest.approx(pd.Series([97, 48.5, 0])[snapshots].tolist())
    assert price.tolist() == pytest.approx(
        pd.Series([60, 65, 1000])[snapshots].tolist()
    )


class TestSnapshotPiecewiseCosts:
    @pytest.mark.parametrize("snapshots", [[0, 1, 2], [1]])
    def test_snapshot_marginal_cost(self, snapshots: list[int]) -> None:
        assert_snapshot_curve_dispatch(snapshot_curve_network(), snapshots)

    @pytest.mark.parametrize("static_first", [True, False])
    def test_mixed_static_and_snapshot_curves(self, static_first: bool) -> None:
        n = pypsa.Network(snapshots=range(3))
        n.add("Bus", "bus")
        components = [
            ("backup", {"marginal_cost": {0: 0, 1: 1000}}),
            ("gen", {"marginal_cost": snapshot_curve(), "p_max_pu": [1, 0.5, 0]}),
        ]
        for name, kwargs in components if static_first else components[::-1]:
            n.add("Generator", name, bus="bus", p_nom=100, **kwargs)
        n.add("Load", "load", bus="bus", p_set=[97, 48.5, 10])
        assert_snapshot_curve_dispatch(n, [0, 1, 2])

    def test_snapshot_curve_netcdf_roundtrip(self, tmp_path: Any) -> None:
        n = snapshot_curve_network()
        n.export_to_netcdf(tmp_path / "network.nc")
        m = pypsa.Network(tmp_path / "network.nc")
        pd.testing.assert_frame_equal(
            m.c.generators.piecewise["marginal_cost"],
            n.c.generators.piecewise["marginal_cost"],
        )
        assert_snapshot_curve_dispatch(m, [0, 1, 2])

    @pytest.mark.parametrize(
        ("drop", "match"),
        [
            ({"index": 1, "level": "snapshot"}, "cover all network snapshots"),
            ({"index": [(1, 4)]}, "end at p_pu=1"),
        ],
    )
    def test_snapshot_curve_invalid(self, drop: dict, match: str) -> None:
        curve = snapshot_curve().drop(**drop)
        with pytest.raises(ValueError, match=match):
            snapshot_curve_network(curve)

    def test_snapshot_curve_static_attribute_raises(self) -> None:
        curve = snapshot_curve().set_axis(["p_nom", "capital_cost"], axis=1)
        n = pypsa.Network(snapshots=range(3))
        n.add("Bus", "bus")
        with pytest.raises(ValueError, match="time-varying"):
            n.add("Generator", "gen", bus="bus", capital_cost=curve)

    def test_snapshot_curve_investment_periods_raise(self) -> None:
        n = pypsa.Network(snapshots=range(3))
        n.investment_periods = [2030]
        n.add("Bus", "bus")
        with pytest.raises(NotImplementedError, match="investment periods"):
            n.add("Generator", "gen", bus="bus", marginal_cost=snapshot_curve())
