# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for piecewise linear part-load efficiency curves in the optimisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pypsa


@pytest.fixture
def piecewise_efficiency_network() -> pypsa.Network:
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Carrier", "gas", co2_emissions=1.0)
    n.add(
        "Generator",
        "gen",
        carrier="gas",
        bus="bus0",
        p_nom=70,
        marginal_cost=20,
        efficiency={0.1: 0.3, 0.5: 0.4, 1.0: 0.6},
    )
    n.add(
        "GlobalConstraint",
        "co2_limit",
        sense="<=",
        carrier_attribute="co2_emissions",
        constant=160,
    )
    n.add("Load", "load", bus="bus0", p_set=50)
    return n


def test_piecewise_efficiency_uses_equality_formulation(
    piecewise_efficiency_network: pypsa.Network,
) -> None:
    model = piecewise_efficiency_network.optimize.create_model(
        include_objective_constant=False
    )
    formulation = model._piecewise_formulations["Generator-p_primary_piecewise"]
    assert formulation.method == "incremental"


def test_piecewise_efficiency_rejects_lp_equality(
    piecewise_efficiency_network: pypsa.Network,
) -> None:
    with pytest.raises(ValueError, match="method='lp' requires exactly one tuple"):
        piecewise_efficiency_network.optimize.create_model(
            include_objective_constant=False,
            piecewise_options=[
                {
                    "component": "Generator",
                    "attribute": "efficiency",
                    "sign": "==",
                    "method": "lp",
                }
            ],
        )


def test_piecewise_efficiency_co2_constraint_with_only_segmented_gens(
    piecewise_efficiency_network: pypsa.Network,
) -> None:
    """Regression: when every CO2-emitting generator has piecewise efficiency,
    ``linear_names`` is empty and ``primary_energy`` must still be initialised."""
    n = piecewise_efficiency_network
    n.optimize()
    assert n.c.generators.dynamic.p["gen"].iloc[0] == pytest.approx(50.0, rel=1e-3)


def test_piecewise_efficiency_two_primary_energy_constraints(
    piecewise_efficiency_network: pypsa.Network,
) -> None:
    """Regression: multiple primary-energy constraints share one piecewise aux variable."""
    n = piecewise_efficiency_network
    n.add(
        "GlobalConstraint",
        "co2_limit2",
        sense="<=",
        carrier_attribute="co2_emissions",
        constant=170,
    )
    status, _ = n.optimize()
    assert status == "ok"
    assert n.generators_t.p["gen"].iloc[0] == pytest.approx(50.0, rel=1e-3)


def test_piecewise_efficiency_gen() -> None:
    """gen0 is cheaper but cannot meet all load without hitting co2 limit. Gen1 comes in based on its piecewise curve rate."""
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Carrier", "gas", co2_emissions=1.0)
    x_points = np.array([0.1, 0.5, 1.0])
    y_points = np.array([0.3, 0.4, 0.6])
    n.add(
        "Generator",
        "gen0",
        carrier="gas",
        bus="bus0",
        p_nom=70,
        marginal_cost=15,
        efficiency=0.35,
    )
    n.add(
        "Generator",
        "gen1",
        carrier="gas",
        bus="bus0",
        p_nom=70,
        marginal_cost=20,
        efficiency=pd.DataFrame({"p_pu": x_points, "efficiency": y_points}),
    )
    n.add(
        "GlobalConstraint",
        "co2_limit",
        sense="<=",
        carrier_attribute="co2_emissions",
        constant=160,
    )
    n.add("Load", "load", bus="bus0", p_set=80)
    n.optimize()
    p = n.generators_t.p.iloc[0]
    fuel = p["gen0"] / 0.35 + np.interp(
        p["gen1"], 70 * x_points, 70 * x_points / y_points
    )
    assert fuel <= 160 * (1 + 1e-6)


class TestPiecewiseMultiPort2Bus:
    @pytest.fixture
    def base_network(self) -> pypsa.Network:
        n = pypsa.Network()
        n.set_snapshots(range(3))
        n.add("Bus", ["bus0", "bus1"])
        n.add("Carrier", "gas")
        n.add("Generator", "gen0", carrier="gas", bus="bus0", p_nom=150)
        n.add("Load", "load", bus="bus1", p_set=[20, 30, 40])
        return n

    @pytest.fixture
    def base_multiport_attrs(self) -> dict:
        return {
            "name": "multiport1",
            "carrier": "gas",
            "bus0": "bus0",
            "bus1": "bus1",
            "p_nom": 100,
            "p_nom_extendable": False,
            "marginal_cost": 20,
        }

    @pytest.fixture
    def piecewise_two_port_link_network(
        self, base_network: pypsa.Network, base_multiport_attrs: dict
    ) -> tuple[pypsa.Network, str]:
        base_network.add(
            "Link", efficiency={0.1: 0.3, 0.5: 0.4, 1.0: 0.6}, **base_multiport_attrs
        )
        return base_network, "Link"

    @pytest.fixture
    def piecewise_two_port_process_network_r0(
        self, base_network: pypsa.Network, base_multiport_attrs: dict
    ) -> tuple[pypsa.Network, str]:
        base_network.add(
            "Process",
            rate0={0.1: -1 / 0.3, 0.5: -1 / 0.4, 1.0: -1 / 0.6},
            rate1=1,
            **base_multiport_attrs,
        )
        return base_network, "Process"

    @pytest.fixture
    def piecewise_two_port_process_network_r1(
        self, base_network: pypsa.Network, base_multiport_attrs: dict
    ) -> tuple[pypsa.Network, str]:
        base_network.add(
            "Process", rate1={0.1: 0.3, 0.5: 0.4, 1.0: 0.6}, **base_multiport_attrs
        )
        return base_network, "Process"

    @pytest.mark.parametrize(
        "fixture_name",
        ["piecewise_two_port_link_network", "piecewise_two_port_process_network_r1"],
    )
    def test_piecewise_efficiency_two_port_out(
        self, request, fixture_name: str
    ) -> None:
        n, comp = request.getfixturevalue(fixture_name)
        n.optimize()
        expected_p = pd.Series(
            {
                0: 50,  # output of 20 units at bus1 requires 50% load rate (40% efficient)
                1: 62.5,  # output of 30 units at bus1 requires 62.5% load rate (48% efficient)
                2: 75,  # output of 30 units at bus1 requires 75% load rate (53% efficient)
            },
            name="multiport1",
        ).rename_axis(index="snapshot")
        pd.testing.assert_series_equal(n.c[comp].dynamic.p.squeeze(), expected_p)

        expected_p1 = (
            -1 * n.model.variables[f"{comp}-p1_piecewise"].solution.to_pandas()
        )
        pd.testing.assert_frame_equal(n.c[comp].dynamic.p1, expected_p1)

    @pytest.mark.parametrize("fixture_name", ["piecewise_two_port_process_network_r0"])
    def test_piecewise_efficiency_two_port_in(self, request, fixture_name: str) -> None:
        n, comp = request.getfixturevalue(fixture_name)
        status, _ = n.optimize()
        assert status == "ok"
        expected_p = pd.Series(
            {0: 20.0, 1: 30.0, 2: 40.0}, name="multiport1"
        ).rename_axis(index="snapshot")
        pd.testing.assert_series_equal(n.c[comp].dynamic.p.squeeze(), expected_p)

        # withdrawal is linearised in p between breakpoints (10, 100/3) and (50, 125),
        expected_p0 = pd.Series(
            {
                0: 56.25,  # chord at p=20
                1: 79.1667,  # chord at p=30
                2: 102.083,  # chord at p=40
            },
            name="multiport1",
        ).rename_axis(index="snapshot")
        pd.testing.assert_series_equal(n.c[comp].dynamic.p0.squeeze(), expected_p0)

        expected_p1 = -1 * expected_p
        pd.testing.assert_series_equal(n.c[comp].dynamic.p1.squeeze(), expected_p1)

    def test_piecewise_efficiency_with_delay(
        self, base_network: pypsa.Network, base_multiport_attrs: dict
    ) -> None:
        """Regression: piecewise port relations are defined once on the undelayed
        dispatch variable; mixed delay groups must not collide and the delayed
        group shifts the auxiliary variable in time."""
        n = base_network
        n.add(
            "Process",
            rate1={0.1: 0.3, 0.5: 0.4, 1.0: 0.6},
            delay1=1,
            **base_multiport_attrs,
        )
        n.add(
            "Process",
            rate1={0.1: 0.3, 0.5: 0.4, 1.0: 0.6},
            **{**base_multiport_attrs, "name": "multiport2"},
        )
        status, _ = n.optimize()
        assert status == "ok"
        load = pd.Series([20.0, 30.0, 40.0]).rename_axis(index="snapshot")
        supply = -n.c.processes.dynamic.p1.sum(axis=1)
        pd.testing.assert_series_equal(supply, load, check_names=False)


class TestPiecewiseMultiPort3Bus:
    @pytest.fixture
    def base_network(self) -> pypsa.Network:
        n = pypsa.Network()
        n.set_snapshots(range(3))
        n.add("Bus", ["bus0", "bus1", "bus2"])
        n.add("Carrier", "gas")
        n.add("Generator", "gen0", carrier="gas", bus="bus0", p_nom=100)
        # load1 set to have demands that match having a 50% efficient port
        n.add("Load", "load1", bus="bus1", p_set=[25, 31.25, 37.5])
        # load2 to be met by the piecewise efficient port
        n.add("Load", "load2", bus="bus2", p_set=[20, 30, 40])
        return n

    @pytest.fixture
    def base_multiport_attrs(self) -> dict:
        return {
            "name": "multiport1",
            "carrier": "gas",
            "bus0": "bus0",
            "bus1": "bus1",
            "bus2": "bus2",
            "p_nom": 100,
            "p_nom_extendable": False,
            "marginal_cost": 20,
        }

    @pytest.fixture
    def piecewise_two_port_link_network(
        self, base_network: pypsa.Network, base_multiport_attrs: dict
    ) -> tuple[pypsa.Network, str]:
        base_network.add(
            "Link",
            efficiency=0.5,
            efficiency2={0.1: 0.3, 0.5: 0.4, 1.0: 0.6},
            **base_multiport_attrs,
        )
        return base_network, "Link"

    @pytest.fixture
    def piecewise_two_port_process_network_r1(
        self, base_network: pypsa.Network, base_multiport_attrs: dict
    ) -> tuple[pypsa.Network, str]:
        base_network.add(
            "Process",
            rate1=0.5,
            rate2={0.1: 0.3, 0.5: 0.4, 1.0: 0.6},
            **base_multiport_attrs,
        )
        return base_network, "Process"

    @pytest.mark.parametrize(
        "fixture_name",
        ["piecewise_two_port_link_network", "piecewise_two_port_process_network_r1"],
    )
    def test_piecewise_efficiency_two_port_out(
        self, request, fixture_name: str
    ) -> None:
        n, comp = request.getfixturevalue(fixture_name)
        n.optimize()
        expected_p = pd.Series(
            {
                0: 50,  # output of 20 units at bus1 requires 50% load rate (40% efficient)
                1: 62.5,  # output of 30 units at bus1 requires 62.5% load rate (48% efficient)
                2: 75,  # output of 30 units at bus1 requires 75% load rate (53% efficient)
            },
            name="multiport1",
        ).rename_axis(index="snapshot")
        pd.testing.assert_series_equal(n.c[comp].dynamic.p.squeeze(), expected_p)

        expected_p1 = -1 * expected_p / 2
        pd.testing.assert_series_equal(n.c[comp].dynamic.p1.squeeze(), expected_p1)

        expected_p2 = (
            -1 * n.model.variables[f"{comp}-p2_piecewise"].solution.to_pandas()
        )
        pd.testing.assert_frame_equal(n.c[comp].dynamic.p2, expected_p2)
        # The auxiliary variable feeds p2; it must not leak as its own output.
        assert "p2_piecewise_opt" not in n.c[comp].dynamic
