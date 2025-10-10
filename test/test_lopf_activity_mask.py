# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

import pypsa


def test_optimize(ac_dc_network):
    n = ac_dc_network

    inactive_links = ["DC link"]

    n.c.links.static.loc[inactive_links, "active"] = False

    status, _ = n.optimize(snapshots=n.snapshots)

    assert status == "ok"

    assert n.c.links.dynamic.p0.loc[:, inactive_links].eq(0).all().all()

    assert "Bremen Converter" in n.model.variables["Link-p_nom"].coords["name"]
    assert "DC link" not in n.model.variables["Link-p_nom"].coords["name"]

    assert "Bremen Converter" in n.model.variables["Link-p"].coords["name"]
    assert "DC link" not in n.model.variables["Link-p"].coords["name"]

    assert (
        "Bremen Converter" in n.model.constraints["Link-ext-p_nom-lower"].coords["name"]
    )
    assert "DC link" not in n.model.constraints["Link-ext-p_nom-lower"].coords["name"]


def test_optimize_with_power_flow(scipy_network):
    """
    Test the functionality of the 'active' attribute in PyPSA components.

    This test function verifies that the 'active' attribute of network components
    (specifically lines in this case) is correctly respected during optimization,
    non-linear power flow, and linear power flow calculations.

    The test performs the following checks:
    1. Optimization respects the 'active' status of lines.
    2. Non-linear power flow calculations adhere to the 'active' attribute.
    3. Linear power flow (LPF) results are consistent with the 'active' status.

    The test is performed for both active (True) and inactive (False) scenarios
    to ensure proper behavior in both cases.
    """

    @pytest.mark.parametrize("line_active", [True, False])
    def test_scenario(line_active):
        n = scipy_network.copy()
        switchable_lines = n.c.lines.static.index[100]
        n.c.lines.static.loc[switchable_lines, "active"] = line_active

        # Test optimization and non-linear power flow
        res = n.optimize.optimize_and_run_non_linear_powerflow(
            snapshots=n.snapshots[:1]
        )

        assert res["status"] == "ok", f"Optimization failed with status {res['status']}"
        assert res["converged"].all().all(), "Non-linear power flow did not converge"

        expected_flow = (
            not np.isclose(n.c.lines.dynamic.p0.loc[:, switchable_lines], 0).all().all()
        )
        msg = f"'active' attribute not respected in optimization/non-linear power flow: expected {'non-zero' if line_active else 'zero'} flow"
        assert expected_flow == line_active, msg

        # Test linear power flow
        n.lpf()
        expected_flow = (
            not np.isclose(n.c.lines.dynamic.p0.loc[:, switchable_lines], 0).all().all()
        )
        msg = f"'active' attribute not respected in linear power flow: expected {'non-zero' if line_active else 'zero'} flow"
        assert expected_flow == line_active, msg

        msg = "Power balance not maintained"
        assert np.isclose(n.c.buses.dynamic.p.sum().sum(), 0, atol=1e-5), msg

    test_scenario(True)
    test_scenario(False)


def test_generators_with_active_attribute():
    """Test generators with active=False attribute."""
    n = pypsa.Network()

    n.set_snapshots(pd.date_range("2023-01-01", periods=2, freq="h"))

    n.add("Bus", "bus", carrier="AC")
    n.add("Carrier", "AC")

    n.add("Load", "load", bus="bus", p_set=100)

    # Add one active generator to meet demand
    n.add(
        "Generator", "gen_active", bus="bus", p_nom=200, marginal_cost=20, active=True
    )

    # Add one inactive generator
    n.add(
        "Generator",
        "gen_inactive",
        bus="bus",
        p_nom=150,
        marginal_cost=10,
        active=False,
    )

    status, _ = n.optimize()

    assert status == "ok"

    # Inactive generator should have zero output
    assert (n.c.generators.dynamic.p["gen_inactive"] == 0).all()

    # Inactive generator should not be in model variables
    assert "gen_inactive" not in n.model.variables["Generator-p"].coords["name"]
    assert "gen_active" in n.model.variables["Generator-p"].coords["name"]


def test_stores_with_active_attribute():
    """Test stores with active=False attribute."""
    n = pypsa.Network()

    n.set_snapshots(pd.date_range("2023-01-01", periods=3, freq="h"))

    n.add("Bus", "bus0", carrier="AC")
    n.add("Bus", "bus1", carrier="AC")
    n.add("Carrier", "electricity")
    n.add("Carrier", "AC")

    n.add("Load", "load", bus="bus0", p_set=100)
    n.add(
        "Generator",
        "gen",
        bus="bus0",
        p_nom=200,
        marginal_cost=10,
        carrier="electricity",
    )

    # Add stores with different active states
    n.add(
        "Store",
        "store_active",
        bus="bus0",
        e_nom=500,
        e_cyclic=True,
        active=True,
        carrier="electricity",
    )
    n.add(
        "Store",
        "store_inactive",
        bus="bus1",
        e_nom=300,
        e_cyclic=False,
        active=False,
        carrier="electricity",
    )

    status, _ = n.optimize()

    assert status == "ok"

    # Inactive store should have zero flow
    assert (n.c.stores.dynamic.p["store_inactive"] == 0).all()

    # Inactive store should not be in model variables
    assert "store_inactive" not in n.model.variables["Store-p"].coords["name"]
    assert "store_active" in n.model.variables["Store-p"].coords["name"]


def test_mixed_components_with_active_attribute():
    """Test mixed component types with active=False attributes."""
    n = pypsa.Network()

    n.set_snapshots(pd.date_range("2023-01-01", periods=3, freq="h"))

    # Create buses
    for i in range(3):
        n.add("Bus", f"bus{i}", carrier="AC")

    n.add("Carrier", "electricity")
    n.add("Carrier", "AC")

    n.add("Load", "load", bus="bus0", p_set=100)

    # Add generators - some active, some not
    n.add(
        "Generator",
        "gen_active",
        bus="bus0",
        p_nom=150,
        marginal_cost=30,
        carrier="electricity",
        active=True,
    )
    n.add(
        "Generator",
        "gen_inactive",
        bus="bus1",
        p_nom=120,
        marginal_cost=25,
        carrier="electricity",
        active=False,
    )

    # Add links - some active, some not
    n.add(
        "Link",
        "link_active",
        bus0="bus1",
        bus1="bus2",
        p_nom=90,
        marginal_cost=5,
        active=True,
        carrier="electricity",
    )
    n.add(
        "Link",
        "link_inactive",
        bus0="bus0",
        bus1="bus1",
        p_nom=70,
        marginal_cost=2,
        active=False,
        carrier="electricity",
    )

    # Add stores - some active, some not
    n.add(
        "Store",
        "store_active",
        bus="bus0",
        e_nom=300,
        e_cyclic=True,
        marginal_cost=1,
        active=True,
        carrier="electricity",
    )
    n.add(
        "Store",
        "store_inactive",
        bus="bus1",
        e_nom=200,
        e_cyclic=False,
        marginal_cost=3,
        active=False,
        carrier="electricity",
    )

    status, _ = n.optimize()

    assert status == "ok"

    # Test inactive components have zero flows
    assert (n.c.generators.dynamic.p["gen_inactive"] == 0).all()
    assert (n.c.links.dynamic.p0["link_inactive"] == 0).all()
    assert (n.c.stores.dynamic.p["store_inactive"] == 0).all()

    # Test inactive components not in model variables
    assert "gen_inactive" not in n.model.variables["Generator-p"].coords["name"]
    assert "link_inactive" not in n.model.variables["Link-p"].coords["name"]
    assert "store_inactive" not in n.model.variables["Store-p"].coords["name"]


def test_inactive_stores_with_global_operational_limit():
    """Test that inactive stores are excluded from global operational limits."""
    n = pypsa.Network()

    n.set_snapshots(pd.date_range("2023-01-01", periods=3, freq="h"))

    n.add("Bus", "bus0", carrier="AC")
    n.add("Bus", "bus1", carrier="AC")
    n.add("Carrier", "electricity")
    n.add("Carrier", "AC")

    n.add("Load", "load", bus="bus0", p_set=100)
    n.add(
        "Generator",
        "gen",
        bus="bus0",
        p_nom=200,
        marginal_cost=10,
        carrier="electricity",
    )

    # Add active and inactive stores with the same carrier
    n.add(
        "Store",
        "store_active",
        bus="bus0",
        e_nom=500,
        e_cyclic=False,
        e_initial=100,
        active=True,
        carrier="electricity",
    )
    n.add(
        "Store",
        "store_inactive",
        bus="bus1",
        e_nom=300,
        e_cyclic=False,
        e_initial=50,
        active=False,
        carrier="electricity",
    )

    # Add global operational limit for electricity carrier
    n.add(
        "GlobalConstraint",
        "electricity_limit",
        type="operational_limit",
        carrier_attribute="electricity",
        sense="<=",
        constant=1000,  # Generous limit to allow optimization to succeed
    )

    status, _ = n.optimize()

    assert status == "ok"

    # Test inactive store has zero flow
    assert (n.c.stores.dynamic.p["store_inactive"] == 0).all()

    # Test inactive store is not in model variables
    assert "store_inactive" not in n.model.variables["Store-p"].coords["name"]
    assert "store_active" in n.model.variables["Store-p"].coords["name"]

    # Test inactive store is not in model constraints related to energy
    assert "store_inactive" not in n.model.variables["Store-e"].coords["name"]
    assert "store_active" in n.model.variables["Store-e"].coords["name"]
