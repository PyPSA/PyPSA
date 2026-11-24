# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest

import pypsa
from pypsa import Network, SubNetwork


@pytest.fixture
def scipy_subnetwork(scipy_network: Network) -> SubNetwork:
    n = scipy_network
    n.determine_network_topology()
    return n.c.sub_networks.static.obj.iloc[0]


@pytest.fixture
def ac_dc_subnetwork(ac_dc_network: Network) -> SubNetwork:
    n = ac_dc_network
    n.determine_network_topology()
    return n.c.sub_networks.static.obj.iloc[1]


@pytest.fixture
def ac_dc_subnetwork_inactive(ac_dc_network: Network) -> SubNetwork:
    n = ac_dc_network
    n.c.lines.static.loc["2", "active"] = False
    n.determine_network_topology()
    return n.c.sub_networks.static.obj.iloc[1]


def test_network(scipy_subnetwork: SubNetwork) -> None:
    assert isinstance(scipy_subnetwork.n, pypsa.Network)


def test_name(scipy_subnetwork: SubNetwork) -> None:
    assert scipy_subnetwork.name == "0"


def test_snapshots(scipy_subnetwork: SubNetwork) -> None:
    assert scipy_subnetwork.snapshots.equals(scipy_subnetwork.n.snapshots)


def test_snapshot_weightings(scipy_subnetwork: SubNetwork) -> None:
    assert scipy_subnetwork.snapshot_weightings.equals(
        scipy_subnetwork.n.snapshot_weightings
    )


def test_investment_periods(scipy_subnetwork: SubNetwork) -> None:
    assert scipy_subnetwork.investment_periods.equals(
        scipy_subnetwork.n.investment_periods
    )


def test_investment_period_weightings(scipy_subnetwork: SubNetwork) -> None:
    assert scipy_subnetwork.investment_period_weightings.equals(
        scipy_subnetwork.n.investment_period_weightings
    )


def test_df(scipy_subnetwork: SubNetwork) -> None:
    buses = scipy_subnetwork.components.buses.static
    assert not buses.empty
    assert buses.index.isin(scipy_subnetwork.n.c.buses.static.index).all()

    component_names = ["Line", "Transformer", "Generator", "Load"]
    for c_name in component_names:
        df = scipy_subnetwork.components[c_name].static
        assert not df.empty
        assert df.index.isin(scipy_subnetwork.n.c[c_name].static.index).all()

    with pytest.raises(ValueError):
        scipy_subnetwork.components["Link"].static

    with pytest.raises(ValueError):
        scipy_subnetwork.components["GlobalConstraint"].static


def test_incidence_matrix(ac_dc_subnetwork: SubNetwork) -> None:
    lines = ac_dc_subnetwork.components["Line"].static
    buses = ac_dc_subnetwork.components["Bus"].static
    A = ac_dc_subnetwork.incidence_matrix()
    assert A.shape == (len(buses), len(lines))


def test_incidence_matrix_inactive(ac_dc_subnetwork_inactive: SubNetwork) -> None:
    lines = ac_dc_subnetwork_inactive.components.lines.static
    buses = ac_dc_subnetwork_inactive.components.buses.static
    A = ac_dc_subnetwork_inactive.incidence_matrix()
    assert A.shape == (len(buses), len(lines[lines["active"]]))



def test_calculate_B_H_zero_impedance(
    ac_dc_subnetwork: SubNetwork, caplog: pytest.LogCaptureFixture
) -> None:
    ac_dc_subnetwork.n.c.lines.static.loc["2", ["x", "r"]] = 0
    ac_dc_subnetwork.calculate_B_H()
    assert "series impedances are zero" in caplog.text

@pytest.mark.parametrize(
    ("controls", "slack_generator", "slack_bus", "controls_after"),
    [
        ({}, None, "a", {}),
        ({"gb": "PQ", "gc": "PV"}, "gb", "b", {"gb": "Slack", "gc": "PV"}),
        ({"gb": "PV", "gc": "Slack"}, "gc", "c", {"gb": "PV", "gc": "Slack"}),
        ({"gb": "Slack", "gc": "Slack"}, "gb", "b", {"gb": "Slack", "gc": "PV"}),
    ],
)
def test_find_slack_bus(controls, slack_generator, slack_bus, controls_after) -> None:
    """Exactly one generator ends up as slack; the slack bus is its bus."""
    n = pypsa.Network()
    n.add("Bus", ["a", "b", "c"])
    n.add("Line", ["ab", "bc"], bus0=["a", "b"], bus1=["b", "c"], x=0.1)
    buses = {"gb": "b", "gc": "c"}
    for name, control in controls.items():
        n.add("Generator", name, bus=buses[name], control=control)
    n.determine_network_topology()
    sub_network = n.c.sub_networks.static.obj.iloc[0]

    sub_network.find_slack_bus()

    assert sub_network.slack_generator == slack_generator
    assert sub_network.slack_bus == slack_bus
    assert n.c.generators.static.control.to_dict() == controls_after
