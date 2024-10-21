import pytest

import pypsa
from pypsa import Network, SubNetwork


@pytest.fixture
def scipy_subnetwork(scipy_network: Network) -> SubNetwork:
    n = scipy_network
    n.determine_network_topology()
    return n.sub_networks.obj.iloc[0]


@pytest.fixture
def ac_dc_subnetwork(ac_dc_network: Network) -> SubNetwork:
    n = ac_dc_network
    n.determine_network_topology()
    return n.sub_networks.obj.iloc[1]


@pytest.fixture
def ac_dc_subnetwork_inactive(ac_dc_network: Network) -> SubNetwork:
    n = ac_dc_network
    n.lines.loc["2", "active"] = False
    n.determine_network_topology()
    return n.sub_networks.obj.iloc[1]


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
    buses = scipy_subnetwork.static("Bus")
    assert not buses.empty
    assert buses.index.isin(scipy_subnetwork.n.buses.index).all()

    component_names = ["Line", "Transformer", "Generator", "Load"]
    for c_name in component_names:
        df = scipy_subnetwork.static(c_name)
        assert not df.empty
        assert df.index.isin(scipy_subnetwork.n.static(c_name).index).all()

    with pytest.raises(ValueError):
        scipy_subnetwork.static("Link")

    with pytest.raises(ValueError):
        scipy_subnetwork.static("GlobalConstraint")


def test_incidence_matrix(ac_dc_subnetwork: SubNetwork) -> None:
    lines = ac_dc_subnetwork.static("Line")
    buses = ac_dc_subnetwork.static("Bus")
    A = ac_dc_subnetwork.incidence_matrix()
    assert A.shape == (len(buses), len(lines))


def test_incidence_matrix_inactive(ac_dc_subnetwork_inactive: SubNetwork) -> None:
    lines = ac_dc_subnetwork_inactive.static("Line")
    buses = ac_dc_subnetwork_inactive.static("Bus")
    A = ac_dc_subnetwork_inactive.incidence_matrix()
    assert A.shape == (len(buses), len(lines[lines["active"]]))
