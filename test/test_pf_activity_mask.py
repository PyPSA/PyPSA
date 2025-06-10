import pytest


@pytest.fixture
def sub_network_full(scipy_network):
    n = scipy_network.copy()
    return n.sub_networks.obj.iloc[0]


@pytest.fixture
def sub_network_filtered(scipy_network):
    n = scipy_network.copy()
    n.lines.loc["2", "active"] = False
    return n.sub_networks.obj.iloc[0]


def test_different_shape_incidence_matrix(sub_network_full, sub_network_filtered):
    k_full = sub_network_full.incidence_matrix()
    k_filtered = sub_network_filtered.incidence_matrix()

    assert k_full.shape[0] == k_filtered.shape[0]
    assert k_full.shape[1] == k_filtered.shape[1]


def test_subnetwork_full_pf(sub_network_full):
    sub_network_full.pf(sub_network_full.snapshots[:3])


def test_subnetwork_filtered_pf(sub_network_filtered):
    sub_network_filtered.pf(sub_network_filtered.snapshots[:3])
    n = sub_network_filtered.n
    assert n.lines_t.p0.loc[:, ~n.lines.active].eq(0).all().all()
