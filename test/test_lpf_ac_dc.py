from pathlib import Path

import pytest
from numpy.testing import assert_array_almost_equal as equal

import pypsa


@pytest.fixture
def ac_dc_network_r():
    csv_folder = Path(__file__).parent / "data" / "ac-dc-meshed" / "results-lpf"
    return pypsa.Network(csv_folder)


def test_lpf(ac_dc_network, ac_dc_network_r):
    n = ac_dc_network
    n_r = ac_dc_network_r
    n.links_t.p_set = n_r.links_t.p_set

    n.lpf(snapshots=n.snapshots)

    equal(n.generators_t.p[n.generators.index], n_r.generators_t.p[n.generators.index])
    equal(n.lines_t.p0[n.lines.index], n_r.lines_t.p0[n.lines.index])
    equal(n.links_t.p0[n.links.index], n_r.links_t.p0[n.links.index])


def test_lpf_chunks(ac_dc_network, ac_dc_network_r):
    n = ac_dc_network
    n_r = ac_dc_network_r
    n.links_t.p_set = n_r.links_t.p_set

    for snapshot in n.snapshots:
        n.lpf(snapshot)

    equal(n.generators_t.p[n.generators.index], n_r.generators_t.p[n.generators.index])
    equal(n.lines_t.p0[n.lines.index], n_r.lines_t.p0[n.lines.index])
    equal(n.links_t.p0[n.links.index], n_r.links_t.p0[n.links.index])
