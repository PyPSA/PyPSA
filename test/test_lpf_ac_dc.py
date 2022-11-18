# -*- coding: utf-8 -*-
import os

import pytest
from numpy.testing import assert_array_almost_equal as equal

import pypsa


@pytest.fixture
def ac_dc_network_r():
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ac-dc-meshed",
        "ac-dc-data",
        "results-lpf",
    )
    return pypsa.Network(csv_folder)


def test_lpf(ac_dc_network, ac_dc_network_r):
    n = ac_dc_network
    n_r = ac_dc_network_r
    n.links_t.p_set = n_r.links_t.p_set

    n.lpf(snapshots=n.snapshots)

    equal(
        n.generators_t.p[n.generators.index].iloc[:2],
        n_r.generators_t.p[n.generators.index].iloc[:2],
    )
    equal(n.lines_t.p0[n.lines.index].iloc[:2], n_r.lines_t.p0[n.lines.index].iloc[:2])
    equal(n.links_t.p0[n.links.index].iloc[:2], n_r.links_t.p0[n.links.index].iloc[:2])


def test_lpf_chunks(ac_dc_network, ac_dc_network_r):
    n = ac_dc_network
    n_r = ac_dc_network_r

    for snapshot in n.snapshots[:2]:
        n.lpf(snapshot)

    n.lpf(snapshots=n.snapshots)

    equal(n.generators_t.p[n.generators.index], n_r.generators_t.p[n.generators.index])
    equal(n.lines_t.p0[n.lines.index], n_r.lines_t.p0[n.lines.index])
    equal(n.links_t.p0[n.links.index], n_r.links_t.p0[n.links.index])
