# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

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
    n.c.links.dynamic.p_set = n_r.c.links.dynamic.p_set

    n.lpf(snapshots=n.snapshots)

    equal(
        n.c.generators.dynamic.p[n.c.generators.static.index],
        n_r.c.generators.dynamic.p[n.c.generators.static.index],
    )
    equal(
        n.c.lines.dynamic.p0[n.c.lines.static.index],
        n_r.c.lines.dynamic.p0[n.c.lines.static.index],
    )
    equal(
        n.c.links.dynamic.p0[n.c.links.static.index],
        n_r.c.links.dynamic.p0[n.c.links.static.index],
    )


def test_lpf_chunks(ac_dc_network, ac_dc_network_r):
    n = ac_dc_network
    n_r = ac_dc_network_r
    n.c.links.dynamic.p_set = n_r.c.links.dynamic.p_set

    for snapshot in n.snapshots:
        n.lpf(snapshot)

    equal(
        n.c.generators.dynamic.p[n.c.generators.static.index],
        n_r.c.generators.dynamic.p[n.c.generators.static.index],
    )
    equal(
        n.c.lines.dynamic.p0[n.c.lines.static.index],
        n_r.c.lines.dynamic.p0[n.c.lines.static.index],
    )
    equal(
        n.c.links.dynamic.p0[n.c.links.static.index],
        n_r.c.links.dynamic.p0[n.c.links.static.index],
    )


@pytest.mark.parametrize("method", ["lpf", "pf"])
def test_pf_subnetwork_without_transformer(method):
    # https://github.com/PyPSA/PyPSA/issues/1698
    # one sub-network has a transformer, the other only lines
    n = pypsa.Network()
    n.add("Bus", ["a", "c", "d"], v_nom=110)
    n.add("Bus", "b", v_nom=220)
    n.add("Transformer", "t", bus0="a", bus1="b", x=0.1, s_nom=100)
    n.add("Line", "l", bus0="c", bus1="d", x=0.1, s_nom=100)
    n.add("Generator", ["ga", "gc"], bus=["a", "c"], control="Slack")
    n.add("Load", ["lb", "ld"], bus=["b", "d"], p_set=10)
    n.determine_network_topology()

    getattr(n, method)()

    equal(n.c.transformers.dynamic.p0["t"], [10.0])
    equal(n.c.lines.dynamic.p0["l"], [10.0])
