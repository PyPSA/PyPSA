# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pandas as pd
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


def test_lpf_contingency_unaffected_subnetwork():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2026-01-01", periods=1, freq="h"))
    for area, load in [("a", 100.0), ("b", 60.0)]:
        n.add("Bus", [area + "0", area + "1"], v_nom=110)
        n.add(
            "Generator",
            area + "g",
            bus=area + "0",
            control="Slack",
            p_nom=1000,
            p_set=load,
        )
        n.add("Load", area + "d", bus=area + "1", p_set=load)
        for k in (0, 1):
            n.add(
                "Line",
                area + "l" + str(k),
                bus0=area + "0",
                bus1=area + "1",
                r=0.01,
                x=0.1,
                s_nom=200,
            )
    p0 = n.lpf_contingency(n.snapshots[0], branch_outages=[("Line", "al0")])
    assert not p0[("Line", "al0")].isna().any()
    equal(p0[("Line", "al0")].to_numpy(), [0.0, 100.0, 30.0, 30.0])
