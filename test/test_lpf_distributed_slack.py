# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pypsa
from numpy.testing import assert_allclose


def test_lpf_distributed_slack_by_capacity() -> None:
    n = pypsa.Network()
    n.add("Bus", ["a", "b"], v_nom=110)
    n.add("Line", "line", bus0="a", bus1="b", x=0.1, s_nom=100)
    n.add(
        "Generator",
        ["slack", "remote"],
        bus=["a", "b"],
        control=["Slack", "PQ"],
        p_set=[10.0, 10.0],
        p_nom=[100.0, 300.0],
    )
    n.add("Load", "load", bus="a", p_set=30.0)

    n.lpf(distribute_slack=True, slack_weights="p_nom")

    assert_allclose(
        n.c.generators.dynamic.p.loc[n.snapshots[0], ["slack", "remote"]],
        [12.5, 17.5],
    )
    assert_allclose(n.c.buses.dynamic.p.loc[n.snapshots[0]].sum(), 0.0)
