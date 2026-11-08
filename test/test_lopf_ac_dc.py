# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as equal

import pypsa


def test_optimize(ac_dc_network, ac_dc_network_r):
    n = ac_dc_network
    n_r = ac_dc_network_r

    status, _ = n.optimize(snapshots=n.snapshots)

    assert status == "ok"

    equal(
        n.c.generators.dynamic.p.loc[:, n.c.generators.static.index],
        n_r.c.generators.dynamic.p.loc[:, n.c.generators.static.index],
        decimal=2,
    )

    equal(
        n.c.lines.dynamic.p0.loc[:, n.c.lines.static.index],
        n_r.c.lines.dynamic.p0.loc[:, n.c.lines.static.index],
        decimal=2,
    )

    equal(
        n.c.links.dynamic.p0.loc[:, n.c.links.static.index],
        n_r.c.links.dynamic.p0.loc[:, n.c.links.static.index],
        decimal=2,
    )


def chain_network_with_infinite_reactance() -> pypsa.Network:
    n = pypsa.Network(snapshots=range(3))
    n.add("Bus", ["a", "b", "c"])
    n.add("Line", "ab", bus0="a", bus1="b", x=0.1, s_nom=100)
    n.add("Line", "bc", bus0="b", bus1="c", x=np.inf, s_nom=100)
    n.add("Generator", ["ga", "gc"], bus=["a", "c"], p_nom=100, marginal_cost=[1, 2])
    n.add("Load", ["lb", "lc"], bus=["b", "c"], p_set=[30, 10])
    return n


@pytest.mark.parametrize("case", ["losses", "infinite_reactance"])
def test_optimize_v_ang_zero_at_slack(ac_dc_network, case):
    if case == "losses":
        n = ac_dc_network
        n.c.lines.static.s_nom_max = 1e4
        n.optimize(transmission_losses=True)
    else:
        n = chain_network_with_infinite_reactance()
        n.optimize()

    p = n.c.buses.dynamic.p
    for sub in n.c.sub_networks.static.obj:
        buses = sub.buses_o
        theta = n.c.buses.dynamic.v_ang[buses]
        residual = theta.to_numpy() @ sub.B.toarray() - p[buses].to_numpy()
        equal(theta[sub.slack_bus], 0, decimal=12)
        equal(residual[:, theta.abs().max().to_numpy() > 0], 0, decimal=8)
    if case == "losses":
        assert not np.allclose(p.sum(axis=1), 0)


def test_optimize_link_only_network_skips_topology():
    n = pypsa.Network(snapshots=range(3))
    n.add("Bus", ["a", "b"])
    n.add("Link", "ab", bus0="a", bus1="b", p_nom=100, p_min_pu=-1)
    n.add("Generator", "ga", bus="a", p_nom=100, marginal_cost=1)
    n.add("Load", "lb", bus="b", p_set=30)
    n.optimize()

    assert n.c.sub_networks.static.empty
    assert (n.c.buses.dynamic.v_ang == 0).all().all()
    assert list(n.c.buses.dynamic.v_ang.columns) == ["a", "b"]
