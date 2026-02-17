# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

import pypsa


@pytest.fixture
def base_network():
    n = pypsa.Network()
    n.set_snapshots(range(6))
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Generator", "gen0", bus="bus0", p_nom=200, marginal_cost=10)
    n.add("Generator", "gen1", bus="bus1", p_nom=200, marginal_cost=50)
    return n


def test_link_delay_cyclic(base_network):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add(
        "Link",
        "link",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        efficiency=0.9,
        delay=2,
        cyclic_delay=True,
    )
    n.optimize()
    p0 = n.c.links.dynamic.p0["link"].values
    p1 = n.c.links.dynamic.p1["link"].values
    for t in range(6):
        np.testing.assert_allclose(p1[t], -0.9 * p0[(t - 2) % 6])


def test_link_delay_non_cyclic(base_network):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add(
        "Link",
        "link",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        efficiency=0.9,
        delay=2,
        cyclic_delay=False,
    )
    n.optimize()
    p0 = n.c.links.dynamic.p0["link"].values
    p1 = n.c.links.dynamic.p1["link"].values
    np.testing.assert_allclose(p1[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(p1[1], 0.0, atol=1e-10)
    for t in range(2, 6):
        np.testing.assert_allclose(p1[t], -0.9 * p0[t - 2])
    np.testing.assert_allclose(p0[4], 0.0, atol=1e-10)
    np.testing.assert_allclose(p0[5], 0.0, atol=1e-10)


def test_link_delay_zero_unchanged(base_network):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add(
        "Link",
        "link",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        efficiency=0.9,
        delay=0,
    )
    n.optimize()
    p0 = n.c.links.dynamic.p0["link"].values
    p1 = n.c.links.dynamic.p1["link"].values
    np.testing.assert_allclose(p1, -0.9 * p0)


def test_link_multiport_different_delays(base_network):
    n = base_network
    n.add("Bus", "bus2")
    n.add("Generator", "gen2", bus="bus2", p_nom=200, marginal_cost=50)
    n.add("Load", "load1", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add("Load", "load2", bus="bus2", p_set=[5, 10, 15, 20, 25, 30])
    n.add(
        "Link",
        "link",
        bus0="bus0",
        bus1="bus1",
        bus2="bus2",
        p_nom=100,
        efficiency=0.9,
        efficiency2=0.8,
        delay=1,
        delay2=3,
        cyclic_delay=True,
        cyclic_delay2=True,
    )
    n.optimize()
    p0 = n.c.links.dynamic.p0["link"].values
    p1 = n.c.links.dynamic.p1["link"].values
    p2 = n.c.links.dynamic.p2["link"].values
    for t in range(6):
        np.testing.assert_allclose(p1[t], -0.9 * p0[(t - 1) % 6])
        np.testing.assert_allclose(p2[t], -0.8 * p0[(t - 3) % 6])


def test_link_delay_mixed_delayed_and_non_delayed(base_network):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add(
        "Link",
        "delayed",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        efficiency=0.9,
        delay=2,
        cyclic_delay=True,
    )
    n.add(
        "Link",
        "instant",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        efficiency=0.8,
        delay=0,
    )
    n.optimize()
    p0_d = n.c.links.dynamic.p0["delayed"].values
    p1_d = n.c.links.dynamic.p1["delayed"].values
    p0_i = n.c.links.dynamic.p0["instant"].values
    p1_i = n.c.links.dynamic.p1["instant"].values
    for t in range(6):
        np.testing.assert_allclose(p1_d[t], -0.9 * p0_d[(t - 2) % 6])
    np.testing.assert_allclose(p1_i, -0.8 * p0_i)


def test_link_delay_extendable(base_network):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add(
        "Link",
        "link",
        bus0="bus0",
        bus1="bus1",
        p_nom_extendable=True,
        p_nom_max=200,
        capital_cost=1,
        efficiency=0.9,
        delay=2,
        cyclic_delay=True,
    )
    n.optimize()
    p0 = n.c.links.dynamic.p0["link"].values
    p1 = n.c.links.dynamic.p1["link"].values
    assert n.c.links.static.at["link", "p_nom_opt"] > 0
    for t in range(6):
        np.testing.assert_allclose(p1[t], -0.9 * p0[(t - 2) % 6])


def test_link_delay_equal_to_horizon(base_network):
    n = base_network
    n.add("Load", "load", bus="bus1", p_set=[10, 20, 30, 40, 50, 60])
    n.add(
        "Link",
        "link",
        bus0="bus0",
        bus1="bus1",
        p_nom=100,
        efficiency=0.9,
        delay=6,
        cyclic_delay=True,
    )
    n.optimize()
    p0 = n.c.links.dynamic.p0["link"].values
    p1 = n.c.links.dynamic.p1["link"].values
    np.testing.assert_allclose(p1, -0.9 * p0)
