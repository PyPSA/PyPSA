# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest
from numpy.testing import assert_allclose

import pypsa

LOAD_P = [100.0, 160.0]
LOAD_Q = 10.0
P_SET = {"g1": 20.0, "g2": 60.0}


@pytest.fixture
def single_bus():
    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=LOAD_P, q_set=LOAD_Q)
    n.add(
        "Generator",
        ["g1", "g2"],
        bus="bus",
        control=["Slack", "PV"],
        p_set=list(P_SET.values()),
        p_nom=[100, 300],
        p_nom_opt=[300, 100],
    )
    return n


@pytest.mark.parametrize(
    ("kwargs", "g1_share"),
    [
        ({"distribute_slack": False}, 1.0),
        ({"distribute_slack": True, "slack_weights": "p_set"}, 0.25),
        ({"distribute_slack": True, "slack_weights": "p_nom"}, 0.25),
        ({"distribute_slack": True, "slack_weights": "p_nom_opt"}, 0.75),
        ({"distribute_slack": True, "slack_weights": {"0": {"g1": 3, "g2": 1}}}, 0.75),
        ({"distribute_slack": True, "slack_weights": {"0": {"g1": 1}}}, 1.0),
    ],
)
def test_single_bus_slack_takes_imbalance(single_bus, kwargs, g1_share):
    """Generators share the active power imbalance by weight; the slack takes all reactive power."""
    n = single_bus
    n.pf(**kwargs)

    imbalance = pd.Series(LOAD_P, n.snapshots) - sum(P_SET.values())
    p = n.c.generators.dynamic.p
    assert_allclose(p.g1, P_SET["g1"] + g1_share * imbalance)
    assert_allclose(p.g2, P_SET["g2"] + (1 - g1_share) * imbalance)
    assert_allclose(n.c.generators.dynamic.q.g1, LOAD_Q)
    assert_allclose(n.c.generators.dynamic.q.g2, 0.0)
    assert_allclose(n.c.buses.dynamic.v_mag_pu.bus, 1.0)
    assert_allclose(n.c.buses.dynamic.v_ang.bus, 0.0)


def test_isolated_bus_balanced_separately(single_bus):
    """An isolated bus is balanced on its own, next to a meshed sub-network with losses."""
    n = single_bus
    n.add("Bus", ["a", "b"], v_nom=110)
    n.add("Line", "ab", bus0="a", bus1="b", x=10, r=1)
    n.add("Generator", "ga", bus="a", control="Slack", p_set=0)
    n.add("Load", "lb", bus="b", p_set=50.0)

    n.pf()

    p = n.c.generators.dynamic.p
    losses = n.c.lines.dynamic.p0.ab + n.c.lines.dynamic.p1.ab
    assert len(n.c.sub_networks.static) == 2
    assert_allclose(p.g1 + p.g2, LOAD_P)
    assert_allclose(p.ga, 50.0 + losses)
    assert (losses > 0).all()


@pytest.mark.parametrize("slack_weights", ["p_nom", "p_set"])
def test_single_bus_zero_slack_weights_raise(single_bus, slack_weights):
    """Zero weights at a bus cannot take its imbalance, even if other buses have weights."""
    n = single_bus
    n.add("Bus", "other")
    n.add("Load", "l_other", bus="other", p_set=10.0)
    n.add("Generator", "g_other", bus="other", p_set=10.0, p_nom=100)
    n.c.generators.static.loc[["g1", "g2"], ["p_nom", "p_set"]] = 0.0

    with pytest.raises(ValueError, match="at bus bus are zero"):
        n.pf(distribute_slack=True, slack_weights=slack_weights)


def test_single_bus_unknown_slack_weights_raise(single_bus):
    with pytest.raises(ValueError, match="Invalid slack weights"):
        single_bus.pf(distribute_slack=True, slack_weights="foo")
