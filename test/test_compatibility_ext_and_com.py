# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as equal

import pypsa


@pytest.fixture
def base_network():
    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Bus", "bus")
    n.add("Generator", "coal", bus="bus", p_nom=1000, marginal_cost=2)
    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])
    return n


def test_compatibility_ext_and_comt():
    """Test complex network with various generator combinations and objective verification."""
    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Bus", "bus")
    n.add("Bus", "bus2")

    n.add(
        "Line",
        "line",
        bus0="bus",
        bus1="bus2",
        x=0.01,
        r=0.01,
        capital_cost=1,
        s_nom_max=7000,
        s_nom_mod=0.5,
        s_nom_extendable=True,
    )

    n.add(
        "Generator",
        "coal-com-non_mod-non_ext",
        bus="bus2",
        committable=True,
        ramp_limit_up=1,
        ramp_limit_down=0.95,
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
        start_up_cost=2,
    )

    n.add(
        "Generator",
        "gas-com-mod-ext",
        bus="bus",
        committable=True,
        p_nom_extendable=True,
        ramp_limit_up=0.5,
        marginal_cost=70,
        capital_cost=1,
        stand_by_cost=10,
        p_nom_mod=500,
        p_min_pu=0.1,
        start_up_cost=2,
    )

    n.add(
        "Generator",
        "gas-com-non_mod-ext",
        bus="bus",
        committable=True,
        p_nom_extendable=True,
        ramp_limit_up=0.9,
        marginal_cost=60,
        stand_by_cost=10,
        p_min_pu=0.1,
        capital_cost=1,
        start_up_cost=2,
    )

    n.add(
        "Generator",
        "gas-non_com-mod-non_ext",
        bus="bus",
        ramp_limit_up=0.8,
        marginal_cost=60,
        stand_by_cost=10,
        p_min_pu=0.1,
        p_nom=1000,
        p_nom_mod=250,
        start_up_cost=2,
    )

    n.add(
        "Generator",
        "gas-com-mod-non_ext",
        bus="bus",
        committable=True,
        ramp_limit_up=0.8,
        ramp_limit_down=0.9,
        marginal_cost=15,
        stand_by_cost=10,
        p_min_pu=0.1,
        p_nom=1000,
        p_nom_mod=250,
        start_up_cost=2,
    )

    n.add(
        "Generator",
        "gas-non_com-mod-ext",
        bus="bus",
        p_nom_extendable=True,
        ramp_limit_up=0.8,
        ramp_limit_down=0.9,
        marginal_cost=15,
        stand_by_cost=10,
        p_min_pu=0.1,
        p_nom_mod=250,
        capital_cost=0.5,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    n.optimize(transmission_losses={"mode": "tangents", "segments": 1})

    f_obj = (n.c.generators.static.p_nom_opt * n.c.generators.static.capital_cost).sum()
    f_obj += (
        (n.c.generators.dynamic.p * n.c.generators.static.marginal_cost).sum().sum()
    )
    f_obj += (
        (n.c.generators.dynamic.status * n.c.generators.static.stand_by_cost)
        .sum()
        .sum()
    )
    f_obj += (
        (n.c.generators.dynamic.start_up * n.c.generators.static.start_up_cost)
        .sum()
        .sum()
    )
    f_obj += (
        (n.c.generators.dynamic.shut_down * n.c.generators.static.shut_down_cost)
        .sum()
        .sum()
    )
    f_obj += (n.c.lines.static.s_nom_opt * n.c.lines.static.capital_cost).sum()

    equal(f_obj, n.objective + n.objective_constant)


def test_ext_and_com_single():
    """Test single modular+extendable+committable generator with status counting."""
    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_nom_extendable=True,
        marginal_cost=1,
        capital_cost=1,
        p_nom_mod=200,
        p_nom_max=10000,
        p_min_pu=0.1,
        stand_by_cost=1,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 0, 800])

    n.optimize()

    equal(n.c.generators.static.p_nom_opt["coal"], 6000)
    equal(n.c.generators.dynamic.status.to_numpy().flatten(), [20, 30, 0, 4])


def test_unit_commitment_mod():
    """Test unit commitment with modular generators."""
    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_nom_extendable=True,
        p_min_pu=0.3,
        p_nom_mod=10000,
        p_nom_min=10000,
        p_nom_max=10000,
        marginal_cost=20,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        p_nom=1000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    n.optimize()

    expected_status = np.array([[1, 1, 1, 0], [0, 0, 0, 1]], dtype=float).T
    equal(n.c.generators.dynamic.status.values, expected_status)

    expected_dispatch = np.array([[4000, 6000, 5000, 0], [0, 0, 0, 800]], dtype=float).T
    equal(n.c.generators.dynamic.p.values, expected_dispatch)


@pytest.mark.parametrize(
    ("gen_config", "load", "expected_dispatch"),
    [
        # Committable only
        (
            {"p_nom": 5000, "marginal_cost": 1, "committable": True, "p_min_pu": 0.2},
            [4000, 6000, 5000, 800],
            [[0, 1000, 0, 800], [4000, 5000, 5000, 0]],
        ),
        # Modular + committable
        (
            {
                "p_nom": 5000,
                "capital_cost": 1,
                "marginal_cost": 0,
                "committable": True,
                "p_min_pu": 0.2,
            },
            [4000, 6000, 5000, 20],
            [[0, 1000, 0, 20], [4000, 5000, 5000, 0]],
        ),
        # Extendable + modular
        (
            {
                "p_nom_mod": 1,
                "p_nom_max": 5000,
                "capital_cost": 1,
                "marginal_cost": 0,
                "p_nom_extendable": True,
            },
            [4000, 6000, 5000, 800],
            [[0, 1000, 0, 0], [4000, 5000, 5000, 800]],
        ),
        # Extendable + committable + modular
        (
            {
                "p_nom_mod": 200,
                "p_nom_max": 5000,
                "capital_cost": 1,
                "marginal_cost": 0,
                "p_nom_extendable": True,
                "committable": True,
                "p_min_pu": 0.2,
            },
            [4000, 6000, 5000, 20],
            [[0, 1000, 0, 20], [4000, 5000, 5000, 0]],
        ),
    ],
)
def test_generator_combinations(base_network, gen_config, load, expected_dispatch):
    """Test various combinations of extendable/committable/modular generators."""
    n = base_network
    n.loads_t.p_set["load"] = load
    n.add("Generator", "flex", bus="bus", **gen_config)

    n.optimize()

    equal(n.c.generators.dynamic.p.values, np.array(expected_dispatch, dtype=float).T)


def test_com_ext_mod_p_nom():
    """Test that p_nom is ignored when extendable+modular."""
    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Bus", "bus")
    n.add("Generator", "coal", bus="bus", p_nom=5000, marginal_cost=100)

    n.add(
        "Generator",
        "ffg",
        bus="bus",
        p_nom=0,
        p_nom_mod=200,
        p_nom_max=10000,
        p_nom_extendable=True,
        committable=True,
        capital_cost=1,
        marginal_cost=0,
        p_min_pu=0.1,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    n.optimize()

    expected_dispatch = np.array([[0, 0, 0, 0], [4000, 6000, 5000, 800]], dtype=float).T
    equal(n.c.generators.dynamic.p.values, expected_dispatch)


def test_p_nom_not_multiple_of_mod_raises():
    """Test that non-multiple p_nom/p_nom_mod raises error."""
    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Bus", "bus")

    n.add(
        "Generator",
        "ffg",
        bus="bus",
        p_nom=6001,
        p_nom_mod=2,
        capital_cost=1,
        marginal_cost=0.5,
        committable=True,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    with pytest.raises(ValueError):
        n.optimize()
