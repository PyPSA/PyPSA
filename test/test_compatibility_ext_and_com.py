import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as equal

import pypsa


def test_compatibility_ext_and_comt():
    """
    This test is based on https://pypsa.readthedocs.io/en/latest/examples/unit-
    commitment.html and is not very comprehensive.
    """
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

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

    n.optimize(transmission_losses=1)

    f_obj = (n.generators.p_nom_opt * n.generators.capital_cost).sum()
    f_obj += (n.generators_t.p * n.generators.marginal_cost).sum().sum()
    f_obj += (n.generators_t.status * n.generators.stand_by_cost).sum().sum()
    f_obj += (n.generators_t.start_up * n.generators.start_up_cost).sum().sum()
    f_obj += (n.generators_t.shut_down * n.generators.shut_down_cost).sum().sum()

    equal(f_obj, n.objective + n.objective_constant)


def test_ext_and_com_single():
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

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
        # Without this, we might decide to use more units than necessary
        stand_by_cost=1,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 0, 800])

    n.optimize()

    equal(n.generators.p_nom_opt["coal"], 6000)

    equal(n.generators_t.status.to_numpy().flatten(), [20, 30, 0, 4])


def test_unit_commitment_mod():
    """
    This test is based on https://pypsa.readthedocs.io/en/latest/examples/unit-
    commitment.html and is not very comprehensive.
    """
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

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

    equal(n.generators_t.status.values, expected_status)

    expected_dispatch = np.array([[4000, 6000, 5000, 0], [0, 0, 0, 800]], dtype=float).T

    equal(n.generators_t.p.values, expected_dispatch)


# Trivial + non-extendable, non-modular & committable
# Already tested in test_lopf_unit_commitment.py
def test_com():
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        p_nom=1000,
        marginal_cost=2,
    )

    n.add(
        "Generator",
        "FFG",
        bus="bus",
        p_nom=5000,
        marginal_cost=1,
        committable=True,
        p_min_pu=0.2,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    n.optimize()

    expected_dispatch = np.array(
        [[0, 1000, 0, 800], [4000, 5000, 5000, 0]], dtype=float
    ).T

    equal(n.generators_t.p.values, expected_dispatch)


# Trivial + non-extendable, modular & non-committable
# Already test in test_lopf_modularity.py


# Trivial + non-extendable, modular & committable
def test_mod_com():
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        p_nom=1000,
        marginal_cost=2,
    )

    n.add(
        "Generator",
        "ffg",
        bus="bus",
        p_nom=5000,
        capital_cost=1,
        marginal_cost=0,
        committable=True,
        p_min_pu=0.2,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 20])

    n.optimize()

    expected_dispatch = np.array(
        [[0, 1000, 0, 20], [4000, 5000, 5000, 0]], dtype=float
    ).T

    equal(n.generators_t.p.values, expected_dispatch)


# Trivial + extendable, non-modular & non-committable
# Already tested

# Trivial + extendable, non-modular & committable
# Already tested


# Trivial + extendable, modular & non-committable
def test_ext_mod():
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        p_nom=1000,
        marginal_cost=2,
    )

    n.add(
        "Generator",
        "PV",
        bus="bus",
        p_nom_mod=1,
        p_nom_max=5000,
        capital_cost=1,
        marginal_cost=0,
        p_nom_extendable=True,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    n.optimize()

    expected_dispatch = np.array(
        [[0, 1000, 0, 0], [4000, 5000, 5000, 800]], dtype=float
    ).T

    equal(n.generators_t.p.values, expected_dispatch)


# Trivial + extendable, committable & modular
def test_ext_com_mod():
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        p_nom=1000,
        marginal_cost=2,
    )

    n.add(
        "Generator",
        "ffg",
        bus="bus",
        p_nom_mod=200,
        p_nom_max=5000,
        capital_cost=1,
        marginal_cost=0,
        p_nom_extendable=True,
        committable=True,
        p_min_pu=0.2,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 20])

    n.optimize()

    expected_dispatch = np.array(
        [[0, 1000, 0, 20], [4000, 5000, 5000, 0]], dtype=float
    ).T

    equal(n.generators_t.p.values, expected_dispatch)


# If we have com + ext + mod but p_nom is defined, p_nom should be ignored
def test_com_ext_mod_p_nom():
    n = pypsa.Network()
    snapshots = range(4)
    n.set_snapshots(snapshots)
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "coal",
        bus="bus",
        p_nom=5000,
        marginal_cost=100,
    )

    n.add(
        "Generator",
        "ffg",
        bus="bus",
        # This should be ignored
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

    equal(n.generators_t.p.values, expected_dispatch)


def test_p_nom_p_nom_mod():
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "ffg",
        bus="bus",
        # p_nom is not a multiple of p_nom_mod
        # therefore, it is discarded
        p_nom=6001,
        p_nom_mod=2,
        capital_cost=1,
        marginal_cost=0.5,
        committable=True,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    with pytest.raises(ValueError):
        n.optimize()
