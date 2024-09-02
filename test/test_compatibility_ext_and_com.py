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

    n.add(
        "Generator",
        "coal-com-non_mod-non_ext",
        bus="bus",
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
        marginal_cost=15,
        stand_by_cost=10,
        p_min_pu=0.1,
        p_nom_mod=250,
        capital_cost=0.5,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    n.optimize()

    f_obj = (n.generators.p_nom_opt * n.generators.capital_cost).sum()
    f_obj += (n.generators_t.p * n.generators.marginal_cost).sum().sum()
    f_obj += (n.generators_t.status * n.generators.stand_by_cost).sum().sum()
    f_obj += (n.generators_t.start_up * n.generators.start_up_cost).sum().sum()
    f_obj += (n.generators_t.shut_down * n.generators.shut_down_cost).sum().sum()

    equal(f_obj, n.objective + n.objective_constant)
