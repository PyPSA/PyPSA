import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal as equal

import pypsa


def test_unit_commitment():
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
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
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


def test_minimum_up_time():
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
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        up_time_before=0,
        min_up_time=3,
        p_nom=1000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 800, 5000, 3000])

    n.optimize()

    expected_status = np.array([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=float).T

    equal(n.generators_t.status.values, expected_status)

    expected_dispatch = np.array(
        [[3900, 0, 4900, 3000], [100, 800, 100, 0]], dtype=float
    ).T

    equal(n.generators_t.p.values, expected_dispatch)


def test_minimum_up_time_up_time_before():
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
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        up_time_before=1,
        min_up_time=4,
        p_nom=1000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 800, 5000, 3000])

    n.optimize()

    expected_status = np.array([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=float).T

    equal(n.generators_t.status.values, expected_status)

    expected_dispatch = np.array(
        [[3900, 0, 4900, 3000], [100, 800, 100, 0]], dtype=float
    ).T

    equal(n.generators_t.p.values, expected_dispatch)


def test_minimum_down_time():
    """
    This test is based on https://pypsa.readthedocs.io/en/latest/examples/unit-
    commitment.html and is not very comprehensive.
    """
    n = pypsa.Network()

    n.set_snapshots(range(4))

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        min_down_time=2,
        down_time_before=1,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        p_nom=4000,
    )

    n.add("Load", "load", bus="bus", p_set=[3000, 800, 3000, 8000])

    n.optimize()

    expected_status = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=float).T

    equal(n.generators_t.status.values, expected_status)

    expected_dispatch = np.array([[0, 0, 3000, 8000], [3000, 800, 0, 0]], dtype=float).T

    equal(n.generators_t.p.values, expected_dispatch)


def test_minimum_down_time_up_time_before():
    """
    This test is based on https://pypsa.readthedocs.io/en/latest/examples/unit-
    commitment.html and is not very comprehensive.
    """
    n = pypsa.Network()

    n.set_snapshots(range(4))

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        min_down_time=3,
        down_time_before=2,
        up_time_before=0,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        p_nom=4000,
    )

    n.add("Load", "load", bus="bus", p_set=[3000, 800, 3000, 8000])

    n.optimize()

    expected_status = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=float).T

    equal(n.generators_t.status.values, expected_status)

    expected_dispatch = np.array([[0, 0, 3000, 8000], [3000, 800, 0, 0]], dtype=float).T

    equal(n.generators_t.p.values, expected_dispatch)


def test_start_up_costs():
    n = pypsa.Network()

    n.snapshots = range(4)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.1,
        up_time_before=0,
        marginal_cost=20,
        start_up_cost=3000,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.0,
        up_time_before=1,
        start_up_cost=1000,
        p_nom=10000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    n.optimize()

    assert n.objective == 359000


def test_shut_down_costs():
    n = pypsa.Network()

    n.snapshots = range(4)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        shut_down_cost=1000,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        shut_down_cost=1000,
        p_min_pu=0.1,
        p_nom=1000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    n.optimize()

    assert n.objective == 358000


def test_unit_commitment_rolling_horizon():
    n = pypsa.Network()
    n.snapshots = range(7)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.1,
        up_time_before=1,
        marginal_cost=20,
        min_up_time=2,
        min_down_time=2,
        start_up_cost=10000,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.01,
        start_up_cost=100,
        up_time_before=0,
        p_nom=10000,
    )
    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 800, 5000, 3000, 950, 800])

    n.optimize(snapshots=[0, 1, 2])
    n.optimize(snapshots=[2, 3, 4])
    n.optimize(snapshots=[4, 5, 6])

    expected_status = np.array(
        [[1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1]], dtype=float
    ).T
    equal(n.generators_t.status.values, expected_status)

    expected_dispatch = np.array(
        [[4000, 6000, 0, 0, 0, 0, 0], [0, 0, 800, 5000, 3000, 950, 800]]
    ).T

    equal(n.generators_t.p.values, expected_dispatch)


def test_linearized_unit_commitment():
    n = pypsa.Network()
    n.snapshots = pd.date_range("2022-01-01", "2022-02-09", freq="d")

    load = np.zeros(len(n.snapshots))
    load[:5] = 5
    load[5:10] = 6
    load[10:15] = 8
    load[15:20] = 10
    load[20:30] = 7
    load[30:40] = 6
    load *= 100

    n.add("Bus", "bus")

    for seed, i in enumerate(range(40), start=1):
        rng = np.random.default_rng(seed)  # Create a random number generator
        p_min_pu = rng.integers(1, 5) / 10
        marginal_cost = rng.integers(1, 11) * 10
        min_up_time = rng.integers(0, 6)
        min_down_time = rng.integers(0, 6)
        p_nom = rng.integers(1, 10) * 5
        start_up_cost = rng.integers(1, 5) * 100

        # the constraint tightening proposed in Baldick et al. depends on start_up_cost
        # and shut_down_cost being equal therefore, we force them to be equal for first
        # 20 generators
        shut_down_cost = rng.integers(1, 5) * 100 if i >= 20 else start_up_cost

        n.add(
            "Generator",
            f"{i}",
            bus="bus",
            committable=True,
            up_time_before=0,
            p_min_pu=p_min_pu,
            marginal_cost=marginal_cost,
            min_up_time=min_up_time,
            min_down_time=min_down_time,
            p_nom=p_nom,
            start_up_cost=start_up_cost,
            shut_down_cost=shut_down_cost,
        )
    n.add("Load", "load", bus="bus", p_set=load)

    n.optimize(linearized_unit_commitment=True)

    MILP_objective = 1241100
    assert round(n.objective / MILP_objective, 2) == 1


def test_link_unit_commitment():
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", ["gas", "electricity"])

    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)

    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        committable=True,
        p_min_pu=0.1,
        efficiency=0.5,
        up_time_before=0,
        min_up_time=3,
        start_up_cost=3333,
        p_nom=12000,
    )

    n.add(
        "Generator",
        "wind",
        bus="electricity",
        p_nom=800,
    )

    n.add("Load", "load", bus="electricity", p_set=[4000, 6000, 800, 5000])

    n.optimize()

    expected_status = [1.0, 1.0, 1.0, 1.0]

    equal(n.links_t.status["OCGT"].values, expected_status)

    expected_dispatch = [3200.0, 5200.0, 600.0, 4200.0]

    equal(-n.links_t.p1["OCGT"].values, expected_dispatch)

    assert round(n.objective, 1) == 267333.0


def test_dynamic_ramp_rates():
    """
    This test checks that dynamic ramp rates are correctly applied when
    considering a unit outage represented by p_max_pu.
    """
    n = pypsa.Network()

    snapshots = range(15)
    n.set_snapshots(snapshots)
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=100)

    # vary marginal price of gen1 to induce ramping
    gen1_marginal = pd.Series(100, index=n.snapshots)
    gen1_marginal[[4, 5, 6, 10, 11, 12]] = 200

    static_ramp_up = 0.8
    static_ramp_down = 1
    p_max_pu = pd.Series(1, index=n.snapshots).astype(float)
    p_max_pu.loc[n.snapshots[0:6]] = 0.5  # 50% capacity outage for 6 periods

    n.add(
        "Generator",
        "gen1",
        bus="bus",
        p_nom=100,
        p_max_pu=p_max_pu,
        ramp_limit_up=static_ramp_up * p_max_pu,
        ramp_limit_down=static_ramp_down * p_max_pu,
        marginal_cost=gen1_marginal,
    )

    n.add("Generator", "gen2", bus="bus", p_nom=100, marginal_cost=150)

    n.optimize()

    assert (n.generators_t.p.diff().loc[0:6, "gen1"]).max() <= 0.5 * 80
    assert (n.generators_t.p.diff().loc[0:6, "gen1"]).min() >= -0.5 * 100
    assert (n.generators_t.p.diff().loc[6:, "gen1"]).max() <= 80
    assert (n.generators_t.p.diff().loc[6:, "gen1"]).min() >= -100
