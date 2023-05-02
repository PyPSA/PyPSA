# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
from conftest import SUPPORTED_APIS, optimize
from numpy.testing import assert_array_almost_equal as equal

import pypsa


@pytest.mark.parametrize("api", SUPPORTED_APIS)
def test_unit_commitment(api):
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

    optimize(n, api)

    expected_status = np.array([[1, 1, 1, 0], [0, 0, 0, 1]], dtype=float).T

    equal(n.generators_t.status.values, expected_status)

    expected_dispatch = np.array([[4000, 6000, 5000, 0], [0, 0, 0, 800]], dtype=float).T

    equal(n.generators_t.p.values, expected_dispatch)


@pytest.mark.parametrize("api", ["pyomo", "linopy"])
def test_minimum_up_time(api):
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

    optimize(n, api)

    expected_status = np.array([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=float).T

    equal(n.generators_t.status.values, expected_status)

    expected_dispatch = np.array(
        [[3900, 0, 4900, 3000], [100, 800, 100, 0]], dtype=float
    ).T

    equal(n.generators_t.p.values, expected_dispatch)


@pytest.mark.parametrize("api", ["pyomo", "linopy"])
def test_minimum_up_time_up_time_before(api):
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

    optimize(n, api)

    expected_status = np.array([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=float).T

    equal(n.generators_t.status.values, expected_status)

    expected_dispatch = np.array(
        [[3900, 0, 4900, 3000], [100, 800, 100, 0]], dtype=float
    ).T

    equal(n.generators_t.p.values, expected_dispatch)


@pytest.mark.parametrize("api", ["pyomo", "linopy"])
def test_minimum_down_time(api):
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

    optimize(n, api)

    expected_status = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=float).T

    equal(n.generators_t.status.values, expected_status)

    expected_dispatch = np.array([[0, 0, 3000, 8000], [3000, 800, 0, 0]], dtype=float).T

    equal(n.generators_t.p.values, expected_dispatch)


@pytest.mark.parametrize("api", ["pyomo", "linopy"])
def test_minimum_down_time_up_time_before(api):
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

    optimize(n, api)

    expected_status = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=float).T

    equal(n.generators_t.status.values, expected_status)

    expected_dispatch = np.array([[0, 0, 3000, 8000], [3000, 800, 0, 0]], dtype=float).T

    equal(n.generators_t.p.values, expected_dispatch)


@pytest.mark.parametrize("api", ["pyomo", "linopy"])
def test_start_up_costs(api):
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

    optimize(n, api)

    assert n.objective == 359000


@pytest.mark.parametrize("api", ["pyomo", "linopy"])
def test_shut_down_costs(api):
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

    optimize(n, api)

    assert n.objective == 358000


@pytest.mark.parametrize("api", ["pyomo", "linopy"])
def test_unit_commitment_rolling_horizon(api):
    n = pypsa.Network()

    n.snapshots = range(6)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.1,
        up_time_before=0,
        marginal_cost=20,
        min_up_time=3,
        start_up_cost=10000,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=50,
        p_min_pu=0.0,
        up_time_before=1,
        min_down_time=2,
        start_up_cost=1000,
        p_nom=10000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800, 1000, 2000])

    optimize(n, api, snapshots=[0, 1, 2])

    optimize(n, api, snapshots=[3, 4, 5])

    expected_status = np.array([[1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=float).T

    equal(n.generators_t.status.values, expected_status)

    expected_dispatch = np.array(
        [[4000, 6000, 5000, 0, 1000, 2000], [0, 0, 0, 800, 0, 0]]
    ).T

    equal(n.generators_t.p.values, expected_dispatch)


@pytest.mark.parametrize("api", ["linopy"])
def test_linearized_unit_commitment(api):
    n = pypsa.Network()
    n.snapshots = pd.date_range("2022-01-01", "2022-02-09", freq="d")

    load = np.zeros(len(n.snapshots))
    load[0:5] = 5
    load[5:10] = 6
    load[10:15] = 8
    load[15:20] = 10
    load[20:30] = 7
    load[30:40] = 6
    load *= 100

    n.add("Bus", "bus")

    seed = 1
    for i in range(40):
        np.random.seed(seed)
        p_min_pu = np.random.randint(1, 5) / 10
        marginal_cost = np.random.randint(1, 11) * 10
        min_up_time = np.random.randint(0, 6)
        min_down_time = np.random.randint(0, 6)
        p_nom = np.random.randint(1, 10) * 5
        start_up_cost = np.random.randint(1, 5) * 100
        shut_down_cost = np.random.randint(1, 5) * 100

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
        seed += 1

    n.add("Load", "load", bus="bus", p_set=load)

    optimize(n, api, linearized_unit_commitment=True)

    MILP_objective = 1510000
    assert round(n.objective / MILP_objective, 2) == 1


@pytest.mark.parametrize("api", ["linopy"])
def test_link_unit_commitment(api):
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.madd("Bus", ["gas", "electricity"])

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

    optimize(n, api)

    expected_status = [1.0, 1.0, 1.0, 1.0]

    equal(n.links_t.status["OCGT"].values, expected_status)

    expected_dispatch = [3200.0, 5200.0, 600.0, 4200.0]

    equal(-n.links_t.p1["OCGT"].values, expected_dispatch)

    assert round(n.objective, 1) == 267333.0
