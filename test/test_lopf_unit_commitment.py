# -*- coding: utf-8 -*-
import numpy as np
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

    nu = pypsa.Network()

    snapshots = range(4)

    nu.set_snapshots(snapshots)

    nu.add("Bus", "bus")

    nu.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
    )

    nu.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        p_nom=1000,
    )

    nu.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    optimize(nu, api)

    expected_status = np.array([[1, 1, 1, 0], [0, 0, 0, 1]], dtype=float).T

    equal(nu.generators_t.status.values, expected_status)

    expected_dispatch = np.array([[4000, 6000, 5000, 0], [0, 0, 0, 800]], dtype=float).T

    equal(nu.generators_t.p.values, expected_dispatch)


@pytest.mark.parametrize("api", ["pyomo", "linopy"])
def test_minimum_up_time(api):
    """
    This test is based on https://pypsa.readthedocs.io/en/latest/examples/unit-
    commitment.html and is not very comprehensive.
    """

    nu = pypsa.Network()

    snapshots = range(4)

    nu.set_snapshots(snapshots)

    nu.add("Bus", "bus")

    nu.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
    )

    nu.add(
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

    nu.add("Load", "load", bus="bus", p_set=[4000, 800, 5000, 3000])

    optimize(nu, api)

    expected_status = np.array([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=float).T

    equal(nu.generators_t.status.values, expected_status)

    expected_dispatch = np.array(
        [[3900, 0, 4900, 3000], [100, 800, 100, 0]], dtype=float
    ).T

    equal(nu.generators_t.p.values, expected_dispatch)


@pytest.mark.parametrize("api", ["pyomo", "linopy"])
def test_minimum_up_time_up_time_before(api):
    """
    This test is based on https://pypsa.readthedocs.io/en/latest/examples/unit-
    commitment.html and is not very comprehensive.
    """

    nu = pypsa.Network()

    snapshots = range(4)

    nu.set_snapshots(snapshots)

    nu.add("Bus", "bus")

    nu.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
    )

    nu.add(
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

    nu.add("Load", "load", bus="bus", p_set=[4000, 800, 5000, 3000])

    optimize(nu, api)

    expected_status = np.array([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=float).T

    equal(nu.generators_t.status.values, expected_status)

    expected_dispatch = np.array(
        [[3900, 0, 4900, 3000], [100, 800, 100, 0]], dtype=float
    ).T

    equal(nu.generators_t.p.values, expected_dispatch)


@pytest.mark.parametrize("api", ["pyomo", "linopy"])
def test_minimum_down_time(api):
    """
    This test is based on https://pypsa.readthedocs.io/en/latest/examples/unit-
    commitment.html and is not very comprehensive.
    """

    nu = pypsa.Network()

    nu.set_snapshots(range(4))

    nu.add("Bus", "bus")

    nu.add(
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

    nu.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        p_nom=4000,
    )

    nu.add("Load", "load", bus="bus", p_set=[3000, 800, 3000, 8000])

    optimize(nu, api)

    expected_status = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=float).T

    equal(nu.generators_t.status.values, expected_status)

    expected_dispatch = np.array([[0, 0, 3000, 8000], [3000, 800, 0, 0]], dtype=float).T

    equal(nu.generators_t.p.values, expected_dispatch)


@pytest.mark.parametrize("api", ["pyomo", "linopy"])
def test_minimum_down_time_up_time_before(api):
    """
    This test is based on https://pypsa.readthedocs.io/en/latest/examples/unit-
    commitment.html and is not very comprehensive.
    """

    nu = pypsa.Network()

    nu.set_snapshots(range(4))

    nu.add("Bus", "bus")

    nu.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        min_down_time=3,
        down_time_before=2,
        p_nom=10000,
    )

    nu.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        p_nom=4000,
    )

    nu.add("Load", "load", bus="bus", p_set=[3000, 800, 3000, 8000])

    optimize(nu, api)

    expected_status = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=float).T

    equal(nu.generators_t.status.values, expected_status)

    expected_dispatch = np.array([[0, 0, 3000, 8000], [3000, 800, 0, 0]], dtype=float).T

    equal(nu.generators_t.p.values, expected_dispatch)


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
