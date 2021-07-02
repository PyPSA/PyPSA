
import pypsa
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal as equal

solver_name = "glpk"


@pytest.mark.parametrize("pyomo", [True, False])
def test_part_load(pyomo):
    """This test is based on
    https://pypsa.org/examples/unit-commitment.html
    and is not very comprehensive."""

    nu = pypsa.Network()

    snapshots = range(4)

    nu.set_snapshots(snapshots)

    nu.add("Bus", "bus")

    nu.add("Generator", "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000)

    nu.add("Generator", "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        p_nom=1000)

    nu.add("Load", "load",
        bus="bus",
        p_set=[4000,6000,5000,800])

    nu.lopf(nu.snapshots, solver_name=solver_name, pyomo=pyomo)

    expected_status = np.array([[1,1,1,0],[0,0,0,1]], dtype=float).T

    equal(nu.generators_t.status.values, expected_status)

    expected_dispatch = np.array([[4000,6000,5000,0],[0,0,0,800]], dtype=float).T

    equal(nu.generators_t.p.values, expected_dispatch)


def test_minimum_up_time():
    """This test is based on
    https://pypsa.org/examples/unit-commitment.html
    and is not very comprehensive."""

    nu = pypsa.Network()

    snapshots = range(4)

    nu.set_snapshots(snapshots)

    nu.add("Bus", "bus")

    nu.add("Generator", "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000)

    nu.add("Generator", "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        up_time_before=0,
        min_up_time=3,
        p_nom=1000)

    nu.add("Load", "load",
        bus="bus",
        p_set=[4000,800,5000,3000])

    nu.lopf(nu.snapshots,solver_name=solver_name)

    expected_status = np.array([[1,0,1,1],[1,1,1,0]],dtype=float).T

    equal(nu.generators_t.status.values,expected_status)

    expected_dispatch = np.array([[3900,0,4900,3000],[100,800,100,0]],dtype=float).T

    equal(nu.generators_t.p.values,expected_dispatch)


def test_minimum_down_time():
    """This test is based on
    https://pypsa.org/examples/unit-commitment.html
    and is not very comprehensive."""


    nu = pypsa.Network()

    nu.set_snapshots(range(4))

    nu.add("Bus", "bus")

    nu.add("Generator", "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        min_down_time=2,
        down_time_before=1,
        p_nom=10000)

    nu.add("Generator", "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        p_nom=4000)

    nu.add("Load", "load",
        bus="bus",
        p_set=[3000,800,3000,8000])

    nu.lopf(nu.snapshots,solver_name=solver_name)

    expected_status = np.array([[0,0,1,1],[1,1,0,0]],dtype=float).T

    equal(nu.generators_t.status.values,expected_status)

    expected_dispatch = np.array([[0,0,3000,8000],[3000,800,0,0]],dtype=float).T

    equal(nu.generators_t.p.values,expected_dispatch)

