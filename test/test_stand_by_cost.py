import numpy as np
from numpy.testing import assert_array_almost_equal as equal

import pypsa


def test_stand_by_cost():
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
        stand_by_cost=10,
        p_min_pu=0.1,
        p_nom=1000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    n.optimize()

    cost = (
        n.generators_t.p * n.generators.marginal_cost
        + (
            n.generators_t.status.reindex(columns=n.generators.index, fill_value=0)
            * n.generators.stand_by_cost
        )
    ).mul(n.snapshot_weightings.generators, axis=0)

    expected_cost = np.array([80000, 120000, 100000, 56010], dtype=float).T

    equal(cost.sum(1), expected_cost)

    equal(sum(expected_cost), n.objective)
