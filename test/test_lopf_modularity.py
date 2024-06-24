import numpy as np
from numpy.testing import assert_array_almost_equal as equal

import pypsa


def test_modular_components():
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
        "gas",
        bus="bus",
        p_nom_extendable=True,
        marginal_cost=10,
        p_nom_max=5000,
        p_nom_mod=1000,
    )

    n.add(
        "Store",
        "Store_unit",
        bus="bus",
        e_nom_extendable=True,
        e_nom_max=2000,
        e_nom_mod=100,
        capital_cost=100,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 5000, 6000, 800])

    n.optimize()

    expected_p_nom_opt_gen = np.array([5000], dtype=float).T
    expected_e_nom_opt_store = np.array([1000], dtype=float).T

    equal(n.generators.p_nom_opt, expected_p_nom_opt_gen)
    equal(n.stores.e_nom_opt, expected_e_nom_opt_store)
