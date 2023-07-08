# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
from conftest import SUPPORTED_APIS, optimize
from numpy.testing import assert_array_almost_equal as equal

import pypsa


@pytest.mark.parametrize("api", SUPPORTED_APIS)
def test_stand_by_cost(api):
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
        p_nom=1000,
    )

    n.add(
        "Store",
        "Store_unit",
        bus="bus",
        e_nom_extendable=True,
        e_nom_max=2000,
        e_nom=100,
        capital_cost=100,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 5000, 6000, 800])

    n.lopf(n.snapshots, pyomo=False)

    expected_n_opt_gen = np.array([5], dtype=float).T
    expected_n_opt_store = np.array([10], dtype=float).T

    equal(n.generators.n_opt, expected_n_opt_gen)
    equal(n.stores.n_opt, expected_n_opt_store)
