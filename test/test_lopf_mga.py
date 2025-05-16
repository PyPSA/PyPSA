import pytest
from numpy.testing import assert_almost_equal as almost_equal

import pypsa


def test_mga():
    n = pypsa.Network()

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        marginal_cost=20,
        capital_cost=200,
        p_nom_extendable=True,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        marginal_cost=40,
        capital_cost=230,
        p_nom_extendable=True,
    )

    n.add("Load", "load", bus="bus", p_set=100)

    # can only run MGA on solved networks
    with pytest.raises(ValueError):
        n.optimize.optimize_mga()

    n.optimize()

    opt_capacity = n.generators.p_nom_opt
    opt_cost = (n.statistics.capex() + n.statistics.opex()).sum()

    weights = {"Generator": {"p_nom": {"coal": 1}}}
    slack = 0.05
    n.optimize.optimize_mga(slack=0.05, weights=weights)

    mga_capacity = n.generators.p_nom_opt
    mga_cost = (n.statistics.capex() + n.statistics.opex()).sum()

    assert mga_capacity["coal"] <= opt_capacity["coal"]
    almost_equal(mga_cost / opt_cost, 1 + slack)
