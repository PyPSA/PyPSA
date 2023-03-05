# -*- coding: utf-8 -*-
import numpy as np

import pypsa


def test_344():
    """
    Overridden multi-links but empty n.links.
    """
    override = pypsa.descriptors.Dict(
        {k: v.copy() for k, v in pypsa.components.component_attrs.items()}
    )
    override["Link"].loc["bus2"] = [
        "string",
        np.nan,
        np.nan,
        "2nd bus",
        "Input (optional)",
    ]
    override["Link"].loc["efficiency2"] = [
        "static or series",
        "per unit",
        1.0,
        "2nd bus efficiency",
        "Input (optional)",
    ]
    override["Link"].loc["p2"] = ["series", "MW", 0.0, "2nd bus output", "Output"]

    network = pypsa.Network(override_component_attrs=override)

    network.add("Bus", "a")
    network.add("Load", "a", bus="a", p_set=5)
    network.add("Generator", "a", bus="a", p_nom=5)

    network.lopf(pyomo=False)


def test_331():
    n = pypsa.Network()
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=10)
    n.add("Generator", "generator1", bus="bus", p_nom=15, marginal_cost=10)
    n.lopf(pyomo=False)
    n.add("Generator", "generator2", bus="bus", p_nom=5, marginal_cost=5)
    n.lopf(pyomo=False)
    assert "generator2" in n.generators_t.p


def test_nomansland_bus(caplog):
    n = pypsa.Network()
    n.add("Bus", "bus")

    n.add("Load", "load", bus="bus", p_set=10)
    n.add("Generator", "generator1", bus="bus", p_nom=15, marginal_cost=10)

    n.consistency_check()
    assert (
        "The following buses have no attached components" not in caplog.text
    ), "warning should not trigger..."

    n.add("Bus", "extrabus")

    n.consistency_check()
    assert (
        "The following buses have no attached components" in caplog.text
    ), "warning is not working..."

    try:
        n.lopf(pyomo=False)
    except:
        print("to be fixed - unconnected bus throws error in non-pyomo version.")

    try:
        n.lopf(pyomo=True)
    except:
        print("to be fixed - unconnected bus throws error in pyomo version.")


def test_515():
    """
    Time-varying marginal costs removed.
    """
    marginal_costs = [0, 10]

    n = pypsa.Network()
    n.set_snapshots(range(2))

    n.add("Bus", "bus")
    n.add("Generator", "gen", bus="bus", p_nom=1, marginal_cost=marginal_costs)
    n.add("Load", "load", bus="bus", p_set=1)

    n.lopf(pyomo=False)

    assert n.objective == 10
