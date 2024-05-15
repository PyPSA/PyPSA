# -*- coding: utf-8 -*-
import numpy as np

import pypsa


def test_331():
    n = pypsa.Network()
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=10)
    n.add("Generator", "generator1", bus="bus", p_nom=15, marginal_cost=10)
    n.optimize()
    n.add("Generator", "generator2", bus="bus", p_nom=5, marginal_cost=5)
    n.optimize()
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

    n.optimize()


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

    n.optimize()

    assert n.objective == 10


def test_779():
    """
    Importing from xarray dataset.
    """
    n1 = pypsa.Network()
    n1.add("Bus", "bus")
    xarr = n1.export_to_netcdf()
    n2 = pypsa.Network()
    n2.import_from_netcdf(xarr)
