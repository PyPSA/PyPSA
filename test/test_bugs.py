import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal as almost_equal

import pypsa


def test_1144():
    """
    See https://github.com/PyPSA/PyPSA/issues/1144.
    """
    n = pypsa.examples.ac_dc_meshed()
    n.generators["build_year"] = [2020, 2020, 2030, 2030, 2040, 2040]
    n.investment_periods = [2020, 2030, 2040]
    capacity = n.statistics.installed_capacity(comps="Generator")
    assert capacity[2020].sum() < capacity[2030].sum() < capacity[2040].sum()


def test_890():
    """
    See https://github.com/PyPSA/PyPSA/issues/890.
    """
    n = pypsa.examples.scigrid_de()
    n.calculate_dependent_values()

    n.lines = n.lines.reindex(columns=n.components["Line"]["attrs"].index[1:])
    n.lines["type"] = np.nan
    n.buses = n.buses.reindex(columns=n.components["Bus"]["attrs"].index[1:])
    n.buses["frequency"] = 50

    n.set_investment_periods([2020, 2030])

    weighting = pd.Series(1, n.buses.index)
    busmap = n.cluster.busmap_by_kmeans(bus_weightings=weighting, n_clusters=50)
    nc = n.cluster.cluster_by_busmap(busmap)

    C = n.cluster.get_clustering_from_busmap(busmap)
    nc = C.n

    almost_equal(n.investment_periods, nc.investment_periods)
    almost_equal(n.investment_period_weightings, nc.investment_period_weightings)


def test_331():
    """
    See https://github.com/PyPSA/PyPSA/issues/331.
    """
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
    assert "The following buses have no attached components" not in caplog.text, (
        "warning should not trigger..."
    )

    n.add("Bus", "extrabus")

    n.consistency_check()
    assert "The following buses have no attached components" in caplog.text, (
        "warning is not working..."
    )

    n.optimize()


def test_515():
    """
    Time-varying marginal costs removed.
    See https://github.com/PyPSA/PyPSA/issues/515.
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
    See https://github.com/PyPSA/PyPSA/issues/779.
    """
    n1 = pypsa.Network()
    n1.add("Bus", "bus")
    xarr = n1.export_to_netcdf()
    n2 = pypsa.Network()
    n2.import_from_netcdf(xarr)


def test_multiport_assignment_defaults_single_add():
    """
    Add a single link to a network, then add a second link with additional
    ports.

    Check that the default values are assigned to the first link.
    """
    n = pypsa.Network()
    n.add("Bus", "bus")
    n.add("Bus", "bus2")
    n.add("Link", "link", bus0="bus", bus1="bus2")
    n.add("Link", "link2", bus0="bus", bus1="bus2", bus2="bus")
    assert n.links.loc["link", "bus2"] == ""


def test_multiport_assignment_defaults_multiple_add():
    """
    Add a single link to a network, then add a second link with additional
    ports.

    Check that the default values are assigned to the first link.
    """
    n = pypsa.Network()
    n.add("Bus", "bus")
    n.add("Bus", "bus2")
    n.add("Link", ["link"], bus0="bus", bus1="bus2")
    n.add("Link", ["link2"], bus0="bus", bus1="bus2", bus2="bus")
    assert n.links.loc["link", "bus2"] == ""
