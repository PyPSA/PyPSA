# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as almost_equal

import pypsa

try:
    import openpyxl  # noqa: F401
    import python_calamine  # noqa: F401

    excel_installed = True
except ImportError:
    excel_installed = False


def test_1144():
    """
    See https://github.com/PyPSA/PyPSA/issues/1144.
    """
    n = pypsa.examples.ac_dc_meshed()
    n.c.generators.static["build_year"] = [2020, 2020, 2030, 2030, 2040, 2040]
    n.investment_periods = [2020, 2030, 2040]
    capacity = n.statistics.installed_capacity(components="Generator")
    assert capacity[2020].sum() < capacity[2030].sum() < capacity[2040].sum()


def test_890():
    """
    See https://github.com/PyPSA/PyPSA/issues/890.
    """
    n = pypsa.examples.scigrid_de()
    n.calculate_dependent_values()

    n.c.lines.static = n.c.lines.static.reindex(
        columns=n.components["Line"]["defaults"].index[1:]
    )
    n.c.lines.static["type"] = np.nan
    n.c.buses.static = n.c.buses.static.reindex(
        columns=n.components["Bus"]["defaults"].index[1:]
    )
    n.c.buses.static["frequency"] = 50

    n.set_investment_periods([2020, 2030])

    weighting = pd.Series(1, n.c.buses.static.index)
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
    assert "generator2" in n.c.generators.dynamic.p


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
    assert n.c.links.static.loc["link", "bus2"] == ""


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
    assert n.c.links.static.loc["link", "bus2"] == ""


@pytest.mark.skipif(not excel_installed, reason="openpyxl not installed")
def test_1268(tmpdir):
    """
    Excel import without snapshots sheet should not raise KeyError.
    See https://github.com/PyPSA/PyPSA/issues/1268.
    """
    fn = str(tmpdir / "no_snapshots.xlsx")

    buses = pd.DataFrame({"v_nom": [132]}, index=["bus1"])
    with pd.ExcelWriter(fn, engine="openpyxl") as writer:
        buses.to_excel(writer, sheet_name="buses")

    n = pypsa.Network()
    n.import_from_excel(fn)
    assert len(n.c.buses.static) == 1


def test_1319():
    """
    Copying a solved network should work after setting solver_model to None.
    See https://github.com/PyPSA/PyPSA/issues/1319.
    """
    n = pypsa.examples.ac_dc_meshed()
    n.optimize()

    # Should raise error when trying to copy with solver_model attached
    with pytest.raises(
        ValueError, match="Copying a solved network with an attached solver model"
    ):
        n.copy()

    # Should work after setting solver_model to None
    n.model.solver_model = None
    n_copy = n.copy()  # Should not raise an error
    assert n_copy is not n
    assert len(n_copy.buses) == len(n.c.buses.static)


def test_1411():
    """
    Investment periods should sync when setting snapshots with MultiIndex.
    See https://github.com/PyPSA/PyPSA/issues/1411.
    """
    # Test 1: Setting MultiIndex snapshots directly on fresh network
    n = pypsa.Network()
    n.add("Bus", "bus0")
    timesteps = pd.date_range("2013-03-01", periods=3, freq="D")
    snapshots = pd.MultiIndex.from_product(
        [[2020, 2030], timesteps], names=["period", "timestep"]
    )
    n.set_snapshots(snapshots)

    # Check that _investment_periods_data is synchronized with both periods
    assert n._investment_periods_data.index.tolist() == [2020, 2030]

    # Test 2: Extending snapshots to add new period
    n2 = pypsa.Network()
    n2.set_snapshots(pd.date_range("2013-03-01", periods=3, freq="D"))
    n2.add("Bus", "bus0")

    # Convert to multi-period with period 0
    snapshots_multi = pd.MultiIndex.from_product(
        [[0], n2.snapshots], names=["period", "timestep"]
    )
    n2.set_snapshots(snapshots_multi)
    n2.set_investment_periods([0])

    # Check that after set_investment_periods, data is synchronized
    assert n2._investment_periods_data.index.tolist() == [0]

    # Extend snapshots to add period 2040
    new_snapshots = pd.date_range("2013-03-01", periods=3, freq="D")
    extended = pd.MultiIndex.from_tuples(
        list(n2.snapshots) + [(2040, t) for t in new_snapshots],
        names=["period", "timestep"],
    )
    n2.set_snapshots(extended)

    # Check that _investment_periods_data is synchronized
    assert n2._investment_periods_data.index.tolist() == [0, 2040]


def test_1420(tmp_path):
    """
    Network pickling should not cause RecursionError in xarray accessor.
    See https://github.com/PyPSA/PyPSA/issues/1420.
    """
    n = pypsa.Network()
    n.add("Bus", "bus")
    n.add("Generator", "gen", bus="bus", p_nom=100)

    pickle_file = tmp_path / "network.pkl"

    with Path(pickle_file).open("wb") as out:
        pickle.dump(n, out)

    with Path(pickle_file).open("rb") as inp:
        n_loaded = pickle.load(inp)

    # Verify network was loaded correctly
    assert len(n_loaded.c.buses.static) == 1
    assert len(n_loaded.c.generators.static) == 1
    # tmp_path is automatically cleaned up by pytest
