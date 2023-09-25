# -*- coding: utf-8 -*-
import os
from pathlib import Path

import geopandas
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as equal
from shapely.geometry import Point

import pypsa


@pytest.mark.parametrize("meta", [{"test": "test"}, {"test": {"test": "test"}}])
def test_netcdf_io(scipy_network, tmpdir, meta):
    fn = os.path.join(tmpdir, "netcdf_export.nc")
    scipy_network.meta = meta
    scipy_network.export_to_netcdf(fn)
    reloaded = pypsa.Network(fn)
    assert reloaded.meta == scipy_network.meta


def test_geo_wkt_netcdf(geo_components_network, tmpdir):
    """
    This function validates the conversion of shapely.Geometry to and from WKT strings during the process of writing and reading to a netCDF file.

    It appends a GeoSeries to all geo_components and verifies its consistency after writing to and reading from the netCDF file.
    """
    fn = os.path.join(tmpdir, "netcdf_export.nc")

    num_components = len(geo_components_network.buses)

    geo_series = geopandas.GeoSeries([])

    for i in range(num_components):
        geo_series[i] = Point(i, i)

    network_buses_gpd = geo_components_network.buses

    # making sure the index and crs is same as in the network
    geo_series.index = network_buses_gpd.index
    geo_series.crs = network_buses_gpd.crs
    geo_series = geo_series.to_crs(network_buses_gpd.crs)

    geo_components_network.buses.set_geometry(geo_series)
    geo_components_network.lines.set_geometry(geo_series)
    geo_components_network.links.set_geometry(geo_series)
    geo_components_network.transformers.set_geometry(geo_series)

    geo_components_network.export_to_netcdf(fn)
    reloaded = pypsa.Network(fn)

    # geom_euals_exact gives us a array of boolean values for each value of the GeoSeries
    assert False not in reloaded.buses.geometry.geom_equals_exact(geo_series, 0.000001)
    assert False not in reloaded.lines.geometry.geom_equals_exact(geo_series, 0.000001)
    assert False not in reloaded.links.geometry.geom_equals_exact(geo_series, 0.000001)
    assert False not in reloaded.transformers.geometry.geom_equals_exact(
        geo_series, 0.000001
    )


def test_netcdf_io_Path(scipy_network, tmpdir):
    fn = Path(os.path.join(tmpdir, "netcdf_export.nc"))
    scipy_network.export_to_netcdf(fn)
    pypsa.Network(fn)


def test_netcdf_io_datetime(tmpdir):
    fn = os.path.join(tmpdir, "temp.nc")
    exported_sns = pd.date_range(start="2013-03-01", end="2013-03-02", freq="h")
    n = pypsa.Network()
    n.set_snapshots(exported_sns)
    n.export_to_netcdf(fn)
    imported_sns = pypsa.Network(fn).snapshots

    assert (imported_sns == exported_sns).all()


@pytest.mark.parametrize("meta", [{"test": "test"}, {"test": {"test": "test"}}])
def test_csv_io(scipy_network, tmpdir, meta):
    fn = os.path.join(tmpdir, "csv_export")
    scipy_network.meta = meta
    scipy_network.export_to_csv_folder(fn)
    pypsa.Network(fn)
    reloaded = pypsa.Network(fn)
    assert reloaded.meta == scipy_network.meta


def test_csv_io_Path(scipy_network, tmpdir):
    fn = Path(os.path.join(tmpdir, "csv_export"))
    scipy_network.export_to_csv_folder(fn)
    pypsa.Network(fn)


@pytest.mark.parametrize("meta", [{"test": "test"}, {"test": {"test": "test"}}])
def test_hdf5_io(scipy_network, tmpdir, meta):
    fn = os.path.join(tmpdir, "hdf5_export.h5")
    scipy_network.meta = meta
    scipy_network.export_to_hdf5(fn)
    pypsa.Network(fn)
    reloaded = pypsa.Network(fn)
    assert reloaded.meta == scipy_network.meta


def test_hdf5_io_Path(scipy_network, tmpdir):
    fn = Path(os.path.join(tmpdir, "hdf5_export.h5"))
    scipy_network.export_to_hdf5(fn)
    pypsa.Network(fn)


def test_netcdf_io_multiindexed(ac_dc_network_multiindexed, tmpdir):
    fn = os.path.join(tmpdir, "netcdf_export.nc")
    ac_dc_network_multiindexed.export_to_netcdf(fn)
    m = pypsa.Network(fn)
    pd.testing.assert_frame_equal(
        m.generators_t.p,
        ac_dc_network_multiindexed.generators_t.p,
    )
    pd.testing.assert_frame_equal(
        m.snapshot_weightings,
        ac_dc_network_multiindexed.snapshot_weightings[
            m.snapshot_weightings.columns
        ],  # reset order
    )


def test_csv_io_multiindexed(ac_dc_network_multiindexed, tmpdir):
    fn = os.path.join(tmpdir, "csv_export")
    ac_dc_network_multiindexed.export_to_csv_folder(fn)
    m = pypsa.Network(fn)
    pd.testing.assert_frame_equal(
        m.generators_t.p,
        ac_dc_network_multiindexed.generators_t.p,
    )


def test_hdf5_io_multiindexed(ac_dc_network_multiindexed, tmpdir):
    fn = os.path.join(tmpdir, "hdf5_export.h5")
    ac_dc_network_multiindexed.export_to_hdf5(fn)
    m = pypsa.Network(fn)
    pd.testing.assert_frame_equal(
        m.generators_t.p,
        ac_dc_network_multiindexed.generators_t.p,
    )


@pytest.mark.parametrize("use_pandapower_index", [True, False])
@pytest.mark.parametrize("extra_line_data", [True, False])
def test_import_from_pandapower_network(
    pandapower_custom_network,
    pandapower_cigre_network,
    extra_line_data,
    use_pandapower_index,
):
    nets = [pandapower_custom_network, pandapower_cigre_network]
    for net in nets:
        network = pypsa.Network()
        network.import_from_pandapower_net(
            net,
            use_pandapower_index=use_pandapower_index,
            extra_line_data=extra_line_data,
        )
        assert len(network.buses) == len(net.bus)
        assert len(network.generators) == (
            len(net.gen) + len(net.sgen) + len(net.ext_grid)
        )
        assert len(network.loads) == len(net.load)
        assert len(network.transformers) == len(net.trafo)
        assert len(network.shunt_impedances) == len(net.shunt)


def test_netcdf_from_url():
    url = "https://github.com/PyPSA/PyPSA/raw/master/examples/scigrid-de/scigrid-with-load-gen-trafos.nc"
    pypsa.Network(url)


def test_netcdf_io_no_compression(scipy_network, tmpdir):
    fn = os.path.join(tmpdir, "netcdf_export.nc")
    scipy_network.export_to_netcdf(fn, float32=False, compression=None)
    scipy_network_compressed = pypsa.Network(fn)
    assert (
        (scipy_network.loads_t.p_set == scipy_network_compressed.loads_t.p_set)
        .all()
        .all()
    )


def test_netcdf_io_custom_compression(scipy_network, tmpdir):
    fn = os.path.join(tmpdir, "netcdf_export.nc")
    digits = 5
    compression = dict(zlib=True, complevel=9, least_significant_digit=digits)
    scipy_network.export_to_netcdf(fn, compression=compression)
    scipy_network_compressed = pypsa.Network(fn)
    assert (
        (
            (scipy_network.loads_t.p_set - scipy_network_compressed.loads_t.p_set).abs()
            < 1 / 10**digits
        )
        .all()
        .all()
    )


def test_netcdf_io_typecast(scipy_network, tmpdir):
    fn = os.path.join(tmpdir, "netcdf_export.nc")
    scipy_network.export_to_netcdf(fn, float32=True, compression=None)
    pypsa.Network(fn)


def test_netcdf_io_typecast_and_compression(scipy_network, tmpdir):
    fn = os.path.join(tmpdir, "netcdf_export.nc")
    scipy_network.export_to_netcdf(fn, float32=True)
    pypsa.Network(fn)


def test_io_time_dependent_efficiencies(tmpdir):
    n = pypsa.Network()
    s = [1, 0.95, 0.99]
    n.snapshots = range(len(s))
    n.add("Bus", "bus")
    n.add("Generator", "gen", bus="bus", efficiency=s)
    n.add("Store", "sto", bus="bus", standing_loss=s)
    n.add(
        "StorageUnit",
        "su",
        bus="bus",
        efficiency_store=s,
        efficiency_dispatch=s,
        standing_loss=s,
    )

    fn = os.path.join(tmpdir, "network-time-eff.nc")
    n.export_to_netcdf(fn)
    m = pypsa.Network(fn)

    assert not m.stores_t.standing_loss.empty
    assert not m.storage_units_t.standing_loss.empty
    assert not m.generators_t.efficiency.empty
    assert not m.storage_units_t.efficiency_store.empty
    assert not m.storage_units_t.efficiency_dispatch.empty

    equal(m.stores_t.standing_loss, n.stores_t.standing_loss)
    equal(m.storage_units_t.standing_loss, n.storage_units_t.standing_loss)
    equal(m.generators_t.efficiency, n.generators_t.efficiency)
    equal(m.storage_units_t.efficiency_store, n.storage_units_t.efficiency_store)
    equal(m.storage_units_t.efficiency_dispatch, n.storage_units_t.efficiency_dispatch)
