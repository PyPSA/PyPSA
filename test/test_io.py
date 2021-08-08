import pypsa
import pytest
import os
import pandas as pd
import numpy as np

@pytest.fixture
def network():
    csv_folder_name = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "scigrid-de",
        "scigrid-with-load-gen-trafos",
    )
    return pypsa.Network(csv_folder_name)


@pytest.fixture
def network_mi():
    csv_folder_name = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ac-dc-meshed",
        "ac-dc-data",
    )
    n = pypsa.Network(csv_folder_name)
    n.snapshots = pd.MultiIndex.from_product([[2013], n.snapshots])
    n.generators_t.p.loc[:,:] = np.random.rand(*n.generators_t.p.shape)
    return n


def test_netcdf_io(network, tmpdir):
    fn = os.path.join(tmpdir, "netcdf_export.nc")
    network.export_to_netcdf(fn)
    pypsa.Network(fn)


def test_csv_io(network, tmpdir):
    fn = os.path.join(tmpdir, "csv_export")
    network.export_to_csv_folder(fn)
    pypsa.Network(fn)


def test_hdf5_io(network, tmpdir):
    fn = os.path.join(tmpdir, "hdf5_export.h5")
    network.export_to_hdf5(fn)
    pypsa.Network(fn)


def test_netcdf_io_multiindexed(network_mi, tmpdir):
    fn = os.path.join(tmpdir, "netcdf_export.nc")
    network_mi.export_to_netcdf(fn)
    m = pypsa.Network(fn)
    pd.testing.assert_frame_equal(m.generators_t.p, network_mi.generators_t.p)


def test_csv_io(network_mi, tmpdir):
    fn = os.path.join(tmpdir, "csv_export")
    network_mi.export_to_csv_folder(fn)
    m = pypsa.Network(fn)
    pd.testing.assert_frame_equal(m.generators_t.p, network_mi.generators_t.p)


def test_hdf5_io(network_mi, tmpdir):
    fn = os.path.join(tmpdir, "hdf5_export.h5")
    network_mi.export_to_hdf5(fn)
    m = pypsa.Network(fn)
    pd.testing.assert_frame_equal(m.generators_t.p, network_mi.generators_t.p)
