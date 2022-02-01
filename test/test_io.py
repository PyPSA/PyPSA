import pypsa
import os
import pandas as pd


def test_netcdf_io(scipy_network, tmpdir):
    fn = os.path.join(tmpdir, "netcdf_export.nc")
    scipy_network.export_to_netcdf(fn)
    pypsa.Network(fn)


def test_csv_io(scipy_network, tmpdir):
    fn = os.path.join(tmpdir, "csv_export")
    scipy_network.export_to_csv_folder(fn)
    pypsa.Network(fn)


def test_hdf5_io(scipy_network, tmpdir):
    fn = os.path.join(tmpdir, "hdf5_export.h5")
    scipy_network.export_to_hdf5(fn)
    pypsa.Network(fn)


def test_netcdf_io_multiindexed(ac_dc_network_multiindexed, tmpdir):
    fn = os.path.join(tmpdir, "netcdf_export.nc")
    ac_dc_network_multiindexed.export_to_netcdf(fn)
    m = pypsa.Network(fn)
    pd.testing.assert_frame_equal(
        m.generators_t.p, ac_dc_network_multiindexed.generators_t.p
    )


def test_csv_io_multiindexed(ac_dc_network_multiindexed, tmpdir):

    fn = os.path.join(tmpdir, "csv_export")
    ac_dc_network_multiindexed.export_to_csv_folder(fn)
    m = pypsa.Network(fn)
    pd.testing.assert_frame_equal(
        m.generators_t.p, ac_dc_network_multiindexed.generators_t.p
    )


def test_hdf5_io_multiindexed(ac_dc_network_multiindexed, tmpdir):
    fn = os.path.join(tmpdir, "hdf5_export.h5")
    ac_dc_network_multiindexed.export_to_hdf5(fn)
    m = pypsa.Network(fn)
    pd.testing.assert_frame_equal(
        m.generators_t.p, ac_dc_network_multiindexed.generators_t.p
    )
