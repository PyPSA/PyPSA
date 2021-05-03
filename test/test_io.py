import pypsa
import pytest
import os

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
    
