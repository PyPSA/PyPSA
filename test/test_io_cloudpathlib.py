import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import pypsa

pytest.importorskip("cloudpathlib", reason="cloudpathlib not installed")

from cloudpathlib import AnyPath, CloudPath, implementation_registry
from cloudpathlib.local import (
    local_azure_blob_implementation,
    local_gs_implementation,
    local_s3_implementation,
)


@pytest.fixture(
    params=[
        local_s3_implementation,
        local_gs_implementation,
        local_azure_blob_implementation,
    ],
    ids=["s3", "gs", "azure"],
)
def cloudpathlib_local_implementation(request):
    implementation = request.param
    with (
        patch.dict(implementation_registry, {implementation.name: implementation}),
        # NOTE: Bypassing "AzureBlobClient does not support anonymous instantiation"
        patch.dict("os.environ", {"AZURE_STORAGE_CONNECTION_STRING": ""}),
    ):
        yield implementation._path_class
    implementation._client_class.reset_default_storage_dir()


@pytest.fixture
def cloudpath_local_bucket(cloudpathlib_local_implementation):
    local_cls = cloudpathlib_local_implementation
    local_bucket = local_cls(f"{local_cls.cloud_prefix}test-local-bucket")
    assert isinstance(AnyPath(local_bucket), CloudPath)
    return local_bucket


@pytest.fixture
def cloudpath_network_parameterized_ext(request, cloudpath_local_bucket):
    ext = request.param
    cloudpath = cloudpath_local_bucket / f"network{ext}"
    assert cloudpath.suffix == request.param
    return cloudpath


@pytest.fixture(params=[True, False], ids=["uri", "cloudpath"])
def cloudpath_network(request, cloudpath_network_parameterized_ext):
    if request.param:
        return cloudpath_network_parameterized_ext.as_uri()
    return cloudpath_network_parameterized_ext


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows leads to permission errors"
)
class TestIOCloudpath:
    @pytest.mark.parametrize(
        "cloudpath_network_parameterized_ext", [".nc"], indirect=True, ids=["netcdf"]
    )
    def test_io_cloudpath_netcdf(self, cloudpath_network, scipy_network):
        scipy_network.export_to_netcdf(cloudpath_network)
        n = pypsa.Network()
        n.import_from_netcdf(cloudpath_network)

    @pytest.mark.parametrize(
        "cloudpath_network_parameterized_ext", [".h5"], indirect=True, ids=["hdf5"]
    )
    def test_io_cloudpath_hdf5(self, cloudpath_network, scipy_network):
        scipy_network.export_to_hdf5(cloudpath_network)
        # FIXME: why is this needed for hdf5? cloudpathlib is claiming that the local
        # cached file from export is newer on disk than the cloud file being imported
        with patch.dict("os.environ", {"CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD": "1"}):
            n = pypsa.Network()
            n.import_from_hdf5(cloudpath_network)

    @pytest.mark.parametrize(
        "cloudpath_network_parameterized_ext", [""], indirect=True, ids=["csv_folder"]
    )
    def test_io_cloudpath_csv_folder(self, cloudpath_network, scipy_network):
        scipy_network.export_to_csv_folder(cloudpath_network)
        n = pypsa.Network()
        n.import_from_csv_folder(cloudpath_network)


def test_cloudpathlib_anypath_uses_pathlib_path_locally():
    path = AnyPath(".")
    assert isinstance(path, Path)
    assert not isinstance(path, CloudPath)
