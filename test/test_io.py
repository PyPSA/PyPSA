# -*- coding: utf-8 -*-
import os
from pathlib import Path

import pandas as pd
import pytest

import pypsa


@pytest.mark.parametrize("meta", [{"test": "test"}, {"test": {"test": "test"}}])
def test_netcdf_io(scipy_network, tmpdir, meta):
    fn = os.path.join(tmpdir, "netcdf_export.nc")
    scipy_network.meta = meta
    scipy_network.export_to_netcdf(fn)
    reloaded = pypsa.Network(fn)
    assert reloaded.meta == scipy_network.meta


def test_netcdf_io_Path(scipy_network, tmpdir):
    fn = Path(os.path.join(tmpdir, "netcdf_export.nc"))
    scipy_network.export_to_netcdf(fn)
    pypsa.Network(fn)


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
        m.generators_t.p, ac_dc_network_multiindexed.generators_t.p
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
        m.generators_t.p, ac_dc_network_multiindexed.generators_t.p
    )


def test_hdf5_io_multiindexed(ac_dc_network_multiindexed, tmpdir):
    fn = os.path.join(tmpdir, "hdf5_export.h5")
    ac_dc_network_multiindexed.export_to_hdf5(fn)
    m = pypsa.Network(fn)
    pd.testing.assert_frame_equal(
        m.generators_t.p, ac_dc_network_multiindexed.generators_t.p
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
