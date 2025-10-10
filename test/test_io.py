# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import sys

import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from numpy.testing import assert_array_almost_equal as equal

import pypsa

try:
    import tables  # noqa: F401

    tables_installed = True
except ImportError:
    tables_installed = False

try:
    import openpyxl  # noqa: F401
    import python_calamine  # noqa: F401

    excel_installed = True
except ImportError:
    excel_installed = False


def custom_equals(n1, n2, ignore_attrs=None):
    """
    Custom equality check that allows certain attributes to be different.
    Parameters
    ----------
    n1, n2 : pypsa.Network
        Networks to compare
    ignore_attrs : list of str, optional
        List of attribute names that are allowed to be different.
    """
    if not ignore_attrs:
        return n1.equals(n2, log_mode="strict")

    # Copy networks to avoid modifying originals
    n1 = n1.copy()
    n2 = n2.copy()

    for attr in ignore_attrs:
        for net in (n1, n2):
            obj = net
            parts = attr.split(".")
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    break
            else:
                if hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], None)

    return n1.equals(n2, log_mode="strict")


# TODO classes could be further parametrized
class TestCSVDir:
    @pytest.mark.parametrize(
        "meta",
        [
            {"test": "test"},
            {"test": "test", "test2": "test2"},
            {"test": {"test": "test", "test2": "test2"}},
        ],
    )
    def test_csv_io(self, scipy_network, tmpdir, meta):
        fn = tmpdir / "csv_export"
        scipy_network.meta = meta
        scipy_network.export_to_csv_folder(fn)
        pypsa.Network(fn)
        reloaded = pypsa.Network(fn)
        assert reloaded.meta == scipy_network.meta

    @pytest.mark.parametrize(
        "meta",
        [
            {"test": "test"},
            {"test": "test", "test2": "test2"},
            {"test": {"test": "test", "test2": "test2"}},
        ],
    )
    def test_csv_io_quotes(self, scipy_network, tmpdir, meta):
        fn = tmpdir / "csv_export"
        scipy_network.meta = meta
        scipy_network.export_to_csv_folder(fn, quotechar="'")
        imported = pypsa.Network()
        imported.import_from_csv_folder(fn, quotechar="'")
        assert imported.meta == scipy_network.meta

    def test_csv_io_Path(self, scipy_network, tmpdir):
        fn = tmpdir / "csv_export"
        scipy_network.export_to_csv_folder(fn)
        pypsa.Network(fn)

    def test_csv_io_multiindexed(self, ac_dc_periods, tmpdir):
        fn = tmpdir / "csv_export"
        ac_dc_periods.export_to_csv_folder(fn)
        m = pypsa.Network(fn)
        pd.testing.assert_frame_equal(
            m.c.generators.dynamic.p,
            ac_dc_periods.c.generators.dynamic.p,
        )

    def test_csv_io_shapes(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "csv_export"
        ac_dc_shapes.export_to_csv_folder(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            ac_dc_shapes.c.shapes.static,
            check_less_precise=True,
        )

    def test_csv_io_shapes_with_missing(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "csv_export"
        n = ac_dc_shapes.copy()
        n.c.shapes.static.loc["Manchester", "geometry"] = None
        n.export_to_csv_folder(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            n.c.shapes.static,
            check_less_precise=True,
        )

    @pytest.mark.skipif(
        sys.version_info < (3, 13) or sys.platform not in ["linux", "darwin"],
        reason="Unstable test in CI. Remove with 1.0",
    )
    def test_io_equality(self, networks_including_solved, tmp_path):
        """
        Test if the network is equal after export and import using CSV format.
        """
        n = networks_including_solved
        if n.has_scenarios:
            with pytest.raises(
                NotImplementedError,
                match="Stochastic networks are not supported*",
            ):
                n.export_to_csv_folder(tmp_path / "network")
            return
        n.export_to_csv_folder(tmp_path / "network")
        n3 = pypsa.Network(tmp_path / "network")
        # Allow difference for solved networks
        # TODO: Remove _components.links with #1128
        ignore = (
            [
                "_components.sub_networks.static.obj",
                "_components.links",
                "_components.lines",
            ]
            if n.model is not None
            else []
        )
        assert custom_equals(n, n3, ignore_attrs=ignore)


class TestNetcdf:
    @pytest.mark.parametrize(
        "meta",
        [
            {"test": "test"},
            {"test": "test", "test2": "test2"},
            {"test": {"test": "test", "test2": "test2"}},
        ],
    )
    def test_netcdf_io(self, scipy_network, tmpdir, meta):
        fn = tmpdir / "netcdf_export.nc"
        scipy_network.meta = meta
        scipy_network.export_to_netcdf(fn)
        reloaded = pypsa.Network(fn)
        assert reloaded.meta == scipy_network.meta

    def test_netcdf_io_Path(self, scipy_network, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        scipy_network.export_to_netcdf(fn)
        pypsa.Network(fn)

    def test_netcdf_io_datetime(self, tmpdir):
        fn = tmpdir / "temp.nc"
        exported_sns = pd.date_range(start="2013-03-01", end="2013-03-02", freq="h")
        n = pypsa.Network()
        n.set_snapshots(exported_sns)
        n.export_to_netcdf(fn)
        imported_sns = pypsa.Network(fn).snapshots

        assert (imported_sns == exported_sns).all()

    def test_netcdf_io_multiindexed(self, ac_dc_periods, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        ac_dc_periods.export_to_netcdf(fn)
        m = pypsa.Network(fn)
        pd.testing.assert_frame_equal(
            m.c.generators.dynamic.p,
            ac_dc_periods.c.generators.dynamic.p,
        )
        pd.testing.assert_frame_equal(
            m.snapshot_weightings,
            ac_dc_periods.snapshot_weightings[
                m.snapshot_weightings.columns
            ],  # reset order
        )

    def test_netcdf_io_shapes(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        ac_dc_shapes.export_to_netcdf(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            ac_dc_shapes.c.shapes.static,
            check_less_precise=True,
        )

    def test_netcdf_io_shapes_with_missing(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        n = ac_dc_shapes.copy()
        n.c.shapes.static.loc["Manchester", "geometry"] = None
        n.export_to_netcdf(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            n.c.shapes.static,
            check_less_precise=True,
        )

    def test_netcdf_from_url(self):
        url = "https://github.com/PyPSA/PyPSA/raw/master/examples/networks/scigrid-de/scigrid-de.nc"
        pypsa.Network(url)

    def test_netcdf_io_no_compression(self, scipy_network, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        scipy_network.export_to_netcdf(fn, float32=False, compression=None)
        scipy_network_compressed = pypsa.Network(fn)
        assert (
            (
                scipy_network.c.loads.dynamic.p_set
                == scipy_network_compressed.c.loads.dynamic.p_set
            )
            .all()
            .all()
        )

    def test_netcdf_io_custom_compression(self, scipy_network, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        digits = 5
        compression = {"zlib": True, "complevel": 9, "least_significant_digit": digits}
        scipy_network.export_to_netcdf(fn, compression=compression)
        scipy_network_compressed = pypsa.Network(fn)
        assert (
            (
                (
                    scipy_network.c.loads.dynamic.p_set
                    - scipy_network_compressed.c.loads.dynamic.p_set
                ).abs()
                < 1 / 10**digits
            )
            .all()
            .all()
        )

    def test_netcdf_io_typecast(self, scipy_network, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        scipy_network.export_to_netcdf(fn, float32=True, compression=None)
        pypsa.Network(fn)

    def test_netcdf_io_typecast_and_compression(self, scipy_network, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        scipy_network.export_to_netcdf(fn, float32=True)
        pypsa.Network(fn)

    @pytest.mark.skipif(
        sys.version_info < (3, 13) or sys.platform not in ["linux", "darwin"],
        reason="Unstable test in CI. Remove with 1.0",
    )
    def test_io_equality(self, networks_including_solved, tmp_path):
        """
        Test if the network is equal after export and import using netCDF format.
        """
        n = networks_including_solved
        n.export_to_netcdf(tmp_path / "network.nc")
        n2 = pypsa.Network(tmp_path / "network.nc")
        # Allow difference for solved networks
        # TODO: Remove _components.links with #1128
        ignore = (
            [
                "_components.sub_networks.static.obj",
                "_components.links",
                "_components.lines",
            ]
            if n.model is not None
            else []
        )
        assert custom_equals(n, n2, ignore_attrs=ignore)


@pytest.mark.skipif(not tables_installed, reason="PyTables not installed")
class TestHDF5:
    @pytest.mark.parametrize(
        "meta",
        [
            {"test": "test"},
            {"test": "test", "test2": "test2"},
            {"test": {"test": "test", "test2": "test2"}},
        ],
    )
    def test_hdf5_io(self, scipy_network, tmpdir, meta):
        fn = tmpdir / "hdf5_export.h5"
        scipy_network.meta = meta
        scipy_network.export_to_hdf5(fn)
        pypsa.Network(fn)
        reloaded = pypsa.Network(fn)
        assert reloaded.meta == scipy_network.meta

    def test_hdf5_io_Path(self, scipy_network, tmpdir):
        fn = tmpdir / "hdf5_export.h5"
        scipy_network.export_to_hdf5(fn)
        pypsa.Network(fn)

    def test_hdf5_io_multiindexed(self, ac_dc_periods, tmpdir):
        fn = tmpdir / "hdf5_export.h5"
        ac_dc_periods.export_to_hdf5(fn)
        m = pypsa.Network(fn)
        pd.testing.assert_frame_equal(
            m.c.generators.dynamic.p,
            ac_dc_periods.c.generators.dynamic.p,
        )

    def test_hdf5_io_shapes(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "hdf5_export.h5"
        ac_dc_shapes.export_to_hdf5(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            ac_dc_shapes.c.shapes.static,
            check_less_precise=True,
        )

    def test_hdf5_io_shapes_with_missing(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "hdf5_export.h5"
        n = ac_dc_shapes.copy()
        n.c.shapes.static.loc["Manchester", "geometry"] = None
        n.export_to_hdf5(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            n.c.shapes.static,
            check_less_precise=True,
        )

    @pytest.mark.skipif(
        sys.version_info < (3, 13) or sys.platform not in ["linux", "darwin"],
        reason="Unstable test in CI. Remove with 1.0",
    )
    def test_io_equality(self, networks_including_solved, tmp_path):
        """
        Test if the network is equal after export and import using HDF5 format.
        """
        n = networks_including_solved
        if n.has_scenarios:
            with pytest.raises(
                NotImplementedError,
                match="Stochastic networks are not supported*",
            ):
                n.export_to_hdf5(tmp_path / "network.h5")
            return
        n.export_to_hdf5(tmp_path / "network.h5")
        n5 = pypsa.Network(tmp_path / "network.h5")
        # Allow difference for solved networks
        # TODO: Remove _components.links with #1128
        ignore = (
            [
                "_components.sub_networks.static.obj",
                "_components.links",
                "_components.lines",
            ]
            if n.model is not None
            else []
        )
        assert custom_equals(n, n5, ignore_attrs=ignore)


@pytest.mark.skipif(not excel_installed, reason="openpyxl not installed")
class TestExcelIO:
    @pytest.mark.parametrize(
        "meta",
        [
            {"test": "test"},
            {"test": "test", "test2": "test2"},
            {"test": {"test": "test", "test2": "test2"}},
        ],
    )
    def test_excel_io(self, scipy_network, tmpdir, meta):
        fn = tmpdir / "excel_export.xlsx"
        scipy_network.meta = meta
        scipy_network.export_to_excel(fn)
        reloaded = pypsa.Network(fn)
        assert reloaded.meta == scipy_network.meta

    def test_excel_io_Path(self, scipy_network, tmpdir):
        fn = tmpdir / "excel_export.xlsx"
        scipy_network.export_to_excel(fn)
        pypsa.Network(fn)

    def test_excel_io_datetime(self, tmpdir):
        fn = tmpdir / "temp.xlsx"
        exported_sns = pd.date_range(start="2013-03-01", end="2013-03-02", freq="h")
        n = pypsa.Network()
        n.set_snapshots(exported_sns)
        n.export_to_excel(fn)
        imported_sns = pypsa.Network(fn).snapshots
        assert (imported_sns == exported_sns).all()

    def test_excel_io_multiindexed(self, ac_dc_periods, tmpdir):
        fn = tmpdir / "excel_export.xlsx"
        ac_dc_periods.export_to_excel(fn)
        m = pypsa.Network(fn)
        pd.testing.assert_frame_equal(
            m.c.generators.dynamic.p,
            ac_dc_periods.c.generators.dynamic.p,
        )
        pd.testing.assert_frame_equal(
            m.snapshot_weightings,
            ac_dc_periods.snapshot_weightings[m.snapshot_weightings.columns],
            check_dtype=False,  # TODO Remove once validation layer leads to safer types
        )

    def test_excel_io_shapes(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "excel_export.xlsx"
        ac_dc_shapes.export_to_excel(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            ac_dc_shapes.c.shapes.static,
            check_less_precise=True,
        )

    def test_excel_io_shapes_with_missing(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "excel_export.xlsx"
        n = ac_dc_shapes.copy()
        n.c.shapes.static.loc["Manchester", "geometry"] = None
        n.export_to_excel(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            n.c.shapes.static,
            check_less_precise=True,
        )

    @pytest.mark.skipif(
        sys.version_info < (3, 13) or sys.platform not in ["linux", "darwin"],
        reason="Unstable test in CI. Remove with 1.0",
    )
    def test_io_equality(self, networks_including_solved, tmp_path):
        """
        Test if the network is equal after export and import using Excel format.
        """
        n = networks_including_solved
        if n.has_scenarios:
            with pytest.raises(
                NotImplementedError,
                match="Stochastic networks are not supported*",
            ):
                n.export_to_excel(tmp_path / "network.xlsx")
            return
        n.export_to_excel(tmp_path / "network.xlsx")
        n4 = pypsa.Network(tmp_path / "network.xlsx")
        # Allow difference for solved networks
        # TODO: Remove _components.links with #1128
        ignore = (
            [
                "_components.sub_networks.static.obj",
                "_components.links",
                "_components.lines",
            ]
            if n.model is not None
            else []
        )
        assert custom_equals(n, n4, ignore_attrs=ignore)

    def test_io_time_dependent_efficiencies_excel(self, tmpdir):
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
        fn = tmpdir / "network-time-eff.xlsx"
        n.export_to_excel(fn)
        m = pypsa.Network(fn)
        assert not m.c.stores.dynamic.standing_loss.empty
        assert not m.c.storage_units.dynamic.standing_loss.empty
        assert not m.c.generators.dynamic.efficiency.empty
        assert not m.c.storage_units.dynamic.efficiency_store.empty
        assert not m.c.storage_units.dynamic.efficiency_dispatch.empty
        equal(m.c.stores.dynamic.standing_loss, n.c.stores.dynamic.standing_loss)
        equal(
            m.c.storage_units.dynamic.standing_loss,
            n.c.storage_units.dynamic.standing_loss,
        )
        equal(m.c.generators.dynamic.efficiency, n.c.generators.dynamic.efficiency)
        equal(
            m.c.storage_units.dynamic.efficiency_store,
            n.c.storage_units.dynamic.efficiency_store,
        )
        equal(
            m.c.storage_units.dynamic.efficiency_dispatch,
            n.c.storage_units.dynamic.efficiency_dispatch,
        )


@pytest.mark.skipif(
    sys.version_info < (3, 13) or sys.platform not in ["linux", "darwin"],
    reason="Unstable test in CI. Remove with 1.0",
)
def test_io_equality(networks_including_solved, tmp_path):
    """
    Test if the network is equal after export and import.
    """
    n = networks_including_solved
    n.export_to_netcdf(tmp_path / "network.nc")
    n2 = pypsa.Network(tmp_path / "network.nc")
    # Allow difference for solved networks
    # TODO: Remove _components.links with #1128
    ignore = (
        [
            "_components.sub_networks.static.obj",
            "_components.links",
            "_components.lines",
        ]
        if n.model is not None
        else []
    )
    assert custom_equals(n, n2, ignore_attrs=ignore)

    # Only check with supported io formats
    if n.has_scenarios:
        return

    n.export_to_csv_folder(tmp_path / "network")
    n3 = pypsa.Network(tmp_path / "network")
    assert custom_equals(n, n3, ignore_attrs=ignore)

    if excel_installed:
        n.export_to_excel(tmp_path / "network.xlsx")
        n4 = pypsa.Network(tmp_path / "network.xlsx")
        assert custom_equals(n, n4, ignore_attrs=ignore)

    if tables_installed:
        n.export_to_hdf5(tmp_path / "network.h5")
        n5 = pypsa.Network(tmp_path / "network.h5")
        assert custom_equals(n, n5, ignore_attrs=ignore)


@pytest.mark.skipif(
    sys.version_info < (3, 13) or sys.platform not in ["linux", "darwin"],
    reason="Only check once since it is an optional test when examples are updated.",
)
@pytest.mark.parametrize(
    "example_network",
    [
        "ac-dc-meshed",
        "storage-hvdc",
        "scigrid-de",
        "model-energy",
    ],
)
def test_examples_against_master(tmp_path, example_network):
    # Test examples are unchanged
    n = pypsa.Network(f"examples/networks/{example_network}/{example_network}")
    # Test examples vs master
    example_network = pypsa.Network(
        f"https://github.com/PyPSA/PyPSA/raw/master/examples/networks/{example_network}/{example_network}.nc"
    )
    assert n.equals(example_network, log_mode="strict")


@pytest.mark.parametrize(
    "example_network",
    [
        "ac-dc-meshed",
        "storage-hvdc",
        "scigrid-de",
        "model-energy",
    ],
)
def test_examples_consistency(tmp_path, example_network):
    # Test examples are unchanged
    n = pypsa.Network(f"examples/networks/{example_network}/{example_network}")
    n.export_to_csv_folder(tmp_path / "network")
    n2 = pypsa.Network(tmp_path / "network")
    assert n.equals(n2, log_mode="strict")


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Test requires Python 3.12 or higher"
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
        n = pypsa.Network()
        n.import_from_pandapower_net(
            net,
            use_pandapower_index=use_pandapower_index,
            extra_line_data=extra_line_data,
        )
        assert len(n.c.buses.static) == len(net.bus)
        assert len(n.c.generators.static) == (
            len(net.gen) + len(net.sgen) + len(net.ext_grid)
        )
        assert len(n.loads) == len(net.load)
        assert len(n.c.transformers.static) == len(net.trafo)
        assert len(n.c.shunt_impedances.static) == len(net.shunt)


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

    fn = tmpdir / "network-time-eff.nc"
    n.export_to_netcdf(fn)
    m = pypsa.Network(fn)

    assert not m.c.stores.dynamic.standing_loss.empty
    assert not m.c.storage_units.dynamic.standing_loss.empty
    assert not m.c.generators.dynamic.efficiency.empty
    assert not m.c.storage_units.dynamic.efficiency_store.empty
    assert not m.c.storage_units.dynamic.efficiency_dispatch.empty

    equal(m.c.stores.dynamic.standing_loss, n.c.stores.dynamic.standing_loss)
    equal(
        m.c.storage_units.dynamic.standing_loss, n.c.storage_units.dynamic.standing_loss
    )
    equal(m.c.generators.dynamic.efficiency, n.c.generators.dynamic.efficiency)
    equal(
        m.c.storage_units.dynamic.efficiency_store,
        n.c.storage_units.dynamic.efficiency_store,
    )
    equal(
        m.c.storage_units.dynamic.efficiency_dispatch,
        n.c.storage_units.dynamic.efficiency_dispatch,
    )


def test_sort_attrs():
    """Ensure _sort_attrs preserves attribute order semantics."""
    from pypsa.network.io import _sort_attrs

    axis_labels = pd.Index(["c", "a", "b", "d"])
    attrs_list = ["a", "b", "c"]
    ordered = _sort_attrs(axis_labels, attrs_list)
    assert list(ordered) == ["a", "b", "c", "d"]

    # Ignore attributes that are not present on the axis
    attrs_list = ["a", "x", "b", "y"]
    ordered = _sort_attrs(axis_labels, attrs_list)
    assert list(ordered) == ["a", "b", "c", "d"]

    # Missing attrs_list should leave order untouched
    ordered = _sort_attrs(axis_labels, [])
    assert ordered.equals(axis_labels)

    # Empty axis behaves like a no-op
    empty_axis = pd.Index([])
    ordered = _sort_attrs(empty_axis, ["a", "b"])
    assert ordered.equals(empty_axis)

    # Works with non-unique Index types (e.g. MultiIndex)
    axis_labels = pd.MultiIndex.from_product([["a", "b"], ["x", "y"]])
    attrs_list = pd.MultiIndex.from_product([["b", "a"], ["y"]])
    ordered = _sort_attrs(axis_labels, attrs_list)
    assert list(ordered) == [
        ("b", "y"),
        ("a", "y"),
        ("a", "x"),
        ("b", "x"),
    ]


def test_version_warning(caplog):
    # Assert no info logged with "version"
    n = pypsa.examples.ac_dc_meshed()
    assert "Importing network from PyPSA version" not in caplog.text

    n._pypsa_version = "0.10.0"
    n.export_to_netcdf("test.nc")
    pypsa.Network("test.nc")
    assert "Importing network from PyPSA version v0.10.0" in caplog.text
