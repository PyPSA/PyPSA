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
    def test_csv_io_quotes(self, scipy_network, tmpdir, meta, quotechar="'"):
        fn = tmpdir / "csv_export"
        scipy_network.meta = meta
        scipy_network.export_to_csv_folder(fn, quotechar=quotechar)
        imported = pypsa.Network()
        imported.import_from_csv_folder(fn, quotechar=quotechar)
        assert imported.meta == scipy_network.meta

    def test_csv_io_Path(self, scipy_network, tmpdir):
        fn = tmpdir / "csv_export"
        scipy_network.export_to_csv_folder(fn)
        pypsa.Network(fn)

    def test_csv_io_multiindexed(self, ac_dc_network_mi, tmpdir):
        fn = tmpdir / "csv_export"
        ac_dc_network_mi.export_to_csv_folder(fn)
        m = pypsa.Network(fn)
        pd.testing.assert_frame_equal(
            m.generators_t.p,
            ac_dc_network_mi.generators_t.p,
        )

    def test_csv_io_shapes(self, ac_dc_network_shapes, tmpdir):
        fn = tmpdir / "csv_export"
        ac_dc_network_shapes.export_to_csv_folder(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.shapes,
            ac_dc_network_shapes.shapes,
            check_less_precise=True,
        )

    def test_csv_io_shapes_with_missing(self, ac_dc_network_shapes, tmpdir):
        fn = tmpdir / "csv_export"
        n = ac_dc_network_shapes.copy()
        n.shapes.loc["Manchester", "geometry"] = None
        n.export_to_csv_folder(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.shapes,
            n.shapes,
            check_less_precise=True,
        )

    @pytest.mark.skipif(
        sys.version_info < (3, 13) or sys.platform not in ["linux", "darwin"],
        reason="Unstable test in CI. Remove with 1.0",
    )
    def test_io_equality(self, network_all, tmp_path):
        """
        Test if the network is equal after export and import using CSV format.
        """
        n = network_all
        n.export_to_csv_folder(tmp_path / "network")
        n3 = pypsa.Network(tmp_path / "network")
        assert n.equals(n3, log_mode="strict")


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

    def test_netcdf_io_multiindexed(self, ac_dc_network_mi, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        ac_dc_network_mi.export_to_netcdf(fn)
        m = pypsa.Network(fn)
        pd.testing.assert_frame_equal(
            m.generators_t.p,
            ac_dc_network_mi.generators_t.p,
        )
        pd.testing.assert_frame_equal(
            m.snapshot_weightings,
            ac_dc_network_mi.snapshot_weightings[
                m.snapshot_weightings.columns
            ],  # reset order
        )

    def test_netcdf_io_shapes(self, ac_dc_network_shapes, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        ac_dc_network_shapes.export_to_netcdf(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.shapes,
            ac_dc_network_shapes.shapes,
            check_less_precise=True,
        )

    def test_netcdf_io_shapes_with_missing(self, ac_dc_network_shapes, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        n = ac_dc_network_shapes.copy()
        n.shapes.loc["Manchester", "geometry"] = None
        n.export_to_netcdf(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.shapes,
            n.shapes,
            check_less_precise=True,
        )

    def test_netcdf_from_url(self):
        url = "https://github.com/PyPSA/PyPSA/raw/master/examples/scigrid-de/scigrid-with-load-gen-trafos.nc"
        pypsa.Network(url)

    def test_netcdf_io_no_compression(self, scipy_network, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        scipy_network.export_to_netcdf(fn, float32=False, compression=None)
        scipy_network_compressed = pypsa.Network(fn)
        assert (
            (scipy_network.loads_t.p_set == scipy_network_compressed.loads_t.p_set)
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
                    scipy_network.loads_t.p_set - scipy_network_compressed.loads_t.p_set
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
    def test_io_equality(self, network_all, tmp_path):
        """
        Test if the network is equal after export and import using netCDF format.
        """
        n = network_all
        n.export_to_netcdf(tmp_path / "network.nc")
        n2 = pypsa.Network(tmp_path / "network.nc")
        assert n.equals(n2, log_mode="strict")


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

    def test_hdf5_io_multiindexed(self, ac_dc_network_mi, tmpdir):
        fn = tmpdir / "hdf5_export.h5"
        ac_dc_network_mi.export_to_hdf5(fn)
        m = pypsa.Network(fn)
        pd.testing.assert_frame_equal(
            m.generators_t.p,
            ac_dc_network_mi.generators_t.p,
        )

    def test_hdf5_io_shapes(self, ac_dc_network_shapes, tmpdir):
        fn = tmpdir / "hdf5_export.h5"
        ac_dc_network_shapes.export_to_hdf5(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.shapes,
            ac_dc_network_shapes.shapes,
            check_less_precise=True,
        )

    def test_hdf5_io_shapes_with_missing(self, ac_dc_network_shapes, tmpdir):
        fn = tmpdir / "hdf5_export.h5"
        n = ac_dc_network_shapes.copy()
        n.shapes.loc["Manchester", "geometry"] = None
        n.export_to_hdf5(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.shapes,
            n.shapes,
            check_less_precise=True,
        )

    @pytest.mark.skipif(
        sys.version_info < (3, 13) or sys.platform not in ["linux", "darwin"],
        reason="Unstable test in CI. Remove with 1.0",
    )
    def test_io_equality(self, network_all, tmp_path):
        """
        Test if the network is equal after export and import using HDF5 format.
        """
        n = network_all
        n.export_to_hdf5(tmp_path / "network.h5")
        n5 = pypsa.Network(tmp_path / "network.h5")
        assert n.equals(n5, log_mode="strict")


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

    def test_excel_io_multiindexed(self, ac_dc_network_mi, tmpdir):
        fn = tmpdir / "excel_export.xlsx"
        ac_dc_network_mi.export_to_excel(fn)
        m = pypsa.Network(fn)
        pd.testing.assert_frame_equal(
            m.generators_t.p,
            ac_dc_network_mi.generators_t.p,
        )
        pd.testing.assert_frame_equal(
            m.snapshot_weightings,
            ac_dc_network_mi.snapshot_weightings[m.snapshot_weightings.columns],
            check_dtype=False,  # TODO Remove once validation layer leads to safer types
        )

    def test_excel_io_shapes(self, ac_dc_network_shapes, tmpdir):
        fn = tmpdir / "excel_export.xlsx"
        ac_dc_network_shapes.export_to_excel(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.shapes,
            ac_dc_network_shapes.shapes,
            check_less_precise=True,
        )

    def test_excel_io_shapes_with_missing(self, ac_dc_network_shapes, tmpdir):
        fn = tmpdir / "excel_export.xlsx"
        n = ac_dc_network_shapes.copy()
        n.shapes.loc["Manchester", "geometry"] = None
        n.export_to_excel(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.shapes,
            n.shapes,
            check_less_precise=True,
        )

    @pytest.mark.skipif(
        sys.version_info < (3, 13) or sys.platform not in ["linux", "darwin"],
        reason="Unstable test in CI. Remove with 1.0",
    )
    def test_io_equality(self, network_all, tmp_path):
        """
        Test if the network is equal after export and import using Excel format.
        """
        n = network_all
        n.export_to_excel(tmp_path / "network.xlsx")
        n4 = pypsa.Network(tmp_path / "network.xlsx")
        assert n.equals(n4, log_mode="strict")

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
        assert not m.stores_t.standing_loss.empty
        assert not m.storage_units_t.standing_loss.empty
        assert not m.generators_t.efficiency.empty
        assert not m.storage_units_t.efficiency_store.empty
        assert not m.storage_units_t.efficiency_dispatch.empty
        equal(m.stores_t.standing_loss, n.stores_t.standing_loss)
        equal(m.storage_units_t.standing_loss, n.storage_units_t.standing_loss)
        equal(m.generators_t.efficiency, n.generators_t.efficiency)
        equal(m.storage_units_t.efficiency_store, n.storage_units_t.efficiency_store)
        equal(
            m.storage_units_t.efficiency_dispatch, n.storage_units_t.efficiency_dispatch
        )


@pytest.mark.skipif(
    sys.version_info < (3, 13) or sys.platform not in ["linux", "darwin"],
    reason="Unstable test in CI. Remove with 1.0",
)
def test_io_equality(network_all, tmp_path):
    """
    Test if the network is equal after export and import.
    """
    n = network_all
    n.export_to_netcdf(tmp_path / "network.nc")
    n2 = pypsa.Network(tmp_path / "network.nc")
    assert n.equals(n2, log_mode="strict")

    n.export_to_csv_folder(tmp_path / "network")
    n3 = pypsa.Network(tmp_path / "network")
    assert n.equals(n3, log_mode="strict")

    if excel_installed:
        n.export_to_excel(tmp_path / "network.xlsx")
        n4 = pypsa.Network(tmp_path / "network.xlsx")
        assert n.equals(n4, log_mode="strict")

    if tables_installed:
        n.export_to_hdf5(tmp_path / "network.h5")
        n5 = pypsa.Network(tmp_path / "network.h5")
        assert n.equals(n5, log_mode="strict")


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
        assert len(n.buses) == len(net.bus)
        assert len(n.generators) == (len(net.gen) + len(net.sgen) + len(net.ext_grid))
        assert len(n.loads) == len(net.load)
        assert len(n.transformers) == len(net.trafo)
        assert len(n.shunt_impedances) == len(net.shunt)


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


def test_sort_attrs():
    """Test _sort_attrs function for sorting DataFrame columns/index."""
    from pypsa.network.io import _sort_attrs

    # Test sorting columns (axis=1)
    df = pd.DataFrame(
        {"c": [1, 2, 3], "a": [4, 5, 6], "b": [7, 8, 9], "d": [10, 11, 12]}
    )

    # Sort columns according to attrs_list
    attrs_list = ["a", "b", "c"]
    result = _sort_attrs(df, attrs_list, axis=1)
    expected_order = ["a", "b", "c", "d"]  # d is appended at end
    assert list(result.columns) == expected_order

    # Test with attrs not in DataFrame (should be ignored)
    attrs_list = ["a", "x", "b", "y"]
    result = _sort_attrs(df, attrs_list, axis=1)
    expected_order = ["a", "b", "c", "d"]  # x, y ignored; c, d appended
    assert list(result.columns) == expected_order

    # Test sorting index (axis=0)
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=["c", "a", "b"])
    attrs_list = ["a", "b"]
    result = _sort_attrs(df, attrs_list, axis=0)
    expected_order = ["a", "b", "c"]  # c is appended at end
    assert list(result.index) == expected_order

    # Test empty attrs_list
    result = _sort_attrs(df, [], axis=0)
    assert list(result.index) == ["c", "a", "b"]  # original order preserved

    # Test empty DataFrame
    empty_df = pd.DataFrame()
    result = _sort_attrs(empty_df, ["a", "b"], axis=1)
    assert result.empty
